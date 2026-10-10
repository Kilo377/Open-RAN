"""Generate a real LLM candidate corpus for Experiment 2.

The output JSONL can be consumed by ``run_candidate_repair_eval.py`` through
``candidate_pool.corpus_path``. This keeps API sampling separate from repair
evaluation, so a corpus can be frozen and audited before producing tables.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import queue
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiments.candidate_repair_eval import (
    PARAM_POLICY_LIBRARY,
    CandidateRepairFixture,
    build_default_available_xapps,
    build_exp2_chunk_descriptors,
    build_exp2_intent_templates,
    build_exp2_scenario_descriptors,
    candidate_fixture_from_record,
    candidate_fixture_to_record,
    write_candidate_fixture_jsonl,
)
from models.intent_orchestration import RewardParameterRange
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
from ric.non_rt.ollama_gptoss_client import StructuredJsonLlmClient


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "candidate_repair" / "exp2_llm_candidate_corpus_v1" / "candidates.jsonl"
DEFAULT_RAW_OUTPUT = (
    PROJECT_ROOT / "outputs" / "candidate_repair" / "exp2_llm_candidate_corpus_v1" / "raw_responses.jsonl"
)
DEFAULT_PROMPT_MODES = ["schema_only", "memory_retrieved", "noisy_memory"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real LLM candidate fixtures for Experiment 2.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT))
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--api-key-env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-s", type=float, default=35.0)
    parser.add_argument("--startup-timeout-s", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=2500)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument(
        "--hard-timeout-s",
        type=float,
        default=0.0,
        help="Optional process-level timeout per LLM request; timed-out requests are recorded and sampling continues.",
    )
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Do not request OpenAI-compatible JSON object response formatting.",
    )
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--samples-per-context", type=int, default=2)
    parser.add_argument("--prompt-modes", nargs="*", default=list(DEFAULT_PROMPT_MODES))
    parser.add_argument("--chunk-ids", nargs="*", default=[])
    parser.add_argument("--intent-slugs", nargs="*", default=[])
    parser.add_argument("--scenario-ids", nargs="*", default=[])
    parser.add_argument(
        "--limit-contexts",
        type=int,
        default=0,
        help="Optional cap over chunk/intent/scenario contexts for pilot runs.",
    )
    parser.add_argument(
        "--context-offset",
        type=int,
        default=0,
        help="Skip this many shuffled chunk/intent/scenario contexts before sampling.",
    )
    parser.add_argument(
        "--context-count",
        type=int,
        default=0,
        help="Sample this many shuffled contexts after --context-offset. Overrides --limit-contexts when nonzero.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSONL files, skipping successful context/sample records and retrying failures.",
    )
    parser.add_argument(
        "--no-flush-each",
        action="store_true",
        help="Only write output files at the end instead of after every request.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_by_attr(items: list[Any], attr_name: str, selected_values: list[str]) -> list[Any]:
    selected = {str(item).strip() for item in list(selected_values or []) if str(item).strip() != ""}
    if len(selected) == 0:
        return list(items)
    return [item for item in items if str(getattr(item, attr_name, "") or "") in selected]


def _xapp_catalog_text(available_xapps: list[str], *, max_variants_per_category: int = 4) -> str:
    by_category: dict[str, list[str]] = {}
    for xapp_name in sorted(set(available_xapps)):
        by_category.setdefault(canonical_xapp_category(xapp_name), []).append(xapp_name)
    rows = []
    for category, variants in sorted(by_category.items()):
        rows.append(f"- {category}: {', '.join(variants[:max_variants_per_category])}")
    return "\n".join(rows)


def _policy_text() -> str:
    rows = []
    for field_name, policy in sorted(PARAM_POLICY_LIBRARY.items()):
        if "values" in policy:
            rows.append(f"- {field_name}: discrete {list(policy.get('values', []) or [])}")
        else:
            rows.append(
                f"- {field_name}: numeric [{policy.get('lower', '')}, {policy.get('upper', '')}]"
            )
    return "\n".join(rows)


def _memory_hint(prompt_mode: str, *, chunk_tags: list[str], scenario_tags: list[str], intent_family: str) -> str:
    tags = set(chunk_tags + scenario_tags)
    if prompt_mode == "schema_only":
        return "No retrieved memory is available. Use only the schema and current context."
    useful = [
        "Past successful candidates kept hard KPI guard xApps when BLER or queue stress was high.",
        "Past failed candidates over-selected resource-heavy xApps and created resource overflow.",
        "Transfer guard avoided aggressive steering when high_mobility and blockage tags appeared together.",
    ]
    if "energy_saving" in tags:
        useful.append("Energy-saving memory succeeded only when mobility and URLLC protection stayed active.")
    if intent_family == "urllc_reliability":
        useful.append("URLLC memory favors latency scheduler plus URLLC guard before capacity-only actions.")
    if prompt_mode == "noisy_memory":
        useful.append("Noisy retrieved memory also suggests deep sleep and aggressive steering, but it may be stale.")
    return " ".join(useful)


def _system_prompt() -> str:
    return (
        "You are a Non-RT RIC orchestration planner. Produce one JSON candidate "
        "for xApp selection and tunable reward/contract bounds. The candidate may "
        "contain mistakes; a downstream repair compiler will validate it. "
        "Your entire response must be one JSON object. The first character must be "
        "{ and the final character must be }. Do not include prose, markdown, or analysis."
    )


def _user_prompt(
    *,
    chunk: Any,
    intent: Any,
    scenario: Any,
    prompt_mode: str,
    sample_index: int,
    available_xapps: list[str],
) -> str:
    return (
        "Return exactly one JSON object with this exact shape. Do not write any text before or after it:\n"
        "{\n"
        '  "selected_xapps": ["xapp_variant_name", "..."],\n'
        '  "parameter_bounds": [\n'
        '    {"target_xapp_name": "xapp_variant_name", "field_name": "latency_bias", '
        '"lower_bound": 1.0, "upper_bound": 1.4, "discrete_values": []}\n'
        "  ],\n"
        '  "rationale": "short reason",\n'
        '  "issue_hint": "natural_llm_candidate"\n'
        "}\n\n"
        f"Context id: {chunk.chunk_id}:{intent.intent_slug}:{scenario.scenario_id}:{prompt_mode}:{sample_index}\n"
        f"Chunk: {chunk.name}; tags={chunk.tags}\n"
        f"Intent family: {intent.intent_family}\n"
        f"Intent text: {intent.intent_text}\n"
        f"Risk posture: {intent.risk_posture}\n"
        f"Scenario: {scenario.scenario_id}; tags={scenario.tags}; observation={scenario.observation_overrides}\n"
        f"Prompt mode: {prompt_mode}\n"
        f"Retrieved memory hint: {_memory_hint(prompt_mode, chunk_tags=chunk.tags, scenario_tags=scenario.tags, intent_family=intent.intent_family)}\n\n"
        "Available xApp variants by category:\n"
        f"{_xapp_catalog_text(available_xapps)}\n\n"
        "Supported tunable fields:\n"
        f"{_policy_text()}\n\n"
        "Use 2 to 6 xApps. Prefer variant names from the catalog. "
        "Bounds must target selected xApps when possible. "
        "For discrete fields, use discrete_values and omit numeric bounds."
    )


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _float_or_none(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _range_from_llm(raw: object) -> RewardParameterRange | None:
    if not isinstance(raw, dict):
        return None
    field_name = str(raw.get("field_name", "") or raw.get("field", "") or "").strip()
    target_xapp_name = str(raw.get("target_xapp_name", "") or raw.get("xapp", "") or "").strip()
    if field_name == "":
        return None
    return RewardParameterRange(
        field_name=field_name,
        target_xapp_name=target_xapp_name or None,
        scope=str(raw.get("scope", "") or "cluster"),
        lower_bound=_float_or_none(raw.get("lower_bound", raw.get("lower"))),
        upper_bound=_float_or_none(raw.get("upper_bound", raw.get("upper"))),
        step=_float_or_none(raw.get("step")),
        discrete_values=list(raw.get("discrete_values", raw.get("values", [])) or []),
        blocked_discrete_values=list(raw.get("blocked_discrete_values", []) or []),
        max_delta=_float_or_none(raw.get("max_delta")),
        tuning_mode=str(raw.get("tuning_mode", "") or "llm_candidate"),
    )


def _fixture_from_llm_payload(
    *,
    payload: dict[str, Any],
    chunk: Any,
    intent: Any,
    scenario: Any,
    prompt_mode: str,
    sample_index: int,
) -> CandidateRepairFixture:
    xapps = [
        str(item).strip()
        for item in _as_list(payload.get("selected_xapps", payload.get("xapps", [])))
        if str(item).strip() != ""
    ]
    if len(xapps) == 0 and str(payload.get("target_xapp_name", "") or "").strip() != "":
        xapps = [str(payload.get("target_xapp_name", "") or "").strip()]
    raw_bounds = payload.get("parameter_bounds", payload.get("bounds", []))
    if not raw_bounds and str(payload.get("field_name", "") or "").strip() != "":
        raw_bounds = [payload]
    bounds = [
        item
        for item in (_range_from_llm(raw) for raw in _as_list(raw_bounds))
        if item is not None
    ]
    issue_hint = str(payload.get("issue_hint", "") or "natural_llm_candidate").strip()
    candidate_id = f"{chunk.chunk_id}:{intent.intent_slug}:{scenario.scenario_id}:{prompt_mode}:s{int(sample_index)}"
    return CandidateRepairFixture(
        candidate_id=candidate_id,
        issue_type=issue_hint,
        requested_xapps=xapps,
        requested_bounds=bounds,
        chunk_id=chunk.chunk_id,
        intent_id=intent.intent_id,
        intent_slug=intent.intent_slug,
        intent_family=intent.intent_family,
        scenario_id=f"{chunk.chunk_id}:{scenario.scenario_id}",
        scenario_tags=list(dict.fromkeys(list(chunk.tags) + list(scenario.tags))),
        candidate_source=f"llm_{prompt_mode}",
        stress_error_types=[],
        notes=str(payload.get("rationale", "") or ""),
    )


def _contexts(args: argparse.Namespace) -> list[tuple[Any, Any, Any]]:
    chunks = _selected_by_attr(build_exp2_chunk_descriptors(), "chunk_id", list(args.chunk_ids or []))
    intents = _selected_by_attr(build_exp2_intent_templates(), "intent_slug", list(args.intent_slugs or []))
    scenarios = _selected_by_attr(build_exp2_scenario_descriptors(), "scenario_id", list(args.scenario_ids or []))
    contexts = [(chunk, intent, scenario) for chunk in chunks for intent in intents for scenario in scenarios]
    rng = random.Random(int(args.seed))
    rng.shuffle(contexts)
    offset = max(0, int(args.context_offset or 0))
    if offset > 0:
        contexts = contexts[offset:]
    context_count = int(args.context_count or 0)
    if context_count <= 0:
        context_count = int(args.limit_contexts or 0)
    if context_count > 0:
        contexts = contexts[:context_count]
    return contexts


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")


def _load_resume_state(
    *,
    output_path: Path,
    raw_output_path: Path,
    enabled: bool,
) -> tuple[dict[str, CandidateRepairFixture], dict[str, dict[str, Any]]]:
    if not enabled:
        return {}, {}
    fixtures_by_id: dict[str, CandidateRepairFixture] = {}
    for record in _read_jsonl_records(output_path):
        fixture = candidate_fixture_from_record(record)
        if fixture.candidate_id != "":
            fixtures_by_id[fixture.candidate_id] = fixture
    raw_by_context: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl_records(raw_output_path):
        context_id = str(record.get("context_id", "") or "")
        if context_id != "":
            raw_by_context[context_id] = record
    return fixtures_by_id, raw_by_context


def _flush_outputs(
    *,
    output_path: Path,
    raw_output_path: Path,
    fixtures_by_id: dict[str, CandidateRepairFixture],
    raw_by_context: dict[str, dict[str, Any]],
) -> None:
    fixtures = [fixtures_by_id[key] for key in sorted(fixtures_by_id)]
    raw_records = [raw_by_context[key] for key in sorted(raw_by_context)]
    write_candidate_fixture_jsonl(output_path, fixtures)
    _write_jsonl_records(raw_output_path, raw_records)


def _llm_request_worker(
    result_queue: Any,
    client_kwargs: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> None:
    try:
        payload = StructuredJsonLlmClient(**client_kwargs).generate_json_or_raise(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        result_queue.put(("ok", payload))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _generate_with_hard_timeout(
    *,
    client_kwargs: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    hard_timeout_s: float,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_llm_request_worker,
        args=(result_queue, client_kwargs, system_prompt, user_prompt),
    )
    process.start()
    process.join(timeout=max(1.0, float(hard_timeout_s)))
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        result_queue.close()
        raise TimeoutError(f"LLM request exceeded hard timeout of {float(hard_timeout_s):.1f}s")
    try:
        status, result = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(
            f"LLM request worker exited with code {process.exitcode} without a result"
        ) from exc
    finally:
        result_queue.close()
    if status != "ok":
        raise RuntimeError(str(result))
    if not isinstance(result, dict):
        raise RuntimeError("LLM request worker returned a non-object JSON payload")
    return result


def main() -> None:
    args = _parse_args()
    prompt_modes = [str(item).strip() for item in list(args.prompt_modes or []) if str(item).strip() != ""]
    contexts = _contexts(args)
    planned = len(contexts) * len(prompt_modes) * max(1, int(args.samples_per_context))
    output_path = Path(str(args.output)).resolve()
    raw_output_path = Path(str(args.raw_output)).resolve()
    fixtures_by_id, raw_by_context = _load_resume_state(
        output_path=output_path,
        raw_output_path=raw_output_path,
        enabled=bool(args.resume),
    )
    completed_ok = sum(1 for record in raw_by_context.values() if bool(record.get("ok", False)))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned_candidates": planned,
                    "context_count": len(contexts),
                    "context_offset": int(args.context_offset or 0),
                    "context_count_limit": int(args.context_count or args.limit_contexts or 0),
                    "prompt_modes": prompt_modes,
                    "samples_per_context": int(args.samples_per_context),
                    "output": str(output_path),
                    "raw_output": str(raw_output_path),
                    "resume": bool(args.resume),
                    "existing_fixture_count": len(fixtures_by_id),
                    "existing_successful_raw_count": completed_ok,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    extra_body: dict[str, Any] = {}
    if not bool(args.no_json_response_format):
        extra_body["response_format"] = {"type": "json_object"}
    client_kwargs: dict[str, Any] = {
        "provider": str(args.provider),
        "model_name": str(args.model),
        "api_key_env_var": str(args.api_key_env_var),
        "temperature": float(args.temperature),
        "timeout_s": float(args.timeout_s),
        "startup_timeout_s": float(args.startup_timeout_s),
        "max_output_tokens": int(args.max_output_tokens),
        "request_retries": int(args.request_retries),
        "retry_backoff_s": 1.0,
        "reasoning_effort": str(args.reasoning_effort),
        "extra_body": extra_body,
    }
    client = StructuredJsonLlmClient(
        **client_kwargs,
    )
    available_xapps = build_default_available_xapps()
    completed = 0
    skipped = 0
    for chunk, intent, scenario in contexts:
        for prompt_mode in prompt_modes:
            for sample_index in range(1, max(1, int(args.samples_per_context)) + 1):
                completed += 1
                context_id = f"{chunk.chunk_id}:{intent.intent_slug}:{scenario.scenario_id}:{prompt_mode}:s{sample_index}"
                if bool(raw_by_context.get(context_id, {}).get("ok", False)):
                    skipped += 1
                    print(f"[exp2-llm-corpus] skip {completed}/{planned} {context_id}")
                    continue
                print(f"[exp2-llm-corpus] {completed}/{planned} {context_id}")
                user_prompt = _user_prompt(
                    chunk=chunk,
                    intent=intent,
                    scenario=scenario,
                    prompt_mode=prompt_mode,
                    sample_index=sample_index,
                    available_xapps=available_xapps,
                )
                raw_record: dict[str, Any] = {
                    "context_id": context_id,
                    "chunk_id": chunk.chunk_id,
                    "intent_slug": intent.intent_slug,
                    "scenario_id": scenario.scenario_id,
                    "prompt_mode": prompt_mode,
                    "sample_index": sample_index,
                    "ok": False,
                }
                try:
                    if float(args.hard_timeout_s or 0.0) > 0.0:
                        payload = _generate_with_hard_timeout(
                            client_kwargs=client_kwargs,
                            system_prompt=_system_prompt(),
                            user_prompt=user_prompt,
                            hard_timeout_s=float(args.hard_timeout_s),
                        )
                    else:
                        payload = client.generate_json_or_raise(
                            system_prompt=_system_prompt(),
                            user_prompt=user_prompt,
                        )
                    fixture = _fixture_from_llm_payload(
                        payload=payload,
                        chunk=chunk,
                        intent=intent,
                        scenario=scenario,
                        prompt_mode=prompt_mode,
                        sample_index=sample_index,
                    )
                    fixtures_by_id[fixture.candidate_id] = fixture
                    raw_record.update({"ok": True, "payload": payload, "candidate": candidate_fixture_to_record(fixture)})
                except Exception as exc:  # pragma: no cover - exercised by live API runs
                    raw_record.update({"error": str(exc)})
                    print(f"[exp2-llm-corpus] failed {context_id}: {exc}")
                raw_by_context[context_id] = raw_record
                if not bool(args.no_flush_each):
                    _flush_outputs(
                        output_path=output_path,
                        raw_output_path=raw_output_path,
                        fixtures_by_id=fixtures_by_id,
                        raw_by_context=raw_by_context,
                    )

    _flush_outputs(
        output_path=output_path,
        raw_output_path=raw_output_path,
        fixtures_by_id=fixtures_by_id,
        raw_by_context=raw_by_context,
    )
    print(f"Skipped {skipped} completed context/sample records.")
    print(f"Wrote {len(fixtures_by_id)} LLM candidate fixtures to {output_path}")
    print(f"Wrote {len(raw_by_context)} raw response records to {raw_output_path}")


if __name__ == "__main__":
    main()
