"""Structured JSON LLM client used by intent-oriented rApps."""

from __future__ import annotations

import json
import importlib
import os
import re
import time
from typing import Any
from urllib import request

from ric.non_rt.intent_orchestration_helpers import extract_json_object

class StructuredJsonLlmClient:
    """Small JSON-oriented client for Ollama or OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        provider: str = "ollama",
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 20.0,
        startup_timeout_s: float | None = None,
        max_output_tokens: int = 384,
        request_retries: int = 1,
        retry_backoff_s: float = 1.0,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.provider = str(provider or "ollama").strip().lower()
        if self.provider not in {"ollama", "openai_compatible", "deepseek"}:
            raise ValueError(
                "Unsupported LLM provider. Expected 'ollama', 'openai_compatible', or 'deepseek', "
                f"received '{self.provider}'."
            )
        default_base_url = (
            "http://127.0.0.1:11434"
            if self.provider == "ollama"
            else "https://api.deepseek.com"
            if self.provider == "deepseek"
            else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        default_model_name = (
            "gpt-oss:20b"
            if self.provider == "ollama"
            else "deepseek-v4-pro"
            if self.provider == "deepseek"
            else "qwen3.5-plus"
        )
        self.base_url = str(base_url or default_base_url).rstrip("/")
        self.model_name = str(model_name or default_model_name)
        self.api_key = str(api_key or "").strip()
        default_api_key_env_var = "DEEPSEEK_API_KEY" if self.provider == "deepseek" else "DASHSCOPE_API_KEY"
        self.api_key_env_var = str(api_key_env_var or default_api_key_env_var).strip()
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.startup_timeout_s = float(startup_timeout_s) if startup_timeout_s is not None else float(timeout_s)
        self.max_output_tokens = max(64, int(max_output_tokens))
        self.request_retries = max(0, int(request_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = str(reasoning_effort or "").strip()
        self.extra_body = dict(extra_body or {})
        self._has_successful_request = False

    def _is_openai_compatible_provider(self) -> bool:
        return self.provider in {"openai_compatible", "deepseek"}

    def _normalized_model_name(self) -> str:
        return str(self.model_name or "").strip().lower()

    @property
    def model_label(self) -> str:
        return f"{self.provider}:{self.model_name}"

    @property
    def generated_by_tag(self) -> str:
        raw_value = f"llm_{self.provider}_{self.model_name}"
        normalized = re.sub(r"[^a-z0-9]+", "_", raw_value.lower()).strip("_")
        return normalized or "llm_unknown"

    def _request_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        use_json_format: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "system": str(system_prompt),
            "prompt": str(user_prompt),
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        }
        if use_json_format:
            payload["format"] = "json"
        return payload

    def _execute_ollama_request(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        req = request.Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=float(timeout_s)) as response:
            return json.loads(response.read().decode("utf-8"))

    def _resolve_api_key(self) -> str:
        if self.api_key != "":
            return self.api_key
        env_api_key = os.getenv(self.api_key_env_var, "").strip()
        if env_api_key != "":
            return env_api_key
        raise RuntimeError(
            f"Environment variable '{self.api_key_env_var}' is required for "
            f"{self.model_label} requests."
        )

    def _openai_client(self) -> Any:
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for OpenAI-compatible LLM access. "
                "Install dependencies from requirements.txt before running intent orchestration."
            ) from exc
        openai_client_cls = getattr(openai_module, "OpenAI", None)
        if openai_client_cls is None:
            raise RuntimeError("Installed 'openai' package does not expose the OpenAI client class.")
        return openai_client_cls(
            api_key=self._resolve_api_key(),
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

    def _truncate_response_excerpt(self, response_text: str, max_chars: int = 320) -> str:
        collapsed = " ".join(str(response_text or "").split())
        if len(collapsed) <= max_chars:
            return collapsed
        return collapsed[: max_chars - 3] + "..."

    def _stringify_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text_value = ""
                if isinstance(item, dict):
                    text_value = str(item.get("text", "") or item.get("content", "") or "")
                else:
                    text_value = str(getattr(item, "text", "") or getattr(item, "content", "") or "")
                if text_value != "":
                    parts.append(text_value)
            return "".join(parts)
        if isinstance(content, dict):
            return str(content.get("text", "") or content.get("content", "") or "")
        return str(content or "")

    def _parse_response_object(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        response_text = str(raw.get("response", "") or "")
        parsed = extract_json_object(response_text)
        if parsed is not None:
            return parsed
        thinking_text = str(raw.get("thinking", "") or "")
        parsed = extract_json_object(thinking_text)
        if parsed is not None:
            return parsed
        if isinstance(raw, dict) and isinstance(raw.get("response"), dict):
            return dict(raw["response"])
        return None

    def _timeout_for_attempt(self, attempt_index: int) -> float:
        base_timeout = self.startup_timeout_s if not self._has_successful_request else self.timeout_s
        growth_factor = 1.0 + 0.5 * float(attempt_index)
        return max(self.timeout_s, float(base_timeout) * growth_factor)

    def _execute_openai_compatible_request(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        timeout_s: float,
    ) -> tuple[dict[str, Any] | None, str]:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        extra_body = dict(self.extra_body)
        if self.provider == "deepseek":
            if self.enable_thinking:
                extra_body.setdefault("thinking", {"type": "enabled"})
            if self.reasoning_effort != "":
                request_kwargs["reasoning_effort"] = self.reasoning_effort
        elif self.provider == "openai_compatible" and self.enable_thinking:
            extra_body.setdefault("enable_thinking", True)
        if len(extra_body) > 0:
            request_kwargs.update(extra_body)

        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(request_kwargs).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._resolve_api_key()}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=float(timeout_s)) as response:
            completion = json.loads(response.read().decode("utf-8"))
        choices = list(dict(completion or {}).get("choices", []) or [])
        if not choices:
            return None, ""
        message = dict(choices[0]).get("message", {})
        response_text = self._stringify_content(
            dict(message or {}).get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
        parsed = extract_json_object(response_text)
        if parsed is not None:
            return parsed, response_text
        reasoning_text = self._stringify_content(
            dict(message or {}).get("reasoning_content", "")
            if isinstance(message, dict)
            else getattr(message, "reasoning_content", "")
        )
        parsed = extract_json_object(reasoning_text)
        if parsed is not None:
            return parsed, response_text if response_text != "" else reasoning_text
        return None, response_text if response_text != "" else reasoning_text

    def generate_json_or_raise(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError(
                f"LLM client is disabled. Intent orchestration rApps require a live {self.model_label} deployment."
            )
        strict_text_prompt = (
            f"{str(user_prompt)}\n\n"
            "Return exactly one valid JSON object. "
            "Do not output markdown. "
            "Do not output explanations. "
            "Do not output reasoning. "
            "Do not output code fences."
        )
        last_text_response = ""
        last_json_response = ""
        last_exc: Exception | None = None
        if self._is_openai_compatible_provider():
            for attempt in range(self.request_retries + 1):
                timeout_s = self._timeout_for_attempt(attempt)
                try:
                    parsed, response_text = self._execute_openai_compatible_request(
                        system_prompt=system_prompt,
                        user_prompt=strict_text_prompt,
                        timeout_s=timeout_s,
                    )
                    if parsed is not None:
                        self._has_successful_request = True
                        return parsed
                    last_text_response = response_text
                except Exception as exc:
                    last_exc = exc
                if attempt < self.request_retries:
                    time.sleep(self.retry_backoff_s)

            if last_exc is not None and last_text_response == "":
                raise RuntimeError(
                    f"Failed to reach {self.model_label} endpoint at {self.base_url}. "
                    f"Last timeout budget was {self._timeout_for_attempt(self.request_retries):.1f}s."
                ) from last_exc

            excerpt = self._truncate_response_excerpt(last_text_response)
            raise RuntimeError(
                f"{self.model_label} did not return a valid JSON object for the "
                f"intent orchestration request. Response excerpt: {excerpt}"
            )

        for attempt in range(self.request_retries + 1):
            timeout_s = self._timeout_for_attempt(attempt)
            try:
                text_payload = self._request_payload(
                    system_prompt=system_prompt,
                    user_prompt=strict_text_prompt,
                    use_json_format=False,
                )
                raw = self._execute_ollama_request(text_payload, timeout_s=timeout_s)
                parsed = self._parse_response_object(raw)
                if parsed is not None:
                    self._has_successful_request = True
                    return parsed
                last_text_response = str(raw.get("response", "") or "")
                json_payload = self._request_payload(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    use_json_format=True,
                )
                raw = self._execute_ollama_request(json_payload, timeout_s=timeout_s)
                parsed = self._parse_response_object(raw)
                if parsed is not None:
                    self._has_successful_request = True
                    return parsed
                last_json_response = str(raw.get("response", "") or "")
            except Exception as exc:
                last_exc = exc
            if attempt < self.request_retries:
                time.sleep(self.retry_backoff_s)

        if last_exc is not None and last_text_response == "" and last_json_response == "":
            raise RuntimeError(
                f"Failed to reach {self.model_label} endpoint at "
                f"{self.base_url}. "
                f"Last timeout budget was {self._timeout_for_attempt(self.request_retries):.1f}s. "
                "If inference is slow, increase 'llm_timeout_s' or 'llm_startup_timeout_s' "
                "in the intent orchestration RIC config."
            ) from last_exc

        fallback_excerpt = self._truncate_response_excerpt(last_text_response)
        excerpt = self._truncate_response_excerpt(last_json_response)
        raise RuntimeError(
            f"{self.model_label} did not return a valid JSON object for the "
            "intent orchestration request. "
            f"Text-mode response excerpt: {fallback_excerpt}. "
            f"JSON-mode response excerpt: {excerpt}"
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any] | None:
        try:
            return self.generate_json_or_raise(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception:
            return None


OllamaGptOssClient = StructuredJsonLlmClient
