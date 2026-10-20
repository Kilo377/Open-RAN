# Agent notes

This branch contains the physical OAI / B210 / RM520N-GL lab under `lab/oai-gnb/`.
Read its README and STARTUP guide before changing deployment behavior.

- Python scripts use the standard library; validate changed scripts with `python3 -m py_compile` and `--help` where available.
- Keep the OAI source commit and container image digest pinned in `upstream.lock.json`.
- New and legacy gNB processes must not share the B210. Preserve the recovery timer and validated rollback path.
- Reuse the existing 5GC and SIM subscription unless the task explicitly changes them.
- Validate throughput with real module-bound, payload-verified data; distinguish unsupported settings from zero throughput.
- Keep public results under `validation/`; keep runtime state and raw logs outside the repository.
- Never commit SSH passwords, agent tokens, SIM K/OPc, database dumps, private keys or raw subscriber logs.
- Documentation and repository cleanup do not require restarting the radio or rerunning physical throughput tests.

Keep user-facing explanations concise and avoid unnecessary structure.
