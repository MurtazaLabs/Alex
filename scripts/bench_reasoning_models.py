"""Bench OpenRouter models for structured JSON output (Stage C follow-up).

Runs the REAL C1 hypothesis-generation task (same system prompt, schema hint,
and a live deterministic EUR/USD structure brief) against each candidate via
the project's LLMClient, then grades every response deterministically:

- network/HTTP success, JSON parse success
- schema conformance: hypotheses[] with pair/text/confidence(/regime)
- semantics: correct pair, confidence in 0..1, sane non-degenerate text
- latency

Also proves paid-model billing works (any successful paid call settles that).

Usage:
  .venv311/Scripts/python.exe scripts/bench_reasoning_models.py            # all candidates
  .venv311/Scripts/python.exe scripts/bench_reasoning_models.py m1 m3      # subset by index
"""

import asyncio
import json
import sys
import time
from typing import Any, Dict, List

from forex_ai_agent.config import get_settings
from forex_ai_agent.llm.client import LLMClient

# Cheap paid models with structured_outputs support (from the live catalog),
# best-value order. One per family first, two OpenAI entries (cheapest two).
CANDIDATES: List[str] = [
    "openai/gpt-oss-20b",              # $0.02/$0.09 — cheapest with structure support
    "openai/gpt-5-nano",               # $0.05/$0.40
    "google/gemini-2.5-flash-lite",    # $0.10/$0.40
    "google/gemma-4-31b-it",           # $0.09/$0.34 (paid sibling of the free one)
    "deepseek/deepseek-v4-flash",      # $0.05/$0.09
    "mistralai/mistral-small-3.2-24b-instruct",  # $0.09/$0.25
    "qwen/qwen3-32b",                  # $0.08/$0.28
    # free survivors worth re-testing with the funded key
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",           # current primary, for comparison
]

# Keep free-model ids OUT of the tier override if the user passes a subset.


async def main() -> None:
    from forex_ai_agent.agent.hypothesis_gen import (
        _LLM_SCHEMA_HINT,
        _LLM_SYSTEM_PROMPT,
        _validate_candidate,
        HypothesisGenConfig,
        build_brief,
    )
    from forex_ai_agent.agent.tools import _build_structure_snapshot

    wanted = set(sys.argv[1:])
    candidates = [c for c in CANDIDATES if not wanted or c in wanted]
    if wanted and not any(c in wanted for c in CANDIDATES):
        candidates = list(wanted)  # allow arbitrary ids from argv

    print("Building live deterministic brief (EUR/USD H1)…")
    snapshot = await _build_structure_snapshot("EUR/USD", "H1")
    briefs = [build_brief(snapshot)]
    config = HypothesisGenConfig(pairs=["EUR/USD"])

    settings = get_settings()
    client = LLMClient()
    rows: List[Dict[str, Any]] = []

    for model in candidates:
        # Point the whole OpenRouter tier at exactly this model for the bench.
        settings.alex_llm_reasoning_model = model
        settings.alex_llm_reasoning_fallbacks = ""
        t0 = time.time()
        row: Dict[str, Any] = {"model": model}
        # Cache-bust: the client's cache key doesn't include the model, so
        # each candidate gets a distinguishable payload.
        context = {"briefs": briefs, "requested_pairs": ["EUR/USD"], "bench_model": model}
        try:
            raw = await asyncio.wait_for(
                client.reason_json(_LLM_SYSTEM_PROMPT, context, schema_hint=_LLM_SCHEMA_HINT),
                timeout=100.0,
            )
            row["latency_s"] = round(time.time() - t0, 1)
            row["parse"] = isinstance(raw, dict)
            hyps = raw.get("hypotheses") if isinstance(raw, dict) else None
            if not isinstance(hyps, list) or not hyps:
                row["grade"] = "BAD_SCHEMA"
                row["detail"] = f"raw type={type(raw).__name__}, hypotheses={type(hyps).__name__}"
            else:
                ok, bad = [], []
                for h in hyps:
                    norm, reason = _validate_candidate(h, {"EUR/USD": "EUR/USD"}, config)
                    (ok if reason is None else bad).append((h, norm, reason))
                row["accepted"] = len(ok)
                row["rejected"] = len(bad)
                first = ok[0][1] if ok else None
                text = (first or {}).get("text", "")
                degenerate = len(set(text.split())) < 4 or text.count("ells") > 3
                row["grade"] = (
                    "PASS" if (ok and not degenerate) else
                    "DEGENERATE" if ok else "ALL_REJECTED"
                )
                row["detail"] = (text[:110] + "…") if text else str(bad[:1])
        except Exception as exc:  # noqa: BLE001 — the bench grades failures too
            row["latency_s"] = round(time.time() - t0, 1)
            row["grade"] = f"FAIL"
            row["detail"] = f"{type(exc).__name__}: {exc}"[:160]
        rows.append(row)
        print(
            f"[{len(rows)}/{len(candidates)}] {model:<45} {row['grade']:<11} "
            f"{row.get('latency_s', '?')}s  {row.get('detail', '')}"
        )

    await client.aclose()

    print("\n== SUMMARY (best first) ==")
    rank = {"PASS": 0, "ALL_REJECTED": 1, "BAD_SCHEMA": 2, "DEGENERATE": 3, "FAIL": 4}
    rows.sort(key=lambda r: (rank.get(r["grade"], 9), r.get("latency_s", 999)))
    for r in rows:
        print(f"  {r['grade']:<11} {r['model']:<45} {r.get('latency_s', '?')}s")

    with open("scripts/bench_results.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print("\nfull details -> scripts/bench_results.json")


if __name__ == "__main__":
    asyncio.run(main())
