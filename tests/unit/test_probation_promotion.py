"""Stage C4 unit tests: deterministic probation promotion review.

Pins: ranking is ready-first and storage-order-independent; the package is
built from stored numbers only; narration is an optional LLM prose layer with
an honest deterministic fallback; the review NEVER promotes — approve/reject
stay human-only (no state-changing call exists on this path, pinned by
source scan and by a fake store whose mutating methods would explode).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from forex_ai_agent.agent import probation_promotion as pp


def _row(status="passed", version_id="aaaaaaaa", obs=20, req=20, win=0.55, pf=1.2, name="Strat", pair="EUR/USD"):
    return SimpleNamespace(
        version_id=version_id,
        strategy_name=name,
        pair=pair,
        status=status,
        observed_trades=obs,
        required_trades=req,
        observed_days=15,
        required_days=14,
        days_elapsed=15,
        win_rate=win,
        profit_factor=pf,
        worst_daily_loss_pct=0.01,
        last_metrics={"win_rate": win, "profit_factor": pf},
        eligible_for_approval=(status == "passed"),
        notes="all probation gates met" if status == "passed" else "",
        started_at="2026-09-01T00:00:00",
        completed_at=None,
    )


class _ExplodingStore:
    """Any state-changing call is a constitution violation on this path."""

    async def list(self):
        return []

    async def approve(self, *_a, **_k):
        raise AssertionError("review must never approve")

    async def reject(self, *_a, **_k):
        raise AssertionError("review must never reject")


def test_rank_ready_orders_passed_first_then_active_by_progress():
    rows = [
        _row(status="active", version_id="cc", obs=5, req=20),
        _row(status="passed", version_id="bb"),
        _row(status="active", version_id="aa", obs=15, req=20),
        _row(status="expired", version_id="dd"),
    ]
    ranked = pp.rank_ready(rows)
    assert [r.version_id for r in ranked] == ["bb", "aa", "cc", "dd"]
    # order-independent: reversed input, same ranking
    assert [r.version_id for r in pp.rank_ready(list(reversed(rows)))] == ["bb", "aa", "cc", "dd"]


async def test_review_packages_only_ready_rows_with_narration_fallback():
    async def _rows():
        return [
            _row(status="passed", version_id="bb", name="Ready One"),
            _row(status="active", version_id="cc", obs=5, req=20),
            _row(status="expired", version_id="dd"),
        ]

    store = SimpleNamespace(list=_rows)
    async def _explode(*_a, **_k):
        raise RuntimeError("no llm in tests")

    review = await pp.review_promotions(
        pp.PromotionConfig(limit=5, include_narration=True), store=store, llm=_explode,
    )
    assert review["ready_count"] == 1
    assert [c["version_id"] for c in review["candidates"]] == ["bb"]
    cand = review["candidates"][0]
    assert cand["ready"] is True
    assert cand["win_rate"] == 0.55 and cand["profit_factor"] == 1.2
    # narration attempted but no LLM key in tests → deterministic fallback prose
    assert review["narration_used"] is False
    assert "Probation gates MET" in cand["narration"]
    assert review["disclaimer"] and "human" in review["disclaimer"].lower()


async def test_review_include_expired_adds_informational_rows():
    async def _rows():
        return [
            _row(status="passed", version_id="bb"),
            _row(status="expired", version_id="dd"),
        ]

    store = SimpleNamespace(list=_rows)
    review = await pp.review_promotions(
        pp.PromotionConfig(limit=5, include_narration=False, include_expired=True),
        store=store,
    )
    ids = [c["version_id"] for c in review["candidates"]]
    assert ids == ["bb", "dd"]
    assert review["ready_count"] == 1
    assert review["candidates"][1]["ready"] is False


async def test_review_never_promotes_and_survives_broken_store():
    # exploding store: if the review ever called approve/reject, the test blows up
    review = await pp.review_promotions(pp.PromotionConfig(), store=_ExplodingStore())
    assert review["reviewed"] == 0

    class _Boom:
        async def list(self):
            raise RuntimeError("db down")

    review2 = await pp.review_promotions(pp.PromotionConfig(), store=_Boom())
    assert review2["candidates"] == []
    assert "unavailable" in (review2["narration_error"] or "")


def test_candidate_dict_json_safe_and_bounded():
    import json

    c = pp._candidate_dict(_row())
    json.loads(json.dumps(c))  # no exotic types
    assert set(c) <= {
        "version_id", "strategy_name", "pair", "status", "ready",
        "observed_trades", "required_trades", "observed_days", "required_days",
        "days_elapsed", "win_rate", "profit_factor", "worst_daily_loss_pct",
        "notes", "started_at", "narration",
    }


def test_hard_blocks_by_construction_source_scan():
    src = inspect.getsource(pp)
    for banned in ("place_order", "execute_trade", "risk_manager",
                   "MAX_PERMISSION_LEVEL = 4", ".approve(", ".reject("):
        assert banned not in src, banned
