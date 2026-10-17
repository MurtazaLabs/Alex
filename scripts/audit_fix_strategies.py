"""One-shot audit + batch-fix of the spec-invalid saved strategies (C2 follow-up).

Audit findings (2026-09-26):
- 24 strategies (14 "Test", 10 "smoke") have EMPTY rules {} — pure smoke-test
  debris from system runs, created 2026-09-14..21. Unfixable (nothing there).
- 1 real strategy, b5199d22 "EMA_Trend_Follower" (status active): freeform
  rules ("EMA_9 > EMA_21 and ADX_14 > 25") that predate the spec format. The
  spec whitelist has NO adx and NO configurable EMA periods (ema is pinned to
  EMA-20), so a faithful translation is IMPOSSIBLE — the honest fix is a NEW
  sibling strategy in the current spec format (EMA20/SMA50 trend + pullback,
  direction both), parented to b5199d22, with the original archived (kept,
  not deleted) and annotated.

Safety: not a git repo — every touched JSON is backed up first to
strategies/versions/_pre_audit_backup_<ts>/.

Run:  .venv311/Scripts/python.exe scripts/audit_fix_strategies.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from forex_ai_agent.strategy.spec import parse_spec  # noqa: E402
from forex_ai_agent.strategy.versioning import create_version_manager  # noqa: E402

VERSIONS = Path("strategies/versions")
KEEP_STATUSES = ("active", "draft")  # everything else (archived/deprecated) is left alone


def audit() -> None:
    print("== AUDIT ==")
    vm = None
    import asyncio

    vm = asyncio.run(create_version_manager())
    empty, salvageable, already_ok = [], [], []
    for s in vm.list_versions():
        _, errors = parse_spec(s.rules)
        if not errors:
            already_ok.append(s.id)
            continue
        if not s.rules:
            empty.append(s.id)
        else:
            salvageable.append((s.id, s.name, s.status))
        print(f"  {s.id} | {s.name[:26]:26} | status={s.status:8} | "
              f"{'EMPTY RULES' if not s.rules else '; '.join(errors[:2])}")
    print(f"\nsummary: {len(already_ok)} already spec-valid, {len(empty)} empty-rule drafts, "
          f"{len(salvageable)} salvageable non-empty")


def backup(paths) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = VERSIONS / f"_pre_audit_backup_{ts}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, dest / p.name)
    print(f"backup: {len(paths)} file(s) -> {dest}")
    return dest


def main() -> None:
    if not VERSIONS.exists():
        print("no strategies/versions directory — nothing to do")
        return

    audit()

    files = sorted(VERSIONS.glob("*.json"))
    by_id = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            by_id[f.stem] = (f, json.load(fh))

    # ---- Pass 1: archive empty-rule debris ---------------------------------
    debris = [
        (f, d)
        for f, d in by_id.values()
        if d.get("status") in KEEP_STATUSES and not d.get("rules")
    ]
    if debris:
        backup([f for f, _ in debris])
        for f, d in debris:
            d["status"] = "archived"
            d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            d["tags"] = sorted(set(d.get("tags") or []) | {"audit-archived-empty-rules"})
            with open(f, "w", encoding="utf-8", newline="") as fh:
                json.dump(d, fh, indent=2)
        print(f"archived {len(debris)} empty-rule drafts (tagged audit-archived-empty-rules)")

    # ---- Pass 2: salvage b5199d22 via a spec-format sibling ----------------
    orig = by_id.get("b5199d22")
    if orig and orig[1].get("status") == "active":
        f, d = orig
        backup([f])

        # 1) Archive the original FIRST (kept, not deleted, annotated honestly).
        #    Must happen BEFORE create_version(parent_id=...) — that call appends
        #    to the parent's children_ids and re-saves the file; rewriting it
        #    afterwards from this stale dict would break the lineage.
        d["status"] = "archived"
        d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        d["tags"] = sorted(set(d.get("tags") or []) | {"audit-archived-untranslatable-rules"})
        with open(f, "w", encoding="utf-8", newline="") as fh:
            json.dump(d, fh, indent=2)

        # 2) The runnable sibling in the current spec format. EMA20/SMA50 is the
        #    whitelisted closest relative of the original EMA9/21 crossover;
        #    ADX is not in the whitelist, so it cannot be carried over (stated
        #    honestly in the description).
        sibling_rules = {
            "name": "EMA Trend Follower (spec format)",
            "description": (
                "Spec-format translation of EMA_Trend_Follower (b5199d22): the "
                "original EMA9/21 crossover + ADX>25 filter cannot be expressed in "
                "the deterministic spec whitelist (no adx, EMA pinned to 20), so "
                "this uses the closest whitelisted relatives: trend up when price "
                "and EMA20 are above SMA50, enter long when price pulls back below "
                "EMA20 (mean-reversion entry within uptrend). SL/TP from ATR "
                "brackets; risk owned by the deterministic engine."
            ),
            "direction": "long",
            "timeframe": "H1",
            "entry_conditions": [
                {"left": "price", "verb": "above", "right": "sma_slow"},
                {"left": "ema", "verb": "above", "right": "sma_slow"},
                {"left": "price", "verb": "crosses_below", "right": "ema"},
            ],
            "atr_sl_mult": 1.5,
            "atr_tp_mult": 3.0,
            "min_score_pct": 0.6,
        }
        spec, errors = parse_spec(sibling_rules)
        assert errors == [], f"sibling spec must validate: {errors}"

        import asyncio

        async def _create():
            vm = await create_version_manager()
            v = vm.create_version(
                name="EMA Trend Follower (spec format)",
                description=sibling_rules["description"],
                rules=sibling_rules,
                parameters={"atr_sl_mult": 1.5, "atr_tp_mult": 3.0, "direction": "long", "timeframe": "H1"},
                parent_id="b5199d22",
            )
            v.status = "draft"
            v.tags = ["audit-migrated-from-b5199d22"]
            vm._save_version(v)
            return v.id

        new_id = asyncio.run(_create())
        print(f"created spec-format sibling {new_id} (parent b5199d22, status draft)")
        print("archived original b5199d22 (kept, annotated; rationale in sibling description)")
    else:
        print("b5199d22 not active or missing — salvage pass skipped")

    # ---- Post-verify --------------------------------------------------------
    print("\n== POST-VERIFY ==")
    import asyncio

    async def _verify():
        vm = await create_version_manager()
        rows = []
        for s in vm.list_versions():
            _, errors = parse_spec(s.rules)
            rows.append((s.status, s.id, s.name, errors))
        return rows

    rows = asyncio.run(_verify())
    active = [r for r in rows if r[0] not in ("archived", "deprecated")]
    bad = [r for r in active if r[3]]
    for status, sid, name, errors in active:
        print(f"  {status:8} {sid} {name[:40]:40} {'OK' if not errors else 'BAD: ' + '; '.join(errors[:2])}")
    print(f"\nnon-archived strategies: {len(active)}, spec-invalid among them: {len(bad)}")
    if bad:
        print("RESULT: FAIL — invalid strategies remain in the active pool")
        sys.exit(1)
    print("RESULT: PASS — every non-archived strategy is a runnable spec")
    print("note: backups are in strategies/versions/_pre_audit_backup_*/ — delete them when satisfied")


if __name__ == "__main__":
    main()
