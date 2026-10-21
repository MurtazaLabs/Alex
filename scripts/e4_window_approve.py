"""E4 window approvals -- the human's per-trade approval for the window.

The window runner produces OrderIntents (audit log) but NEVER sends one
on its own: an intent reaches the wire only after a line for its key is
appended here by this tool (a human action). The runner then grants the
single-use token in its own process and the transmitter's full gate
chain still applies.

ASCII output only (Windows console). Usage:

  python scripts/e4_window_approve.py --list
  python scripts/e4_window_approve.py --key <intent_key> --approver murtaza
  python scripts/e4_window_approve.py --key <intent_key> --declaration "ok"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_ai_agent.execution.live.window import (  # noqa: E402
    DEFAULT_WINDOW_STATE_DIR,
)

STATE_DIR = Path(DEFAULT_WINDOW_STATE_DIR)
INTENTS = STATE_DIR / "intents.jsonl"
APPROVALS = STATE_DIR / "approvals.jsonl"
TRANSMISSIONS = STATE_DIR / "transmissions.jsonl"


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _list() -> int:
    now = datetime.utcnow()
    intents = _read_jsonl(INTENTS)
    approvals = {str(a.get("intent_key")) for a in _read_jsonl(APPROVALS)}
    done = {str(t.get("intent_key")) for t in _read_jsonl(TRANSMISSIONS)}
    print("\n=== E4 WINDOW INTENTS ===\n")
    pending = approved = 0
    for record in intents:
        key = str(record.get("key"))
        if record.get("status") != "VALID" or key in done:
            continue
        try:
            expired = datetime.fromisoformat(str(record["expires_at"])) <= now
        except Exception:
            expired = True
        state = "EXPIRED" if expired else (
            "APPROVED (awaiting dispatch)" if key in approvals else "PENDING APPROVAL"
        )
        if state == "PENDING APPROVAL":
            pending += 1
        elif state.startswith("APPROVED"):
            approved += 1
        print(f"  {key[:12]}  {record.get('side'):4s} "
              f"{float(record.get('volume', 0)):.2f} lots @ "
              f"{float(record.get('entry', 0)):.5f}  "
              f"sl={float(record.get('sl', 0)):.5f} "
              f"tp={float(record.get('tp', 0)):.5f}  "
              f"expires={record.get('expires_at', '?')}  [{state}]")
    print(f"\npending approval: {pending} | approved awaiting dispatch: {approved}")
    if not intents:
        print("no intents recorded (runner has not produced any yet)")
    return 0


def _approve(args: argparse.Namespace) -> int:
    now = datetime.utcnow()
    intents = {str(r.get("key")): r for r in _read_jsonl(INTENTS)}
    record = intents.get(args.key)
    if record is None:
        print(f"REFUSED: no recorded intent with key {args.key}")
        return 2
    if record.get("status") != "VALID":
        print(f"REFUSED: intent status is {record.get('status')} "
              f"({record.get('refusal_reason', '')})")
        return 2
    try:
        if datetime.fromisoformat(str(record["expires_at"])) <= now:
            print("REFUSED: intent already expired (TTL is 300s by design)")
            return 2
    except Exception:
        print("REFUSED: intent expiry unreadable")
        return 2
    existing = {str(a.get("intent_key")) for a in _read_jsonl(APPROVALS)}
    if args.key in existing:
        print("REFUSED: intent already approved")
        return 2
    done = {str(t.get("intent_key")) for t in _read_jsonl(TRANSMISSIONS)}
    if args.key in done:
        print("REFUSED: intent already transmitted")
        return 2

    approval = {
        "approval_id": uuid4().hex[:12],
        "intent_key": args.key,
        "symbol": str(record.get("symbol", "")),
        "side": str(record.get("side", "")),
        "volume": float(record.get("volume", 0.0)),
        "approver": args.approver,
        "approved_at": now.isoformat(),
        "declaration": args.declaration or "",
    }
    _append_jsonl(APPROVALS, approval)
    print("INTENT APPROVED (human record appended)")
    print(f"  approval_id: {approval['approval_id']}")
    print(f"  intent:      {args.key[:12]}  {approval['side']} "
          f"{approval['volume']:.2f} lots {approval['symbol']}")
    print(f"  approver:    {args.approver}")
    print("  the runner will pick this up on its next cycle")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="E4 window approvals")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--key", default="")
    parser.add_argument("--approver", default="murtaza")
    parser.add_argument("--declaration", default="")
    args = parser.parse_args()
    if args.key:
        return _approve(args)
    return _list()


if __name__ == "__main__":
    sys.exit(main())
