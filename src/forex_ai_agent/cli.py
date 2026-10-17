"""CLI Interface for the Forex AI Agent.

Conversational interface over the real pipeline:

    watch EUR/USD GBP/JPY   → persist pairs to the watchlist (feed-verified)
    signals                 → run the full pipeline for every watched pair
    explain EUR/USD         → best-direction setup with 7-section explanation
    status                  → feed health, LLM budget, watchlist, risk settings
    research <topic>        → Phase-2 autonomy roadmap stub

Usage: python -m forex_ai_agent.cli
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Any, Dict, List, Optional

from click import echo, prompt, secho, style

# Windows consoles often default to cp1252; Alex's output is UTF-8 (✓, box art).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # already-closed or exotic streams
            pass

# Local-first database: default to SQLite when no DATABASE_URL and no .env exist,
# so the CLI works out of the box. Must run before any package import.
if not os.environ.get("DATABASE_URL") and not os.path.exists(".env"):
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///alex_local.db")

from forex_ai_agent.config.settings import get_settings  # noqa: E402
from forex_ai_agent.core.event_bus import InMemoryEventBus  # noqa: E402
from forex_ai_agent.pipeline import Pipeline  # noqa: E402

PAIR_RE = re.compile(r"^[A-Z]{3}/[A-Z]{3}$")
LOCAL_USER = "alex-local"
DEFAULT_WATCHLIST = "default"


def ok(msg: str) -> None:
    echo(style("✓  ", fg="green") + msg)


def err(msg: str) -> None:
    echo(style("✗  ", fg="red") + msg)


def info(msg: str) -> None:
    echo(style("ℹ  ", fg="cyan") + msg)


class CLI:
    """Conversational CLI driving the real pipeline on one persistent event loop."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.event_bus = InMemoryEventBus()
        self.pipeline = Pipeline(event_bus=self.event_bus)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._run(self.pipeline.initialize())
        # In-memory fallback when the DB is unavailable.
        self._local_pairs: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _run(self, coro: Any) -> Any:
        return self._loop.run_until_complete(coro)

    # ------------------------------------------------------------------
    # Watchlist persistence (DB-backed, graceful in-memory fallback)
    # ------------------------------------------------------------------

    async def _watched_pairs_async(self) -> List[str]:
        try:
            from forex_ai_agent.data.database import get_session_factory
            from forex_ai_agent.data.repositories import UserRepository, WatchlistRepository

            async with get_session_factory()() as session:
                users = UserRepository(session)
                user = await users.get_by_username(LOCAL_USER)
                if user is None:
                    return []
                lists = WatchlistRepository(session)
                wl = await lists.get_by_name(user.id, DEFAULT_WATCHLIST)
                return list(wl.pairs or []) if wl else []
        except Exception as exc:  # noqa: BLE001 — no DB → in-memory mode
            if self._local_pairs is None:
                info(f"Database unavailable ({type(exc).__name__}); watchlist is session-only")
                self._local_pairs = []
            return list(self._local_pairs)

    async def _save_pairs_async(self, pairs: List[str]) -> bool:
        """Persist pairs to the default watchlist. Returns False when in-memory."""
        try:
            from forex_ai_agent.data.database import get_session_factory
            from forex_ai_agent.data.models.user import User
            from forex_ai_agent.data.repositories import UserRepository, WatchlistRepository

            async with get_session_factory()() as session:
                users = UserRepository(session)
                user = await users.get_by_username(LOCAL_USER)
                if user is None:
                    user = await users.add(
                        User(
                            username=LOCAL_USER,
                            email=f"{LOCAL_USER}@localhost",
                            password_hash="!",
                            permission_level=0,
                        )
                    )
                lists = WatchlistRepository(session)
                wl = await lists.get_by_name(user.id, DEFAULT_WATCHLIST)
                if wl is None:
                    wl = await lists.create(user.id, DEFAULT_WATCHLIST, pairs)
                else:
                    wl = await lists.add_pairs(wl, pairs)
                return True
        except Exception:  # noqa: BLE001
            return False

    async def _remove_pair_async(self, pair: str) -> bool:
        try:
            from forex_ai_agent.data.database import get_session_factory
            from forex_ai_agent.data.repositories import UserRepository, WatchlistRepository

            async with get_session_factory()() as session:
                users = UserRepository(session)
                user = await users.get_by_username(LOCAL_USER)
                if user is None:
                    return True
                lists = WatchlistRepository(session)
                wl = await lists.get_by_name(user.id, DEFAULT_WATCHLIST)
                if wl and pair in (wl.pairs or []):
                    wl.pairs = [p for p in wl.pairs if p != pair]
                    await session.commit()
                return True
        except Exception:  # noqa: BLE001
            if self._local_pairs is not None:
                self._local_pairs = [p for p in self._local_pairs if p != pair]
            return True

    def watched_pairs(self) -> List[str]:
        return self._run(self._watched_pairs_async())

    # ------------------------------------------------------------------
    # REPL
    # ------------------------------------------------------------------

    def run(self) -> None:
        banner = (
            f"ALEX — Forex AI Agent  (feed: {self.pipeline.data_source}, "
            f"risk {self.settings.default_risk_percent}%/trade)"
        )
        secho("=" * 62, fg="cyan")
        secho(banner, fg="cyan", bold=True)
        echo("Type 'help' for commands. Live trading is disabled (permission level 0).")
        secho("=" * 62, fg="cyan")

        while True:
            try:
                user_input = prompt("alex> ", default="").strip()
            except (KeyboardInterrupt, EOFError):
                echo(style("\nGoodbye!", fg="cyan"))
                break
            if not user_input:
                continue
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]
            try:
                self._dispatch(cmd, args)
            except KeyboardInterrupt:
                echo(style("\nGoodbye!", fg="cyan"))
                break
            except Exception as exc:  # noqa: BLE001 — the REPL must survive anything
                err(f"{type(exc).__name__}: {exc}")

    def _dispatch(self, cmd: str, args: List[str]) -> None:
        if cmd in ("help", "h", "?"):
            self._show_help()
        elif cmd == "watch":
            self._handle_watch(args)
        elif cmd in ("unwatch", "stop", "remove"):
            self._handle_unwatch(args)
        elif cmd in ("pairs", "watchlist"):
            self._handle_pairs()
        elif cmd == "signals":
            self._handle_signals(args)
        elif cmd == "explain":
            self._handle_explain(args)
        elif cmd == "status":
            self._handle_status()
        elif cmd == "risk":
            self._handle_risk(args)
        elif cmd == "research":
            self._handle_research(args)
        elif cmd == "strategies":
            self._handle_strategies()
        elif cmd == "chat":
            self._handle_chat(args)
        elif cmd == "approve":
            self._run(self._approve_reject(args[0] if args else "", approve=True))
        elif cmd == "reject":
            self._run(self._approve_reject(args[0] if args else "", approve=False))
        elif cmd in ("exit", "quit", "q"):
            raise KeyboardInterrupt
        else:
            err(f"Unknown command: '{cmd}'. Type 'help'.")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _show_help(self) -> None:
        echo(
            "\nCommands:\n"
            "  watch <pairs...>     Watch pairs, e.g. watch EUR/USD GBP/JPY (persists)\n"
            "  unwatch <pair>       Stop watching a pair\n"
            "  pairs                Show watched pairs\n"
            "  signals [--tf H1]    Run the pipeline for all watched pairs, both directions\n"
            "  explain <pair>       Best-direction setup + 7-section explanation + narrative\n"
            "  status               Feed health, LLM budget, settings, watchlist\n"
            "  risk [<percent>]     Show or set account risk percent (0.1–5.0)\n"
            "  research <pair>      One autonomy-loop run: brief → LLM spec → backtest gate\n"
            "  strategies           Strategy pipeline: probation queue + statuses\n"
            "  chat                 Conversational Alex (AgentCore tool loop)\n"
            "  approve <id>         YOUR approval of a passed candidate (p ← passed)\n"
            "  reject <id>          YOUR rejection of a candidate\n"
            "  exit                 Quit\n"
        )

    def _handle_watch(self, args: List[str]) -> None:
        if not args:
            err("Usage: watch <PAIR1> [PAIR2] ...")
            return
        valid: List[str] = []
        for raw in args:
            pair = raw.upper() if "/" in raw else f"{raw[:3].upper()}/{raw[3:].upper()}"
            if not PAIR_RE.match(pair):
                err(f"'{raw}' is not a valid pair (expected e.g. EUR/USD)")
                continue
            valid.append(pair)
        if not valid:
            return
        # Feed-verify each pair.
        for pair in valid:
            try:
                quote = self._run(self.pipeline.data_provider.fetch_current_price(pair))
                price = f" (last {quote.close:.5f})" if quote else ""
                ok(f"{pair}: feed reachable{price}")
            except Exception as exc:  # noqa: BLE001
                err(f"{pair}: no feed data ({exc}) — added anyway")
        persisted = self._run(self._save_pairs_async(valid))
        current = self.watched_pairs()
        ok("Watching: " + ", ".join(current))
        if not persisted:
            info("(session-only — database unavailable)")

    def _handle_unwatch(self, args: List[str]) -> None:
        if not args:
            err("Usage: unwatch <PAIR>")
            return
        pair = args[0].upper()
        self._run(self._remove_pair_async(pair))
        ok(f"No longer watching {pair}. Now: " + (", ".join(self.watched_pairs()) or "(none)"))

    def _handle_pairs(self) -> None:
        pairs = self.watched_pairs()
        if pairs:
            ok("Watched pairs: " + ", ".join(pairs))
        else:
            info("No pairs watched. Try: watch EUR/USD GBP/JPY")

    def _handle_signals(self, args: List[str]) -> None:
        timeframe = "H1"
        if "--tf" in args:
            idx = args.index("--tf")
            if idx + 1 < len(args):
                timeframe = args[idx + 1].upper()
        pairs = self.watched_pairs()
        if not pairs:
            info("No pairs watched. Try: watch EUR/USD")
            return
        for pair in pairs:
            for direction in ("LONG", "SHORT"):
                result = self._run(self.pipeline.generate_signal(pair, direction, timeframe))
                if result.get("status") != "ok":
                    err(f"{pair} {direction}: {result.get('message', result.get('status'))}")
                    continue
                self._render_signal_card(result)

    def _handle_explain(self, args: List[str]) -> None:
        if not args:
            err("Usage: explain <PAIR>")
            return
        pair = args[0].upper()
        timeframe = args[1].upper() if len(args) > 1 else "H1"
        best: Optional[Dict[str, Any]] = None
        for direction in ("LONG", "SHORT"):
            result = self._run(self.pipeline.generate_signal(pair, direction, timeframe))
            if result.get("status") != "ok":
                err(f"{pair} {direction}: {result.get('message', result.get('status'))}")
                continue
            score = (result.get("signal_candidate") or {}).get("score_pct") or 0.0
            if best is None or score > ((best.get("signal_candidate") or {}).get("score_pct") or 0.0):
                best = result
        if best is None:
            return
        self._render_signal_card(best)
        echo("")
        secho("EXPLANATION", fg="yellow", bold=True)
        echo(best.get("explanation", ""))
        echo("")
        secho("NARRATIVE", fg="magenta", bold=True)
        echo(best.get("narrative", ""))

    def _handle_status(self) -> None:
        status = self._run(self.pipeline.get_pipeline_status())
        secho("ALEX STATUS", fg="cyan", bold=True)
        echo(f"  Data source : {status.get('data_source')}")
        feeds = status.get("feed_health") or []
        if feeds:
            echo("  Feeds       :")
            for f in feeds:
                marker = "✓" if f["status"].startswith(("HEALTHY", "LAST")) else "✗"
                last_err = f' — last error: {f["last_error"]}' if f.get("last_error") else ""
                echo(f"    [{marker}] {f['name']:<12} {f['status']}{last_err}")
        llm = status.get("llm") or {}
        echo(
            "  LLM         : enabled={} openrouter={} tokens={}/{} (cached prompts: {})".format(
                llm.get("llm_enabled"),
                llm.get("openrouter_configured"),
                llm.get("tokens_used_today", 0),
                llm.get("daily_budget"),
                llm.get("cached_prompts", 0),
            )
        )
        echo(f"  Database    : {'connected' if status.get('db_initialized') else 'unavailable (session mode)'}")
        pairs = self.watched_pairs()
        echo(f"  Watchlist   : {', '.join(pairs) if pairs else '(none)'}")
        s = status.get("settings") or {}
        echo(
            f"  Risk        : {s.get('risk_percent')} per trade, min R:R {s.get('min_rr')}, "
            f"max exposure {s.get('max_exposure')}, permission ≤ {s.get('max_permission_level')}"
        )
        # Next high-impact news event (best effort).
        try:
            events = self.pipeline.news_agent.get_upcoming_events(hours_ahead=24)
            high = [e for e in events if getattr(e, "impact", None) and e.impact.value == "HIGH"]
            if high:
                nxt = high[0]
                echo(f"  Next HI news: {nxt.currency} {nxt.title} at {nxt.scheduled_time:%Y-%m-%d %H:%M} UTC")
        except Exception:  # noqa: BLE001
            pass

    def _handle_risk(self, args: List[str]) -> None:
        if not args:
            ok(f"Account risk : {self.settings.default_risk_percent}%")
            ok(f"Min R:R      : {self.settings.min_risk_reward}")
            ok(f"Max exposure : {self.settings.max_total_exposure_pct}%")
            ok(f"Max position : {self.settings.max_position_size_lots} lots")
            return
        try:
            pct = float(args[0])
        except ValueError:
            err("Invalid number")
            return
        if not 0.1 <= pct <= 5.0:
            err("Risk percent must be between 0.1 and 5.0")
            return
        err("Persisting risk changes is not supported in this build (edit .env: DEFAULT_RISK_PERCENT)")

    def _handle_research(self, args: List[str]) -> None:
        """Run one autonomy-loop iteration for one (or all watched) pair(s)."""
        from forex_ai_agent.autonomy.proposal import (
            fetch_history_for,
            propose_for_pair,
        )

        pairs = [a.upper() for a in args if "/" in a] or self.watched_pairs()
        if not pairs:
            err("No pair given and watchlist empty. Usage: research EUR/USD [--days 180]")
            return
        days = 180
        if "--days" in args:
            try:
                days = max(60, int(args[args.index("--days") + 1]))
            except (IndexError, ValueError):
                pass
        for pair in pairs:
            secho(f"\n⟳ RESEARCH {pair}", fg="cyan", bold=True)
            info("building brief → proposing → backtesting (this can take a minute)…")
            try:
                history = self._run(fetch_history_for(self.pipeline, pair, days=days))
                result = self._run(
                    propose_for_pair(self.pipeline, pair, history, use_llm=True)
                )
            except Exception as exc:  # noqa: BLE001
                secho(f"  ✗ research failed: {type(exc).__name__}: {exc}", fg="red")
                continue
            self._render_proposal(pair, result)

    def _handle_strategies(self) -> None:
        """Show probation queue + approvals (accrues new paper trades first)."""
        from forex_ai_agent.autonomy.probation import ProbationStore
        from forex_ai_agent.autonomy.probation_runner import ProbationRunner

        # Replay-based accrual: every strategies view first credits any paper
        # trades closed since the last check (idempotent via watermark).
        try:
            outcomes = self._run(ProbationRunner().run_once())
            for o in outcomes:
                if o.get("accrued"):
                    info(
                        f"probation {o['version_id'][:8]}: +{o['accrued']} paper trades "
                        f"({o.get('observed_trades', 0)}/{o.get('required_trades', '?')})"
                    )
        except Exception as exc:  # noqa: BLE001 — display must survive accrual failure
            info(f"probation accrual skipped ({type(exc).__name__}: {exc})")

        store = ProbationStore()
        rows = self._run(store.list())
        if not rows:
            info("No strategy proposals yet. Run: research <PAIR>")
            return
        secho("STRATEGY PIPELINE", fg="cyan", bold=True)
        for r in rows:
            progress = f"{r.observed_trades}/{r.required_trades} trades"
            flag = "✓" if r.eligible_for_approval else " "
            secho(
                f"  [{flag}] {r.strategy_name[:38]:<38} {r.pair:<8} {r.status:<10} {progress}",
                fg="green" if r.status == "approved" else (
                    "red" if r.status in {"failed", "rejected"} else "yellow"
                ),
            )
            if r.notes:
                echo(f"        ↳ {r.notes}")
        eligible = [r for r in rows if r.eligible_for_approval]
        if eligible:
            echo("")
            ok(f"{len(eligible)} awaiting YOUR approval: " + ", ".join(r.version_id[:8] for r in eligible))
            info("approve <version_id>  |  reject <version_id>")

    def _handle_chat(self, args: List[str]) -> None:
        """Conversational chat with AgentCore (same brain as alex-ui)."""
        from forex_ai_agent.agent import AgentCore
        from forex_ai_agent.llm.client import LLMClient
        from forex_ai_agent.agent.memory import ConversationMemory

        secho("\n💬 ALEX CHAT — Type 'exit' to return to main REPL", fg="cyan", bold=True)
        echo("   Ask questions like: 'What's EUR/USD structure?' or 'Run backtest on strategy abc123'")
        echo("")

        # Use pipeline's event loop
        loop = self._loop
        memory = ConversationMemory()
        core = AgentCore(
            pipeline=self.pipeline,
            llm=LLMClient(),
            memory=memory,
            max_steps=6,
            user_permission_level=0,  # CLI runs at OBSERVE level
        )

        while True:
            try:
                user_input = prompt("chat> ", default="").strip()
            except (KeyboardInterrupt, EOFError):
                echo(style("\nReturning to main REPL...", fg="cyan"))
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                echo(style("Returning to main REPL...", fg="cyan"))
                break

            try:
                result = loop.run_until_complete(core.run_turn(user_input))
                if result.success:
                    secho(f"\n{result.answer}\n", fg="green")
                    if result.tools_used:
                        info(f"Tools used: {', '.join(result.tools_used)} ({result.steps} steps)")
                else:
                    err(f"\nError: {result.error or result.answer}")
            except Exception as exc:
                err(f"{type(exc).__name__}: {exc}")

    async def _approve_reject(self, version_id: str, approve: bool) -> None:
        from forex_ai_agent.autonomy.probation import ProbationStore

        store = ProbationStore()
        rows = await store.list()
        match = [r for r in rows if r.version_id.startswith(version_id)]
        if not match:
            err(f"no probation matching {version_id!r}")
            return
        if len(match) > 1:
            err("ambiguous prefix; candidates: " + ", ".join(r.version_id for r in match))
            return
        target = match[0]
        try:
            status = (
                await store.approve(target.version_id)
                if approve
                else await store.reject(target.version_id)
            )
        except ValueError as exc:
            err(str(exc))
            return
        if approve:
            ok(f"APPROVED {status.strategy_name} ({status.version_id[:8]}) — activate via strategies page")
        else:
            ok(f"REJECTED {status.strategy_name} ({status.version_id[:8]})")

    def _render_proposal(self, pair: str, result: Any) -> None:
        for msg in result.messages:
            info(msg)
        report = result.validation
        if report is None:
            err("no validation report (brief or data failed)")
            return
        status = "PASSED GATE" if report.passed else "REJECTED"
        secho(
            f"  Gate: {status}  ({report.data_range.get('bars', '?')} bars, "
            f"{report.data_range.get('start', '?')[:10]} → {report.data_range.get('end', '?')[:10]})",
            fg="green" if report.passed else "red",
            bold=True,
        )
        fb = report.full_backtest or {}
        if fb:
            echo(
                f"  Full backtest: trades={fb.get('total_trades', 0)} "
                f"win={fb.get('win_rate', 0):.0%} PF={fb.get('profit_factor', 0):.2f} "
                f"sharpe={fb.get('sharpe_ratio', 0):.2f}"
            )
        wf = report.walk_forward or {}
        if wf:
            echo(
                f"  Walk-forward: OOS sharpe={wf.get('oos_sharpe', 0):.2f} "
                f"OOS trades={wf.get('oos_trades', 0)} "
                f"OOS maxDD={wf.get('oos_max_drawdown', 0):.0%} "
                f"overfit={wf.get('overfitting_score', 0):.2f}"
            )
        for reason in report.reasons:
            secho(f"  ✗ {reason}", fg="red")
        if report.passed and result.version is not None:
            secho(
                f"  → versioned {result.version.id[:8]} — entering paper probation",
                fg="green",
            )
            echo("    (probation: 20 paper trades / 14 days → then YOUR approval)")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_signal_card(self, result: Dict[str, Any]) -> None:
        sig = result.get("signal_candidate") or {}
        risk = result.get("risk_validation") or {}
        checks = risk.get("checks") or {}
        failed = [n for n, c in checks.items() if not c.get("passed")]
        width = 72
        border = "─" * (width - 2)

        def row(text: str) -> None:
            echo("│ " + text[: width - 4].ljust(width - 4) + " │")

        echo("┌" + border + "┐")
        row(
            f"{sig.get('pair')} · {sig.get('timeframe')} · {sig.get('direction')}  "
            f"[{result.get('data_source')}] · {result.get('session')}"
        )
        score = sig.get("score_pct")
        score_str = f"{score:.0%}" if isinstance(score, (int, float)) else "—"
        row(f"state: {sig.get('status', '?')}   score: {score_str}")
        if sig.get("setup_type") == "NO_TRADE":
            row("no setup formed — evidence does not meet thresholds")
        else:
            row(f"entry {sig.get('entry_zone_low')} – {sig.get('entry_zone_high')}   SL {sig.get('stop_loss')}")
            row(
                f"TP1 {sig.get('take_profit_1')}   R:R {sig.get('risk_reward')}   "
                f"size {risk.get('position_size_lots')} lots"
            )
        if checks:
            row(
                f"risk: {len(checks) - len(failed)}/{len(checks)} checks pass"
                + (f"   FAILED: {', '.join(failed)}" if failed else "")
            )
        inv = str(sig.get("invalidation_condition") or "")
        row(f"invalidation: {inv}")
        echo("└" + border + "┘")
        narrative = result.get("narrative")
        if narrative:
            echo(f"  ↳ {narrative}\n")


def main() -> None:
    try:
        cli = CLI()
    except Exception as exc:  # noqa: BLE001
        err(f"Failed to initialize Alex: {exc}")
        sys.exit(1)
    cli.run()


if __name__ == "__main__":
    main()
