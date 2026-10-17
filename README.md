# Forex AI Agent

**FOREX Research & Intelligence Agent** — a multi-agent, event-driven forex trading and
research system.

> **Core principle:** The LLM is *never* the primary trading decision-maker. Deterministic
> code handles all calculations, risk, signals and execution logic. The LLM handles
> reasoning, research, explanation and orchestration.

## Permission Levels

| Level | Capability |
|-------|------------|
| 0 | Research only |
| 1 | Market monitoring |
| 2 | Signal generation |
| 3 | Paper trading |
| 4 | Live trading (disabled by default, requires explicit approval) |

## Project Layout

```
src/forex_ai_agent/
├── config/        # Pydantic-settings configuration
├── core/          # Event bus, permission manager, orchestrator, state manager
├── data/
│   ├── models/    # SQLAlchemy 2.0 async models (see docs/DATABASE_SCHEMA.md)
│   ├── providers/ # Market data providers (Mock, MT5 placeholder)
│   └── repositories/  # Async data access layer
├── analysis/
│   └── indicators/    # SMA, EMA, MACD, RSI, ATR, Bollinger, price action
├── agents/        # Specialist agents (Phase 1: stubs)
├── interfaces/
│   └── api/       # FastAPI routes (watchlist CRUD, pairs, health)
└── main.py        # FastAPI application factory (in src/forex_ai_agent/)
docs/                # Full design documentation (Obsidian vault)
migrations/          # Alembic migrations
tests/               # Unit + integration tests
```

## Quick Start (Docker)

```bash
docker compose up --build
```

This starts PostgreSQL 15, Redis 7 and the API on port 8000. Migrations run automatically
on container start.

## Quick Start — Alex CLI (no infra needed)

With no `.env` and no `DATABASE_URL`, Alex auto-falls back to local SQLite and
runs the whole pipeline — no Postgres or Redis required:

```bash
pip install -e ".[dev,feeds]"      # Python 3.11
python -m forex_ai_agent.cli

alex> watch EUR/USD GBP/JPY        # feed-verified, persisted watchlist
alex> signals                      # full pipeline per pair × LONG/SHORT
alex> explain EUR/USD              # 7-section explanation + LLM narrative
alex> status                       # feed health, LLM budget, settings
alex> research EUR/USD             # autonomy loop: brief → LLM spec → backtest gate
alex> strategies                   # probation queue (✅ = awaiting YOUR approval)
alex> approve <id>                 # human-only activation decision
```

Data flows through the resilient feed chain `mt5 → yahoo → twelvedata → mock`
(configure `MT5_*` in `.env` for real broker data; without keys it lands on
Yahoo/mock with a loud status warning). Full MT5 account setup:
**docs/Forex docmentation/MT5_CONNECTION_GUIDE.md**. Narration uses OpenRouter when
`OPENROUTER_API_KEY` is set; otherwise a deterministic template narrates from
the same evidence.

## Dashboard

```bash
run_dashboard.bat                  # or: .venv311\Scripts\python -m streamlit run dashboard\app.py
```

## Local Development

```bash
py -3.11 -m venv .venv311
# Windows: .venv311\Scripts\activate
.venv311\Scripts\python -m pip install -e ".[dev,feeds,dashboard]"
cp .env.example .env

# Run tests
pytest

# Lint
ruff check src tests
black --check src tests

# Run the API without Docker (requires PostgreSQL + Redis locally,
# or set DATABASE_URL to sqlite+aiosqlite:///./forex_ai.db for a quick smoke test)
uvicorn forex_ai_agent.main:app --reload
```

## Phase 1 Scope (implemented)

- [x] Project scaffolding, configuration, logging
- [x] SQLAlchemy 2.0 async models + Alembic migrations
- [x] Redis event bus (pub/sub) with in-memory fallback
- [x] Permission manager (levels 0–4) + completed orchestrator pipeline
- [x] Resilient data feed chain (MT5 / Yahoo / Twelve Data / Mock, health + failback)
- [x] Basic indicators: SMA, EMA, MACD, RSI, ATR, Bollinger, price action
- [x] Signal pipeline on real data (rules → risk → explanation → LLM narration)
- [x] LLM client (OpenRouter + Ollama, two tiers, budget + privacy guard)
- [x] Watchlist management API + pair registry
- [x] Repository pattern + unit/integration tests (suite fully green)

See `docs/Forex docmentation/DEVELOPMENT_PHASES.md` for the full roadmap,
`docs/Forex docmentation/ALEX_REASONING_PLAN.md` for the reasoning architecture
and the Phase-2 autonomy loop (LLM-proposed strategies → backtest gate → paper
probation → human approval), and `docs/Forex docmentation/ALEX_AGENT_LAYER_PLAN.md`
for the approved conversational agent layer (Alex Chat, tool registry, permission
+ confirmation policy, experiments, memory, bounded autonomy). The **next build
order** (reliability → research intelligence → autonomy → paper → execution) is
in `docs/Forex docmentation/ALEX_RESEARCH_ROADMAP.md`.