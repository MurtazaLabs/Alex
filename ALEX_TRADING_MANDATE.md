# Alex — Personal Trading Mandate v1.0

## Objective

Operate as an autonomous intraday forex trader within a human-defined financial mandate. Prioritize capital preservation, positive net expectancy, operational reliability, and risk-adjusted long-term growth.

No return is guaranteed or assumed.

## Initial Operating Scope

- Account: MT5 demo, $100 starting balance
- Instrument: EUR/USD
- Style: Intraday
- Live capital: Undecided
- Autonomy target: Full routine trading autonomy after validation
- Initial live status: Not authorized by this mandate

## Proposed Risk Policy

- Risk per trade: 0.25% of current equity
- Aggregate open stop-loss risk: 0.75% of equity
- Daily loss shutdown: 1% of start-of-day equity
- Weekly loss shutdown: 2% of start-of-week equity
- Peak-to-trough equity drawdown halt: 5%
- Maximum concurrent positions: 3, subject to exposure limits

These values require implementation, testing, and explicit approval before activation.

If minimum trade size violates the risk budget, refuse the trade.

## Strategy Policy

- Trade only registered, validated, explicitly approved strategies.
- Use market-regime detection only after integration and validation.
- Permit no-trade decisions at all times.
- Never deploy an unvalidated strategy or parameter change.
- Retire strategies only through authorized lifecycle transitions.

## Execution Policy

- Deterministic code controls order size, stops, limits, and execution.
- LLMs cannot submit orders or override risk policy.
- Unknown broker outcomes trigger reconciliation and fail-closed handling.
- Verify actual fills and broker positions.
- Prevent duplicate submissions and recover safely after restarts.

## Autonomy Policy

- No permission self-promotion.
- Live deployment requires explicit human sign-off.
- Per-trade approval may be removed only through a separate authorized transition after validation.
- Critical faults, breached limits, or inconsistent account state halt new entries and invoke the prescribed recovery process.

## Improvement Policy

- Analyze every trade and relevant execution cost.
- Research new strategies in isolation from live execution.
- Require out-of-sample and forward validation for changes.
- Preserve audit history, strategy lineage, and rollback capability.

## Reporting

- Daily: P&L, equity, drawdown, positions, costs, and incidents.
- Weekly: Strategy performance, expectancy, execution quality, exposure, and policy compliance.
- Immediate alerts: Critical faults, risk breaches, and unexpected broker/account state.

## Governance

Every permission change, live deployment, risk-policy change, and strategy approval requires a recorded, attributable authorization and supporting evidence.

---

## Implementation Record (engine side)

Approved by the human on 2026-10-18 for implementation before the 30-day
window opens. Implemented in code the same day; verified by the unit suite
(830/830) plus the E3 live verification (21/21, real MT5 data). The
machine-readable source of truth is `src/forex_ai_agent/execution/risk_mandate.py`
(`ACTIVE_MANDATE`); `PolicyCaps` and `LiveCaps` reference it.

Enforcement map:

- Per-trade 0.25% + min-lot budget floor: `ExecutionPolicy._validate`
  (refuses - never clamps size up - when the budget cannot fund one
  minimum lot; `INTENT_REFUSED_RISK_BUDGET`).
- Aggregate open stop risk 0.75%: `_validate` counts broker positions with
  stop losses PLUS recorded-but-unsent VALID intents; unverifiable open
  risk fails closed (`INTENT_REFUSED_UNVERIFIED_RISK`).
- Daily 1% / weekly 2% / drawdown 5% halts: baselines persisted in
  `logs/live_window/risk_baselines.json` (`RiskBaselineStore`); the
  window runner halts the cycle BEFORE creating intents when a limit is
  breached. Daily/weekly halts expire with their day/week rollover; a
  drawdown halt persists until a human clears it
  (`e4_window_run.py --clear-trip drawdown --reason "..."`, which rebases
  the high-water mark).
- Max 3 concurrent positions: `_validate` + window caps.

Deferred / pending (NOT part of this implementation):

- Removing per-trade human approval: requires the separate authorized
  transition in the Autonomy Policy after validation.
- Real capital deployment: the current $100 STANDARD demo cannot fund a
  compliant trade (0.25% of $100 = $0.25 budget vs $2.00 minimum-lot risk
  at a 20-pip stop), so the engine correctly refuses everything - zero
  trades is the mandate-correct outcome. The planned path is an Exness
  Standard Cent demo (about $60: 0.25% = 15 USC; min lot risks about 1
  USC at a 10-pip stop, a compliant 0.15-lot trade risks 15 USC). Broker/symbol wiring (EURUSDc), the lot band,
  and the `.env` credentials update are a separate reviewed change and
  were not made here.
