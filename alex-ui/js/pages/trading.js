/**
 * Alex OS — Trading space: governor status, order intents, approval tokens,
 * kill switch. The human's control panel for live execution.
 */
(function () {
  'use strict';

  const { el, esc, fmtWhen } = UI;
  let pollTimer = null;

  /* ---------------- Governor status ---------------- */

  async function loadGovernor() {
    const box = el('gov-status');
    try {
      const s = await API.get('/execution/governor/status');
      const stateClass = s.state === 'ARMED' ? 'badge-long' : s.state === 'HALTED' ? 'badge-short' : 'badge-muted';
      const caps = s.policy_caps || {};
      box.innerHTML = `
        <div class="flex-between mb-4">
          <div>
            <span class="badge ${stateClass}" style="font-size:var(--fs-md)">${esc(s.state)}</span>
            ${s.tripped ? `<span class="badge badge-short">TRIP: ${esc(s.trip_reason || '?')}</span>` : ''}
          </div>
          <div class="flex gap-2">
            ${s.state !== 'ARMED' ? '<button class="btn btn-sm btn-success" onclick="Trading.arm()">Arm</button>' : ''}
            ${s.state === 'ARMED' ? '<button class="btn btn-sm btn-danger" onclick="Trading.trip()">Kill Switch</button>' : ''}
          </div>
        </div>
        <div class="evidence">
          <div class="evidence-row"><div class="k">Risk / trade</div><div class="v">${caps.max_risk_per_trade_pct}%</div></div>
          <div class="evidence-row"><div class="k">Daily loss cap</div><div class="v">${caps.max_daily_loss_pct}%</div></div>
          <div class="evidence-row"><div class="k">Max positions</div><div class="v">${caps.max_concurrent_positions}</div></div>
          <div class="evidence-row"><div class="k">Min lot</div><div class="v">${caps.min_lot_size}</div></div>
          <div class="evidence-row"><div class="k">Intent expiry</div><div class="v">${caps.intent_expiry_seconds}s</div></div>
          <div class="evidence-row"><div class="k">Pending intents</div><div class="v">${s.pending_intents}</div></div>
        </div>`;
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  /* ---------------- Intents ---------------- */

  function intentCard(i) {
    const valid = i.status === 'VALID';
    const sideBadge = i.side === 'BUY' ? 'badge-long' : 'badge-short';
    return `
      <div class="card mb-3" style="border-left:3px solid ${valid ? (i.side === 'BUY' ? 'var(--green)' : 'var(--red)') : 'var(--text-muted)'}">
        <div class="flex-between mb-3">
          <div>
            <strong style="font-size:var(--fs-lg)">${esc(i.symbol)}</strong>
            <span class="badge ${sideBadge}">${esc(i.side)}</span>
            ${!valid ? `<span class="badge badge-short">${esc(i.status)}</span>` : ''}
          </div>
          <span class="text-muted" style="font-size:var(--fs-xs)">expires ${fmtWhen(i.expires_at)}</span>
        </div>
        <div class="evidence">
          <div class="evidence-row"><div class="k">Volume</div><div class="v">${i.volume} lots</div></div>
          <div class="evidence-row"><div class="k">Entry</div><div class="v">${i.entry}</div></div>
          <div class="evidence-row"><div class="k">Stop Loss</div><div class="v" style="color:var(--red)">${i.sl} (${i.sl_pips} pips)</div></div>
          <div class="evidence-row"><div class="k">Take Profit</div><div class="v" style="color:var(--green)">${i.tp}</div></div>
          <div class="evidence-row"><div class="k">Risk</div><div class="v">$${i.risk_usd} (${(i.risk_usd / i.equity * 100).toFixed(1)}% of $${i.equity})</div></div>
          <div class="evidence-row"><div class="k">R:R</div><div class="v">${i.rr_ratio}:1</div></div>
          ${i.strategy_id ? `<div class="evidence-row"><div class="k">Strategy</div><div class="v">${esc(i.strategy_id)}</div></div>` : ''}
          ${i.refusal_reason ? `<div class="evidence-row"><div class="k">Refused</div><div class="v" style="color:var(--red)">${esc(i.refusal_reason)}</div></div>` : ''}
        </div>
        ${valid ? `<div class="mt-3"><button class="btn btn-success" onclick="Trading.approve('${esc(i.key)}')">Approve & Grant Token</button></div>` : ''}
      </div>`;
  }

  async function loadIntents() {
    const box = el('intent-list');
    try {
      const data = await API.get('/execution/intents?limit=10');
      const intents = data.intents || [];
      box.innerHTML = intents.length
        ? intents.map(intentCard).join('')
        : '<div class="empty-state"><span class="big">📭</span>No intents yet. Signals that pass validation appear here.</div>';
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  /* ---------------- Tokens ---------------- */

  async function loadTokens() {
    const box = el('token-list');
    try {
      const data = await API.get('/execution/tokens');
      const tokens = data.tokens || [];
      box.innerHTML = tokens.length
        ? `<div class="table-wrap"><table class="table">
            <thead><tr><th>Token</th><th>Intent</th><th>Order</th><th>Granted by</th><th>Expires</th><th></th></tr></thead>
            <tbody>${tokens.map((t) => `
              <tr>
                <td><code>${esc(t.token_id.slice(0, 12))}…</code></td>
                <td><code>${esc(t.intent_key.slice(0, 8))}…</code></td>
                <td>${esc(t.side)} ${t.volume} ${esc(t.symbol)}</td>
                <td>${esc(t.granted_by)}</td>
                <td class="text-muted nowrap">${fmtWhen(t.expires_at)}</td>
                <td><button class="btn btn-sm btn-danger" onclick="Trading.revoke('${esc(t.token_id)}')">Revoke</button></td>
              </tr>`).join('')}</tbody></table></div>`
        : '<div class="empty-state">No active tokens.</div>';
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  /* ---------------- Create intent (manual) ---------------- */

  async function createIntent() {
    const body = {
      symbol: el('ci-symbol').value.trim().toUpperCase(),
      side: el('ci-side').value,
      entry: parseFloat(el('ci-entry').value),
      stop_loss: parseFloat(el('ci-sl').value),
      take_profit: parseFloat(el('ci-tp').value),
      strategy_id: el('ci-strategy').value.trim(),
      comment: 'manual',
    };
    if (!body.symbol || !body.entry || !body.stop_loss || !body.take_profit) {
      UI.toast.error('Fill all fields');
      return;
    }
    try {
      const res = await API.post('/execution/intent', body);
      if (res.status === 'VALID') {
        UI.toast.success(`Intent created: ${res.side} ${res.symbol} ${res.volume} lots`);
      } else {
        UI.toast.warning(`Intent refused: ${res.refusal_reason}`);
      }
      loadIntents();
    } catch (err) {
      UI.toast.error(err.message);
    }
  }

  /* ---------------- Actions ---------------- */

  async function approve(key) {
    try {
      const token = await API.post(`/execution/intent/${encodeURIComponent(key)}/approve`, {
        granted_by: 'murtaza',
      });
      UI.toast.success(`Token granted: ${token.token_id.slice(0, 12)}…`);
      loadTokens();
    } catch (err) {
      UI.toast.error(err.message);
    }
  }

  async function revoke(tokenId) {
    try {
      await API.post(`/execution/token/${encodeURIComponent(tokenId)}/revoke`, { revoked_by: 'murtaza' });
      UI.toast.success('Token revoked');
      loadTokens();
    } catch (err) {
      UI.toast.error(err.message);
    }
  }

  async function arm() {
    try {
      await API.post('/execution/governor/arm', {});
      UI.toast.success('Governor armed');
      loadGovernor();
    } catch (err) {
      UI.toast.error(err.message);
    }
  }

  async function trip() {
    try {
      await API.post('/execution/governor/trip', { reason: 'manual' });
      UI.toast.warning('Kill switch tripped');
      loadGovernor();
    } catch (err) {
      UI.toast.error(err.message);
    }
  }

  /* ---------------- Mount ---------------- */

  async function mount(container) {
    container.innerHTML = `
      <div class="card mb-4">
        <div class="card-title">Execution Governor</div>
        <div id="gov-status"><div class="skeleton" style="width:100%"></div></div>
      </div>

      <div class="card mb-4">
        <div class="card-title">Create Order Intent</div>
        <div class="form-grid" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:var(--sp-3)">
          <div class="form-field"><label>Symbol</label><input id="ci-symbol" value="EUR/USD" class="input"></div>
          <div class="form-field"><label>Side</label><select id="ci-side" class="input"><option>BUY</option><option>SELL</option></select></div>
          <div class="form-field"><label>Strategy ID</label><input id="ci-strategy" placeholder="optional" class="input"></div>
          <div class="form-field"><label>Entry</label><input id="ci-entry" type="number" step="0.00001" class="input" placeholder="1.10000"></div>
          <div class="form-field"><label>Stop Loss</label><input id="ci-sl" type="number" step="0.00001" class="input" placeholder="1.09800"></div>
          <div class="form-field"><label>Take Profit</label><input id="ci-tp" type="number" step="0.00001" class="input" placeholder="1.10400"></div>
        </div>
        <button class="btn btn-primary mt-3" onclick="Trading.createIntent()">Create Intent</button>
      </div>

      <div class="card mb-4">
        <div class="card-title">Pending Intents</div>
        <div id="intent-list"><div class="skeleton" style="width:100%"></div></div>
      </div>

      <div class="card mb-4">
        <div class="card-title">Active Approval Tokens</div>
        <div id="token-list"><div class="skeleton" style="width:100%"></div></div>
      </div>`;

    await Promise.all([loadGovernor(), loadIntents(), loadTokens()]);
    pollTimer = setInterval(() => { loadGovernor(); loadIntents(); loadTokens(); }, 15000);
    return () => clearInterval(pollTimer);
  }

  window.Trading = { approve, revoke, arm, trip, createIntent };

  window.Router.register('trading', {
    title: 'Trading',
    subtitle: 'Order intents, approval tokens, kill switch',
    icon: '💰',
    mount,
  });
})();
