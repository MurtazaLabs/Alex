/**
 * Alex OS — Account space: read-only MT5 broker snapshot.
 */
(function () {
  'use strict';

  const { el, esc, fmtMoney } = UI;

  async function mount(container) {
    container.innerHTML = `
      <div class="card-grid grid-4 mb-4">
        <div class="card stat-card"><div class="stat-label">Equity</div><div class="stat-value" id="ac-equity">…</div><div class="stat-sub">mark-to-market</div></div>
        <div class="card stat-card"><div class="stat-label">Balance</div><div class="stat-value" id="ac-balance">…</div><div class="stat-sub">closed P&amp;L booked</div></div>
        <div class="card stat-card"><div class="stat-label">Free Margin</div><div class="stat-value" id="ac-freemargin">…</div><div class="stat-sub">available to trade</div></div>
        <div class="card stat-card"><div class="stat-label">Margin Level</div><div class="stat-value" id="ac-marginlevel">…</div><div class="stat-sub">equity / margin</div></div>
      </div>
      <div class="card-grid grid-2">
        <div class="card">
          <div class="card-title">Broker Connection</div>
          <div id="ac-conn"><div class="skeleton" style="width:70%"></div><div class="skeleton" style="width:50%"></div></div>
        </div>
        <div class="card">
          <div class="card-title">Session</div>
          <div id="ac-session"></div>
        </div>
      </div>
      <div class="hint mt-4">Read-only by design: the UI carries no order endpoints. Live execution stays locked until you explicitly enable it.</div>
    `;

    const a = await API.get('/alex/account');
    el('ac-equity').textContent = a.available ? fmtMoney(a.equity, a.currency) : '—';
    el('ac-balance').textContent = a.available ? fmtMoney(a.balance, a.currency) : '—';
    el('ac-freemargin').textContent = a.available ? fmtMoney(a.free_margin, a.currency) : '—';
    el('ac-marginlevel').textContent = a.available && a.margin_level != null ? `${Number(a.margin_level).toFixed(2)}%` : '—';

    el('ac-conn').innerHTML = a.available
      ? `<div class="evidence">
          <div class="evidence-row"><div class="k">Status</div><div class="v"><span class="badge badge-success">Connected</span></div></div>
          <div class="evidence-row"><div class="k">Login</div><div class="v mono">${esc(a.login ?? '—')}</div></div>
          <div class="evidence-row"><div class="k">Server</div><div class="v mono">${esc(a.server ?? '—')}</div></div>
          <div class="evidence-row"><div class="k">Leverage</div><div class="v mono">1:${esc(a.leverage ?? '—')}</div></div>
          <div class="evidence-row"><div class="k">Currency</div><div class="v mono">${esc(a.currency ?? '—')}</div></div>
          <div class="evidence-row"><div class="k">Open positions</div><div class="v mono">${esc(a.positions ?? 0)}</div></div>
        </div>`
      : `<div class="empty-state"><span class="big">🔌</span>${esc(a.note || 'MT5 terminal not reachable')}<div class="mt-2 text-muted">Start the MT5 terminal, then Refresh.</div></div>`;

    const u = State.user || {};
    el('ac-session').innerHTML = `
      <div class="evidence">
        <div class="evidence-row"><div class="k">Signed in as</div><div class="v mono">${esc(u.username || '—')}</div></div>
        <div class="evidence-row"><div class="k">Permission level</div><div class="v mono">${esc(u.permission_level ?? '—')}</div></div>
        <div class="evidence-row"><div class="k">Live feed</div><div class="v mono">${esc(State.feed || '—')}</div></div>
      </div>`;
  }

  window.Router.register('account', {
    title: 'Account', subtitle: 'MT5 broker snapshot · read-only', icon: '👤', mount,
  });
})();
