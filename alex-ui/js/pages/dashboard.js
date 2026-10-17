/**
 * Alex OS — Dashboard space.
 */
(function () {
  'use strict';

  const { el, esc, fmtMoney, fmtWhen } = UI;

  async function mount(container) {
    container.innerHTML = `
      <div class="card-grid grid-4 mb-4" id="dash-stats">
        <div class="card stat-card"><div class="stat-label">Data Feed</div><div class="stat-value" id="ds-feed">…</div><div class="stat-sub" id="ds-feed-sub">MT5 → Yahoo → TwelveData → Mock</div></div>
        <div class="card stat-card"><div class="stat-label">Account Balance</div><div class="stat-value" id="ds-balance">…</div><div class="stat-sub" id="ds-balance-sub">from MT5 broker</div></div>
        <div class="card stat-card"><div class="stat-label">Probation Queue</div><div class="stat-value" id="ds-probation">…</div><div class="stat-sub">strategies earning approval</div></div>
        <div class="card stat-card"><div class="stat-label">Research Runs</div><div class="stat-value" id="ds-research">…</div><div class="stat-sub">logged experiments</div></div>
      </div>

      <div class="card-grid grid-2">
        <div class="card">
          <div class="flex-between"><div class="card-title">Recent Signals</div>
            <button class="btn btn-sm" onclick="location.hash='#/signals'">All signals →</button></div>
          <div id="dash-signals"><div class="skeleton" style="width:80%"></div><div class="skeleton" style="width:65%"></div><div class="skeleton" style="width:72%"></div></div>
        </div>
        <div class="card">
          <div class="flex-between"><div class="card-title">Feed Health</div>
            <button class="btn btn-sm" onclick="window.Router.navigate('settings',{force:true})">Details →</button></div>
          <div id="dash-feed-health"><div class="skeleton" style="width:70%"></div><div class="skeleton" style="width:60%"></div></div>
        </div>
      </div>

      <div class="card mt-4">
        <div class="card-title">Quick Actions</div>
        <div class="flex" style="flex-wrap:wrap">
          <button class="btn btn-primary" id="qa-signal">⚡ Generate signal</button>
          <button class="btn" id="qa-research">🔬 Run research</button>
          <button class="btn" id="qa-chat">💬 Ask Alex</button>
        </div>
        <div class="hint mt-2">Everything runs on real MT5 data. LLM narration stays off for speed.</div>
      </div>
    `;

    const ov = await API.get('/alex/overview');

    el('ds-feed').textContent = String(ov.data_source || '—').toLowerCase();
    const acct = ov.account || {};
    el('ds-balance').textContent = acct.available ? fmtMoney(acct.balance, acct.currency) : '—';
    el('ds-balance-sub').textContent = acct.available ? `login ${acct.login} · ${acct.server}` : (acct.note || 'MT5 terminal not reachable');
    el('ds-probation').textContent = String(ov.probation_active ?? 0);
    el('ds-research').textContent = String(ov.research_runs ?? 0);
    Shell.setFeed(ov.data_source);

    const sig = ov.recent_signals || [];
    el('dash-signals').innerHTML = sig.length
      ? `<div class="table-wrap"><table class="table"><tbody>${sig.map((s) => `
          <tr>
            <td><strong>${esc(s.pair || '?')}</strong></td>
            <td><span class="badge badge-long">${esc(s.direction || 'LONG')}</span></td>
            <td class="text-secondary">${esc((s.setup || '').slice(0, 90))}</td>
            <td class="text-muted nowrap">${fmtWhen(s.created)}</td>
          </tr>`).join('')}</tbody></table></div>`
      : `<div class="empty-state"><span class="big">🛰️</span>No signals yet — generate one below.</div>`;

    const fh = ov.feed_health || [];
    el('dash-feed-health').innerHTML = fh.length
      ? fh.map((f) => {
          const name = esc(f.name || f.provider || '?');
          const ok = f.healthy !== false;
          return `<div class="flex-between" style="padding:6px 0;border-bottom:1px dashed var(--border-subtle)">
            <span class="flex"><span class="dot ${ok ? 'dot-ok' : 'dot-bad'}"></span><span class="mono">${name}</span>${f.last_resort ? ' <span class="badge badge-warning">last resort</span>' : ''}</span>
            <span class="text-muted">✓ ${f.total_successes ?? 0} · ✗ ${f.total_failures ?? 0}</span>
          </div>`;
        }).join('')
      : '<div class="empty-state">Feed health not reported.</div>';

    el('qa-signal').addEventListener('click', () => location.hash = '#/signals');
    el('qa-research').addEventListener('click', () => location.hash = '#/research');
    el('qa-chat').addEventListener('click', () => location.hash = '#/chat');
  }

  window.Router.register('dashboard', {
    title: 'Dashboard', subtitle: 'Live overview of your trading agent', icon: '▦', mount,
  });
})();
