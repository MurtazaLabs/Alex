/**
 * Alex OS — Settings space: pipeline internals, feed health, LLM, risk posture.
 */
(function () {
  'use strict';

  const { el, esc } = UI;

  function healthRows(list) {
    if (!list || !list.length) return '<div class="text-muted">No feed data.</div>';
    return list.map((f) => {
      const name = esc(f.name || f.provider || '?');
      const ok = f.healthy !== false;
      const badge = f.last_resort
        ? '<span class="badge badge-warning">LAST RESORT</span>'
        : (ok ? '<span class="badge badge-success">healthy</span>' : '<span class="badge badge-danger">failing</span>');
      return `<div class="evidence-row">
        <div class="k"><span class="dot ${ok ? 'dot-ok' : 'dot-bad'}"></span> <span class="mono">${name}</span></div>
        <div class="v flex">${badge}
          <span class="text-muted">✓ ${f.total_successes ?? 0} · ✗ ${f.total_failures ?? 0}</span>
          ${f.last_error ? `<span class="text-muted" style="margin-left:auto;font-size:var(--fs-xs)">${esc(String(f.last_error).slice(0, 60))}</span>` : ''}
        </div></div>`;
    }).join('');
  }

  async function mount(container) {
    container.innerHTML = `
      <div class="card-grid grid-2 mb-4">
        <div class="card"><div class="card-title">Feed Chain (failover order)</div><div id="set-feed"><div class="skeleton" style="width:70%"></div><div class="skeleton" style="width:55%"></div></div></div>
        <div class="card"><div class="card-title">LLM Engine</div><div id="set-llm"><div class="skeleton" style="width:60%"></div></div></div>
      </div>
      <div class="card">
        <div class="card-title">Pipeline Status</div>
        <div id="set-status"><div class="skeleton" style="width:80%"></div><div class="skeleton" style="width:60%"></div></div>
      </div>
    `;

    const st = await API.get('/alex/status');

    el('set-feed').innerHTML =
      `<div class="evidence">
        <div class="evidence-row"><div class="k">Active source</div><div class="v"><span class="badge badge-info">${esc(String(st.data_source || '—').toLowerCase())}</span></div></div>
      </div>
      <div class="mt-4">${healthRows(st.feed_health)}</div>`;

    const llm = st.llm || {};
    el('set-llm').innerHTML = `<div class="evidence">
      <div class="evidence-row"><div class="k">Narration model</div><div class="v mono">${esc(llm.model || llm.narration_model || '—')}</div></div>
      <div class="evidence-row"><div class="k">Available</div><div class="v">${llm.available === false ? '<span class="badge badge-warning">fallback templates</span>' : '<span class="badge badge-success">yes</span>'}</div></div>
      <div class="evidence-row"><div class="k">Daily budget</div><div class="v mono">${esc(llm.daily_token_budget ?? '—')}</div></div>
    </div>`;

    const rows = Object.entries(st)
      .filter(([k]) => !['feed_health', 'llm'].includes(k))
      .map(([k, v]) => {
        const val = (v === null || v === undefined) ? '—'
          : (typeof v === 'object' ? JSON.stringify(v) : String(v));
        return `<div class="evidence-row"><div class="k mono">${esc(k)}</div><div class="v">${esc(val.slice(0, 160))}</div></div>`;
      }).join('');
    el('set-status').innerHTML = `<div class="evidence">${rows}</div>`;
  }

  window.Router.register('settings', {
    title: 'Settings', subtitle: 'System internals and health', icon: '⚙', mount,
  });
})();
