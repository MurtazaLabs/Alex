/**
 * Alex OS — Signals space: pipeline runs + the durable signal trail.
 */
(function () {
  'use strict';

  const { el, esc, fmtWhen, verdictBadge } = UI;

  function evidenceHtml(res) {
    const ev = res.evidence || {};
    const rows = [];
    const push = (k, v) => { if (v !== undefined && v !== null && v !== '') rows.push([k, v]); };
    push('Direction', res.direction);
    push('Timeframe', res.timeframe);
    push('Status', res.signal_status || res.status);
    if (ev.trend) push('Trend', JSON.stringify(ev.trend));
    if (ev.structure) push('Structure', JSON.stringify(ev.structure));
    if (ev.indicators) push('Indicators', JSON.stringify(ev.indicators));
    if (ev.entry) push('Entry', JSON.stringify(ev.entry));
    if (ev.stop_loss) push('Stop loss', ev.stop_loss);
    if (ev.take_profit) push('Take profit', ev.take_profit);
    if (res.narrative) push('Narrative', res.narrative);
    if (res.explanation) push('Explanation', `<pre style="white-space:pre-wrap;margin:0">${esc(res.explanation)}</pre>`);
    if (!rows.length) return '<div class="empty-state">No evidence payload returned.</div>';
    return `<div class="evidence">${rows.map(([k, v]) =>
      `<div class="evidence-row"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`).join('')}</div>`;
  }

  async function generateSignal() {
    const pair = el('sig-pair').value.trim().toUpperCase();
    const dir = el('sig-dir').value;
    const tf = el('sig-tf').value;
    const out = el('sig-result');
    if (!/^[A-Z]{3}\/?[A-Z]{3}$/.test(pair.replace(/\s/g, ''))) {
      UI.toast.error('Pair must look like EUR/USD');
      return;
    }
    out.innerHTML = '<div class="skeleton" style="width:60%"></div><div class="skeleton" style="width:80%"></div><div class="skeleton" style="width:45%"></div>';
    try {
      const res = await API.post('/alex/signals/generate', { pair: pair.includes('/') ? pair : `${pair.slice(0,3)}/${pair.slice(3)}`, direction: dir, timeframe: tf });
      out.innerHTML = `
        <div class="flex-between mb-4">
          <div><strong style="font-size:var(--fs-lg)">${esc(res.pair || pair)}</strong>
            <span class="badge ${dir === 'LONG' ? 'badge-long' : 'badge-short'}">${esc(dir)}</span>
            <span class="badge badge-muted">${esc(tf)}</span></div>
          <span class="text-muted">${fmtWhen(res.timestamp || new Date().toISOString())}</span>
        </div>
        ${evidenceHtml(res)}`;
      loadHistory();
    } catch (err) {
      out.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  async function loadHistory() {
    const box = el('sig-history');
    try {
      const data = await API.get('/alex/signals?limit=25');
      const rows = data.signals || [];
      box.innerHTML = rows.length
        ? `<div class="table-wrap"><table class="table">
            <thead><tr><th>Pair</th><th>Dir</th><th>Setup / conclusion</th><th>Verdict</th><th>When</th></tr></thead>
            <tbody>${rows.map((s) => `
              <tr>
                <td><strong>${esc(s.pair || '?')}</strong></td>
                <td><span class="badge ${s.direction === 'SHORT' ? 'badge-short' : 'badge-long'}">${esc(s.direction || '—')}</span></td>
                <td class="text-secondary">${esc((s.setup || '').slice(0, 110))}</td>
                <td>${verdictBadge(s.verdict)}</td>
                <td class="text-muted nowrap">${fmtWhen(s.created)}</td>
              </tr>`).join('')}</tbody></table></div>`
        : '<div class="empty-state"><span class="big">🛰️</span>No signals recorded yet.</div>';
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  async function mount(container) {
    container.innerHTML = `
      <div class="card mb-4">
        <div class="card-title">Generate a Signal</div>
        <div class="form-row">
          <div class="field"><label>Pair</label><input class="input" id="sig-pair" placeholder="EUR/USD" value="EUR/USD"></div>
          <div class="field" style="max-width:140px"><label>Direction</label>
            <select class="input" id="sig-dir"><option>LONG</option><option>SHORT</option></select></div>
          <div class="field" style="max-width:120px"><label>Timeframe</label>
            <select class="input" id="sig-tf"><option>M15</option><option selected>H1</option><option>H4</option><option>D1</option></select></div>
          <div class="field" style="max-width:150px;align-self:flex-end"><button class="btn btn-primary" id="sig-go" style="width:100%">⚡ Run pipeline</button></div>
        </div>
        <div class="hint">Deterministic pipeline: structure → indicators → rules → risk. Narration off by default (fast).</div>
      </div>
      <div id="sig-result"></div>
      <div class="card mt-4">
        <div class="card-title">Signal History</div>
        <div id="sig-history"><div class="skeleton" style="width:70%"></div><div class="skeleton" style="width:55%"></div></div>
      </div>
    `;
    el('sig-go').addEventListener('click', generateSignal);
    el('sig-pair').addEventListener('keydown', (e) => { if (e.key === 'Enter') generateSignal(); });
    await loadHistory();
  }

  window.Router.register('signals', {
    title: 'Signals', subtitle: 'Run the deterministic pipeline · inspect evidence', icon: '⚡', mount,
  });
})();
