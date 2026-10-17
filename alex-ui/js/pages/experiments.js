/**
 * Alex OS — Experiments space: the research laboratory UI.
 * Experiment records, verdict cards, parameter diffs, real walk-forward
 * windows (IS→OOS Sharpe per window), and side-by-side strategy comparison.
 */
(function () {
  'use strict';

  const { el, esc, toast, modal, fmtWhen } = UI;

  let _container = null;
  let _experiments = [];

  function verdictBadge(v) {
    const s = String(v || '').toLowerCase();
    if (s === 'pass') return '<span class="badge badge-success">PASS</span>';
    if (s === 'fail') return '<span class="badge badge-danger">FAIL</span>';
    if (s === 'inconclusive') return '<span class="badge badge-warning">INCONCLUSIVE</span>';
    if (s === 'stable') return '<span class="badge badge-success">STABLE</span>';
    if (s === 'mixed') return '<span class="badge badge-warning">MIXED</span>';
    if (s === 'unstable') return '<span class="badge badge-danger">UNSTABLE</span>';
    if (s === 'reject' || s === 'rejected') return '<span class="badge badge-danger">REJECTED</span>';
    if (s === 'adopt' || s === 'adopted') return '<span class="badge badge-success">ADOPTED</span>';
    if (s === 'defer') return '<span class="badge badge-muted">DEFERRED</span>';
    return `<span class="badge badge-muted">${esc(String(v || '—').toUpperCase())}</span>`;
  }

  function metricRows(base, variant) {
    const keys = new Set([...Object.keys(base || {}), ...Object.keys(variant || {})]);
    const fmt = (k, v) => {
      if (v == null || !isFinite(Number(v))) return '—';
      if (/pct|rate|dd|drawdown/i.test(k)) return `${Number(v).toFixed(1)}%`;
      return Number(v).toFixed(2);
    };
    let out = '';
    keys.forEach((k) => {
      const b = base ? base[k] : null;
      const v = variant ? variant[k] : null;
      let arrow = '', cls = '';
      if (b != null && v != null && isFinite(Number(b)) && isFinite(Number(v)) && Number(b) !== Number(v)) {
        const better = Number(v) > Number(b);
        arrow = better ? ' ▲' : ' ▼';
        cls = better ? 'text-ok' : 'text-bad';
      }
      out += `<tr><td class="text-muted">${esc(k)}</td><td class="num">${fmt(k, b)}</td><td class="num ${cls}">${fmt(k, v)}${arrow}</td></tr>`;
    });
    return out || '<tr><td colspan="3" class="text-muted">No metrics recorded.</td></tr>';
  }

  function experimentCard(e) {
    const variants = e.variant_metrics || {};
    const vNames = Object.keys(variants);
    const firstV = vNames.length ? variants[vNames[0]] : {};
    return `
      <div class="card mb-4" data-exp="${esc(e.id)}">
        <div class="flex-between">
          <div>
            <div class="card-title" style="margin:0">Experiment <span class="mono">${esc(e.id)}</span></div>
            <div class="hint">strategy <span class="mono">${esc(e.strategy_id)}</span> · base <span class="mono">${esc(e.base_version || '—')}</span> · ${fmtWhen(e.created_at)}${e.concluded_at ? ` · concluded ${fmtWhen(e.concluded_at)}` : ''}</div>
          </div>
          <div class="flex" style="gap:6px">${verdictBadge(e.verdict)}
            ${e.concluded_at ? '' : `<button class="btn btn-sm" data-conclude="${esc(e.id)}">Conclude…</button>`}
          </div>
        </div>
        <div class="mt-4 grid-2">
          <div>
            <div class="text-muted" style="font-size:var(--fs-xs);margin-bottom:4px">Parameter changes</div>
            <pre class="mono" style="font-size:var(--fs-xs);white-space:pre-wrap">${esc(JSON.stringify(e.parameter_changes || {}, null, 2))}</pre>
          </div>
          <div>
            <div class="text-muted" style="font-size:var(--fs-xs);margin-bottom:4px">Recommendation</div>
            <div style="font-size:var(--fs-sm)">${esc(e.recommendation || '—')}</div>
            ${(e.reasons || []).length ? `<div class="text-muted" style="font-size:var(--fs-xs);margin-top:6px">${(e.reasons || []).map((r) => `• ${esc(r)}`).join('<br>')}</div>` : ''}
          </div>
        </div>
        <div class="mt-4"><div class="table-wrap"><table class="table">
          <thead><tr><th>Metric</th><th>Baseline</th><th>Variant</th></tr></thead>
          <tbody>${metricRows(e.baseline_metrics, firstV)}</tbody>
        </table></div></div>
      </div>`;
  }

  async function loadExperiments() {
    const box = el('exp-list');
    try {
      const data = await API.get('/alex/agent/experiments?limit=25');
      _experiments = data.experiments || [];
      if (!_experiments.length) {
        box.innerHTML = `<div class="empty-state"><span class="big">🧪</span>No experiments yet. Ask Alex in chat: "backtest strategy X" or "create an experiment on X".</div>`;
        return;
      }
      box.innerHTML = _experiments.map(experimentCard).join('');
      box.querySelectorAll('[data-conclude]').forEach((b) =>
        b.addEventListener('click', () => concludeFlow(b.dataset.conclude)));
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  async function concludeFlow(id) {
    const notes = await modal.confirm({
      title: 'Conclude experiment',
      message: `Record your final decision for <code>${esc(id)}</code>.`,
      confirmLabel: 'Adopt',
    });
    if (!notes) return;
    try {
      await API.post(`/alex/agent/experiments/${encodeURIComponent(id)}/conclude`, { decision: 'adopt', notes: '' });
      toast.success(`Experiment ${id} concluded`);
      loadExperiments();
    } catch (err) {
      toast.error(err.message);
    }
  }

  /* ---------------- LLM experiment selection (Stage C2) ---------------- */

  async function selectExperiments() {
    const btn = el('exp-select');
    btn.disabled = true;
    btn.textContent = '✨ Thinking…';
    try {
      const report = await API.post('/alex/agent/experiments/select', {});
      const src = report.llm_used ? 'LLM' : (report.fallback_used ? 'deterministic fallback' : '—');
      if (report.accepted && report.accepted.length) {
        toast.success(`${report.accepted.length} experiment(s) run (${src}) — see console for details`);
        console.info('[select_experiments]', report);
      } else if (report.rejected && report.rejected.length) {
        toast.warning(`No experiments run (${src}) — see console for reasons`);
        console.info('[select_experiments] rejections:', report.rejected);
      } else {
        toast.warning('No experiment proposals');
      }
      loadExperiments();
    } catch (err) {
      toast.error(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '✨ Select (LLM)';
    }
  }

  /* ---------------- Walk-forward panel ---------------- */

  async function runWalkForward() {
    const sid = el('wf-strategy').value.trim();
    const symbol = el('wf-symbol').value.trim();
    const months = Number(el('wf-months').value || 12);
    if (!sid || !symbol) { toast.warning('Strategy ID and symbol are required'); return; }
    const end = new Date();
    const start = new Date(end.getTime() - months * 30.44 * 24 * 3600 * 1000);
    const iso = (d) => d.toISOString().slice(0, 19);
    const out = el('wf-result');
    out.innerHTML = '<div class="text-muted">Running walk-forward… (real backtests, this takes a while)</div>';
    try {
      const r = await API.post('/alex/agent/experiments/walk-forward', {
        strategy_id: sid, symbol,
        start_date: iso(start), end_date: iso(end),
        timeframe: el('wf-tf').value, window_months: 3, step_months: 3,
      });
      renderWF(r);
    } catch (err) {
      out.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  function renderWF(r) {
    const agg = r.aggregate_metrics || {};
    const win = r.windows || [];
    const verdict = verdictBadge(r.verdict);
    const stat = (k, label, fmt) => {
      const v = agg[k];
      return `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value" style="font-size:var(--fs-lg)">${v == null ? '—' : fmt(v)}</div></div>`;
    };
    const html = `
      <div class="flex-between mb-2"><div class="card-title" style="margin:0">Walk-forward — ${esc(r.strategy_name || r.strategy_id)} · ${esc(r.symbol)} ${esc(r.timeframe || '')}</div>${verdict}</div>
      <div class="card-grid grid-4 mb-4">
        ${stat('is_sharpe_mean', 'IS Sharpe mean', (v) => Number(v).toFixed(2))}
        ${stat('oos_sharpe_mean', 'OOS Sharpe mean', (v) => Number(v).toFixed(2))}
        ${stat('oos_pf_mean', 'OOS PF mean', (v) => Number(v).toFixed(2))}
        ${stat('degradation_overfitting', 'Overfitting', (v) => Number(v).toFixed(2))}
      </div>
      ${win.length ? `<div class="table-wrap"><table class="table">
        <thead><tr><th>#</th><th>IS bars</th><th>OOS bars</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS PF</th><th>OOS trades</th><th>OOS DD</th></tr></thead>
        <tbody>${win.map((w) => {
          const oosCls = w.oos_sharpe > 0 ? 'text-ok' : 'text-bad';
          return `<tr><td>${w.window}</td><td class="num">${w.is_bars}</td><td class="num">${w.oos_bars}</td>
          <td class="num">${Number(w.is_sharpe).toFixed(2)}</td><td class="num ${oosCls}">${Number(w.oos_sharpe).toFixed(2)}</td>
          <td class="num">${Number(w.oos_pf).toFixed(2)}</td><td class="num">${w.oos_trades}</td>
          <td class="num">${(Number(w.oos_max_dd) * 100).toFixed(1)}%</td></tr>`;
        }).join('')}</tbody></table></div>` : ''}
      <div class="mt-4"><canvas id="wf-chart" data-h="240"></canvas></div>
      <div class="hint mt-2">Stability score ${r.stability_score != null ? Number(r.stability_score).toFixed(2) : '—'} (0 = stable, 1 = unstable) · ${r.windows_tested} windows</div>
    `;
    const out = el('wf-result');
    out.innerHTML = html;
    const canvas = el('wf-chart');
    if (canvas && win.length) {
      window.AlexCharts.lines(canvas, [
        { name: 'IS Sharpe', values: win.map((w) => w.is_sharpe) },
        { name: 'OOS Sharpe', values: win.map((w) => w.oos_sharpe) },
      ], { height: 240 });
    }
  }

  /* ---------------- Compare panel ---------------- */

  async function runCompare() {
    const raw = el('cmp-ids').value.trim();
    const symbol = el('cmp-symbol').value.trim();
    if (!raw || !symbol) { toast.warning('Need strategy IDs and a symbol'); return; }
    const ids = raw.split(',').map((s) => s.trim()).filter(Boolean);
    const end = new Date();
    const start = new Date(end.getTime() - 180 * 24 * 3600 * 1000);
    const out = el('cmp-result');
    out.innerHTML = '<div class="text-muted">Comparing… (real backtests per strategy)</div>';
    try {
      const results = await Promise.all(ids.map((sid) =>
        API.post('/alex/agent/experiments/backtest', {
          strategy_id: sid, symbol,
          start_date: start.toISOString().slice(0, 19), end_date: end.toISOString().slice(0, 19),
          timeframe: 'H1',
        }).then((r) => ({ sid, r })).catch((e) => ({ sid, err: e.message }))
      ));
      out.innerHTML = `<div class="table-wrap"><table class="table">
        <thead><tr><th>Strategy</th><th>Trades</th><th>Win rate</th><th>PF</th><th>Sharpe</th><th>Return</th></tr></thead>
        <tbody>${results.map(({ sid, r, err }) => {
          if (err) return `<tr><td class="mono">${esc(sid)}</td><td colspan="5" class="text-bad">${esc(err)}</td></tr>`;
          const m = r.metrics || {};
          return `<tr><td class="mono">${esc(sid)}</td><td class="num">${r.trades ?? m.total_trades ?? '—'}</td>
            <td class="num">${m.win_rate != null ? (Number(m.win_rate) * (Number(m.win_rate) <= 1 ? 100 : 1)).toFixed(1) + '%' : '—'}</td>
            <td class="num">${m.profit_factor != null ? Number(m.profit_factor).toFixed(2) : '—'}</td>
            <td class="num">${m.sharpe_ratio != null ? Number(m.sharpe_ratio).toFixed(2) : '—'}</td>
            <td class="num ${Number(m.total_return_pct) >= 0 ? 'text-ok' : 'text-bad'}">${m.total_return_pct != null ? Number(m.total_return_pct).toFixed(1) + '%' : '—'}</td></tr>`;
        }).join('')}</tbody></table></div>`;
    } catch (err) {
      out.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  /* ---------------- Mount ---------------- */

  async function mount(container) {
    _container = container;
    container.innerHTML = `
      <div class="card mb-4">
        <div class="flex-between">
          <div class="card-title" style="margin:0">Walk-forward analysis</div>
          <span class="text-muted" style="font-size:var(--fs-xs)">rolling IS→OOS windows · no re-fitting</span>
        </div>
        <div class="flex mt-2" style="flex-wrap:wrap;gap:8px">
          <input class="input" id="wf-strategy" placeholder="strategy id (e.g. f39b8d40)" style="max-width:220px">
          <input class="input" id="wf-symbol" placeholder="symbol (e.g. GBP/JPY)" style="max-width:140px" value="GBP/JPY">
          <select class="input" id="wf-tf" style="max-width:90px">
            ${['M15', 'H1', 'H4', 'D1'].map((t) => `<option ${t === 'H1' ? 'selected' : ''}>${t}</option>`).join('')}
          </select>
          <select class="input" id="wf-months" style="max-width:120px">
            ${[6, 12, 24].map((m) => `<option value="${m}" ${m === 12 ? 'selected' : ''}>${m} months</option>`).join('')}
          </select>
          <button class="btn btn-primary" id="wf-go">Run</button>
        </div>
        <div id="wf-result" class="mt-4"></div>
      </div>

      <div class="card mb-4">
        <div class="flex-between">
          <div class="card-title" style="margin:0">Compare strategies</div>
          <span class="text-muted" style="font-size:var(--fs-xs)">6-month backtest each · same costs</span>
        </div>
        <div class="flex mt-2" style="flex-wrap:wrap;gap:8px">
          <input class="input" id="cmp-ids" placeholder="ids: f39b8d40, ab12cd34" style="max-width:280px">
          <input class="input" id="cmp-symbol" value="EUR/USD" style="max-width:130px">
          <button class="btn btn-primary" id="cmp-go">Compare</button>
        </div>
        <div id="cmp-result" class="mt-4"></div>
      </div>

      <div class="flex-between mb-2">
        <div class="card-title">Experiment records</div>
        <div class="flex" style="gap:8px">
          <button class="btn btn-sm" id="exp-select" title="LLM chooses what to test next from a deterministic menu; validation + real engines own every number">✨ Select (LLM)</button>
          <button class="btn btn-sm" id="exp-refresh">⟳ Refresh</button>
        </div>
      </div>
      <div id="exp-list"><div class="skeleton" style="width:80%"></div><div class="skeleton" style="width:60%"></div></div>
    `;
    el('wf-go').addEventListener('click', runWalkForward);
    el('cmp-go').addEventListener('click', runCompare);
    el('exp-select').addEventListener('click', selectExperiments);
    el('exp-refresh').addEventListener('click', loadExperiments);
    await loadExperiments();
  }

  window.Router.register('experiments', {
    title: 'Experiments', subtitle: 'Research laboratory · variant diffs · walk-forward · comparison', icon: '🧪', mount,
  });
})();
