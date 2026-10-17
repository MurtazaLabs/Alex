/**
 * Alex OS — Research space: the autonomy loop (brief → proposal → gate) + history.
 */
(function () {
  'use strict';

  const { el, esc, fmtWhen, verdictBadge } = UI;

  function runDetailHtml(run) {
    const v = run.validation || run.validation_json || null;
    if (!v) return '';
    const fb = v.full_backtest || {};
    const wfo = v.walk_forward || {};
    const cell = (x) => (x === undefined || x === null || x === '') ? '—' : esc(String(x));
    return `
      <div class="grid-3 mt-4">
        <div class="card"><div class="card-title">Full backtest</div>
          <div class="evidence">
            <div class="evidence-row"><div class="k">Trades</div><div class="v num">${cell(fb.total_trades)}</div></div>
            <div class="evidence-row"><div class="k">Profit factor</div><div class="v num">${cell(fb.profit_factor)}</div></div>
            <div class="evidence-row"><div class="k">Max DD</div><div class="v num">${cell(fb.max_drawdown_pct)}</div></div>
          </div></div>
        <div class="card"><div class="card-title">Walk-forward (OOS)</div>
          <div class="evidence">
            <div class="evidence-row"><div class="k">Sharpe</div><div class="v num">${cell(wfo.sharpe ?? wfo.oos_sharpe)}</div></div>
            <div class="evidence-row"><div class="k">Windows</div><div class="v num">${cell(wfo.windows)}</div></div>
            <div class="evidence-row"><div class="k">Overfit score</div><div class="v num">${cell(wfo.overfit_score)}</div></div>
          </div></div>
        <div class="card"><div class="card-title">Gate reasons</div>
          ${(v.reasons && v.reasons.length) ? v.reasons.map((r) => `<div class="text-secondary" style="padding:3px 0">• ${esc(r)}</div>`).join('') : '<div class="text-muted">—</div>'}
        </div>
      </div>`;
  }

  async function runResearch() {
    const pair = el('res-pair').value.trim().toUpperCase();
    const days = Number(el('res-days').value) || 365;
    const out = el('res-result');
    if (!/^[A-Z]{3}\/[A-Z]{3}$/.test(pair)) { UI.toast.error('Pair must look like EUR/USD'); return; }
    el('res-go').disabled = true;
    out.innerHTML = `
      <div class="card">
        <div class="flex"><span class="dot dot-ok"></span><strong>Research running on ${esc(pair)}…</strong></div>
        <div class="hint mt-2">Building brief → LLM proposes a strategy spec → deterministic backtest + walk-forward gate. Can take 1–5 minutes.</div>
        <div class="skeleton mt-4" style="width:70%"></div><div class="skeleton" style="width:50%"></div>
      </div>`;
    try {
      const res = await API.post('/alex/research/run', { pair, days });
      const passed = res.validation && res.validation.passed;
      out.innerHTML = `
        <div class="card">
          <div class="flex-between mb-4">
            <div class="flex"><strong style="font-size:var(--fs-lg)">${esc(res.pair || pair)}</strong>
              ${passed ? '<span class="badge badge-success">GATE PASSED</span>' : '<span class="badge badge-danger">GATE REJECTED</span>'}</div>
            <span class="text-muted">${fmtWhen(new Date().toISOString())}</span>
          </div>
          ${(res.messages || []).map((m) => `<div class="text-secondary" style="padding:2px 0">› ${esc(m)}</div>`).join('')}
          ${runDetailHtml(res)}
        </div>`;
      loadHistory();
    } catch (err) {
      out.innerHTML = `<div class="card"><div class="empty-state">⚠️ ${esc(err.message)}</div></div>`;
    } finally {
      el('res-go').disabled = false;
    }
  }

  const DIR_STATUS_ICON = { QUEUED: '⏳', RUNNING: '🔄', DONE_VALIDATED: '✅', DONE_REJECTED: '❌', DONE_ERROR: '⚠️', CANCELLED: '🚫' };

  async function loadDirector() {
    const box = el('res-director');
    if (!box) return;
    try {
      const snap = await API.get('/alex/director/queue');
      const bs = snap.by_status || {};
      const rows = (snap.queue || []).map((i) => `
        <tr>
          <td><span class="badge ${i.priority === 'HIGH' ? 'badge-danger' : i.priority === 'LOW' ? 'badge-muted' : 'badge-warning'}">${esc(i.priority)}</span></td>
          <td><strong>${esc(i.pair)}</strong> <span class="text-muted">${esc(i.timeframe)}</span></td>
          <td>${DIR_STATUS_ICON[i.status] || '⏳'} ${esc(i.status)}</td>
          <td class="text-secondary">${esc(((i.hypothesis || '') || ((i.result || {}).status) || '').slice(0, 60))}</td>
        </tr>`).join('');
      box.innerHTML = `
        <div class="flex-between mb-2">
          <div class="text-secondary">Open: <strong>${snap.open_items ?? 0}</strong>
            · queued ${bs.QUEUED || 0} · validated ${bs.DONE_VALIDATED || 0} · rejected ${bs.DONE_REJECTED || 0} · errors ${bs.DONE_ERROR || 0}</div>
        </div>
        ${rows
          ? `<div class="table-wrap"><table class="table"><thead><tr><th>Priority</th><th>Pair</th><th>Status</th><th>Hypothesis / result</th></tr></thead><tbody>${rows}</tbody></table></div>`
          : '<div class="empty-state"><span class="big">🎬</span>Queue empty — the Director is idle.</div>'}
        <div class="hint mt-2">Budget-capped cycles: research tools only, LLM proposals capped per cycle (deterministic fallback after), every action audited. No live execution, no risk changes — human approval still required.</div>`;
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  async function enqueueDirector() {
    const pair = el('dir-pair').value.trim().toUpperCase();
    const hyp = el('dir-hyp').value.trim();
    if (!/^[A-Z]{3}\/[A-Z]{3}$/.test(pair)) { UI.toast.error('Pair must look like EUR/USD'); return; }
    try {
      const res = await API.post('/alex/director/queue', { pair, hypothesis: hyp });
      if (res.enqueued) { UI.toast.success(`Enqueued ${pair}`); el('dir-hyp').value = ''; }
      else UI.toast.warning(res.reason || 'Not enqueued');
      loadDirector();
    } catch (err) {
      UI.toast.error(err.message);
    }
  }

  async function runDirectorCycle() {
    const btn = el('dir-cycle');
    btn.disabled = true;
    btn.textContent = '🔄 Cycle running…';
    try {
      const report = await API.post('/alex/director/cycle', { max_items: 3 });
      UI.toast.success(`Cycle ${report.cycle_id}: ${report.validated} validated, ${report.rejected} rejected, ${report.errors} errors`);
      loadDirector();
      loadHistory();
    } catch (err) {
      UI.toast.error(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '🎬 Run cycle';
    }
  }

  async function suggestHypotheses() {
    const btn = el('dir-suggest');
    btn.disabled = true;
    btn.textContent = '✨ Thinking…';
    try {
      const report = await API.post('/alex/director/generate-hypotheses', { max_hypotheses: 3 });
      const src = report.llm_used ? 'LLM' : (report.fallback_used ? 'deterministic fallback' : '—');
      const acc = (report.accepted || []).filter((h) => h.enqueued);
      if (acc.length) {
        UI.toast.success(`${acc.length} hypothesis(es) enqueued (${src})`);
        console.info('[generate_hypotheses]', report);
      } else if (report.rejected && report.rejected.length) {
        UI.toast.warning(`No candidates accepted (${src}) — see console for reasons`);
        console.info('[generate_hypotheses] rejections:', report.rejected);
      } else {
        UI.toast.warning('No hypotheses generated');
      }
      loadDirector();
    } catch (err) {
      UI.toast.error(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '✨ Suggest (LLM)';
    }
  }

  const PLAN_STATE_ICON = { done: '✅', in_progress: '🔄', skipped: '➖', pending: '⬜' };
  const PLAN_STATUS_ICON = { COMPLETED: '✅', IN_PROGRESS: '🔄', PENDING: '⏳', FAILED: '⚠️', CANCELLED: '🚫' };

  function planCardHtml(card) {
    const step = (s) => `<span style="margin-right:10px">${PLAN_STATE_ICON[s.state] || '⬜'} ${esc(s.label)}</span>`;
    const pairs = Object.entries(card.pairs_progress || {})
      .map(([pair, p]) => `<span class="badge ${p.status === 'validated' ? 'badge-success' : ['validated', 'rejected', 'error', 'cancelled'].includes(p.status) ? 'badge-danger' : 'badge-muted'}">${esc(pair)}: ${esc(p.status)}</span>`)
      .join(' ');
    const concl = card.conclusion
      ? `<div class="hint mt-2"><strong>Conclusion:</strong> ${esc(card.conclusion.verdict || '?')} — ${esc(card.conclusion.summary || '')}</div>`
      : '';
    return `
      <div class="card mt-2" style="border-left:3px solid var(--primary)">
        <div class="flex-between">
          <div><strong>RESEARCH #${esc(String(card.task_id || '').slice(0, 8))}</strong> — ${esc(card.topic || '')}
            ${PLAN_STATUS_ICON[card.status] || '⏳'} <span class="text-muted">${esc(card.status)}</span></div>
          <span class="text-muted nowrap">${fmtWhen(card.created_at)}</span>
        </div>
        <div class="hint mt-1">${esc(card.hypothesis || '')}</div>
        <div class="text-secondary mt-1">${esc((card.pairs || []).join(', '))} · ${esc((card.timeframes || []).join(', '))} · ${esc(String(card.period_days || '?'))}d</div>
        <div class="progress mt-2" style="height:6px"><div class="progress-bar" style="width:${Number(card.progress_percent) || 0}%"></div></div>
        <div class="flex mt-2" style="flex-wrap:wrap;gap:4px">${(card.experiment_checklist || []).map(step).join('')}</div>
        <div class="flex mt-1" style="flex-wrap:wrap;gap:4px">${(card.validation_checklist || []).map(step).join('')}</div>
        <div class="flex mt-2" style="flex-wrap:wrap;gap:4px">${pairs || '<span class="text-muted">no pairs queued yet</span>'}</div>
        ${concl}
      </div>`;
  }

  async function loadPlans() {
    const box = el('res-plans');
    if (!box) return;
    try {
      const plans = await API.get('/alex/research/plans?limit=10');
      box.innerHTML = (Array.isArray(plans) && plans.length)
        ? plans.map(planCardHtml).join('')
        : '<div class="empty-state"><span class="big">🗺️</span>No structured plans yet — create one above; the Director works it cycle by cycle.</div>';
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  async function createPlan() {
    const btn = el('plan-go');
    const topic = el('plan-topic').value.trim();
    const pairs = el('plan-pairs').value.split(',').map((p) => p.trim()).filter(Boolean);
    if (!topic) { UI.toast.error('Describe what to research'); return; }
    btn.disabled = true;
    try {
      const res = await API.post('/alex/research/plan', { topic, pairs });
      UI.toast.success(`Plan #${String(res.task_id || '').slice(0, 8)} created — pairs queued with the Director`);
      el('plan-topic').value = '';
      el('plan-pairs').value = 'EUR/USD, GBP/USD';
      loadPlans();
      loadDirector();
    } catch (err) {
      UI.toast.error(err.message);
    } finally {
      btn.disabled = false;
    }
  }

  async function loadHistory() {
    const box = el('res-history');
    try {
      const rows = await API.get('/alex/research/history?limit=25');
      box.innerHTML = (Array.isArray(rows) && rows.length)
        ? `<div class="table-wrap"><table class="table">
            <thead><tr><th>Topic / pair</th><th>Verdict</th><th>Conclusion</th><th>When</th></tr></thead>
            <tbody>${rows.map((r) => `
              <tr>
                <td><strong>${esc(r.topic || '?')}</strong></td>
                <td>${verdictBadge(r.verdict)}</td>
                <td class="text-secondary">${esc((r.conclusion || '').slice(0, 120))}</td>
                <td class="text-muted nowrap">${fmtWhen(r.created)}</td>
              </tr>`).join('')}</tbody></table></div>`
        : '<div class="empty-state"><span class="big">🔬</span>No research logged yet — run your first loop above.</div>';
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  async function mount(container) {
    container.innerHTML = `
      <div class="card mb-4">
        <div class="card-title">Run the Research Loop</div>
        <div class="form-row">
          <div class="field"><label>Pair</label><input class="input" id="res-pair" placeholder="EUR/USD" value="EUR/USD"></div>
          <div class="field" style="max-width:140px"><label>History (days)</label>
            <select class="input" id="res-days"><option>180</option><option selected>365</option><option>540</option><option>730</option></select></div>
          <div class="field" style="max-width:170px;align-self:flex-end"><button class="btn btn-primary" id="res-go" style="width:100%">🔬 Propose &amp; validate</button></div>
        </div>
        <div class="hint">LLM proposes only the strategy shape; every number is computed by deterministic code. Failures are logged honestly.</div>
      </div>
      <div id="res-result"></div>
      <div class="card mt-4">
        <div class="flex-between">
          <div class="card-title">Research Plans (B5)</div>
          <div class="flex">
            <input class="input" id="plan-topic" placeholder="What should Alex research?" style="width:260px">
            <input class="input" id="plan-pairs" placeholder="EUR/USD, GBP/USD" value="EUR/USD, GBP/USD" style="width:200px">
            <button class="btn btn-primary" id="plan-go">+ New plan</button>
          </div>
        </div>
        <div id="res-plans"><div class="skeleton" style="width:65%"></div><div class="skeleton" style="width:45%"></div></div>
      </div>
      <div class="card mt-4">
        <div class="flex-between">
          <div class="card-title">Research Director Queue</div>
          <div class="flex">
            <input class="input" id="dir-pair" placeholder="GBP/JPY" style="width:130px">
            <input class="input" id="dir-hyp" placeholder="Hypothesis (optional)" style="width:220px">
            <button class="btn" id="dir-add">+ Enqueue</button>
            <button class="btn" id="dir-suggest" title="LLM proposes what to research; deterministic validation + gate still own every outcome">✨ Suggest (LLM)</button>
            <button class="btn btn-primary" id="dir-cycle">🎬 Run cycle</button>
          </div>
        </div>
        <div id="res-director"><div class="skeleton" style="width:65%"></div><div class="skeleton" style="width:45%"></div></div>
      </div>
      <div class="card mt-4">
        <div class="flex-between">
          <div class="card-title">Research Reports</div>
          <button class="btn" id="rep-go" title="LLM narrates over stored evidence; every number is rendered by code from the records and traceability-checked">📄 Generate report</button>
        </div>
        <div id="res-reports"><div class="skeleton" style="width:65%"></div><div class="skeleton" style="width:45%"></div></div>
      </div>
      <div class="card mt-4">
        <div class="card-title">Research History</div>
        <div id="res-history"><div class="skeleton" style="width:65%"></div><div class="skeleton" style="width:45%"></div></div>
      </div>
    `;
    el('res-go').addEventListener('click', runResearch);
    el('res-pair').addEventListener('keydown', (e) => { if (e.key === 'Enter') runResearch(); });
    el('dir-add').addEventListener('click', enqueueDirector);
    el('dir-suggest').addEventListener('click', suggestHypotheses);
    el('dir-cycle').addEventListener('click', runDirectorCycle);
    el('rep-go').addEventListener('click', generateReport);
    el('plan-go').addEventListener('click', createPlan);
    el('plan-topic').addEventListener('keydown', (e) => { if (e.key === 'Enter') createPlan(); });
    await loadPlans();
    await loadDirector();
    await loadHistory();
    await loadReports();
  }

  /* ---------------- Research reports (Stage C3) ---------------- */

  async function generateReport() {
    const btn = el('rep-go');
    btn.disabled = true;
    btn.textContent = '📄 Writing…';
    try {
      const doc = await API.post('/alex/research/report', { topic: 'Research summary' });
      const src = doc.llm_used ? 'LLM narration' : 'deterministic fallback';
      toast.success(`Report ${String(doc.report_id || '').slice(0, 8)} written (${src})`);
      console.info('[generate_research_report]', doc);
      loadReports();
    } catch (err) {
      toast.error(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = '📄 Generate report';
    }
  }

  async function loadReports() {
    const box = el('res-reports');
    try {
      const data = await API.get('/alex/research/reports?limit=10');
      const reports = data.reports || [];
      if (!reports.length) {
        box.innerHTML = '<div class="empty-state">No reports yet — generate one from stored evidence.</div>';
        return;
      }
      box.innerHTML = reports.map((r) => `
        <div class="flex-between" style="padding:6px 0;border-bottom:1px solid var(--border)">
          <div>
            <strong>${esc(r.title || r.topic)}</strong>
            <span class="text-muted" style="font-size:var(--fs-xs)"> · ${esc(String(r.report_id).slice(0, 8))} · ${r.evidence_count} evidence records · ${r.llm_used ? 'LLM' : 'fallback'}</span>
          </div>
          <span class="text-muted" style="font-size:var(--fs-xs)">${esc(fmtWhen(r.created_at))}</span>
        </div>`).join('');
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
    }
  }

  window.Router.register('research', {
    title: 'Research', subtitle: 'Autonomy loop with an honest statistical gate', icon: '🔬', mount,
  });
})();
