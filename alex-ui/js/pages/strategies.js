/**
 * Alex OS — Strategies space: Strategy Health (B1) + the human approval panel.
 * Health is deterministic (computed server-side from stored state).
 * Approval is YOURS alone — the UI double-confirms every decision.
 */
(function () {
  'use strict';

  const { el, esc, fmtWhen, toast, modal, statusBadge } = UI;

  const LIGHT_ICON = { GREEN: '🟢', AMBER: '🟡', RED: '🔴', GREY: '⚪' };
  const BADGE_CLASS = {
    CANDIDATE: 'badge-success',
    PROBATION: 'badge-info',
    'RESEARCH ONLY': 'badge-warning',
    'RETIRE?': 'badge-danger',
  };

  function healthRow(h) {
    const lights = (h.lights || []).map((l) => {
      const title = `${l.name} — ${esc(l.detail)}${l.as_of ? ` (as of ${esc(String(l.as_of).slice(0, 10))})` : ''}`;
      return `<span title="${title}" style="cursor:help">${LIGHT_ICON[l.status] || '⚪'}</span>`;
    }).join(' ');
    const badgeCls = BADGE_CLASS[h.badge] || 'badge-muted';
    const age = h.staleness && h.staleness.age_days != null ? `${h.staleness.age_days}d` : '—';
    const stale = h.staleness && h.staleness.stale
      ? ' <span class="badge badge-warning">STALE — refresh</span>' : '';
    const autoDown = h.auto_downgraded ? ' <span class="badge badge-warning">auto-downgraded</span>' : '';
    return `<div class="evidence-row" style="gap:12px">
      <div class="k" style="min-width:200px">
        <strong>${esc(h.strategy_name || h.strategy_id)}</strong>
        <div class="text-muted mono" style="font-size:var(--fs-xs)">${esc(h.strategy_id)}${h.pair ? ' · ' + esc(h.pair) : ''}${h.timeframe ? ' ' + esc(h.timeframe) : ''}</div>
      </div>
      <div class="v flex" style="align-items:center;gap:10px;flex-wrap:wrap">
        <span class="badge ${badgeCls}">${esc(h.badge)}${autoDown}</span>
        <span style="letter-spacing:2px">${lights}</span>
        <span class="text-muted" style="font-size:var(--fs-xs)">evidence ${age}${stale}</span>
        <button class="btn btn-sm" data-act="health-detail" data-id="${esc(h.strategy_id)}">Details</button>
      </div>
    </div>`;
  }

  async function showHealthDetail(id) {
    try {
      const res = await API.get(`/alex/strategies/health/${encodeURIComponent(id)}`);
      const h = res.health || {};
      const rows = (h.lights || []).map((l) => {
        const asOf = l.as_of ? esc(String(l.as_of).slice(0, 10)) : 'no date';
        return `<div class="evidence-row">
          <div class="k">${LIGHT_ICON[l.status] || '⚪'} ${esc(l.name)}${l.stale ? ' <span class="badge badge-warning">STALE</span>' : ''}</div>
          <div class="v">${esc(l.detail)} <span class="text-muted">(as of ${asOf})</span></div>
        </div>`;
      }).join('');
      const st = h.staleness || {};
      modal.show({
        title: `Health — ${res.strategy_name || id}`,
        body: `
          <div class="evidence-row"><div class="k">Badge</div><div class="v"><span class="badge ${BADGE_CLASS[h.badge] || 'badge-muted'}">${esc(h.badge)}</span></div></div>
          ${rows}
          <div class="evidence-row"><div class="k">Staleness</div><div class="v">${esc(st.note || '—')}</div></div>
          <div class="evidence-row"><div class="k">Recommendation</div><div class="v">${esc(h.recommendation || '—')}</div></div>`,
        footer: '<button class="btn" id="health-detail-close">Close</button>',
      });
      el('health-detail-close').addEventListener('click', () => modal.hide());
    } catch (err) {
      toast.error(err.message);
    }
  }

  async function decide(versionId, name, decision) {
    const ok = await modal.confirm({
      title: decision === 'approve' ? 'Approve strategy' : 'Reject strategy',
      message: `This will <strong>${decision}</strong> <code>${esc(versionId)}</code>${name ? ` — ${esc(name)}` : ''}. ` +
        (decision === 'approve' ? 'It moves to the approved set.' : 'It will be archived as rejected.'),
      confirmLabel: decision === 'approve' ? 'Approve' : 'Reject',
      danger: decision === 'reject',
    });
    if (!ok) return;
    try {
      await API.post(`/alex/strategies/probation/${encodeURIComponent(versionId)}/${decision}`);
      toast.success(decision === 'approve' ? `Approved ${versionId}` : `Rejected ${versionId}`);
      mountCurrent();
    } catch (err) {
      toast.error(err.message);
    }
  }

  let _container = null;
  function mountCurrent() { if (_container) mount(_container, { force: true }); }

  async function mount(container) {
    _container = container;
    container.innerHTML = `
      <div class="card mb-4">
        <div class="flex-between">
          <div class="card-title" style="margin:0">Strategy Health</div>
          <button class="btn btn-sm" id="health-refresh">⟳ Refresh</button>
        </div>
        <div class="hint mb-4">Deterministic lights from stored evidence — profitability, robustness, sample size, OOS consistency, drawdown, pair generality, regime robustness. No re-runs; stale evidence is flagged and downgraded; the last two lights fill in after a robustness sweep.</div>
        <div id="health-rows"><div class="skeleton" style="width:75%"></div><div class="skeleton" style="width:55%"></div></div>
      </div>
      <div class="card">
        <div class="flex-between">
          <div class="card-title" style="margin:0">Probation Queue</div>
          <button class="btn btn-sm" id="prob-refresh">⟳ Refresh</button>
        </div>
        <div class="hint mb-4">Candidates replay their own spec on real MT5 bars and must earn their quota before you decide. Approval is human-only.</div>
        <div id="prob-rows"><div class="skeleton" style="width:75%"></div><div class="skeleton" style="width:55%"></div></div>
      </div>
      <div class="card">
        <div class="flex-between">
          <div class="card-title" style="margin:0">Ready for Review (C4)</div>
          <button class="btn btn-sm" id="promo-refresh">⟳ Refresh</button>
        </div>
        <div class="hint mb-4">Deterministic ranking of probation candidates whose objective gates are MET — packaged for your approval decision. Nothing is promoted automatically; approve/reject stay human-only.</div>
        <div id="promo-rows"><div class="skeleton" style="width:75%"></div></div>
      </div>
    `;
    el('health-refresh').addEventListener('click', mountCurrent);
    el('prob-refresh').addEventListener('click', mountCurrent);
    el('promo-refresh').addEventListener('click', mountCurrent);

    // --- Strategy Health (stored-state read, <1s) ---
    const healthBox = el('health-rows');
    try {
      const cards = await API.get('/alex/strategies/health');
      if (!cards.length) {
        healthBox.innerHTML = '<div class="empty-state"><span class="big">🩺</span>No saved strategies yet. Research proposals that pass the gate appear here.</div>';
      } else {
        healthBox.innerHTML = cards.map(healthRow).join('');
        healthBox.querySelectorAll('[data-act="health-detail"]').forEach((b) => {
          b.addEventListener('click', () => showHealthDetail(b.dataset.id));
        });
      }
    } catch (err) {
      healthBox.innerHTML = `<div class="empty-state">Health unavailable: ${esc(err.message)}</div>`;
    }

    // --- Probation queue (unchanged) ---
    const rows = await API.get('/alex/strategies/probation');
    const box = el('prob-rows');
    if (!rows.length) {
      box.innerHTML = '<div class="empty-state"><span class="big">🧪</span>No candidates in probation. Run Research to propose one.</div>';
      return;
    }
    box.innerHTML = `<div class="table-wrap"><table class="table">
      <thead><tr><th>Strategy</th><th>Pair</th><th>Status</th><th>Trades</th><th>Progress</th><th></th></tr></thead>
      <tbody>${rows.map((r) => {
        const pct = r.required_trades ? Math.min(100, Math.round((r.observed_trades / r.required_trades) * 100)) : 0;
        const eligible = !!r.eligible_for_approval;
        const live = ['proposing', 'active'].includes(r.status);
        return `<tr>
          <td>
            <div><strong>${esc(r.strategy_name || r.version_id)}</strong></div>
            <div class="text-muted mono" style="font-size:var(--fs-xs)">${esc(r.version_id)}</div>
          </td>
          <td>${esc(r.pair || '—')}</td>
          <td>${statusBadge(r.status)}</td>
          <td class="num">${r.observed_trades}/${r.required_trades}</td>
          <td style="min-width:140px"><div class="progress"><div style="width:${pct}%"></div></div></td>
          <td><div class="actions">
            ${live && eligible ? `
              <button class="btn btn-sm btn-primary" data-act="approve" data-id="${esc(r.version_id)}" data-name="${esc(r.strategy_name || '')}">Approve</button>
              <button class="btn btn-sm btn-danger" data-act="reject" data-id="${esc(r.version_id)}" data-name="${esc(r.strategy_name || '')}">Reject</button>`
            : live ? `<span class="badge badge-info">accruing ${pct}%</span>`
            : `<span class="text-muted">${esc(r.notes || '').slice(0, 40)}</span>`}
          </div></td>
        </tr>`;
      }).join('')}</tbody></table></div>`;

    box.querySelectorAll('[data-act]').forEach((b) => {
      b.addEventListener('click', () => decide(b.dataset.id, b.dataset.name, b.dataset.act));
    });

    const activeCount = rows.filter((r) => ['proposing', 'active'].includes(r.status)).length;
    const badge = el('nav-badge-strategies');
    if (badge) { badge.textContent = String(activeCount); badge.classList.toggle('hidden', activeCount === 0); }

    // --- Promotion review (Stage C4): deterministic, read-only ---
    const promoBox = el('promo-rows');
    try {
      const review = await API.get('/alex/research/promotions?limit=5');
      const cands = review.candidates || [];
      if (!cands.length) {
        promoBox.innerHTML = '<div class="empty-state"><span class="big">📣</span>No probation candidates have met their gates yet.</div>';
      } else {
        promoBox.innerHTML = `<div class="table-wrap"><table class="table">
          <thead><tr><th>Strategy</th><th>Pair</th><th>Gates</th><th>Trades</th><th></th></tr></thead>
          <tbody>${cands.map((c) => `<tr>
            <td>
              <div><strong>${esc(c.strategy_name || c.version_id)}</strong></div>
              <div class="text-muted mono" style="font-size:var(--fs-xs)">${esc(c.version_id)}</div>
            </td>
            <td>${esc(c.pair || '—')}</td>
            <td>${c.ready ? '<span class="badge badge-success">MET</span>' : '<span class="badge badge-info">accruing</span>'}</td>
            <td class="num">${c.observed_trades}/${c.required_trades}</td>
            <td class="text-muted" style="max-width:340px;font-size:var(--fs-xs)">${esc(c.narration || '')}</td>
          </tr>`).join('')}</tbody></table></div>`;
      }
    } catch (err) {
      promoBox.innerHTML = `<div class="empty-state">Promotion review unavailable: ${esc(err.message)}</div>`;
    }
  }

  window.Router.register('strategies', {
    title: 'Strategies', subtitle: 'Health · probation queue · your approval is required', icon: '🧠', mount,
  });
})();
