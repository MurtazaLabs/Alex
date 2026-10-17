/**
 * Alex OS — Knowledge space: Alex's research memory.
 * Full-text search across research tasks, experiments, hypotheses and
 * probation rejections. Results render as typed why-cards.
 */
(function () {
  'use strict';

  const { el, esc, toast, fmtWhen } = UI;

  let _container = null;
  let _lastQuery = '';
  let _debounce = null;

  const TYPE_META = {
    research_task: { icon: '🔬', label: 'Research task' },
    experiment: { icon: '🧪', label: 'Experiment' },
    hypothesis: { icon: '💡', label: 'Hypothesis' },
    probation: { icon: '🛡️', label: 'Gate / probation' },
    rejection: { icon: '⛔', label: 'Rejection' },
    strategy: { icon: '🧠', label: 'Strategy' },
  };

  function resultCard(r, i) {
    const meta = TYPE_META[r.type] || { icon: '📄', label: r.type || 'Record' };
    const score = r.score != null ? `<span class="text-muted" style="font-size:var(--fs-xs)">relevance ${(Number(r.score) * 100).toFixed(0)}%</span>` : '';
    const body = r.snippet || r.summary || r.reason || r.description || r.conclusion || r.recommendation || '';
    const metaBits = [
      r.symbol ? `symbol ${esc(r.symbol)}` : '',
      r.strategy_id ? `strategy <span class="mono">${esc(r.strategy_id)}</span>` : '',
      r.verdict ? verdictBadge(r.verdict) : '',
      r.created_at || r.date ? fmtWhen(r.created_at || r.date) : '',
    ].filter(Boolean).join(' · ');
    return `
      <div class="card mb-2" data-idx="${i}">
        <div class="flex-between">
          <div><span style="margin-right:6px">${meta.icon}</span><strong>${esc(r.title || r.name || r.topic || meta.label)}</strong>
            <span class="badge badge-muted" style="margin-left:8px">${meta.label}</span></div>
          ${score}
        </div>
        ${metaBits ? `<div class="hint mt-1">${metaBits}</div>` : ''}
        ${body ? `<div class="mt-2" style="font-size:var(--fs-sm)">${esc(String(body)).slice(0, 400)}</div>` : ''}
      </div>`;
  }

  function verdictBadge(v) {
    const s = String(v || '').toLowerCase();
    if (s.includes('pass')) return '<span class="badge badge-success">PASS</span>';
    if (s.includes('reject') || s.includes('fail')) return '<span class="badge badge-danger">REJECTED</span>';
    return `<span class="badge badge-muted">${esc(String(v).toUpperCase())}</span>`;
  }

  async function runSearch(q) {
    _lastQuery = q;
    const box = el('kn-results');
    const count = el('kn-count');
    if (!q || q.trim().length < 2) {
      box.innerHTML = `<div class="empty-state"><span class="big">🔎</span>Type at least 2 characters — search Alex's research memory: hypotheses, experiments, rejections, decisions.</div>`;
      count.textContent = '';
      return;
    }
    box.innerHTML = '<div class="skeleton" style="width:70%"></div><div class="skeleton" style="width:55%"></div>';
    try {
      const data = await API.post('/alex/agent/knowledge/search', { query: q.trim(), limit: 20 });
      const results = data.results || [];
      count.textContent = `${results.length} result${results.length === 1 ? '' : 's'}`;
      if (!results.length) {
        box.innerHTML = `<div class="empty-state"><span class="big">🌫️</span>Nothing in the knowledge base matches "${esc(q)}".</div>`;
        return;
      }
      box.innerHTML = results.map(resultCard).join('');
    } catch (err) {
      box.innerHTML = `<div class="empty-state">⚠️ ${esc(err.message)}</div>`;
      count.textContent = '';
    }
  }

  async function mount(container) {
    _container = container;
    container.innerHTML = `
      <div class="card mb-4">
        <div class="flex" style="gap:8px">
          <input class="input" id="kn-q" placeholder="Search the research memory — e.g. 'displacement', 'London sweep rejected', 'GBPJPY experiments'…" style="flex:1">
          <button class="btn btn-primary" id="kn-go">Search</button>
        </div>
        <div class="flex-between mt-2">
          <div class="hint">Full-text over research tasks · experiments · hypotheses · probation rejections — every number traces to a stored record.</div>
          <span class="text-muted mono" id="kn-count" style="font-size:var(--fs-xs)"></span>
        </div>
      </div>
      <div id="kn-results">
        <div class="empty-state"><span class="big">📚</span>Alex's research memory is indexed from real runs: every experiment you ran, every gate decision, every hypothesis tested.</div>
      </div>
    `;
    const input = el('kn-q');
    el('kn-go').addEventListener('click', () => runSearch(input.value));
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(input.value); });
    input.addEventListener('input', () => {
      clearTimeout(_debounce);
      _debounce = setTimeout(() => {
        if (input.value.trim() !== _lastQuery) runSearch(input.value);
      }, 350);
    });
    input.focus();
  }

  window.Router.register('knowledge', {
    title: 'Knowledge', subtitle: "Alex's research memory — searchable, reproducible, honest", icon: '🔍', mount,
  });
})();
