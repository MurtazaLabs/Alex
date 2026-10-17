/**
 * Alex OS — Markets space: live candles, structure overlay, indicators.
 * Data straight from the deterministic core (feed chain + structure engine).
 */
(function () {
  'use strict';

  const { el, esc, toast } = UI;

  const DEFAULT_PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'GBP/JPY', 'EUR/JPY', 'AUD/USD'];
  const TFS = ['M15', 'H1', 'H4', 'D1'];

  let state = { pair: null, tf: 'H1' };
  let _pairs = null;          // cached watchlist-driven pair list
  let _fromWatchlist = false; // false = showing defaults (watchlist empty/unreachable)
  let _container = null;
  let _pollTimer = null;

  /** Pair buttons come from the user's watchlist (via /alex/overview),
   *  falling back to DEFAULT_PAIRS when it's empty or unreachable. */
  async function fetchPairs() {
    try {
      const ov = await API.get('/alex/overview');
      const wl = (ov && Array.isArray(ov.watchlist) && ov.watchlist.length)
        ? ov.watchlist.map(String)
        : [];
      _pairs = wl.length ? wl : DEFAULT_PAIRS.slice();
      _fromWatchlist = wl.length > 0;
    } catch {
      if (!_pairs || !_pairs.length) _pairs = DEFAULT_PAIRS.slice();
    }
    // Keep a manually selected pair visible even if it's not watched.
    if (state.pair && !_pairs.includes(state.pair)) _pairs.unshift(state.pair);
    return _pairs;
  }

  async function fetchCandles() {
    return API.get(
      `/alex/markets/candles?symbol=${encodeURIComponent(state.pair)}&timeframe=${state.tf}&limit=180`
    );
  }

  async function fetchStructure() {
    return API.get(
      `/alex/markets/structure?symbol=${encodeURIComponent(state.pair)}&timeframe=${state.tf}`
    );
  }

  async function fetchIndicators() {
    return API.get(
      `/alex/markets/indicators?symbol=${encodeURIComponent(state.pair)}&timeframe=${state.tf}`
    );
  }

  /* ---------------- Watchlist mutations (shared default watchlist) -------- */

  async function wlAdd(pair) {
    return API.post('/alex/watchlist/pairs', { pair });
  }
  async function wlRemove(pair) {
    // Query param, not path — symbols contain a slash (EUR/USD).
    return API.delete(`/alex/watchlist/pairs?symbol=${encodeURIComponent(pair)}`);
  }

  function drawChart(candles, structure) {
    const canvas = el('mkt-chart');
    if (!canvas) return;
    window.AlexCharts.candles(canvas, candles, { height: 360 });
    // Structure overlay: swings + S/R as a second transparent canvas layer.
    const ov = el('mkt-overlay');
    if (!ov || !structure) return;
    const dpr = window.devicePixelRatio || 1;
    ov.width = canvas.width; ov.height = canvas.height;
    ov.style.width = canvas.style.width; ov.style.height = canvas.style.height;
    const ctx = ov.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const w = canvas.clientWidth || 600, h = 360;
    const padL = 8, padR = 58, padT = 10, padB = 22;
    const volH = Math.round(h * 0.16);
    const plotW = w - padL - padR, plotH = h - padT - padB - volH;
    let lo = Infinity, hi = -Infinity;
    candles.forEach((b) => { lo = Math.min(lo, b.low); hi = Math.max(hi, b.high); });
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const n = candles.length;
    const y = (p) => padT + ((hi - p) / (hi - lo)) * plotH;
    const x = (i) => padL + ((i + 0.5) / n) * plotW;

    // key levels
    ctx.setLineDash([6, 5]); ctx.lineWidth = 1;
    const lvl = (price, color, label) => {
      if (price == null || !isFinite(price) || price < lo || price > hi) return;
      ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(padL, y(price)); ctx.lineTo(padL + plotW, y(price)); ctx.stroke();
      ctx.fillStyle = color; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right';
      ctx.fillText(label, padL + plotW - 4, y(price) - 3);
    };
    lvl(structure.key_resistance, 'rgba(224,82,96,.75)', 'R');
    lvl(structure.key_support, 'rgba(47,191,113,.75)', 'S');
    ctx.setLineDash([]);

    // swing points
    const dot = (arr, color) => (arr || []).forEach((s) => {
      if (s.index == null || s.index < 0 || s.index >= n) return;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x(s.index), y(s.price), 3.2, 0, Math.PI * 2); ctx.fill();
    });
    dot(structure.swing_highs, 'rgba(224,82,96,.9)');
    dot(structure.swing_lows, 'rgba(47,191,113,.9)');

    // last BOS/CHoCH marker
    const ev = structure.last_bos || structure.last_choch;
    if (ev && ev.index != null && ev.index < n && ev.price) {
      ctx.fillStyle = '#e0a84f';
      ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText((structure.last_bos ? 'BOS' : 'CHoCH'), x(ev.index) + 6, y(ev.price) - 5);
      ctx.beginPath(); ctx.arc(x(ev.index), y(ev.price), 4.5, 0, Math.PI * 2); ctx.stroke();
    }
  }

  function structureCard(s) {
    if (!s) return '<div class="empty-state">No structure data.</div>';
    const row = (k, v, cls) => `<div class="flex-between" style="padding:5px 0;border-bottom:1px dashed var(--border-subtle)">
      <span class="text-muted">${k}</span><span class="mono ${cls || ''}">${v}</span></div>`;
    const bos = s.last_bos
      ? `${esc(s.last_bos.kind || 'BOS')} ${esc(String(s.last_bos.direction || '').toUpperCase())} @ ${s.last_bos.level != null ? s.last_bos.level : '—'}`
      : (s.last_choch ? `CHoCH ${esc(String(s.last_choch.direction || '').toUpperCase())} @ ${s.last_choch.level != null ? s.last_choch.level : '—'}` : '—');
    const trendCls = /bull|up/i.test(s.trend || '') ? 'text-ok' : /bear|down/i.test(s.trend || '') ? 'text-bad' : '';
    return (
      row('Trend', esc(String(s.trend || '—').toUpperCase()), trendCls) +
      row('Last price', s.current_price != null ? s.current_price : '—') +
      row('Last event', bos) +
      row('Key support', s.key_support != null ? s.key_support : '—', 'text-ok') +
      row('Key resistance', s.key_resistance != null ? s.key_resistance : '—', 'text-bad') +
      row('Liquidity sweeps', (s.liquidity_sweeps || []).length) +
      row('Swings (H/L)', `${(s.swing_highs || []).length} / ${(s.swing_lows || []).length}`) +
      (s.consolidation ? row('Consolidation', esc(s.consolidation.summary || 'detected')) : '')
    );
  }

  function indicatorsCard(ind) {
    if (!ind) return '<div class="empty-state">No indicator data.</div>';
    const rsi = ind.rsi;
    const rsiCls = rsi == null ? '' : rsi >= 70 ? 'text-bad' : rsi <= 30 ? 'text-ok' : '';
    const row = (k, v, cls) => `<div class="flex-between" style="padding:5px 0;border-bottom:1px dashed var(--border-subtle)">
      <span class="text-muted">${k}</span><span class="mono ${cls || ''}">${v == null ? '—' : v}</span></div>`;
    const macd = ind.macd || {};
    return (
      row('RSI(14)', rsi != null ? Number(rsi).toFixed(1) : null, rsiCls) +
      row('EMA 9 / 21', ind.ema_9 != null && ind.ema_21 != null ? `${Number(ind.ema_9).toFixed(5)} / ${Number(ind.ema_21).toFixed(5)}` : null) +
      row('MACD', macd.macd != null ? Number(macd.macd).toFixed(6) : null) +
      row('MACD signal', macd.signal != null ? Number(macd.signal).toFixed(6) : null) +
      row('ATR(14)', ind.atr != null ? Number(ind.atr).toFixed(5) : null)
    );
  }

  function pairChip(p) {
    const active = p === state.pair ? ' btn-primary' : '';
    // Removable when the list really is the watchlist; defaults-only lists
    // stay removable while more than one pair is showing.
    const removable = _fromWatchlist || _pairs.length > 1;
    const x = removable
      ? `<span class="wl-x" data-remove="${p}" title="Remove ${p} from watchlist">×</span>`
      : '';
    return `<span class="wl-chip"><button class="btn btn-sm${active}" data-pair="${p}">${p}</button>${x}</span>`;
  }

  function onPairsClick(e) {
    const rm = e.target.closest('.wl-x');
    if (rm) { removePair(rm.dataset.remove); return; }
    const btn = e.target.closest('[data-pair]');
    if (btn && btn.dataset.pair !== state.pair) { state.pair = btn.dataset.pair; mountCurrent(); }
  }

  async function removePair(pair) {
    const ok = await UI.modal.confirm({
      title: 'Remove pair',
      message: `Remove <strong>${esc(pair)}</strong> from your watchlist? It disappears from Markets and the CLI watch command.`,
      confirmLabel: 'Remove',
      danger: true,
    });
    if (!ok) return;
    try {
      const res = await wlRemove(pair);
      _pairs = res.pairs && res.pairs.length ? res.pairs.slice() : DEFAULT_PAIRS.slice();
      if (!_pairs.includes(state.pair)) state.pair = _pairs[0] || DEFAULT_PAIRS[0];
      toast.success(`${pair} removed from watchlist`);
      mountCurrent();
    } catch (err) {
      toast.error(err.message || 'Failed to remove pair');
    }
  }

  function toggleAddInput(force) {
    const input = el('mkt-add-input');
    if (!input) return;
    const show = force === undefined ? input.style.display === 'none' : !!force;
    input.style.display = show ? '' : 'none';
    if (show) { input.value = ''; input.focus(); }
  }

  async function submitAddPair() {
    const input = el('mkt-add-input');
    const raw = (input.value || '').trim();
    if (!raw) { toggleAddInput(false); return; }
    if (!/^[A-Za-z]{6}$/.test(raw.replace(/[^A-Za-z]/g, ''))) {
      toast.warning('Use a 6-letter pair, e.g. USDCAD or EUR/USD');
      return;
    }
    try {
      const res = await wlAdd(raw);
      const compact = raw.replace(/[^A-Za-z]/g, '').toUpperCase();
      toast.success(`${compact.slice(0, 3)}/${compact.slice(3)} added to watchlist`);
      _pairs = res.pairs && res.pairs.length ? res.pairs.slice() : DEFAULT_PAIRS.slice();
      _fromWatchlist = true;
      toggleAddInput(false);
      mountCurrent();
    } catch (err) {
      toast.error(err.message || 'Failed to add pair');
    }
  }

  async function load() {
    const loading = el('mkt-loading');
    if (loading) loading.classList.remove('hidden');
    try {
      const [candles, structure, indicators] = await Promise.all([
        fetchCandles().catch(() => null),
        fetchStructure().catch(() => null),
        fetchIndicators().catch(() => null),
      ]);
      if (!candles || !candles.candles || !candles.candles.length) {
        el('mkt-chart-box').innerHTML =
          `<div class="empty-state"><span class="big">📡</span>No candles for ${esc(state.pair)} ${state.tf} — check the feed status.</div>`;
      } else {
        el('mkt-chart-box').innerHTML =
          `<div style="position:relative"><canvas id="mkt-chart"></canvas><canvas id="mkt-overlay" style="position:absolute;inset:0;pointer-events:none"></canvas></div>`;
        drawChart(candles.candles, structure);
      }
      el('mkt-structure').innerHTML = structureCard(structure);
      el('mkt-indicators').innerHTML = indicatorsCard(indicators);
      el('mkt-price-note').textContent = candles && candles.source ? `feed: ${candles.source}` : '';
    } catch (err) {
      toast.error(err.message || 'Failed to load market data');
    } finally {
      if (loading) loading.classList.add('hidden');
    }
  }

  async function mount(container) {
    _container = container;
    await fetchPairs();
    if (!state.pair) state.pair = _pairs[0] || DEFAULT_PAIRS[0];
    container.innerHTML = `
      <div class="card mb-4">
        <div class="flex-between" style="flex-wrap:wrap;gap:10px">
          <div class="flex" id="mkt-pairs" style="flex-wrap:wrap;gap:6px;align-items:center">
            ${_pairs.map(pairChip).join('')}
            <button class="btn btn-sm mkt-add-btn" id="mkt-add" title="Add pair to watchlist" style="padding:5px 10px">＋</button>
            <input class="input" id="mkt-add-input" placeholder="e.g. USDCAD" maxlength="10"
                   style="display:none;width:110px;padding:5px 9px;font-size:var(--fs-sm)">
          </div>
          <div class="flex" style="gap:6px">
            ${TFS.map((t) => `<button class="btn btn-sm ${t === state.tf ? 'btn-primary' : ''}" data-tf="${t}">${t}</button>`).join('')}
            <button class="btn btn-sm" id="mkt-refresh">⟳</button>
          </div>
        </div>
        <div class="hint mt-2">Pairs from your watchlist · candles + structure from the live feed chain · <span id="mkt-price-note" class="mono"></span></div>
      </div>

      <div class="card">
        <div id="mkt-loading" class="hidden text-muted" style="padding:4px 0">Loading market data…</div>
        <div id="mkt-chart-box"></div>
      </div>

      <div class="card-grid grid-2 mt-4">
        <div class="card"><div class="card-title">Structure — ${esc(state.pair)} ${state.tf}</div><div id="mkt-structure"><div class="skeleton" style="width:70%"></div></div></div>
        <div class="card"><div class="card-title">Indicators — ${state.tf}</div><div id="mkt-indicators"><div class="skeleton" style="width:70%"></div></div></div>
      </div>
    `;

    container.querySelector('#mkt-pairs').addEventListener('click', onPairsClick);
    container.querySelector('#mkt-add').addEventListener('click', () => toggleAddInput());
    container.querySelector('#mkt-add-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submitAddPair();
      if (e.key === 'Escape') toggleAddInput(false);
    });
    container.querySelectorAll('[data-tf]').forEach((b) =>
      b.addEventListener('click', () => { state.tf = b.dataset.tf; mountCurrent(); }));
    el('mkt-refresh').addEventListener('click', load);

    await load();
    // Light auto-refresh: candles every 60s while the page is open.
    _pollTimer = setInterval(load, 60_000);
    return () => { if (_pollTimer) clearInterval(_pollTimer); };
  }

  function mountCurrent() { if (_container) mount(_container); }

  window.Router.register('markets', {
    title: 'Markets', subtitle: 'Live candles · structure · indicators — deterministic core, real feed', icon: '📈', mount,
  });
})();
