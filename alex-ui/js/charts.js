/**
 * Alex OS — canvas charts (no libraries).
 * Candlesticks + volume, multi-series lines, sparklines. DPR-aware; colors
 * come from CSS custom properties so the theme stays in one place.
 */
(function () {
  'use strict';

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function prep(canvas, height) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth || 600;
    const h = height || parseInt(canvas.getAttribute('data-h') || '300', 10);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function niceBounds(lo, hi) {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  }

  /** bars: [{time, open, high, low, close, volume}] */
  function candles(canvas, bars, opts = {}) {
    if (!canvas || !bars || !bars.length) return;
    const height = opts.height || 320;
    const { ctx, w, h } = prep(canvas, height);
    const up = cssVar('--up', '#2fbf71');
    const down = cssVar('--down', '#e05260');
    const grid = cssVar('--border-subtle', 'rgba(255,255,255,.07)');
    const text = cssVar('--text-muted', '#8b93a7');

    const padL = 8, padR = 58, padT = 10, padB = 22;
    const volH = opts.volume === false ? 0 : Math.round(h * 0.16);
    const plotW = w - padL - padR;
    const plotH = h - padT - padB - volH;

    let lo = Infinity, hi = -Infinity, maxVol = 0;
    bars.forEach((b) => {
      lo = Math.min(lo, b.low); hi = Math.max(hi, b.high);
      maxVol = Math.max(maxVol, b.volume || 0);
    });
    [lo, hi] = niceBounds(lo, hi);
    const n = bars.length;
    const slot = plotW / n;
    const bodyW = Math.max(1.5, Math.min(11, slot * 0.62));
    const y = (p) => padT + ((hi - p) / (hi - lo)) * plotH;

    // grid + price labels (4 lines)
    ctx.strokeStyle = grid; ctx.fillStyle = text;
    ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
      const p = lo + ((hi - lo) * i) / 4;
      const yy = Math.round(y(p)) + 0.5;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(padL + plotW, yy); ctx.stroke();
      ctx.fillText(p.toFixed(p > 50 ? 1 : 4), padL + plotW + 6, yy + 3);
    }

    // volume
    if (volH > 0) {
      const vBase = h - padB;
      bars.forEach((b, i) => {
        const v = b.volume || 0;
        if (!v || !maxVol) return;
        const vh = (v / maxVol) * (volH - 4);
        ctx.fillStyle = b.close >= b.open ? up : down;
        ctx.globalAlpha = 0.25;
        ctx.fillRect(padL + i * slot + (slot - bodyW) / 2, vBase - vh, bodyW, vh);
        ctx.globalAlpha = 1;
      });
    }

    // candles
    bars.forEach((b, i) => {
      const cx = padL + i * slot + slot / 2;
      const bull = b.close >= b.open;
      ctx.strokeStyle = bull ? up : down;
      ctx.fillStyle = bull ? up : down;
      // wick
      ctx.beginPath();
      ctx.moveTo(Math.round(cx) + 0.5, y(b.high));
      ctx.lineTo(Math.round(cx) + 0.5, y(b.low));
      ctx.stroke();
      // body
      const yO = y(b.open), yC = y(b.close);
      const top = Math.min(yO, yC), bh = Math.max(1, Math.abs(yC - yO));
      ctx.fillRect(cx - bodyW / 2, top, bodyW, bh);
    });

    // last price tag
    const last = bars[bars.length - 1];
    const ly = y(last.close);
    ctx.fillStyle = last.close >= last.open ? up : down;
    ctx.fillRect(padL + plotW + 2, ly - 8, padR - 6, 16);
    ctx.fillStyle = '#0b0e14';
    ctx.textAlign = 'center';
    ctx.fillText(last.close.toFixed(last.close > 50 ? 2 : 4), padL + plotW + 2 + (padR - 6) / 2, ly + 4);

    // time labels (first / mid / last)
    ctx.fillStyle = text; ctx.textAlign = 'center';
    const tLabel = (b) => String(b.time || '').replace('T', ' ').slice(5, 16);
    ctx.fillText(tLabel(bars[0]), padL + 22, h - 6);
    ctx.fillText(tLabel(bars[Math.floor(n / 2)]), padL + plotW / 2, h - 6);
    ctx.fillText(tLabel(last), padL + plotW - 30, h - 6);
  }

  /**
   * lines: [{name, values, color?}] — multi-series line chart (WF windows).
   * Shared x = index; draws a zero line when values cross zero.
   */
  function lines(canvas, seriesList, opts = {}) {
    if (!canvas || !seriesList.length) return;
    const height = opts.height || 260;
    const { ctx, w, h } = prep(canvas, height);
    const grid = cssVar('--border-subtle', 'rgba(255,255,255,.07)');
    const text = cssVar('--text-muted', '#8b93a7');
    const palette = ['#5aa9ff', '#2fbf71', '#e05260', '#e0a84f', '#b18cff'];

    const padL = 8, padR = 56, padT = 12, padB = 20;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    let lo = Infinity, hi = -Infinity;
    seriesList.forEach((s) => s.values.forEach((v) => {
      if (v == null || !isFinite(v)) return;
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    [lo, hi] = niceBounds(lo, hi);
    const n = Math.max(...seriesList.map((s) => s.values.length));
    const x = (i) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
    const y = (v) => padT + ((hi - v) / (hi - lo)) * plotH;

    ctx.font = '10px ui-monospace, monospace';
    for (let i = 0; i <= 4; i++) {
      const p = lo + ((hi - lo) * i) / 4;
      const yy = Math.round(y(p)) + 0.5;
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(padL + plotW, yy); ctx.stroke();
      ctx.fillStyle = text; ctx.textAlign = 'left';
      ctx.fillText(p.toFixed(2), padL + plotW + 6, yy + 3);
    }
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = cssVar('--text-muted', '#8b93a7');
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(padL, y(0)); ctx.lineTo(padL + plotW, y(0)); ctx.stroke();
      ctx.setLineDash([]);
    }

    seriesList.forEach((s, si) => {
      const color = s.color || palette[si % palette.length];
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      let started = false;
      s.values.forEach((v, i) => {
        if (v == null || !isFinite(v)) return;
        if (!started) { ctx.moveTo(x(i), y(v)); started = true; }
        else ctx.lineTo(x(i), y(v));
      });
      ctx.stroke();
      ctx.fillStyle = color;
      s.values.forEach((v, i) => {
        if (v == null || !isFinite(v)) return;
        ctx.beginPath(); ctx.arc(x(i), y(v), 3, 0, Math.PI * 2); ctx.fill();
      });
    });

    // legend
    ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left';
    let lx = padL + 2;
    seriesList.forEach((s, si) => {
      const color = s.color || palette[si % palette.length];
      ctx.fillStyle = color; ctx.fillRect(lx, 2, 10, 3);
      ctx.fillStyle = text; ctx.fillText(s.name, lx + 14, 8);
      lx += 14 + ctx.measureText(s.name).width + 14;
    });
  }

  /** sparkline: last-value emphasis, no axes */
  function spark(canvas, values, color) {
    if (!canvas || !values || values.length < 2) return;
    const { ctx, w, h } = prep(canvas, 36);
    const lo = Math.min(...values), hi = Math.max(...values);
    const x = (i) => (i / (values.length - 1)) * (w - 4) + 2;
    const y = (v) => h - 4 - ((v - lo) / ((hi - lo) || 1)) * (h - 8);
    ctx.strokeStyle = color || cssVar('--accent', '#5aa9ff');
    ctx.lineWidth = 1.5; ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath(); ctx.arc(x(values.length - 1), y(values[values.length - 1]), 2.5, 0, Math.PI * 2); ctx.fill();
  }

  window.AlexCharts = { candles, lines, spark };
})();
