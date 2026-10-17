/**
 * Alex OS — UI helpers: dom, toasts, modal.
 * Visibility is controlled by CSS classes only; transitions are decoration.
 * (This is the fix for the invisible-overlay class of bugs.)
 */
(function () {
  'use strict';

  function el(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  /* ---------------- Toasts ---------------- */

  function toast(message, type = 'info', duration = 4200) {
    const stack = el('toast-stack');
    if (!stack) return;
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.innerHTML = `<div class="toast-msg">${esc(message)}</div><button class="toast-x" aria-label="Dismiss">×</button>`;
    const kill = () => {
      t.classList.add('leaving');
      setTimeout(() => t.remove(), 300); // removal only, visibility via CSS
    };
    t.querySelector('.toast-x').addEventListener('click', kill);
    stack.appendChild(t);
    if (duration > 0) setTimeout(kill, duration);
  }
  toast.success = (m, d) => toast(m, 'success', d);
  toast.error = (m, d) => toast(m, 'error', 6000);
  toast.warning = (m, d) => toast(m, 'warning', d);

  /* ---------------- Modal (single reusable dialog) ---------------- */

  const modal = {
    _onClose: null,
    /** show({title, body, footer, onClose}) — body/footer are HTML strings */
    show({ title = '', body = '', footer = '', onClose = null } = {}) {
      el('modal-title').innerHTML = title;
      el('modal-body').innerHTML = body;
      el('modal-foot').innerHTML = footer;
      el('modal-foot').style.display = footer ? '' : 'none';
      this._onClose = onClose;
      el('modal-overlay').classList.remove('closing');
      el('modal-overlay').classList.add('open');   // CSS displays it — instant
    },
    hide() {
      const ov = el('modal-overlay');
      if (!ov.classList.contains('open')) return;
      ov.classList.add('closing');                 // 140ms CSS fade…
      const cb = this._onClose;
      this._onClose = null;
      setTimeout(() => {                           // …then hard-hide via class
        ov.classList.remove('open', 'closing');
        el('modal-title').innerHTML = '';
        el('modal-body').innerHTML = '';
        el('modal-foot').innerHTML = '';
        if (cb) cb();
      }, 160);
    },
    isOpen: () => el('modal-overlay').classList.contains('open'),
    /** confirm({title, message, confirmLabel, danger}) -> Promise<boolean> */
    confirm({ title = 'Confirm', message = '', confirmLabel = 'Confirm', danger = false } = {}) {
      return new Promise((resolve) => {
        const id = `cf-${Date.now()}`;
        this.show({
          title,
          body: `<p style="margin:0">${message}</p>`,
          footer:
            `<button class="btn" id="${id}-no">Cancel</button>` +
            `<button class="btn ${danger ? 'btn-danger' : 'btn-primary'}" id="${id}-yes">${esc(confirmLabel)}</button>`,
          onClose: () => resolve(false),
        });
        el(`${id}-no`).addEventListener('click', () => { this._onClose = null; this.hide(); resolve(false); });
        el(`${id}-yes`).addEventListener('click', () => { this._onClose = null; this.hide(); resolve(true); });
      });
    },
  };

  el('modal-x').addEventListener('click', () => modal.hide());
  el('modal-overlay').addEventListener('click', (e) => {
    if (e.target === el('modal-overlay') && !el('modal-foot').querySelector('.btn-danger')) modal.hide();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && modal.isOpen()) modal.hide(); });

  /* ---------------- Small render helpers ---------------- */

  function fmtMoney(v, cur) {
    if (v == null || isNaN(Number(v))) return '—';
    return `${cur || ''}$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  function fmtWhen(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return isNaN(d) ? String(iso).slice(0, 16) : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function verdictBadge(v) {
    const s = String(v || '').toLowerCase();
    if (s.includes('pass')) return '<span class="badge badge-success">PASS</span>';
    if (s.includes('reject') || s.includes('fail')) return '<span class="badge badge-danger">REJECTED</span>';
    return '<span class="badge badge-muted">' + esc(v || '—') + '</span>';
  }
  function statusBadge(s) {
    const map = {
      active: ['badge-info', 'ACTIVE'], proposing: ['badge-warning', 'PROPOSING'],
      passed: ['badge-success', 'PASSED'], approved: ['badge-success', 'APPROVED'],
      rejected: ['badge-danger', 'REJECTED'], expired: ['badge-muted', 'EXPIRED'],
    };
    const [cls, label] = map[s] || ['badge-muted', String(s || '—').toUpperCase()];
    return `<span class="badge ${cls}">${esc(label)}</span>`;
  }

  window.UI = { el, esc, toast, modal, fmtMoney, fmtWhen, verdictBadge, statusBadge };
})();
