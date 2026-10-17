/**
 * Alex OS — Shell: navigation registry, status bar, login gate, global actions.
 */
(function () {
  'use strict';

  const { el, esc, toast } = UI;

  const NAV = [
    { id: 'dashboard',  label: 'Dashboard',  icon: '▦' },
    { id: 'markets',    label: 'Markets',    icon: '📈' },
    { id: 'signals',    label: 'Signals',    icon: '⚡' },
    { id: 'trading',    label: 'Trading',    icon: '💰', group: 'Execution' },
    { id: 'strategies', label: 'Strategies', icon: '🧠', group: 'Research' },
    { id: 'experiments', label: 'Experiments', icon: '🧪' },
    { id: 'research',   label: 'Research',   icon: '🔬' },
    { id: 'knowledge',  label: 'Knowledge',  icon: '🔍' },
    { id: 'chat',       label: 'Alex Chat',  icon: '💬' },
    { id: 'account',    label: 'Account',    icon: '👤', group: 'System' },
    { id: 'settings',   label: 'Settings',   icon: '⚙' },
  ];

  function buildNav() {
    const nav = el('nav');
    let group = null;
    let html = '';
    NAV.forEach((item) => {
      if (item.group && item.group !== group) {
        group = item.group;
        html += `<div class="nav-label">${esc(group)}</div>`;
      }
      html +=
        `<button class="nav-item" data-page="${item.id}" onclick="location.hash='#/${item.id}'">` +
          `<span class="ico">${item.icon}</span><span>${esc(item.label)}</span>` +
          (item.id === 'strategies' ? `<span class="nav-badge hidden" id="nav-badge-strategies"></span>` : '') +
        `</button>`;
    });
    nav.innerHTML = html;
  }

  function setFeed(name) {
    State.feed = name;
    const f = String(name || '—').toLowerCase();
    el('st-feed').textContent = f;
    el('sb-feed').textContent = f;
  }

  /* ---------------- Statusbar widgets: feed lights + LLM chain ---------------- */

  function setFeedLights(health) {
    const box = el('st-feed-lights');
    if (!box) return;
    const list = Array.isArray(health) ? health : [];
    if (!list.length) { box.innerHTML = ''; box.title = 'No feed health data'; return; }
    box.innerHTML = list.map((f) => {
      const ok = f.healthy !== false;
      const name = String(f.name || f.provider || '?').toLowerCase();
      return `<span class="dot ${ok ? 'dot-ok' : 'dot-bad'}" title="${name}: ${ok ? 'healthy' : 'down'} (✓${f.total_successes ?? 0} ✗${f.total_failures ?? 0})"></span>`;
    }).join('');
  }

  async function refreshLlmChip() {
    const chip = el('st-llm-model');
    if (!chip) return;
    try {
      const st = await API.get('/alex/status');
      const llm = (st && st.llm) || {};
      const chain = Array.isArray(llm.provider_chain) && llm.provider_chain.length
        ? llm.provider_chain.join('→') : 'openrouter→unorouter';
      const model = String(llm.reasoning_model || '—').split('/').pop();
      const budget = llm.daily_budget ? ` · ${llm.tokens_used_today ?? 0}/${llm.daily_budget} tok` : '';
      chip.textContent = model;
      el('st-llm').title = `LLM chain: ${chain} · ${llm.openrouter_configured ? 'OpenRouter✓' : 'OpenRouter✗'} ${llm.unorouter_configured ? 'UnoRouter✓' : 'UnoRouter✗'}${budget}`;
    } catch {
      chip.textContent = '—';
    }
  }

  function setClock() {
    const tick = () => {
      const c = el('st-clock');
      if (c) c.textContent = new Date().toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };
    tick();
    setInterval(tick, 1000);
  }

  /* ---------------- Login gate ---------------- */

  function showLogin() {
    el('login-overlay').classList.add('open');   // pure CSS display
    el('login-error').classList.remove('show');
    setTimeout(() => el('login-user').focus(), 60);
  }

  function hideLogin() { el('login-overlay').classList.remove('open'); }

  async function doLogin() {
    const btn = el('login-go');
    const user = el('login-user').value.trim();
    const pass = el('login-pass').value;
    const errBox = el('login-error');
    errBox.classList.remove('show');
    if (!user || !pass) {
      errBox.textContent = 'Enter your username and password.';
      errBox.classList.add('show');
      return;
    }
    btn.disabled = true;
    btn.textContent = 'Signing in…';
    try {
      await API.login(user, pass);
      hideLogin();
      el('login-pass').value = '';
      await enterApp();
      toast.success('Welcome back');
    } catch (err) {
      errBox.textContent = err.message || 'Sign-in failed';
      errBox.classList.add('show');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Sign in';
    }
  }

  /* ---------------- App lifecycle ---------------- */

  async function enterApp() {
    try { State.user = await API.get('/auth/me'); } catch { State.user = null; }
    el('st-user').textContent = State.user ? `👤 ${State.user.username}` : '';
    try {
      const ov = await API.get('/alex/overview');
      setFeed(ov.data_source);
      setFeedLights(ov.feed_health);
      el('st-pairs').textContent = (ov.watchlist || []).length || '—';
      if (typeof ov.probation_active === 'number') {
        const b = el('nav-badge-strategies');
        if (b) { b.textContent = String(ov.probation_active); b.classList.toggle('hidden', ov.probation_active === 0); }
      }
    } catch { setFeed(null); }
    refreshLlmChip();
    if (!Router.current()) Router.start('dashboard');
    else Router.navigate(Router.current(), { force: true });
    WS.connect();
  }

  async function logout() {
    await API.logout();
    location.reload();
  }

  /* ---------------- Wire-up ---------------- */

  document.addEventListener('DOMContentLoaded', () => {
    // Scripts executed and the shell exists — boot is proven. Data loading may
    // still be in flight, but that is not a boot failure.
    window.__alexBooted = true;
    window.__bootHide && window.__bootHide();
    buildNav();
    setClock();
    el('login-go').addEventListener('click', doLogin);
    el('login-pass').addEventListener('keydown', (e) => { if (e.key === 'Enter') doLogin(); });
    el('login-user').addEventListener('keydown', (e) => { if (e.key === 'Enter') doLogin(); });
    el('btn-logout').addEventListener('click', logout);
    el('btn-refresh').addEventListener('click', () => Router.navigate(Router.current() || 'dashboard', { force: true }));
    el('nav-toggle').addEventListener('click', () => el('sidebar').classList.toggle('open'));
    window.addEventListener('alex:logout', showLogin);

    // Boot: token? enter straight away. Otherwise show the (visible!) login.
    if (API.isLoggedIn()) {
      enterApp();
    } else {
      showLogin();
    }
  });

  window.Shell = { setFeed, setFeedLights, refreshLlmChip, showLogin, hideLogin, enterApp };
})();
