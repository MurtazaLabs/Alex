/**
 * Alex OS — Space router (Mai OS pattern).
 * Each page registers {title, subtitle, icon, mount(container), unmount?}.
 * Rendered into #main with a CSS-only enter animation.
 */
(function () {
  'use strict';

  const SPACES = {};
  let currentId = null;
  let currentUnmount = null;

  function shell() {
    return {
      main: document.getElementById('main'),
      title: document.getElementById('page-title'),
      sub: document.getElementById('page-sub'),
    };
  }

  const Router = {
    register(id, config) { SPACES[id] = config; },

    async navigate(id, opts = {}) {
      const space = SPACES[id];
      const s = shell();
      if (!space) { console.warn('Unknown space:', id); return; }
      if (id === currentId && !opts.force) return;

      if (typeof currentUnmount === 'function') {
        try { currentUnmount(); } catch (e) { console.warn('unmount failed', currentId, e); }
      }
      currentUnmount = null;
      currentId = id;

      if (typeof space.mount !== 'function') {
        s.main.innerHTML = `<div class="empty-state"><span class="big">🚧</span>Page "${UI.esc(space.title)}" is not implemented yet.</div>`;
        return;
      }

      // Render shell for the space, then hand the container to the page.
      s.main.classList.add('loading');
      s.main.innerHTML =
        `<div class="space-body" id="space-root">` +
          `<div class="space-header">` +
            `<h1><span class="ico">${space.icon || '◈'}</span> ${UI.esc(space.title)}</h1>` +
            (space.subtitle ? `<div class="space-header-sub">${space.subtitle}</div>` : '') +
          `</div>` +
          `<div id="space-content" class="space-enter"></div>` +
        `</div>`;
      s.title.textContent = space.title;
      s.sub.textContent = space.subtitle || '';

      document.querySelectorAll('.nav-item').forEach((n) => n.classList.toggle('active', n.dataset.page === id));

      // CSS animation has already started; JS waits only for data, never for animation.
      try {
        currentUnmount = (await space.mount(document.getElementById('space-content'))) || null;
      } catch (err) {
        console.error('mount failed', id, err);
        document.getElementById('space-content').innerHTML =
          `<div class="empty-state"><span class="big">⚠️</span>${UI.esc(err.message || 'Failed to load this page')}` +
          `<div class="mt-4"><button class="btn" onclick="window.Router.navigate('${id}', {force:true})">Retry</button></div></div>`;
      } finally {
        s.main.classList.remove('loading');
      }
    },

    current: () => currentId,

    start(defaultId = 'dashboard') {
      const apply = () => {
        const id = (location.hash || '').replace(/^#\//, '') || defaultId;
        this.navigate(id, { force: true });
      };
      window.addEventListener('hashchange', apply);
      apply();
    },
  };

  window.Router = Router;
})();
