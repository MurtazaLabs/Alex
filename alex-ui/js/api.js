/**
 * Alex OS — API client (plain script, no modules, no build step).
 * JWT auth with silent refresh; every method returns parsed JSON.
 */
(function () {
  'use strict';

  const BASE = '/api';

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  const tokens = {
    access: localStorage.getItem('access_token') || null,
    refresh: localStorage.getItem('refresh_token') || null,
    set(a, r) {
      this.access = a; this.refresh = r;
      localStorage.setItem('access_token', a);
      localStorage.setItem('refresh_token', r);
    },
    clear() {
      this.access = null; this.refresh = null;
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
    },
  };

  let refreshInFlight = null;

  async function refreshAccessToken() {
    if (!tokens.refresh) return false;
    if (!refreshInFlight) {
      refreshInFlight = (async () => {
        try {
          const res = await fetch(`${BASE}/auth/refresh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: tokens.refresh }),
          });
          if (!res.ok) return false;
          const data = await res.json();
          tokens.set(data.access_token, data.refresh_token);
          return true;
        } catch { return false; }
        finally { setTimeout(() => { refreshInFlight = null; }, 0); }
      })();
    }
    return refreshInFlight;
  }

  async function request(method, path, body, retry = true) {
    const headers = { 'Content-Type': 'application/json' };
    if (tokens.access) headers['Authorization'] = `Bearer ${tokens.access}`;
    let res;
    try {
      res = await fetch(BASE + path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch {
      throw new ApiError(0, 'Cannot reach the server — is run_api.bat running?');
    }
    if (res.status === 401 && retry && tokens.refresh) {
      if (await refreshAccessToken()) return request(method, path, body, false);
      tokens.clear();
      window.dispatchEvent(new CustomEvent('alex:logout'));
      throw new ApiError(401, 'Session expired');
    }
    if (!res.ok) {
      let msg = `Request failed (${res.status})`;
      try {
        const err = await res.json();
        if (err && err.detail) {
          msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
        }
      } catch { /* keep default */ }
      throw new ApiError(res.status, msg);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  window.API = {
    ApiError,
    get: (p) => request('GET', p),
    post: (p, b) => request('POST', p, b === undefined ? {} : b),
    delete: (p) => request('DELETE', p),

    async login(username, password) {
      const data = await request('POST', '/auth/login', { username, password });
      tokens.set(data.access_token, data.refresh_token);
      return data;
    },
    async logout() {
      try { await request('POST', '/auth/logout', {}); } catch { /* best effort */ }
      tokens.clear();
    },
    isLoggedIn: () => !!tokens.access,
  };
})();
