/**
 * Alex OS — shared client state.
 * Small, global, boring — the antidote to scattered DOM lookups.
 */
window.State = {
  user: null,
  feed: null,
  watchlist: [],
  probationCount: null,
  /** per-page message handlers pushed by pages: fn(type, payload) */
  wsHandlers: [],
  onWsMessage(type, payload) {
    this.wsHandlers.forEach((fn) => { try { fn(type, payload); } catch (e) { console.warn('ws handler failed', e); } });
  },
};
