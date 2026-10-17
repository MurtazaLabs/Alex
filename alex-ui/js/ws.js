/**
 * Alex OS — WebSocket manager.
 * Auto-reconnect with capped backoff; routes messages to State handlers.
 */
(function () {
  'use strict';

  let ws = null;
  let attempts = 0;
  let heartbeat = null;
  const MAX_ATTEMPTS = 6;

  function setConn(ok, label) {
    const dot = document.getElementById('st-dot');
    const txt = document.getElementById('st-conn');
    const dot2 = document.getElementById('sb-dot');
    const txt2 = document.getElementById('sb-conn');
    [dot, dot2].forEach((d) => { if (d) d.className = 'dot ' + (ok ? 'dot-ok' : 'dot-bad'); });
    [txt, txt2].forEach((t) => { if (t) t.textContent = label; });
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    try {
      ws = new WebSocket(`${proto}://${location.host}/ws`);
    } catch {
      setConn(false, 'WS unavailable');
      return;
    }

    ws.onopen = () => {
      attempts = 0;
      setConn(true, 'Connected');
      heartbeat = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
      }, 25000);
    };

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      window.State.onWsMessage(msg.type || 'unknown', msg);
    };

    ws.onclose = () => {
      clearInterval(heartbeat);
      setConn(false, attempts >= MAX_ATTEMPTS ? 'Offline' : 'Reconnecting…');
      if (attempts < MAX_ATTEMPTS) {
        attempts += 1;
        setTimeout(connect, Math.min(1000 * 2 ** attempts, 20000));
      }
    };

    ws.onerror = () => { try { ws.close(); } catch { /* noop */ } };
  }

  window.WS = { connect };
})();
