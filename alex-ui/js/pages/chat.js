/**
 * Alex OS — Alex Chat space (real AgentCore brain).
 * Contract: POST /api/agent/chat {message} -> {answer, tools_used, evidence,
 * steps, success, error, pending_confirmation?}
 * Medium-risk actions surface pending_confirmation; the user confirms via
 * POST /api/agent/confirm {token, tool_name, args} (single-use, 10-min TTL).
 */
(function () {
  'use strict';

  const { el, esc, toast } = UI;

  function bubble(role, text, tools, steps) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;flex-direction:column;gap:4px;margin-bottom:14px;' +
      (role === 'user' ? 'align-items:flex-end;' : '');
    const b = document.createElement('div');
    b.style.cssText =
      `max-width:78%;padding:10px 14px;border-radius:14px;line-height:1.5;white-space:pre-wrap;` +
      (role === 'user'
        ? 'background:var(--accent-dim);border:1px solid rgba(47,214,163,.3);border-bottom-right-radius:4px;'
        : 'background:var(--bg-elevated);border:1px solid var(--border-subtle);border-bottom-left-radius:4px;');
    b.textContent = text;
    wrap.appendChild(b);
    if ((tools && tools.length) || steps) {
      const meta = document.createElement('div');
      meta.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;align-items:center;';
      meta.innerHTML =
        (tools || []).map((t) => `<span class="badge badge-info">🔧 ${esc(t)}</span>`).join('') +
        (steps ? `<span class="badge badge-muted">${steps} step${steps > 1 ? 's' : ''}</span>` : '');
      wrap.appendChild(meta);
    }
    el('chat-msgs').appendChild(wrap);
    el('chat-msgs').scrollTop = el('chat-msgs').scrollHeight;
    return wrap;
  }

  /** Attach Confirm/Cancel buttons under a pending-confirmation reply. */
  function attachConfirmation(wrap, pending) {
    const bar = document.createElement('div');
    bar.style.cssText = 'display:flex;gap:8px;margin-top:4px;';
    bar.innerHTML =
      `<button class="btn btn-sm btn-primary">✔ Confirm</button>` +
      `<button class="btn btn-sm">✖ Cancel</button>` +
      `<span class="text-muted" style="font-size:var(--fs-xs);align-self:center">expires in ${Math.round(pending.expires_in_seconds / 60)} min</span>`;
    const [yes, no] = bar.querySelectorAll('button');

    yes.addEventListener('click', async () => {
      yes.disabled = true; no.disabled = true;
      yes.textContent = 'Executing…';
      try {
        const res = await API.post('/alex/agent/confirm', {
          token: pending.token, tool_name: pending.tool, args: pending.args,
        });
        if (res.success) {
          toast.success(`${pending.tool} executed`);
          bubble('alex', `✅ Done — ${pending.tool} executed.\n${res.result ? JSON.stringify(res.result).slice(0, 300) : ''}`);
        } else {
          toast.error(res.error || 'Confirmation failed');
          bubble('alex', `⚠️ Confirmation failed: ${res.error || 'unknown error'}`);
        }
      } catch (err) {
        toast.error(err.message);
        bubble('alex', `⚠️ ${err.message}`);
      }
      bar.remove();
    });

    no.addEventListener('click', () => {
      bar.remove();
      const note = document.createElement('div');
      note.className = 'text-muted';
      note.style.cssText = 'font-size:var(--fs-xs);margin-top:2px;';
      note.textContent = 'Cancelled — nothing was executed. (The token expires unused.)';
      wrap.appendChild(note);
    });

    wrap.appendChild(bar);
  }

  async function send() {
    const input = el('chat-input');
    const text = input.value.trim();
    if (!text || send._busy) return;
    send._busy = true;
    input.value = '';
    bubble('user', text);
    const think = document.createElement('div');
    think.id = 'chat-pending';
    think.style.cssText = 'color:var(--text-muted);font-size:var(--fs-sm);margin-bottom:10px;';
    think.textContent = 'Alex is thinking — running tools…';
    el('chat-msgs').appendChild(think);
    el('chat-msgs').scrollTop = el('chat-msgs').scrollHeight;
    try {
      const res = await API.post('/alex/agent/chat', { message: text });
      think.remove();
      const wrap = bubble('alex', res.answer || '(empty reply)', res.tools_used || [], res.steps);
      if (res.pending_confirmation) {
        attachConfirmation(wrap, res.pending_confirmation);
      } else if (res.success === false && res.error) {
        toast.warning(res.error);
      }
    } catch (err) {
      think.remove();
      bubble('alex', `⚠️ ${err.message}`);
      if (err.status === 0) toast.error('Server unreachable');
    } finally {
      send._busy = false;
      input.focus();
    }
  }

  async function mount(container) {
    container.innerHTML = `
      <div class="card" style="display:flex;flex-direction:column;height:calc(100vh - var(--topbar-h) - var(--statusbar-h) - 120px);min-height:420px;padding:0">
        <div style="padding:var(--sp-4) var(--sp-5);border-bottom:1px solid var(--border-subtle)" class="flex-between">
          <div class="flex">
            <div class="brand-logo" style="width:30px;height:30px;font-size:1rem">A</div>
            <div><strong>Alex</strong><div class="text-muted" style="font-size:var(--fs-xs)">32 tools · deterministic answers · LLM narration</div></div>
          </div>
          <span class="badge badge-info">AgentCore</span>
        </div>
        <div id="chat-msgs" style="flex:1;overflow-y:auto;padding:var(--sp-5)" role="log" aria-live="polite"></div>
        <div style="display:flex;gap:10px;padding:var(--sp-4) var(--sp-5);border-top:1px solid var(--border-subtle)">
          <input class="input" id="chat-input" placeholder="Ask about a pair, run an experiment, request an action…  (Enter to send)">
          <button class="btn btn-primary" id="chat-send">Send</button>
        </div>
      </div>
    `;
    el('chat-send').addEventListener('click', send);
    el('chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
    bubble('alex',
      'I am Alex — I answer from real data by calling deterministic tools (32 available: market analysis, strategy research, backtests, experiments, explanations).\n\n' +
      'Try: "What is happening with EUR/USD?", "List my probation queue", "Explain why X was rejected", or "Backtest strategy <id> with SL=2.0 ATR".\n' +
      'Destructive-ish actions (remove/archive a strategy) will ask you to confirm first.');
    setTimeout(() => el('chat-input').focus(), 80);
  }

  window.Router.register('chat', {
    title: 'Alex Chat', subtitle: 'Conversational control plane over the deterministic core', icon: '💬', mount,
  });
})();
