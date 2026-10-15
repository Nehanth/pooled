import { MODES, requireLocal, invitePath, MemoryStore, systemPrompt } from './state.js';
const $ = id => document.getElementById(id);
const titles = { chat: 'Assistant', code: 'Code workspace', devices: 'Devices', memories: 'Memory', connections: 'Connections', activity: 'Activity', settings: 'Settings & privacy' };
let view = 'chat', model = null, run = null, editId = '', toastTimer, confirmAction = null;
let turns = [], events = [];
// Storage may be unavailable even on property access (private mode / user policy).
let storage;
try { storage = window.localStorage; } catch { storage = { getItem() { throw Error(); }, setItem() { throw Error(); }, removeItem() { throw Error(); } }; }
const memories = new MemoryStore(storage);
function notice(text) { $('notice').textContent = text; $('notice').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('notice').hidden = true; }, 5000); }
function navigate(next, hash = true) {
  view = Object.hasOwn(titles, next) ? next : 'chat';
  document.querySelectorAll('[data-panel]').forEach(el => { el.hidden = el.dataset.panel !== view; });
  document.querySelectorAll('[data-view]').forEach(el => { if (el.dataset.view === view) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current'); });
  $('view-title').textContent = titles[view]; document.title = `Duga · ${titles[view]}`;
  document.body.classList.remove('nav-open'); $('nav-toggle').setAttribute('aria-expanded', 'false');
  if (hash) history.replaceState(null, '', `#${view}`);
  if (view === 'activity') renderEvents();
}
document.querySelectorAll('[data-view]').forEach(el => el.addEventListener('click', () => navigate(el.dataset.view)));
addEventListener('hashchange', () => navigate(location.hash.slice(1), false));
navigate(location.hash.slice(1) || 'chat', false);
$('nav-toggle').addEventListener('click', () => { const open = document.body.classList.toggle('nav-open'); $('nav-toggle').setAttribute('aria-expanded', String(open)); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { document.body.classList.remove('nav-open'); $('nav-toggle').setAttribute('aria-expanded', 'false'); } });
function event(text) { events.unshift({ text, time: new Date() }); events = events.slice(0, 100); renderEvents(); }
function renderEvents() {
  $('activity-list').replaceChildren();
  if (!events.length) { const li = document.createElement('li'); li.textContent = 'No actions yet. Your session starts here.'; $('activity-list').append(li); }
  for (const entry of events) { const li = document.createElement('li'), time = document.createElement('time'), text = document.createElement('span'); time.dateTime = entry.time.toISOString(); time.textContent = entry.time.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }); text.textContent = entry.text; li.append(time,text); $('activity-list').append(li); }
}
$('activity-clear').addEventListener('click', () => { events = []; renderEvents(); });
function download(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type })), a = document.createElement('a');
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function confirm(text, action) { confirmAction = action; $('confirm-text').textContent = text; $('confirm-dialog').showModal(); }
$('confirm-cancel').addEventListener('click', () => $('confirm-dialog').close());
$('confirm-dialog').addEventListener('close', () => { confirmAction = null; });
$('confirm-delete').addEventListener('click', () => { try { confirmAction?.(); $('confirm-dialog').close(); } catch { notice('Could not delete saved data. Browser storage may be blocked.'); } });
function renderMemories() {
  $('memory-count').textContent = memories.items.length; $('memory-list').replaceChildren();
  if (!memories.items.length) { const empty = document.createElement('div'); empty.className = 'empty-state'; empty.textContent = 'A fresh start. Save a note when there’s something you want to keep.'; $('memory-list').append(empty); }
  for (const item of memories.items) {
    const row = document.createElement('article'), text = document.createElement('p'), edit = document.createElement('button'), del = document.createElement('button'); row.className = 'memory-item'; text.textContent = item.text; edit.textContent = 'Edit'; edit.ariaLabel = `Edit note: ${item.text.slice(0,40)}`; del.textContent = 'Delete'; del.className = 'danger'; del.ariaLabel = `Delete note: ${item.text.slice(0,40)}`;
    edit.addEventListener('click', () => { editId = item.id; $('memory-text').value = item.text; $('memory-save').textContent = 'Save changes'; $('memory-cancel').hidden = false; $('memory-text').focus(); });
    del.addEventListener('click', () => confirm('Delete this saved note from this browser?', () => { memories.remove(item.id); if (editId === item.id) resetEdit(); renderMemories(); notice('Note deleted. Existing conversation context is unchanged; start a new conversation to clear it.'); }));
    row.append(text, edit, del); $('memory-list').append(row);
  }
  $('memory-status').textContent = memories.error;
}
function resetEdit() { editId = ''; $('memory-form').reset(); $('memory-save').textContent = 'Save note'; $('memory-cancel').hidden = true; }
$('memory-cancel').addEventListener('click', resetEdit);
$('memory-form').addEventListener('submit', e => { e.preventDefault(); try { memories.save($('memory-text').value, editId); resetEdit(); renderMemories(); $('memory-status').textContent = 'Saved in this browser.'; } catch (err) { $('memory-status').textContent = err.message || 'Save failed.'; } });
$('memory-export').addEventListener('click', () => download('duga-notes.json', JSON.stringify(memories.items, null, 2)));
$('memory-clear').addEventListener('click', () => confirm('Delete all saved notes from this browser? This cannot be undone. Existing chat context is kept until you start a new conversation.', () => { memories.clear(); resetEdit(); renderMemories(); }));
renderMemories();
function clearChat() { turns = []; $('conversation').replaceChildren(); $('welcome').hidden = false; $('prompt').value = ''; $('run-status').textContent = ''; $('use-memories').checked = false; model?.dispose(); model = null; }
$('new-conversation').addEventListener('click', () => { if (run) { notice('Stop the current task before starting a new conversation.'); return; } if (turns.length) confirm('Discard this conversation and release its local model? Saved notes will remain.', () => { clearChat(); navigate('chat'); }); else { navigate('chat'); $('prompt').focus(); } });
$('data-delete').addEventListener('click', () => { if (run) { notice('Stop the current task before deleting session data.'); return; } confirm('Delete local chat, saved notes and activity? Model weight caches and separate room data are not removed.', () => { memories.clear(); resetEdit(); renderMemories(); clearChat(); events = []; renderEvents(); notice('Local assistant data deleted.'); }); });
$('chat-export').addEventListener('click', () => download('duga-chat.md', turns.map(t => `## ${t.role === 'user' ? 'You' : 'Duga'}\n\n${t.text}`).join('\n\n'), 'text/markdown'));
for (const [id, mode] of Object.entries(MODES)) {
  const card = document.createElement('div'), title = document.createElement('h3'), badge = document.createElement('small'), p = document.createElement('p');
  card.className = `mode-card${id === 'local' ? ' active' : ''}`; title.textContent = mode.label; badge.textContent = mode.available ? 'Selected' : 'Not available'; title.append(badge); p.textContent = mode.detail; card.append(title,p); $('mode-options').append(card);
}
$('mode-open').addEventListener('click', () => $('mode-dialog').showModal());
$('voice-open').addEventListener('click', () => $('voice-dialog').showModal());
document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => b.closest('dialog').close()));
document.querySelectorAll('[data-prompt]').forEach(b => b.addEventListener('click', () => { $('prompt').value = b.dataset.prompt; $('prompt').focus(); }));
$('gpu-status').textContent = navigator.gpu ? 'WebGPU API detected · model availability checked when you start.' : 'WebGPU unavailable · try a supported desktop browser or a trusted room.';
$('create-room').addEventListener('click', () => openWorkspace());
function openWorkspace(path = '/room') {
  if (run) { notice('Stop the local task before opening another inference workspace.'); return; }
  const open = () => {
    navigate('code');
    // Release the local GPU allocation before a room may load another model.
    model?.dispose(); model = null;
    $('workspace').src = path; $('workspace').hidden = false; $('workspace-open').hidden = true;
    event('Opened a separate trusted-room workspace. Local conversation and notes were not transferred.');
  };
  if ($('workspace').hasAttribute('src')) {
    confirm('Replace the open room workspace? Unsaved room work may be lost. Its credentials will not be revoked.', open);
  } else open();
}
$('workspace-open').addEventListener('click', () => openWorkspace());
$('join-form').addEventListener('submit', e => { e.preventDefault(); $('invite-error').textContent = ''; try { const path = invitePath($('invite').value.trim(), location.origin); $('invite').value = ''; openWorkspace(path); } catch (err) { $('invite-error').textContent = err.message; } });
function addMessage(role, text) {
  const article = document.createElement('article'), label = document.createElement('div'), body = document.createElement('div'); article.className = `message ${role}`; label.className = 'message-label'; label.textContent = role === 'user' ? 'You' : 'Duga · Local'; body.className = 'message-body'; body.textContent = text; article.append(label,body); $('conversation').append(article); return { article, body };
}
function busy(on) { $('send').disabled = on; $('stop').hidden = !on; $('prompt').readOnly = on; $('use-memories').disabled = on; }
$('stop').addEventListener('click', () => { run?.abort(); $('run-status').textContent = 'Stopping…'; });
$('prompt').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!run) $('chat-form').requestSubmit(); } });
$('chat-form').addEventListener('submit', async e => {
  e.preventDefault(); const text = $('prompt').value.trim(); if (!text || run) return;
  // Never execute two models in one tab: a hidden room could still be running.
  if ($('workspace').hasAttribute('src')) { notice('Disconnect the room workspace before starting Local.'); navigate('code'); return; }
  requireLocal('local');
  const ctrl = new AbortController(); run = ctrl; busy(true); $('welcome').hidden = true;
  turns.push({ role:'user', text }); addMessage('user', text); $('prompt').value = '';
  const reply = addMessage('assistant', ''); let answer = '';
  event('Local task started. No remote inference route selected.');
  try {
    if (!model) {
      $('run-status').textContent = 'Preparing local model…';
      const { loadLocal } = await import('./local.js');
      model = await loadLocal({ signal: ctrl.signal,
        onStatus: t => { $('run-status').textContent = t; },
        onProgress: (done,total) => { if (total > 0) { $('load-progress').hidden = false; $('load-progress').value = Math.min(100, Math.round(done / total * 100)); $('run-status').textContent = `Loading model on this device · ${$('load-progress').value}%`; } },
      });
    }
    ctrl.signal.throwIfAborted(); $('load-progress').hidden = true; $('run-status').textContent = 'Duga is thinking on this device…';
    const selected = $('use-memories').checked ? memories.items : [];
    for await (const delta of model.generate({ system:systemPrompt(selected), turns:turns.map(t => ({ ...t })), signal:ctrl.signal, onStatus:t => { if (!ctrl.signal.aborted) $('run-status').textContent = t; } })) {
      answer += delta; reply.body.textContent = answer;
    }
    if (answer) turns.push({ role:'assistant', text:answer }); else turns.pop();
    if (ctrl.signal.aborted) { reply.body.textContent ||= 'Stopped.'; $('run-status').textContent = 'Stopped. Nothing was sent to a remote worker.'; event('Local task stopped.'); }
    else { reply.body.textContent ||= 'No answer was produced. Try a shorter question.'; $('run-status').textContent = 'Finished on this device. Check important details.'; event('Local task completed. No external actions performed.'); }
  } catch (err) {
    // Failed/partial runs must not leave an unmatched user turn in the next prompt.
    turns.pop(); model?.dispose(); model = null; reply.article.classList.add('failed');
    reply.body.textContent = ctrl.signal.aborted ? 'Stopped. Your message was not sent to a remote worker.' : `Could not complete this local task. ${err.message || 'Try again.'}`;
    $('run-status').textContent = ctrl.signal.aborted ? 'Stopped.' : 'Local unavailable. No automatic fallback. Your question is restored below.';
    if (!ctrl.signal.aborted) $('prompt').value = text;
    event(ctrl.signal.aborted ? 'Local task stopped.' : 'Local task failed. No fallback or external action was attempted.');
  } finally { run = null; busy(false); $('load-progress').hidden = true; }
});
// Explicitly close the iframe so a hidden room does not keep using compute.
const disconnect = document.createElement('button'); disconnect.id = 'workspace-disconnect'; disconnect.textContent = 'Disconnect workspace'; disconnect.className = 'danger';
$('workspace-open').after(disconnect); disconnect.hidden = true;
$('workspace').addEventListener('load', () => { disconnect.hidden = !$('workspace').hasAttribute('src'); });
disconnect.addEventListener('click', () => confirm('Disconnect this room workspace? Unsaved room work may be lost. This does not revoke room credentials.', () => {
  $('workspace').removeAttribute('src'); $('workspace').hidden = true; $('workspace-open').hidden = false; disconnect.hidden = true; event('Room workspace disconnected. Credentials were not revoked.');
}));
addEventListener('pagehide', () => { run?.abort(); model?.dispose(); });

// Native host lifecycle: release computation while keeping explicitly saved notes.
addEventListener('duga:pause', () => {
  run?.abort(); model?.dispose(); model = null;
  const f = $('workspace'); f.removeAttribute('src'); f.hidden = true;
  $('workspace-open').hidden = false; disconnect.hidden = true;
  event('Android paused the workspace. Active computation and room connections stopped.');
});
