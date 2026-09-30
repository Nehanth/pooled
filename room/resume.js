// Keeping a room's work alive through a device that drops out for a while (a phone that locks,
// a tab that reloads, a laptop that sleeps). DOM-free so it can be unit tested; room.js wires it.
//
// The host's generation core runs round the chain; when a device in the chain goes away mid-answer
// the lap in flight fails. Instead of failing the answer (or the Code run's step), the host waits
// for the room to be whole again (the device comes back into its slot, or the host re-deals the
// layers over the devices still there) and runs the same generation again from where it stopped:
// the prompt plus every token already emitted is the new prompt, so nothing already shown changes
// and greedy decoding gives the same text it would have given without the drop.

// How long the host waits for a dropped device before re-dealing without it (when auto re-deal is
// on), and how long a whole recovery may take before the run gives up.
export const REJOIN_GRACE_MS = 60000;
export const RECOVER_MAX_MS = 6 * 60000;
// A link that has carried nothing (not even the 2.5 s ping) for this long is treated as gone.
// Chrome surfaces a dead data channel only when ICE fails (~30 s); a locked iPhone says nothing.
export const LINK_SILENT_MS = 12000;

// Is an error from a generation something a recovery can fix? A lap that timed out or was failed
// because a chain device left, a re-deal, or a device reporting an error. Not: the user's Stop, a
// missing model, a NaN (a broken kernel stays broken), a full context.
export function recoverableError(err) {
  const m = String(err?.message || err || "");
  if (/NaN/.test(m)) return false;
  return /pipeline timeout|\bleft\b|re-deal|re-dealing|reconnect|could not connect/i.test(m);
}

// Run gen(ids, opts) until it finishes, recovering from recoverable failures in between:
//   gen(ids, { ...opts, maxNew, onToken })  -> result { tokens, count, reason, ... } (roomGenerate)
//   recover({ err, attempt, emitted })       -> resolves once the room can generate again, or throws
//   aborted()                                -> the user stopped: no retry
// Tokens emitted before a failure stay emitted (onToken is not called for them again); the retry's
// prompt is ids + those tokens and its budget is what is left of maxNew. The merged result
// reports every token of every attempt, and `resumed` how many recoveries it took.
export async function resumableGenerate(gen, ids, opts = {}, { recover, aborted = () => false, maxTries = 4, onResume = () => {} } = {}) {
  const maxNew = opts.maxNew ?? Infinity;
  const onToken = opts.onToken || (() => {});
  const emitted = [];
  const acc = { count: 0, tPre: 0, tDecode: 0, prefilled: 0, preFrames: 0, copied: 0 };
  for (let attempt = 0; ; attempt++) {
    let r;
    try {
      r = await gen([...ids, ...emitted], {
        ...opts,
        maxNew: maxNew - emitted.length,
        onToken: (t, d) => { emitted.push(t); onToken(t, d); },
      });
    } catch (err) {
      if (aborted() || !recover || attempt + 1 >= maxTries || !recoverableError(err) || emitted.length >= maxNew) throw err;
      await recover({ err, attempt, emitted: emitted.length });
      if (aborted()) throw err;
      onResume({ err, attempt: attempt + 1, emitted: emitted.length });
      continue;
    }
    for (const k of Object.keys(acc)) acc[k] += r[k] || 0;
    return { ...r, ...acc, tokens: emitted.slice(), count: emitted.length, resumed: attempt };
  }
}

// Wait for the room to be whole again after a device in the chain dropped.
//   ready()        the chain is complete and every device has its layers
//   gone()         names of the devices still missing (for the status line)
//   redeal()       re-deal over whoever is here; resolves when the new deal is dealt (or throws)
//   autoRedeal()   whether to re-deal on our own after graceMs
//   status(text)   progress for the screen
// Resolves when ready() holds; throws when the wait runs past maxMs, or aborted().
export async function waitForRoom({ ready, gone = () => [], redeal, autoRedeal = () => true, status = () => {}, aborted = () => false,
  graceMs = REJOIN_GRACE_MS, maxMs = RECOVER_MAX_MS, pollMs = 250, now = () => Date.now(), sleep = (ms) => new Promise((r) => setTimeout(r, ms)) } = {}) {
  const t0 = now();
  let dealt = false, last = "";
  const say = (s) => { if (s !== last) { last = s; status(s); } };
  while (!ready()) {
    if (aborted()) throw new Error("stopped while waiting for the room");
    const waited = now() - t0;
    if (waited > maxMs) throw new Error(`the room did not recover in ${Math.round(maxMs / 1000)} s: re-deal the layers`);
    const missing = gone();
    if (!dealt && missing.length && waited >= graceMs && autoRedeal() && redeal) {
      dealt = true;
      say(`${missing.join(", ")} did not come back: re-dealing the layers without ${missing.length > 1 ? "them" : "it"} (experimental)…`);
      await redeal();
      continue;
    }
    if (missing.length && !dealt) {
      const left = Math.max(0, Math.ceil((graceMs - waited) / 1000));
      say(`waiting for ${missing.join(", ")} to come back${autoRedeal() && redeal ? ` (re-dealing without ${missing.length > 1 ? "them" : "it"} in ${left} s)` : ""}…`);
    } else say(dealt ? "re-dealing the layers…" : "reloading layers…");
    await sleep(pollMs);
  }
}

// Link health: the last time anything arrived from a peer, and whether that is too long ago.
export function linkSilent(lastSeen, now, limit = LINK_SILENT_MS) {
  return lastSeen != null && now - lastSeen > limit;
}

// A device's own view after it was hidden (screen locked, Safari in the background, tab switched):
// was it away long enough that its links may be dead? Short blips (a notification) are not.
export function backFromAway(hiddenAt, now, minMs = 3000) {
  return hiddenAt != null && now - hiddenAt >= minMs;
}

// A worker told to load what it already holds (it came back into its slot after a lock, with its
// GPU buffers intact) skips the download and only rejoins the chain.
export function sameShard(held, msg) {
  return !!held && !!msg && held.model === msg.model && held.ctx === msg.ctx
    && Array.isArray(held.range) && Array.isArray(msg.range) && held.range[0] === msg.range[0] && held.range[1] === msg.range[1];
}

// What a reloaded guest tab needs to walk back into its room (kept in sessionStorage: per tab, it
// survives a reload, including the one iOS does after killing the tab for memory).
export const GUEST_KEY = "pooled-guest";
export const GUEST_TTL_MS = 10 * 60000;
export function guestResume(saved, { now = Date.now(), linkCode = null } = {}) {
  if (!saved || typeof saved !== "object" || !saved.code || !saved.name) return null;
  if (!(now - saved.t < GUEST_TTL_MS)) return null;
  if (linkCode && linkCode !== saved.code) return null;
  return { code: String(saved.code), name: String(saved.name).slice(0, 40), gb: +saved.gb > 0 ? +saved.gb : null };
}
