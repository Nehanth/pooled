// Host/worker inference and checkpoints over caller-owned state and frame delivery.
// No browser globals, engine construction, or PeerJS connection ownership.
import { badF32, packWire, unpackWire, wireStats } from "./wire.js";
import { DROP_ALL } from "./transport.js";
import { lapTimeout } from "./liveness.js";
import { PrefixIndex } from "../harness/prefix.js";

export function createPipeline({ state: ai, transport, options = {}, hooks = {}, checkpointStore: ckptDisk = null, getRoomCode = () => null }) {
  const { sendHidden, sendTo, chainRtt = () => 0 } = transport;
  const { profile = "browser", checkpointMax: CKPT_MAX = 2, fillDrafts: FILL_DRAFTS = true,
    mtpBatch: MTP_BATCH = true, tailFrame: TAIL_FRAME = true, prefillWindow: PREFILL_WINDOW = 6 } = options;
  const { onStatus: aiStatus = () => {}, computePass = () => {}, teleNote = () => {},
    wakeChain = () => {}, keepWarm = () => {}, onFrame = () => {}, prefillFrame = () => undefined,
    onWorkerFrame = () => {}, onError = (err) => aiStatus("⚠ " + err.message) } = hooks;
  // Keep each consumer's diagnostics and checkpoint transport policy during extraction.
  // The node owns its CkptIndex; the browser checkpoint helpers below retain disk copies.
  const node = profile === "node";
  const hostError = (pos) => node ? `NaN after host layers (pos ${pos})` : `NaN after HOST layers (pos ${pos}) — host GPU kernel issue`;
  const peerError = (pos) => `NaN in hidden returned by peers (pos ${pos})` + (node ? "" : " — check peer status lines");
  const headError = (pos) => `NaN in logits (pos ${pos})` + (node ? "" : " — head/lm_head kernel issue on host");
  const defaultStatus = aiStatus;

  function lapWait(key, ms, what) {
    return new Promise((res, rej) => {
      const timer = setTimeout(() => { ai.waiters.delete(key); rej(new Error(`pipeline timeout (${what}): a device in the room stopped answering` + (node ? "" : ". Check that every device's tab is open with its screen on, then ask again"))); }, ms);
      ai.waiters.set(key, {
        res: (h) => { clearTimeout(timer); res(h); },
        rej: (e) => { clearTimeout(timer); rej(e); },
      });
    });
  }

  function failWaiters(err) { for (const [k, w] of ai.waiters) { ai.waiters.delete(k); w.rej(err); } }

  function lapDone(key, h) { const w = ai.waiters.get(key); if (w) { ai.waiters.delete(key); w.res(h); } }

  function sendChain(msg) {
    const ctl = ai.pendingCtl; ai.pendingCtl = {};
    if (node && ai.dropQ.length) {
      const dp = [].concat(ctl.dp ?? []);
      if (dp.includes(DROP_ALL)) ai.dropQ = [];
      else { while (dp.length < 2 && ai.dropQ.length) dp.push(ai.dropQ.shift()); ctl.dp = dp; }
    }
    ai.frames = (ai.frames || 0) + 1;
    if (!node) ai.hostAmax = Math.max(0.9 * (ai.hostAmax || 0), wireStats.lastMax || 0);
    sendHidden(ai.chain[0], { ...msg, ...ctl });
    // the chain saves with this frame: only now is it a checkpoint every device has, so only now
    // does the host's own copy go to disk (a reloaded host must not index a slot no worker saved)
    const saved = !node && ctl.sv != null && ai.ckpt?.items.find((x) => x.key === ctl.sv);
    if (saved) ckptPersist(ctl.sv, saved.ids, saved.pin);
  }

  function resetState() {
    try { ai.engine.reset?.(); } catch {}
    ai.pos = 0;
    ai.fed = [];
    // a pending rollback (the last answer's final verify rejected drafts) must still reach the chain:
    // the save that rides with it records that answer's end state, and without the rollback a
    // worker would save its state with the rejected columns in it (the host saved its rolled-back
    // state), so a later resume of that checkpoint would run the worker's layers on a corrupt state
    const { sv, dp, rb } = ai.pendingCtl || {};
    ai.pendingCtl = ai.chain.length ? { ...(rb != null ? { rb } : {}), ...(sv != null ? { sv } : {}), ...(dp != null ? { dp } : {}), reset: 1 } : {};
  }

  function ckptClear(tellChain = false) {   // engines rebuilt or in an unknown state: nothing saved is usable
    const keys = ai.ckpt ? ai.ckpt.items.map((x) => x.key) : [];
    for (const k of keys) { try { ai.engine?.dropSlot?.(k); } catch {} }
    if (tellChain && keys.length && ai.chain.length) ai.pendingCtl = { ...ai.pendingCtl, dp: [DROP_ALL] };
    if (tellChain) ckptForget([DROP_ALL]);
    ai.ckpt = new PrefixIndex(1 << 30); ai.ckptN = ai.ckptN || 0;
  }

  function ckptEngine() { return !!ai.engine?.saveSlot && ai.engine.hostCkpt !== false; }

  function ckptSave(pin = false) {
    if (!CKPT_MAX || !ai.fed?.length || !ckptEngine()) return;
    if (!ai.ckpt) ckptClear();
    // a worker applies sv before dp, so a save riding with DROP_ALL would be gone at once on every
    // worker: skip it (the host would otherwise index a slot the chain does not have)
    if (ai.chain.length && [].concat(ai.pendingCtl?.dp ?? []).includes(DROP_ALL)) return;
    // a save still waiting for its frame never reached the chain (no frame went out since, e.g. Stop
    // before the first prefill round): the frame header carries one save, so this one supersedes it.
    // Forget it here too, or a later resume could load a slot no worker saved. Its drops stay pending.
    const prev = ai.chain.length ? ai.pendingCtl?.sv : null;
    if (prev != null) {
      // a pinned save superseded at the same tokens (Stop right after it) stays pinned
      const p = ai.ckpt.items.find((x) => x.key === prev);
      if (p?.pin && p.ids.length === ai.fed.length && p.ids.every((t, i) => t === ai.fed[i])) pin = true;
      ai.ckpt.remove(prev); try { ai.engine.dropSlot(prev); } catch {}
      ckptForget([prev]);   // the host's copy is only written once the frame goes out: none should exist, but never keep one
    }
    // the caches hold exactly a checkpoint already on every device (Stop right after the pinned
    // save, solo; or a regenerate stopped before its first frame): keep that one instead of a
    // second slot with the same state, which would only evict an answer checkpoint
    const same = ai.ckpt.items.find((x) => x.ids.length === ai.fed.length && x.ids.every((t, i) => t === ai.fed[i]));
    if (same && (same.pin || !pin)) {
      same.t = ++ai.ckpt.clock;
      // the superseded save must not go out any more: no new sv replaces it here, and the workers
      // would save (and copy to disk) a slot the host no longer indexes
      if (prev != null) { const { sv, ...rest } = ai.pendingCtl; ai.pendingCtl = rest; }
      return;
    }
    const drop = [];
    if (same) { ai.ckpt.remove(same.key); ai.engine.dropSlot(same.key); drop.push(same.key); }   // re-saved pinned below
    if (pin) for (const old of ai.ckpt.pinned()) { ai.ckpt.remove(old.key); ai.engine.dropSlot(old.key); drop.push(old.key); }
    else while (ai.ckpt.unpinned().length >= CKPT_MAX) {
      const old = ai.ckpt.unpinned().reduce((a, b) => (a.t < b.t ? a : b));
      ai.ckpt.remove(old.key); ai.engine.dropSlot(old.key); drop.push(old.key);
    }
    if (drop.length) ckptForget(drop);
    const key = ai.ckptN = (ai.ckptN || 0) % 65534 + 1;   // slot numbers ride the frame header (u16)
    ai.engine.saveSlot(key);
    ai.ckpt.add(ai.fed.slice(), key, { pin });
    if (!ai.chain.length) ckptPersist(key, ai.fed, pin);   // solo: no chain to wait for (sendChain does it otherwise)
    if (ai.chain.length) {
      const dp = [...new Set([...[].concat(ai.pendingCtl?.dp ?? []), ...drop])];
      ai.pendingCtl = { ...ai.pendingCtl, sv: key, ...(dp.length ? { dp } : {}) };
    }
  }

  function ckptResume(ids, reused) {
    if (!CKPT_MAX || !ai.ckpt) return reused;
    const b = ai.ckpt.best(ids);
    if (!b || b.n <= reused) return reused;
    ai.engine.loadSlot(b.key);
    ai.pos = b.n; ai.fed = ids.slice(0, b.n);
    if (ai.chain.length) { const { reset, ...rest } = ai.pendingCtl || {}; ai.pendingCtl = { ...rest, ld: b.key }; }
    return b.n;
  }

  function ckptWhere(slot) { return { room: getRoomCode(), model: ai.model, sig: ai.engine.stateSignature(), slot }; }

  function ckptPersist(key, ids, pin = false) {
    if (!ckptDisk || !ai.engine?.exportSlot || !ai.engine.stateSignature || !getRoomCode()) return;
    const E = ai.engine, where = ckptWhere(key);
    // read when the write's turn comes: a slot dropped or an engine replaced by then is skipped
    ckptDisk.put(where, () => (ai.engine === E ? E.exportSlot(key) : Promise.reject(new Error("engine replaced"))), ids ? { ids: Array.from(ids), ...(pin ? { pin: true } : {}) } : {})
      .catch(() => false);
  }

  function ckptForget(keys) {
    if (!ckptDisk || !getRoomCode() || !keys?.length) return;
    ckptDisk.drop(getRoomCode(), keys.includes(DROP_ALL) ? "all" : keys).catch(() => {});
  }

  async function ckptRestore() {
    if (!ckptDisk || !ai.engine?.importState || !ai.engine.stateSignature || !getRoomCode()) return [];
    const E = ai.engine, host = ai.role === "host", got = [];
    try {
      const have = (await ckptDisk.list(ckptWhere(0))).filter((c) => !host || Array.isArray(c.meta.ids));
      // the host indexes the newest CKPT_MAX answer checkpoints, oldest first so they keep their age
      // order (ckptSave evicts the oldest), and the newest pinned one (the system prompt), pinned again
      // so answer saves still never evict it. A worker does not know which is pinned: it keeps one
      // more for the pinned one and one more in case the host's copy of the newest is gone.
      const pinned = host ? have.filter((c) => c.meta.pin).slice(0, 1) : [];
      for (const c of host ? [...pinned, ...have.filter((c) => !c.meta.pin).slice(0, CKPT_MAX).reverse()] : have.slice(0, CKPT_MAX + 2)) {
        const st = await ckptDisk.get(ckptWhere(c.slot));
        if (!st || ai.engine !== E) continue;
        try { E.importState(st); E.saveSlot(c.slot); } catch { continue; }   // another shape: not ours after all
        got.push(c.slot);
        ai.ckptN = Math.max(ai.ckptN || 0, c.slot);   // never hand out a number a device may still have on disk
        if (host) { if (!ai.ckpt) ckptClear(); ai.ckpt.add(st.meta.ids, c.slot, { pin: !!st.meta.pin }); }
      }
    } catch {}
    if (got.length && ai.engine === E) { E.reset?.(); if (host) { ai.pos = 0; ai.fed = []; } }
    return got;
  }

  function ckptPrune() {
    if (!ai.ckptHeld?.size) return;
    const drop = [], reports = [...ai.ckptHeld].filter(([id]) => ai.chain.includes(id));
    for (const [, slots] of reports) {
      const held = new Set(Array.isArray(slots) ? slots : []);
      for (const x of ai.ckpt?.items.slice() || []) if (!held.has(x.key)) {
        ai.ckpt.remove(x.key); try { ai.engine?.dropSlot?.(x.key); } catch {}
        drop.push(x.key);
      }
    }
    // a device reads back more than the host indexes (a worker keeps CKPT_MAX + 1, and copies the host
    // lost): a slot it holds that the index does not name would sit on its GPU for good, so the chain
    // drops it too. Slot numbers go on past it, so no new save can land on a number still being dropped.
    const kept = new Set((ai.ckpt?.items || []).map((x) => x.key));
    for (const [, slots] of reports) for (const k of Array.isArray(slots) ? slots : []) {
      if (!Number.isInteger(k) || kept.has(k) || drop.includes(k)) continue;
      drop.push(k); ai.ckptN = Math.max(ai.ckptN || 0, k);
    }
    ai.ckptHeld.clear();
    if (!drop.length) return;
    ckptForget(drop);
    const dp = [...new Set([...[].concat(ai.pendingCtl?.dp ?? []), ...drop])];
    ai.pendingCtl = { ...ai.pendingCtl, dp };
  }

  function ckptRejoin() {
    if (!ckptDisk) { ckptClear(true); return; }
    const { sv, ...rest } = ai.pendingCtl || {};
    if (sv == null) return;
    ai.ckpt?.remove(sv);
    try { ai.engine?.dropSlot?.(sv); } catch {}
    ai.pendingCtl = rest;
  }

  function fillDrafts(h, ids, i0, basePos, n) {
    if (!FILL_DRAFTS || !ai.engine?.mtp) return;
    const dim = ai.engine.dims.dim, E = ai.engine, NC = E.NC || 4;
    // Node prompt frames can span several batches; the browser keeps its original fill policy.
    let c = 0;
    if ((node ? E.mtpBatchFill !== false : MTP_BATCH) && E._mtpFillBatch && E.B && (node || n <= NC)) {
      for (; n - c > 1; c += NC) {
        const m = Math.min(NC, n - c);
        for (let k = 0; k < m; k++) E.device.queue.writeBuffer(E.B.x.buf, k * E.B.x.stride, h.subarray((c + k) * dim, (c + k + 1) * dim));
        E._mtpFillBatch(ids, i0 + c, basePos + c, m);
      }
    }
    for (; c < n; c++) {
      const next = ids[i0 + c + 1];
      if (next === undefined) break;
      ai.engine.setHidden(h.subarray(c * dim, (c + 1) * dim));
      ai.engine.mtpRun(null, next, basePos + c + 1, false);   // no readback: queued, returns at once
    }
  }

  async function aiPipeToken(id, needLogits = true, fillNext, desc = null, ahead = null) {
    // A node call keeps the engine it started on; the browser historically reads live state.
    const current = node ? { engine: ai.engine } : ai;
    const pos = ai.pos;
    if (!ai.chain.length && !needLogits) {
      // solo prefill: layers only, no head, no readback; sync every 8 tokens
      current.engine.pos = pos;
      await current.engine.prefillToken(id);
      if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
      ai.pos++; ai.fed?.push(id);
      return null;
    }
    const tHost = ahead?.t0 ?? performance.now();
    if (needLogits) wakeChain(pos);
    let h = ahead?.h || await current.engine.embedRun(id, pos);
    if (badF32(h)) throw new Error(hostError(pos));
    if (ai.chain.length) {
      const hostMs = performance.now() - tHost;
      const returned = lapWait(pos, lapTimeout(ai.lapStat, 30000, chainRtt()), "token");
      sendChain({ t: "ai-hidden", pos, ...packWire(h) });
      ahead?.onSent?.();
      h = await returned;
      if (badF32(h)) throw new Error(peerError(pos));
      noteLap(performance.now() - tHost, hostMs);
      ai.lastHidden = h;
      if (!needLogits && fillNext !== undefined) fillDrafts(h, [id, fillNext], 0, pos, 1);
    } // solo mode: engine holds every layer, embedRun already produced the final hidden
    ai.pos++; ai.fed?.push(id);
    if (!needLogits || ahead?.defer) return null;   // prefill: skip the head entirely
    if (desc) {
      const c = await current.engine.headFromHiddenIds(h, desc);
      if (c.bad) throw new Error(headError(ai.pos));
      return c;
    }
    const logits = await current.engine.headFromHidden(h);
    if (badF32(logits)) throw new Error(headError(ai.pos));
    return logits;
  }

  async function aiPrefill(ids, { aborted = () => !node && ai.abort, onStatus = defaultStatus, desc = null } = {}) {
    const current = node ? { engine: ai.engine } : ai;
    if (!ai.chain.length && current.engine.prefillTokens && ids.length > 1) {
      // solo: batched prefill, several prompt tokens per GPU pass
      current.engine.pos = ai.pos;
      await current.engine.prefillTokens(ids.slice(0, -1));
      ai.pos = current.engine.pos;
      ai.fed.push(...ids.slice(0, -1));
      return aiPipeToken(ids[ids.length - 1], true, undefined, desc);
    }
    let i = 0;
    // the hybrid engine takes any column count per frame (speculative verifies already send 2..8),
    // so the prompt's tail, last token included, goes round the chain as ONE frame instead of one
    // serial lap per token; short follow-ups become a single lap
    const flex = TAIL_FRAME && !!(ai.chain.length && current.engine.specStep && current.engine.embedRunBatch);
    let tailLogits = null;
    if (current.engine.embedRunBatch && (ids.length > 5 || flex)) {
      // split: up to 16 prompt tokens per round, and several rounds in flight at once. Every device
      // runs frames in send order, so round r+1 can enter the host's layers while round r is on a
      // worker: the chain works like a pipeline instead of one device at a time.
      const hdim = current.engine.dims.dim;
      const NC = current.engine.NC || 4;   // columns per GPU pass
      // step down 16 -> 8 -> 4 on the tail: without this a remainder of up to NC-1 tokens costs one
      // network lap each
      const widths = [NC, ...[8, 4].filter((w) => w < NC)];
      const inflight = [];
      const send = async (hb, basePos, i0, n) => {
        if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
        if (ai.chain.length) {
          while (inflight.length >= PREFILL_WINDOW) await inflight.shift();
          const p = lapWait("b" + basePos, 90000, "batch prefill").then((h) => fillDrafts(h, ids, i0, basePos, n));
          p.catch(() => {});
          inflight.push(p);
          sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
        }
        ai.pos = basePos + n;
        ai.fed.push(...ids.slice(i0, i0 + n));
        i += n;
        if (!node) onStatus(`prefill: ${i}/${ids.length} tokens…`);
      };
      try {
        // Node hosts can send wide prompt frames through every device's prefill kernels.
        // The caller supplies its environment override; browser host framing stays unchanged.
        const F = node && ai.chain.length && current.engine.prefillHidden && current.engine.prefillFrame?.() > 0
          ? +(prefillFrame() || current.engine.prefillFrame()) || 0 : 0;
        while (F && ids.length - 1 - i >= 2 * NC && !aborted()) {
          const n = Math.min(F, ids.length - 1 - i), basePos = ai.pos;
          await send(await current.engine.prefillHidden(ids.slice(i, i + n), basePos), basePos, i, n);
        }
        outer: for (const W of widths) while (ids.length - 1 - i >= W) {
          if (aborted()) break outer;
          const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
          const n = nChunks * W, basePos = ai.pos;
          const hb = new Float32Array(n * hdim);
          for (let c = 0; c < nChunks; c++)
            hb.set(await current.engine.embedRunBatch(ids.slice(i + c * W, i + (c + 1) * W), basePos + c * W), c * W * hdim);
          await send(hb, basePos, i, n);
        }
        if (flex && !aborted() && i < ids.length) {
          const n = ids.length - i, basePos = ai.pos, i0 = i;   // n <= 4: what the widths above left
          const hb = await current.engine.embedRunBatch(ids.slice(i), basePos);
          if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
          const p = lapWait("b" + basePos, 90000, "prefill tail");
          p.catch(() => {});
          sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
          ai.pos = basePos + n;
          ai.fed.push(...ids.slice(i0));
          i = ids.length;
          for (const q of inflight) await q;
          const h = await p;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${basePos})`);
          fillDrafts(h, ids, i0, basePos, n);
          const dim = current.engine.dims.dim;
          ai.lastHidden = h.slice((n - 1) * dim, n * dim);
          if (desc) {
            tailLogits = await current.engine.headFromHiddenIds(ai.lastHidden, desc);
            if (!node && tailLogits.bad) throw new Error(headError(ai.pos));
          } else tailLogits = await current.engine.headFromHidden(ai.lastHidden);
          if (!node && !desc && badF32(tailLogits)) throw new Error(headError(ai.pos));
        }
        for (const p of inflight) await p;
      } catch (err) { failWaiters(err); throw err; }
    }
    if (aborted()) return null;
    if (tailLogits) return tailLogits;
    let logits = null;
    for (; i < ids.length; i++) {
      if (aborted()) return null;
      logits = await aiPipeToken(ids[i], i === ids.length - 1, ids[i + 1], desc);
    }
    return logits;
  }

  function noteLap(lapMs, hostMs) {
    const L = ai.lapStat ||= { lap: 0, host: 0, n: 0, max: 0 };
    L.lap = L.n ? 0.7 * L.lap + 0.3 * lapMs : lapMs;
    L.max = Math.max(lapMs, 0.98 * (L.max || 0));   // the slowest recent lap, fading over ~50 laps (lap timeouts)
    L.host = L.n ? 0.7 * L.host + 0.3 * hostMs : hostMs;
    L.n++;
  }

  async function workerFrame(d) {
    const current = node ? { engine: ai.engine } : ai;
    if (!current.engine) return;
    const ctl = {};
    // order matters: a pending rollback belongs to the answer that just ended, the save records
    // that answer's final state, and only then may the state be reset or replaced by a checkpoint
    // (also before a reset: the save may record the state first, and the host saved its own after
    // its rollback)
    if (d.rb != null) { current.engine.restoreDN?.(d.rb); ctl.rb = d.rb; }
    if (d.sv != null) { current.engine.saveSlot?.(d.sv); if (!node) ckptPersist(d.sv); ctl.sv = d.sv; }
    if (d.dp != null) { for (const k of [].concat(d.dp)) k === DROP_ALL ? current.engine.dropAllSlots?.() : current.engine.dropSlot?.(k); if (!node) ckptForget([].concat(d.dp)); ctl.dp = d.dp; }
    if (d.reset) { current.engine.reset?.(); ctl.reset = 1; }
    if (d.ld != null) { current.engine.loadSlot?.(d.ld); ctl.ld = d.ld; }
    const t0 = performance.now();
    if (d.t === "ai-hidden-b") {
      // n hiddens in, my layers (batched), n hiddens on
      const xs = unpackWire(d);
      const nTok = d.n || 4;
      const wdim = current.engine.dims.dim;
      const NC = current.engine.NC || 4;
      // A node host's wide prompt uses prefill kernels; verifies retain per-column state.
      const wide = !d.spec && nTok >= 2 * NC && current.engine.prefillHidden && current.engine.prefillFrame?.() > 0;
      const hb = wide ? await current.engine.prefillHidden(xs, d.basePos) : new Float32Array(nTok * wdim);
      if (!wide) for (let c = 0; c < nTok; c += NC) {
        const m = Math.min(NC, nTok - c);
        hb.set(await current.engine.runHiddenBatch(xs.subarray(c * wdim, (c + m) * wdim), d.basePos + c, d.spec ? { base: c, total: nTok } : false), c * wdim);
      }
      if (badF32(hb)) { if (!node) aiStatus(`⚠ NaN in batched prefill on this device`); sendTo(ai.hostId, { t: "ai-error", message: "NaN in batched prefill" }); }
      if (!node) {
        teleNote(d.spec ? "spec" : "pre", performance.now() - t0);
        computePass(nTok, performance.now() - t0);
      }
      // the verify flag travels with the frame: every device snapshots its recurrent state per
      // column, or a later rollback on it restores a stale snapshot
      const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) };
      if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret-b", ...bmsg });
      else sendHidden(ai.next, { t: "ai-hidden-b", ...bmsg, ...ctl });
      if (!node && d.spec) keepWarm(d.basePos);
    } else {
      // one token: run my layers, forward along the chain
      const hin = unpackWire(d);
      if (!node && badF32(hin)) { aiStatus(`⚠ NaN ARRIVED at this device (pos ${d.pos}) — upstream peer broken`); }
      const h = await current.engine.runHidden(hin, d.pos);
      if (badF32(h)) {
        if (!node) aiStatus(`⚠ NaN PRODUCED by this device (pos ${d.pos}, layers ${ai.range[0]}–${ai.range[1] - 1}) — GPU kernel issue here`);
        sendTo(ai.hostId, { t: "ai-error", message: node ? `NaN produced on node layers ${ai.range[0]}-${ai.range[1] - 1}` : `NaN produced on worker layers ${ai.range[0]}–${ai.range[1] - 1}` });
      }
      if (!node) {
        teleNote("one", performance.now() - t0);
        computePass(1, performance.now() - t0);
      }
      const msg = { pos: d.pos, ...packWire(h) };
      if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret", ...msg });
      else sendHidden(ai.next, { t: "ai-hidden", ...msg, ...ctl });
      if (!node) keepWarm(d.pos);
      if (!node && d.pos % 8 === 0) aiStatus(`serving layers ${ai.range[0]}–${ai.range[1] - 1} — pos ${d.pos}`);
    }
    if (node) onWorkerFrame(performance.now() - t0);
  }
  function handleFrame(d) {
    switch (d.t) {
      case "ai-hidden-b":
      case "ai-hidden":
        if (ai.role !== "worker") break;
        onFrame(d);
        ai.q = ai.q.then(() => workerFrame(d)).catch((err) => {
          onError(err);
          sendTo(ai.hostId, { t: "ai-error", message: err.message });
        });
        break;
      case "ai-hiddenret-b": lapDone("b" + d.basePos, unpackWire(d)); break;
      case "ai-hiddenret": lapDone(d.pos, unpackWire(d)); break;
      default: return false;
    }
    return true;
  }

  return { lapWait, lapDone, failWaiters, sendChain, resetState, ckptEngine, ckptClear, ckptSave, ckptResume,
    ckptPersist, ckptRestore, ckptPrune, ckptRejoin, aiPipeToken, aiPrefill, noteLap,
    fillDrafts, workerFrame, handleFrame };
}
