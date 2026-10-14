// A Qwen3.8 Flash tab's load as three parts that run at the same time where they can (room/model-loader.js loadShard
// qwen4exp): "gpu" (the layers' resident weights onto the GPU), "save" (the offloaded layers' experts copied to this
// computer's disk, room/flashdisk.js expertTierCopy) and "words" (the n-gram table copied there, ingestNgram).
// Before this they ran one after the other (save, words, gpu: 610 s on the Spark from localhost, the GPU last).
//   t0      gpu starts; save's check (what is on disk already: latency, no conversion) and words (its check, then
//           its copy in a worker, one head at a time) start next to it. While save still has a copy ahead, gpu is
//           not held to the room's pace (room.js aiLoadPace): held to a slow device, the whole copy would start
//           only after that device is done
//   gpu     done: save's copy runs (on the main thread, as before: it is CPU-bound, and next to the GPU step it took
//           the GPU's speed from 144 to 10 MB/s), words goes on with two heads at a time. A copy in workers
//           (engine/expert_ingest_pool.js, save.early) starts at t0 instead, once its check answers
//   end     the experts' reader opens, words is awaited (it never fails the load: the table is read over the network
//           instead), then the engine
// Each part's bytes and state go to onParts (the contract with the load card: { parts: [{ id, done, total, state }],
// note? }) and, summed, to the old onProgress, so every caller keeps working.
// Pure apart from timers: tests/unit/flashload_test.js.

// the parts in the order a snapshot lists them
export const PART_ORDER = ["gpu", "save", "words"];
const TERMINAL = new Set(["done", "skip", "fail"]);

// The parts' progress, throttled. sizes: { gpu, save, words } bytes (0: that part has nothing to do: "skip" from the
// first snapshot). onParts(snapshot) and onProg(sumDone, sumTotal, info) are called together, at most every `every`
// ms while bytes move and at once on any state or note change. gpu starts as "run", save and words as "wait". info: { step (gpu while it runs, then the copy that
// runs), stepNo, steps, stepDone, stepTotal, gpuPct (gpu's own percent: 100 once it is done), parts }.
// Rules: a part's done never goes down; its total is fixed, except skip (0) and fail (it keeps what it did: total =
// done); done, skip and fail are final. The first snapshot is sent at once.
export function partsTracker({ sizes = {}, onParts = null, onProg = null, every = 250, now = () => performance.now(),
  setTimer = (f, ms) => setTimeout(f, ms), clearTimer = (t) => clearTimeout(t) } = {}) {
  const parts = PART_ORDER.map((id) => {
    const total = Math.max(0, +sizes[id] || 0);
    // gpu runs from the start; save and words wait until their check answers (save: until the GPU step is done)
    return { id, done: 0, total, state: total > 0 ? (id === "gpu" ? "run" : "wait") : "skip" };
  });
  const by = Object.fromEntries(parts.map((p) => [p.id, p]));
  let note = null, last = -Infinity, timer = null, dirty = false;
  const snapshot = () => ({ parts: parts.map((p) => ({ ...p })), ...(note ? { note } : {}) });
  const emit = () => {
    if (timer != null) { clearTimer(timer); timer = null; }
    dirty = false; last = now();
    const snap = snapshot();
    try { onParts?.(snap); } catch (e) { console.warn("onParts:", e); }
    if (onProg) {
      const g = by.gpu, live = parts.filter((p) => p.state !== "skip");
      const cur = g.state === "run" ? g : parts.find((p) => p.id !== "gpu" && p.state === "run") || parts.find((p) => p.state === "wait") || g;
      const gpuPct = g.state === "done" || g.state === "skip" ? 100 : g.total ? g.done / g.total * 100 : 0;
      onProg(parts.reduce((a, p) => a + p.done, 0), parts.reduce((a, p) => a + p.total, 0),
        { step: cur.id, stepNo: Math.max(1, live.indexOf(cur) + 1), steps: Math.max(1, live.length), stepDone: cur.done, stepTotal: cur.total, gpuPct, parts: snap.parts });
    }
  };
  const soon = () => {
    dirty = true;
    const wait = every - (now() - last);
    if (wait <= 0) emit();
    else if (timer == null) timer = setTimer(() => { timer = null; if (dirty) emit(); }, wait);
  };
  const t = {
    get parts() { return snapshot().parts; },
    get(id) { return { ...by[id] }; },
    // part `id` has done `done` bytes (its own count; clamped to its total, never back)
    at(id, done) {
      const p = by[id];
      if (!p || TERMINAL.has(p.state)) return;
      const d = Math.max(p.done, Math.min(p.total, +done || 0));
      if (d === p.done) return;
      p.done = d; soon();
    },
    // run | wait | done | skip | fail
    state(id, s) {
      const p = by[id];
      if (!p || TERMINAL.has(p.state) || p.state === s) return;
      p.state = s;
      if (s === "skip") { p.done = 0; p.total = 0; }
      else if (s === "fail") p.total = p.done;
      else if (s === "done") p.done = p.total;
      emit();
    },
    note(s) { if ((s || null) === note) return; note = s || null; emit(); },
    // every part still running or waiting: failed (a load that stopped), keeping its bytes
    failRunning() {
      let any = false;
      for (const p of parts) if (p.state === "run" || p.state === "wait") { p.state = "fail"; p.total = p.done; any = true; }
      if (any) emit();
    },
    // the last snapshot now, and no timer left behind
    flush() { if (timer != null || dirty) emit(); },
  };
  emit();
  return t;
}

// The load's parts at the same time. parts: a partsTracker. ac: the load's AbortController (Stop fires it too).
//   gpu({ at, paced })                  -> weights   (today's GPU step; at(bytes) its progress; paced(): whether it may
//                                         be held to the room's pace (room.js aiLoadPace). Not while save still has
//                                         a copy to run after it: held to the slowest device, the whole copy would
//                                         start after that device is done, not next to it)
//   save: { check() -> chk ({ need, haveBytes }), copy(chk, { at, signal }), open(chk), early? } | null   (early: copy
//                                         starts next to the GPU step, as soon as check answers; else after it)
//   words: { check() -> wchk ({ missing: [], presentBytes, dir }), copy(wchk, { at, signal, parallel, after,
//           onState }) } | null   (parallel(): heads at a time, 1 while gpu runs, 2 after; after: settles once
//           save's copy is over, for a copy that must not run next to it)
//   finish({ weights, words: { ok, dir } | null, save: chk | null }) -> the load's result
// words never fails the load: an error marks it "fail" (onWordsFail(e) says why) and finish gets { ok: false }.
// Any other failure (gpu, save's check or copy, open, finish) marks what still runs as failed, aborts the rest and
// waits for it to stop (no worker or lock outlives the load), then throws.
export async function runFlashParts({ parts, ac = new AbortController(), gpu, save = null, words = null, finish, onWordsFail = () => {} }) {
  let gpuRunning = true, saveOver, saveAhead = !!save;
  const saveSettled = new Promise((r) => { saveOver = r; });
  const signal = ac.signal;
  const chkP = save ? Promise.resolve().then(() => save.check()) : null;
  chkP?.then((c) => { if (!c.need) { saveAhead = false; parts.state("save", "skip"); } }, () => {});
  chkP?.then((c) => { if (c.need) parts.at("save", c.haveBytes || 0); }, () => {});
  const wordsP = words ? (async () => {
    try {
      const c = await words.check();
      if (!c.missing.length) { parts.state("words", "skip"); return { ok: true, dir: c.dir }; }
      parts.at("words", c.presentBytes || 0);
      parts.state("words", "run");
      await words.copy(c, { signal, at: (d) => parts.at("words", (c.presentBytes || 0) + d), parallel: () => (gpuRunning ? 1 : 2),
        after: saveSettled, onState: (s) => parts.state("words", s) });
      if (signal.aborted) throw signal.reason;
      parts.state("words", "done");
      return { ok: true, dir: c.dir };
    } catch (e) {
      parts.state("words", "fail");
      if (!signal.aborted) onWordsFail(e);
      return { ok: false, error: e };
    }
  })() : null;
  // save.early: the copy starts once its check answers, next to the GPU step (a copy off the main thread: workers);
  // its failure stops the load at once (the GPU step, unpaced while a copy runs, sees the abort)
  const copyP = save?.early ? chkP.then((c) => {
    if (!c.need || signal.aborted) return;
    parts.state("save", "run");
    return save.copy(c, { signal, at: (d) => parts.at("save", d) });
  }) : null;
  copyP?.then(() => { saveAhead = false; if (!signal.aborted) parts.state("save", "done"); }, (e) => { if (!signal.aborted) ac.abort(e); });
  try {
    const weights = await gpu({ at: (d) => parts.at("gpu", d), paced: () => !saveAhead });
    gpuRunning = false;
    parts.state("gpu", "done");
    let chk = null;
    if (save) {
      chk = await chkP;
      if (chk.need) {
        parts.state("save", "run");
        await (copyP || save.copy(chk, { signal, at: (d) => parts.at("save", d) }));
        if (signal.aborted) throw signal.reason;
        parts.state("save", "done");
      }
      saveOver();
      await save.open(chk);
    }
    saveOver();
    const w = wordsP ? await wordsP : null;
    if (signal.aborted) throw signal.reason;
    return await finish({ weights, words: w, save: chk });
  } catch (e) {
    saveOver();
    parts.failRunning();
    if (!signal.aborted) ac.abort(e);
    await Promise.allSettled([wordsP, chkP, copyP].filter(Boolean));
    throw e;
  } finally {
    parts.flush();
  }
}
