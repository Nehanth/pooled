// room/flashload.js: a Flash tab's three load parts (gpu, save, words) at the same time, and the onParts snapshots.
//   deno test --no-check --allow-read tests/unit/flashload_test.js
import { partsTracker, runFlashParts, PART_ORDER } from "../../room/flashload.js";

const ok = (v, m = "assertion failed") => { if (!v) throw new Error(m); };
const eq = (a, b, m = "") => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${m} ${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const test = Deno.test;
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));
function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; }

// a tracker on a manual clock: every snapshot and onProgress call recorded
function tracked(sizes, { every = 250 } = {}) {
  let t = 0;
  const timers = [], snaps = [], progs = [];
  const tr = partsTracker({ sizes, every, now: () => t, setTimer: (f, ms) => { const x = { f, at: t + ms }; timers.push(x); return x; },
    clearTimer: (x) => { const i = timers.indexOf(x); if (i >= 0) timers.splice(i, 1); },
    onParts: (s) => snaps.push(s), onProg: (d, tot, info) => progs.push([d, tot, info]) });
  const advance = (ms) => { t += ms; for (const x of timers.slice()) if (x.at <= t) { timers.splice(timers.indexOf(x), 1); x.f(); } };
  return { tr, snaps, progs, advance, timers };
}
const byId = (snap) => Object.fromEntries(snap.parts.map((p) => [p.id, p]));
// the contract's rules over a whole run of snapshots
function checkRules(snaps) {
  for (let i = 1; i < snaps.length; i++) {
    const a = byId(snaps[i - 1]), b = byId(snaps[i]);
    for (const id of PART_ORDER) {
      const x = a[id], y = b[id];
      if (y.state === "skip") { eq([y.done, y.total], [0, 0], `${id} skip`); continue; }
      ok(y.done >= x.done, `${id} done went back: ${x.done} -> ${y.done}`);
      if (y.state === "fail") ok(y.total === y.done && y.total <= x.total, `${id} fail keeps what it did`);
      else ok(y.total === x.total, `${id} total changed: ${x.total} -> ${y.total} (${y.state})`);
      if (["done", "skip", "fail"].includes(x.state)) eq(y.state, x.state, `${id} left ${x.state}`);
    }
  }
}

test("tracker: the first snapshot at once, sized up front; gpu runs, the copies wait; nothing to do is skip from the start", () => {
  const { snaps, progs } = tracked({ gpu: 100, save: 0, words: 50 });
  eq(snaps.length, 1);
  eq(snaps[0].parts, [{ id: "gpu", done: 0, total: 100, state: "run" }, { id: "save", done: 0, total: 0, state: "skip" }, { id: "words", done: 0, total: 50, state: "wait" }]);
  eq(progs[0].slice(0, 2), [0, 150]);
  eq([progs[0][2].step, progs[0][2].stepNo, progs[0][2].steps, progs[0][2].gpuPct], ["gpu", 1, 2, 0]);
});

test("tracker: bytes at most every 250 ms (the last one never lost), a state or note change at once", () => {
  const { tr, snaps, advance, timers } = tracked({ gpu: 1000, save: 500, words: 300 });
  tr.at("gpu", 10);                 // 250 ms since the first? no: t = 0, the first was at 0 -> waits
  eq(snaps.length, 1);
  tr.at("gpu", 20); tr.at("gpu", 30);
  eq(timers.length, 1, "one timer for the burst");
  advance(250);
  eq(snaps.length, 2); eq(byId(snaps[1]).gpu.done, 30);
  tr.state("save", "wait");          // already waiting: nothing new
  eq(snaps.length, 2);
  tr.state("save", "run");           // at once
  eq(snaps.length, 3); eq(byId(snaps[2]).save.state, "run");
  tr.note("Checking");
  eq(snaps.length, 4); eq(snaps[3].note, "Checking");
  tr.note("Checking");               // the same note: nothing new
  eq(snaps.length, 4);
  tr.note(null);
  ok(!("note" in snaps[4]), "note cleared");
  tr.at("words", 5);
  advance(100); eq(snaps.length, 5);
  tr.flush();                         // the pending bytes now, no timer left
  eq(snaps.length, 6); eq(byId(snaps[5]).words.done, 5); eq(timers.length, 0);
  // over four seconds of steady bytes: about four a second
  for (let i = 0; i < 400; i++) { tr.at("gpu", 31 + i); advance(10); }
  ok(snaps.length - 6 <= 17 && snaps.length - 6 >= 15, `${snaps.length - 6} snapshots in 4 s`);
});

test("tracker: done never goes back, totals fixed; skip -> 0, fail keeps its bytes, done fills; terminal states stay", () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 200, words: 300 });
  tr.at("gpu", 50); tr.at("gpu", 40); tr.at("gpu", 500);
  eq(tr.get("gpu").done, 100, "clamped to its total, never back");
  tr.state("gpu", "done");
  tr.at("save", 20); tr.state("save", "skip");
  eq(tr.get("save"), { id: "save", done: 0, total: 0, state: "skip" });
  tr.at("words", 120); tr.state("words", "fail");
  eq(tr.get("words"), { id: "words", done: 120, total: 120, state: "fail" });
  tr.state("words", "run"); tr.at("words", 200); tr.state("save", "run");
  eq([tr.get("words").state, tr.get("words").done, tr.get("save").state], ["fail", 120, "skip"]);
  tr.flush();
  checkRules(snaps);
});

test("tracker: onProgress sums the parts; its step is gpu while it runs, then the copy that runs; gpuPct stays 100 after", () => {
  const { tr, progs } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  tr.state("words", "run");
  tr.at("gpu", 50);
  let [d, t, info] = progs.at(-1);
  eq([d, t, info.step, info.gpuPct, info.stepDone, info.stepTotal], [50, 1400, "gpu", 50, 50, 100]);
  tr.at("words", 100);
  tr.state("gpu", "done");
  [d, t, info] = progs.at(-1);
  eq([info.step, info.gpuPct, info.stepNo, info.steps], ["words", 100, 3, 3]);
  tr.state("save", "run"); tr.at("save", 10);
  [d, t, info] = progs.at(-1);
  eq([d, t, info.step, info.gpuPct, info.stepDone, info.stepTotal, info.stepNo], [210, 1400, "save", 100, 10, 1000, 2]);
  ok(Array.isArray(info.parts) && info.parts.length === 3, "the parts ride along");
});

// ---------------------------------------------------------------- runFlashParts

// a scripted load: each job records when it starts and ends, and waits on a gate the test opens
function script({ save = true, words = true, need = true, missing = [3, 4], wordsWorker = true } = {}) {
  const ev = [], gates = {};
  const gate = (k) => (gates[k] ||= deferred());
  const job = (name, fn) => async (...a) => { ev.push(`${name}:start`); try { const r = await fn(...a); ev.push(`${name}:end`); return r; } catch (e) { ev.push(`${name}:fail`); throw e; } };
  const seen = { parallel: [], wordsSignal: null, saveSignal: null };
  const s = {
    ev, gate, seen,
    gpu: job("gpu", async ({ at }) => { at(10); await gate("gpu").promise; at(100); return { pleTable: {} }; }),
    save: save ? {
      check: job("save.check", async () => { await gate("save.check").promise; return { need, haveBytes: need ? 100 : 0 }; }),
      copy: job("save.copy", async (_c, { at, signal }) => {
        seen.saveSignal = signal; at(300);
        await Promise.race([gate("save.copy").promise, new Promise((_, rej) => signal.addEventListener("abort", () => rej(signal.reason), { once: true }))]);
        at(1000);
      }),
      open: job("save.open", async () => {}),
    } : null,
    words: words ? {
      check: job("words.check", async () => { await gate("words.check").promise; return { missing, presentBytes: 100, dir: ["d"] }; }),
      copy: job("words.copy", async (_c, { at, signal, parallel, after }) => {
        seen.wordsSignal = signal;
        if (!wordsWorker) await after;
        seen.parallel.push(parallel()); at(50);
        await Promise.race([gate("words.copy").promise, new Promise((_, rej) => signal.addEventListener("abort", () => rej(signal.reason), { once: true }))]);
        seen.parallel.push(parallel()); at(200);
      }),
    } : null,
    finish: job("finish", async (r) => r),
  };
  return s;
}
const run = (s, parts, extra = {}) => runFlashParts({ parts, gpu: s.gpu, save: s.save, words: s.words, finish: s.finish, ...extra });

test("parallel: gpu starts before the checks answer; save's copy only after gpu; words copies alongside, 1 head then 2", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script();
  const p = run(s, tr);
  await tick();
  ok(s.ev.indexOf("gpu:start") >= 0 && s.ev.includes("save.check:start") && s.ev.includes("words.check:start"), s.ev.join());
  ok(!s.ev.includes("save.check:end"), "gpu runs while the checks are still out");
  s.gate("save.check").resolve(); s.gate("words.check").resolve();
  await tick();
  eq(byId(snaps.at(-1)).save.state, "wait", "save waits for the GPU");
  eq(byId(snaps.at(-1)).save.done, 100, "what is on disk already counts");
  ok(s.ev.includes("words.copy:start") && !s.ev.includes("save.copy:start"), s.ev.join());
  eq(s.seen.parallel, [1], "one head at a time next to the GPU");
  s.gate("gpu").resolve();
  await tick();
  ok(s.ev.indexOf("save.copy:start") > s.ev.indexOf("gpu:end"), "save's copy after the GPU step");
  s.gate("words.copy").resolve(); s.gate("save.copy").resolve();
  const r = await p;
  eq(s.seen.parallel, [1, 2], "two at a time once the GPU is done");
  eq(r.words, { ok: true, dir: ["d"] });
  ok(s.ev.indexOf("save.open:start") > s.ev.indexOf("save.copy:end") && s.ev.indexOf("finish:start") > s.ev.indexOf("words.copy:end"), s.ev.join());
  eq(snaps.at(-1).parts.map((x) => [x.id, x.state, x.done, x.total]), [["gpu", "done", 100, 100], ["save", "done", 1000, 1000], ["words", "done", 300, 300]]);
  checkRules(snaps);
});

test("parallel: a second start skips save and words within the checks; only gpu moves; the bar never goes back", async () => {
  const { tr, snaps, progs } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script({ need: false, missing: [] });
  s.gate("save.check").resolve(); s.gate("words.check").resolve();
  const p = run(s, tr);
  await tick();
  eq(snaps.at(-1).parts.map((x) => x.state), ["run", "skip", "skip"]);
  s.gate("gpu").resolve();
  const r = await p;
  ok(!s.ev.includes("save.copy:start") && !s.ev.includes("words.copy:start"), s.ev.join());
  ok(s.ev.includes("save.open:start"), "the reader still opens");
  eq(r.words, { ok: true, dir: ["d"] });
  checkRules(snaps);
  const pct = progs.map(([d, t]) => (t ? d / t : 0));
  ok(pct.every((x, i) => !i || x >= pct[i - 1]), `percent never goes back: ${pct.map((x) => x.toFixed(2))}`);
  eq(pct.at(-1), 1);
});

test("parallel: gpu fails -> words is aborted and settled before the throw; save (waiting) and words end as fail", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script();
  s.gate("save.check").resolve(); s.gate("words.check").resolve();
  const p = run(s, tr);
  await tick();
  ok(s.ev.includes("words.copy:start"), s.ev.join());
  s.gate("gpu").reject(new Error("GPU out of memory"));
  let err = null;
  try { await p; } catch (e) { err = e; }
  ok(/out of memory/.test(err?.message), String(err));
  ok(s.seen.wordsSignal.aborted, "words saw the abort");
  ok(s.ev.includes("words.copy:fail"), `words stopped before the throw: ${s.ev.join()}`);
  const last = byId(snaps.at(-1));
  eq([last.gpu.state, last.save.state, last.words.state], ["fail", "fail", "fail"]);
  eq(last.words.done, 150, "words keeps what it did");
  checkRules(snaps);
});

test("parallel: words fails -> the load goes on, finish gets { ok: false }, the reason is said", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script();
  for (const k of ["save.check", "words.check", "gpu", "save.copy"]) s.gate(k).resolve();
  s.gate("words.copy").reject(new Error("network down"));
  let said = null;
  const r = await run(s, tr, { onWordsFail: (e) => { said = e.message; } });
  eq([r.words.ok, said], [false, "network down"]);
  const last = byId(snaps.at(-1));
  eq([last.gpu.state, last.save.state, last.words.state], ["done", "done", "fail"]);
  checkRules(snaps);
});

test("parallel: save's copy fails -> the load fails, words is aborted, gpu stays done", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script();
  for (const k of ["save.check", "words.check", "gpu"]) s.gate(k).resolve();
  s.gate("save.copy").reject(new Error("out of storage"));
  let err = null;
  try { await run(s, tr); } catch (e) { err = e; }
  ok(/out of storage/.test(err?.message), String(err));
  ok(s.seen.wordsSignal.aborted && s.ev.includes("words.copy:fail"), s.ev.join());
  const last = byId(snaps.at(-1));
  eq([last.gpu.state, last.save.state, last.words.state], ["done", "fail", "fail"]);
  eq(last.save.done, 300, "save keeps what it did");
  checkRules(snaps);
});

test("parallel: Stop (the load's abort) mid-save stops save and words; nothing is finished", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script(), ac = new AbortController();
  for (const k of ["save.check", "words.check", "gpu"]) s.gate(k).resolve();
  const p = run(s, tr, { ac });
  await tick(); await tick();
  ok(s.ev.includes("save.copy:start"), s.ev.join());
  ac.abort(new Error("the start was stopped"));
  let err = null;
  try { await p; } catch (e) { err = e; }
  ok(/stopped/.test(err?.message), String(err));
  ok(s.seen.saveSignal.aborted && s.ev.includes("save.copy:fail") && s.ev.includes("words.copy:fail") && !s.ev.includes("finish:start"), s.ev.join());
  eq(snaps.at(-1).parts.map((x) => x.state), ["done", "fail", "fail"]);
  checkRules(snaps);
});

test("parallel: without a worker the words copy waits for save's copy (gpu, save, words: never two main-thread copies)", async () => {
  const { tr } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script({ wordsWorker: false });
  for (const k of ["save.check", "words.check", "gpu"]) s.gate(k).resolve();
  const p = run(s, tr);
  await tick(); await tick();
  ok(s.ev.includes("save.copy:start") && s.seen.parallel.length === 0, `words has not copied yet: ${s.ev.join()}`);
  s.gate("save.copy").resolve();
  await tick(); await tick();
  eq(s.seen.parallel, [2], "words copies after save, two heads at a time");
  s.gate("words.copy").resolve();
  await p;
});

test("parallel: no disk tier and no table (a plain Flash share): only gpu", async () => {
  const { tr, snaps } = tracked({ gpu: 100 }, { every: 0 });
  const s = script({ save: false, words: false });
  s.gate("gpu").resolve();
  const r = await run(s, tr);
  eq([r.words, r.save], [null, null]);
  eq(snaps.at(-1).parts.map((x) => x.state), ["done", "skip", "skip"]);
});

// the GPU step as room.js paces it (aiLoadPace): while paced() says so it may not finish before the slowest device
function pacedGpu(s, peer) {
  const seen = [];
  s.gpu = async ({ at, paced }) => {
    s.ev.push("gpu:start"); at(10);
    await tick();
    seen.push(paced());
    if (paced()) await peer.promise;    // held to the slowest device's GPU step
    s.ev.push("gpu:end"); at(100);
    return { pleTable: {} };
  };
  return seen;
}

test("parallel: in a room with a slow device, save's copy does not wait for it (the GPU step is not held while a copy is ahead)", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script(), peer = deferred();
  const seen = pacedGpu(s, peer);
  s.gate("save.check").resolve(); s.gate("words.check").resolve();
  const p = run(s, tr);
  await tick(); await tick(); await tick();
  eq(seen, [false], "not paced: this tab still has its disk copy to run");
  ok(s.ev.includes("save.copy:start"), `save's copy started before the slow device was done: ${s.ev.join()}`);
  s.gate("save.copy").resolve(); s.gate("words.copy").resolve();
  await p;
  checkRules(snaps);
});

test("parallel: nothing left to copy (a second start) -> the GPU step keeps the room's pace", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script({ need: false, missing: [] }), peer = deferred();
  const seen = pacedGpu(s, peer);
  s.gate("save.check").resolve(); s.gate("words.check").resolve();
  const p = run(s, tr);
  await tick(); await tick(); await tick();
  eq(seen, [true], "paced once save's check says nothing to copy");
  ok(!s.ev.includes("gpu:end"), "held for the slow device");
  peer.resolve();
  await p;
  // save and words went wait -> skip: never "run" with bytes that were not there
  for (const id of ["save", "words"]) ok(snaps.every((x) => byId(x)[id].state !== "run"), `${id} never ran`);
  checkRules(snaps);
});

test("parallel: save and words wait until their checks answer (no run with nothing moving)", async () => {
  const { tr, snaps } = tracked({ gpu: 100, save: 1000, words: 300 }, { every: 0 });
  const s = script();
  const p = run(s, tr);
  await tick();
  eq(snaps.at(-1).parts.map((x) => x.state), ["run", "wait", "wait"]);
  s.gate("words.check").resolve();
  await tick();
  eq(byId(snaps.at(-1)).words.state, "run", "words runs once its check says what is missing");
  s.gate("gpu").resolve();
  await tick();
  eq(byId(snaps.at(-1)).save.state, "wait", "save's check is slower than the GPU step: still waiting, not running");
  s.gate("save.check").resolve();
  await tick(); await tick();
  eq(byId(snaps.at(-1)).save.state, "run");
  s.gate("save.copy").resolve(); s.gate("words.copy").resolve();
  await p;
  checkRules(snaps);
});
