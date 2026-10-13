// room/loadsteps.js: a load as parts that may run side by side (Qwen3.8 Flash: save to this computer, save the word
// list, the graphics chip), from today's sequential onProgress or a concurrent loader's onParts, and the plain words
// the load card says about them. The bar must only go forward and reach 100% only at the end; the time left must not
// jump around; no "experts", "OPFS", "n-gram", "kernels" on the screen.
import {
  PART_ORDER, GATE, planSteps, drop, loadTracker, partsFromProg, partsFromSnapshot, progressTracker, roomProgress, etaTracker,
  packParts, unpackParts, leftText, stepBytes, stepLine, rowVerb, partRow, deviceVerb, headline, sentence, footer, eyebrow,
  stoppedText, stopWho, stopIsDisk, cachedStart, firstStart, wholePct, WORDS_NOTE,
} from "../../room/loadsteps.js";

const eq = (a, b, m = "") => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error(`${m} ${ja} != ${jb}`); };
const ok = (c, m = "assertion failed") => { if (!c) throw new Error(m); };
const GB = 2 ** 30;
const states = (r) => Object.fromEntries(r.parts.map((p) => [p.id, p.state]));

// today's loader (room/model-loader.js prog): onProgress(done, total, info) through a loadTracker
function seqLoad(plan) {
  const t = loadTracker(planSteps(plan)), calls = [];
  return {
    t, calls,
    prog(step, d, tot) { const o = t.at(step, tot > 0 ? d / tot : 0); const c = [o.done, o.total, { step, stepNo: o.stepNo, steps: o.steps, stepDone: d, stepTotal: tot }]; calls.push(c); return c; },
  };
}

Deno.test("planSteps / loadTracker (the loader's sizing) still work", () => {
  eq(PART_ORDER, ["save", "words", "gpu"]);
  ok(GATE.has("save") && GATE.has("words") && GATE.has("gpu"), "#388: every part gates ready");
  eq(planSteps({ saveBytes: 63, wordsBytes: 28, gpuBytes: 12 }).map((s) => s.id), ["save", "words", "gpu"]);
  eq(planSteps({ saveBytes: NaN, wordsBytes: -1, gpuBytes: "3" }), [{ id: "gpu", bytes: 3 }]);
  eq(drop(planSteps({ saveBytes: 1, gpuBytes: 2 }), "save"), [{ id: "gpu", bytes: 2 }]);
  const t = loadTracker(planSteps({ saveBytes: 60, wordsBytes: 30, gpuBytes: 10 }));
  eq(t.at("save", 0.5).pct, 30);
  t.drop("words");
  eq(t.at("gpu", 1).pct, 100);
});

Deno.test("partsFromProg: a first start, save -> words -> gpu, in order", () => {
  const L = seqLoad({ saveBytes: 60 * GB, wordsBytes: 30 * GB, gpuBytes: 10 * GB }), st = {};
  let r = partsFromProg(st, ...L.prog("save", 0, 60 * GB));
  eq(states(r), { save: "run", words: "wait", gpu: "wait" });
  eq(r.parts.find((p) => p.id === "gpu").total, null, "a later part's total is not known yet");
  r = partsFromProg(st, ...L.prog("save", 30 * GB, 60 * GB));
  eq(r.parts[0].done, 30 * GB); eq([r.done, r.total], [30 * GB, 100 * GB], "onProgress's own numbers");
  r = partsFromProg(st, ...L.prog("save", 60 * GB, 60 * GB));
  eq(states(r).save, "run", "still running until the next part starts");
  r = partsFromProg(st, ...L.prog("words", 0, 30 * GB));
  eq(states(r), { save: "done", words: "run", gpu: "wait" });
  eq(r.parts[0], { id: "save", state: "done", done: 60 * GB, total: 60 * GB });
  r = partsFromProg(st, ...L.prog("words", 30 * GB, 30 * GB));
  r = partsFromProg(st, ...L.prog("gpu", 0, 10 * GB));
  eq(states(r), { save: "done", words: "done", gpu: "run" });
  r = partsFromProg(st, ...L.prog("gpu", 10 * GB, 10 * GB));
  eq(states(r), { save: "done", words: "done", gpu: "done" });
  eq([r.done, r.total], [100 * GB, 100 * GB]);
});

Deno.test("partsFromProg: a second start skips the saving (0-byte drops)", () => {
  const L = seqLoad({ saveBytes: 60 * GB, wordsBytes: 30 * GB, gpuBytes: 10 * GB }), st = {};
  L.t.drop("save"); L.t.drop("words");
  const r = partsFromProg(st, ...L.prog("gpu", 2 * GB, 10 * GB));
  eq(states(r), { save: "skip", words: "skip", gpu: "run" });
  ok(cachedStart(r.parts) && firstStart(r.parts) === false);
  // only the word list was already there
  const L2 = seqLoad({ saveBytes: 60 * GB, wordsBytes: 30 * GB, gpuBytes: 10 * GB }), st2 = {};
  partsFromProg(st2, ...L2.prog("save", 0, 60 * GB));
  partsFromProg(st2, ...L2.prog("save", 60 * GB, 60 * GB));
  L2.t.drop("words");
  const r2 = partsFromProg(st2, ...L2.prog("gpu", 0, 10 * GB));
  eq(states(r2), { save: "done", words: "skip", gpu: "run" }, "the shrink was the skipped part, not a save that failed");
});

Deno.test("partsFromProg: a drop after bytes is a fail that keeps what it did", () => {
  const L = seqLoad({ saveBytes: 30 * GB, wordsBytes: 28 * GB, gpuBytes: 10 * GB }), st = {};
  partsFromProg(st, ...L.prog("save", 30 * GB, 30 * GB));
  partsFromProg(st, ...L.prog("words", 14 * GB, 28 * GB));
  L.t.drop("words");
  const r = partsFromProg(st, ...L.prog("gpu", 0, 10 * GB));
  eq(states(r), { save: "done", words: "fail", gpu: "run" });
  const w = r.parts.find((p) => p.id === "words");
  eq([w.done, w.total], [14 * GB, 14 * GB], "total = done");
  eq(r.total, 54 * GB, "onProgress's total shrank");
});

Deno.test("partsFromProg: steps - stepNo says which later parts wait", () => {
  // no word list on this device: save then gpu
  eq(states(partsFromProg({}, 0, 70, { step: "save", stepNo: 1, steps: 2, stepDone: 0, stepTotal: 60 })), { save: "run", words: "skip", gpu: "wait" });
  // only the word list, then gpu
  eq(states(partsFromProg({}, 0, 40, { step: "words", stepNo: 1, steps: 2, stepDone: 0, stepTotal: 30 })), { save: "skip", words: "run", gpu: "wait" });
  eq(states(partsFromProg({}, 0, 100, { step: "save", stepNo: 1, steps: 3, stepDone: 0, stepTotal: 60 })), { save: "run", words: "wait", gpu: "wait" });
  // no info: every other model, one part
  eq(partsFromProg({}, 5, 10), { parts: [{ id: "gpu", done: 5, total: 10, state: "run" }], done: 5, total: 10 });
  eq(partsFromProg({}, 10, 10).parts[0].state, "done");
});

Deno.test("partsFromSnapshot: three parts at once; skip; a fail keeps its bytes", () => {
  let r = partsFromSnapshot({ parts: [
    { id: "save", done: 20 * GB, total: 60 * GB, state: "run" }, { id: "words", done: 3 * GB, total: 30 * GB, state: "run" }, { id: "gpu", done: 5 * GB, total: 10 * GB, state: "run" }] });
  eq([r.done, r.total], [28 * GB, 100 * GB]);
  eq(states(r), { save: "run", words: "run", gpu: "run" });
  r = partsFromSnapshot({ parts: [{ id: "save", done: 0, total: 0, state: "skip" }, { id: "words", done: 0, total: 0, state: "skip" }, { id: "gpu", done: 4, total: 10, state: "run" }] });
  eq([r.done, r.total], [4, 10], "skip parts count nothing");
  r = partsFromSnapshot({ parts: [{ id: "save", done: 60, total: 60, state: "done" }, { id: "words", done: 12, total: 30, state: "fail" }, { id: "gpu", done: 10, total: 10, state: "run" }] });
  eq([r.done, r.total], [82, 82], "the fail counts as total = done");
  r = partsFromSnapshot({ parts: [{ id: "x", done: 1, total: 2, state: "run" }, { id: "gpu", done: 99, total: 10, state: "zz" }, null] });
  eq(r.parts, [], "malformed parts are left out");
});

Deno.test("progressTracker: never backwards, 100 only at the end", () => {
  // sequential, through a skip and a fail
  const L = seqLoad({ saveBytes: 30 * GB, wordsBytes: 28 * GB, gpuBytes: 10 * GB }), st = {}, tr = progressTracker();
  let last = -1;
  const push = (r) => { const o = tr.push(r); ok(o.pct >= last, `went back ${last} -> ${o.pct}`); last = o.pct; return o; };
  push(partsFromProg(st, ...L.prog("save", 15 * GB, 30 * GB)));
  push(partsFromProg(st, ...L.prog("save", 30 * GB, 30 * GB)));
  push(partsFromProg(st, ...L.prog("words", 14 * GB, 28 * GB)));
  L.t.drop("words");
  ok(push(partsFromProg(st, ...L.prog("gpu", 0, 10 * GB))).pct < 100, "a failed copy does not make the load 100%");
  eq(push(partsFromProg(st, ...L.prog("gpu", 9.99 * GB, 10 * GB))).pct, 99);
  eq(push(partsFromProg(st, ...L.prog("gpu", 10 * GB, 10 * GB))).pct, 100);
  // concurrent, then a re-deal that starts a new load with a bigger total
  const t2 = progressTracker();
  eq(t2.push(partsFromSnapshot({ parts: [{ id: "gpu", done: 6, total: 10, state: "run" }] })).pct, 60);
  const o = t2.push(partsFromSnapshot({ parts: [{ id: "save", done: 0, total: 40, state: "run" }, { id: "gpu", done: 0, total: 10, state: "wait" }] }));
  eq(o.pct, 60, "held"); eq(o.done, 6);
  eq(t2.push({ parts: [], done: 49, total: 50 }).pct, 98);
  eq(wholePct(49.99, 50), 99); eq(wholePct(50, 50), 100); eq(wholePct(0, 0), 0);
});

Deno.test("roomProgress: byte sums, an older peer, left and slowest", () => {
  const r = roomProgress({
    mac: { parts: [{ id: "gpu", state: "run" }], done: 30, total: 100, left: 300 },
    pc: { parts: [{ id: "gpu", state: "run" }], done: 10, total: 100, left: 600 },
  });
  eq([r.done, r.total, r.pct, r.left, r.slowest], [40, 200, 20, 600, "pc"]);
  // an older peer: pct only, out of the byte sums
  const o = roomProgress({ mac: { parts: [{ id: "gpu", state: "run" }], done: 50, total: 100, left: 60 }, old: { pct: 10 } });
  eq([o.done, o.total, o.pct], [50, 100, 50]);
  eq([o.left, o.slowest], [null, "old"], "a running device without an estimate");
  eq(roomProgress({ a: { pct: 40 }, b: { pct: 60 } }).pct, 50, "only old peers: the pcts average");
  // a done device does not hold the time left
  const d = roomProgress({ mac: { parts: [{ id: "gpu", state: "done" }], done: 100, total: 100, left: null }, pc: { parts: [{ id: "gpu", state: "run" }], done: 50, total: 100, left: 120 } });
  eq([d.left, d.slowest], [120, "pc"]);
  eq(roomProgress({ mac: { parts: [{ id: "gpu", state: "done" }], done: 10, total: 10 } }).left, 0);
  eq(roomProgress({ mac: { parts: [], done: 10, total: 100 } }, 30).pct, 30, "held to its best");
});

Deno.test("etaTracker: quiet until 20 s and 1 GiB, shared link, hysteresis", () => {
  const e = etaTracker();
  const run = (save, gpu) => [{ id: "save", done: save, total: 60 * GB, state: "run" }, { id: "gpu", done: gpu, total: 10 * GB, state: "run" }];
  eq(e.push(0, run(0, 0)), null);
  eq(e.push(5000, run(0.5 * GB, 0.2 * GB)), null, "under 20 s");
  eq(e.push(19000, run(0.6 * GB, 0.3 * GB)), null);
  // 20 s in but under 1 GiB
  const slow = etaTracker(); slow.push(0, run(0, 0)); slow.push(1000, run(1e6, 0));
  eq(slow.push(30000, run(0.5 * GB, 0)), null, "under 1 GiB");
  // save at 100 MB/s, gpu at 50 MB/s: save alone needs (60 - 2) GB / 100 MB/s; both share the link
  const MB = 2 ** 20;
  const f = etaTracker();
  f.push(0, run(0, 0)); f.push(1000, run(100 * MB, 50 * MB));
  const v = f.push(21000, run(2100 * MB, 1050 * MB));
  const perPart = (60 * GB - 2100 * MB) / (100 * MB), shared = (70 * GB - 3150 * MB) / (150 * MB);
  ok(v != null && Math.abs(v - Math.max(perPart, shared)) < 1, `eta ${v}`);
  ok(shared >= perPart - 1e-9 || v >= shared, "the shared-link term is at least the per-part term here");
  // hysteresis: a small change inside 10 s does not move it
  const H = etaTracker({ minMs: 0, minBytes: 0 }), big = (d) => [{ id: "gpu", done: d, total: 100000, state: "run" }];
  H.push(0, big(0)); const s0 = H.push(1000, big(100));   // 999 s
  const s1 = H.push(2000, big(220));                      // rate 110 -> ~889 s: under 20% of 998, held
  ok(Math.abs(s1 - (s0 - 1)) < 1e-6, `held (counts down) ${s0} -> ${s1}`);
  const s2 = H.push(12500, big(1150));                   // 10 s held, and up by more than a minute: it moves
  ok(s2 > s1 + 60, `moves after 10 s ${s1} -> ${s2}`);
  // never raised by less than 60 s: 90 s shown, 11 s later the rate says 89 s (a raise of 10 over the 79 it counted down to)
  const r = (d) => [{ id: "gpu", done: d, total: 10000, state: "run" }], R2 = etaTracker({ minMs: 0, minBytes: 0 });
  R2.push(0, r(0)); eq(R2.push(10000, r(1000)), 90);
  eq(R2.push(21000, r(1990)), 79, "no small raise, even after 10 s held");
  // today's sequential loader: a waiting part with no total counts the whole remainder
  const S = etaTracker({ minMs: 0, minBytes: 0 }), seq = (d) => [{ id: "save", done: d, total: 100, state: "run" }, { id: "gpu", done: 0, total: null, state: "wait" }];
  S.push(0, seq(0), { done: 0, total: 1100 });
  ok(Math.abs(S.push(1000, seq(10), { done: 10, total: 1100 }) - 109) < 1e-6, "1090 left at 10 B/s");
});

Deno.test("packParts / unpackParts", () => {
  const p = [{ id: "save", done: 1.4, total: 10, state: "run" }, { id: "gpu", done: 0, total: null, state: "wait" }];
  eq(unpackParts(packParts(p)), [{ id: "save", done: 1, total: 10, state: "run" }, { id: "gpu", done: 0, total: null, state: "wait" }]);
  eq(unpackParts("x"), null); eq(unpackParts([["evil", 1, 2, "run"], ["gpu", "a", 1, "run"]]), null);
});

Deno.test("leftText at its edges", () => {
  eq(leftText(44), "under a minute"); eq(leftText(45), "about a minute"); eq(leftText(89), "about a minute");
  eq(leftText(90), "about 2 min"); eq(leftText(600), "about 10 min");
});

Deno.test("partRow: labels and values", () => {
  eq(partRow({ id: "save", done: 19.1 * GB, total: 63.6 * GB, state: "run" }), { label: "Saving to this computer", value: "19.1 of 63.6 GB", cls: "run", pct: 30 });
  eq(partRow({ id: "words", state: "wait", done: 0, total: null }).value, "waiting");
  eq(partRow({ id: "gpu", state: "done", done: 1, total: 1 }).value, "✓ done");
  eq(partRow({ id: "gpu", state: "done", done: 1, total: 1 }, false).value, "\u2713", "a meter says it short");
  eq(partRow({ id: "save", state: "skip" }).cls, "skip");
  eq(partRow({ id: "words", state: "fail", done: 1, total: 1 }), { label: "Saving the word list", value: "read online instead", cls: "fb", pct: 100 });
  eq(partRow({ id: "save", state: "fail", done: 1, total: 1 }).value, "stopped");
  eq(partRow({ id: "gpu", state: "run", done: 1, total: 4 }, true, { single: true }).label, "Downloading the model");
  eq(partRow({ id: "gpu", state: "run", done: 1, total: 4 }, false), { label: "Graphics chip", value: "25%", cls: "run", pct: 25 });
  eq(stepBytes(300 * 2 ** 20, 1.2 * GB), "300 MB of 1.2 GB");
});

Deno.test("deviceVerb: every row", () => {
  const P = (s, w, g) => [{ id: "save", state: s, done: 1 }, { id: "words", state: w, done: 1 }, { id: "gpu", state: g, done: 1 }];
  eq(deviceVerb([{ id: "save", state: "wait", done: 0 }, { id: "gpu", state: "wait", done: 0 }]), "getting ready");
  eq(deviceVerb([]), "getting ready");
  eq(deviceVerb(P("run", "wait", "run")), "saving and loading");
  eq(deviceVerb(P("done", "run", "run")), "saving and loading");
  eq(deviceVerb(P("run", "run", "done")), "finishing its save");
  eq(deviceVerb(P("run", "wait", "wait")), "saving to its disk");
  eq(deviceVerb(P("done", "done", "run")), "loading");
  eq(deviceVerb(P("skip", "skip", "run")), "loading from its disk");
  eq(deviceVerb(P("done", "done", "done"), { slowest: "pc", self: "mac" }), "waiting for pc");
  eq(deviceVerb(P("done", "fail", "done"), { roomReady: true, slowest: "pc" }), "✓ ready");
  eq(rowVerb("save"), "saving"); eq(rowVerb(null), "loading");
});

Deno.test("headline / sentence / footer / stopped copy", () => {
  eq(headline({ left: null }, "run"), "Getting ready…");
  eq(headline({ left: 540 }, "pressed"), "Getting ready…");
  eq(headline({ left: 540 }, "run"), "Ready to chat in about 9 min");
  eq(headline({}, "ready", { model: "Qwen3.8 Flash" }), "Qwen3.8 Flash is ready");
  eq(headline({}, "stopped", { why: "QuotaExceededError: the quota has been exceeded" }), "This computer ran out of disk space");
  eq(headline({}, "stopped", { why: "pc couldn't load its part of the model (QuotaExceededError)", me: "mac" }), "pc ran out of disk space");
  eq(headline({}, "stopped", { why: "pc couldn't load its part of the model (lost)" }), "The start stopped");
  // the solo sentence: the first rule that matches
  const me = (parts, done = 1) => [{ name: "mac", you: true, parts, done }];
  const P = (s, w, g) => [{ id: "save", state: s, done: 1 }, { id: "words", state: w, done: 1 }, { id: "gpu", state: g, done: 1 }];
  eq(sentence({}, "pressed", me([])), "Checking this computer and what's already saved.");
  eq(sentence({ left: 9 }, "run", me(P("run", "wait", "run"))), "Saving the model to this computer and loading it at the same time.");
  eq(sentence({ left: 9 }, "run", me(P("run", "wait", "wait"))), "Saving the model to this computer first. This happens once.");
  eq(sentence({ left: 9 }, "run", me(P("done", "run", "done"))), "The graphics chip is loaded. Finishing the save to this computer.");
  eq(sentence({ left: 9 }, "run", me(P("done", "fail", "run"))), "Saved. Now loading it onto the graphics chip.");
  eq(sentence({ left: 9 }, "run", me(P("skip", "skip", "run"))), "Loading it from this computer onto the graphics chip.");
  eq(sentence({ left: 9 }, "run", me([{ id: "gpu", state: "run", done: 1 }])), "Downloading the model and loading it onto the graphics chip.");
  eq(sentence({ left: null }, "run", me(P("run", "wait", "wait"))), "Saving the model to this computer first. This happens once. Working out how long this takes…");
  eq(sentence({}, "ready", me([])), "Running on this computer. Ask it anything below.");
  // two devices
  const two = (mine, theirs) => [{ name: "mac", you: true, parts: mine }, { name: "pc", parts: theirs }];
  eq(sentence({}, "pressed", two([], [])), "mac and pc are checking what's already saved.");
  eq(sentence({ slowest: "pc" }, "run", two(P("run", "wait", "run"), P("run", "skip", "run"))), "Each device saves its own part and loads it.");
  eq(sentence({ slowest: "pc" }, "run", two(P("done", "done", "done"), P("run", "skip", "run"))), "Your part is ready. Waiting for pc to finish.");
  eq(sentence({ slowest: "pc" }, "run", two(P("done", "done", "run"), P("run", "skip", "run")), { paced: true }), "Waiting for pc to catch up.");
  eq(sentence({}, "pressed", two([], []), { redeal: true }), "Sharing out the model again. Nothing is lost.");
  eq(sentence({ slowest: "pc" }, "run", two(P("skip", "skip", "run"), P("skip", "skip", "run")), { first: false }), "Each device loads its own part from its disk.");
  eq(sentence({}, "pressed", two([], []), { host: "pc" }), "pc is getting the model ready.");
  eq(sentence({}, "ready", two([], [])), "Running on mac and pc together. Ask it anything below.");
  eq(sentence({}, "pressed", [...two([], []), { name: "ipad", parts: [] }]), "3 devices are checking what's already saved.");
  // footer
  eq(footer("run", { solo: true, first: true }), "The saving happens once. Next time it starts in about a minute.");
  eq(footer("run", { solo: false, first: true }), "Each device saves only its own part, once. Next time it starts in about a minute.");
  eq(footer("run", { first: false }), "", "no 'first time' note on a second start");
  eq(footer("ready", { first: true, took: 540 }), "Took 9 min · next start about a minute");
  eq(footer("ready", { first: false, took: 41 }), "Took 41 s");
  // eyebrow and stopped
  eq(eyebrow("run", { model: "Qwen3.8 Flash", first: true }), "Starting Qwen3.8 Flash · first start");
  eq(eyebrow("run", { model: "Qwen3.8 Flash", first: false }), "Starting Qwen3.8 Flash · saved from last time");
  eq(eyebrow("run", { model: "Qwen3.8 Flash" }), "Starting Qwen3.8 Flash");
  eq(eyebrow("stopped", { model: "Qwen3.8 Flash" }), "Qwen3.8 Flash stopped");
  eq(eyebrow("ready", { model: "Qwen3.8 Flash", first: true }), "First start \u00b7 saved for next time");
  eq(eyebrow("ready", { model: "Qwen3 1.7B" }), "Started");
  eq(stoppedText("QuotaExceededError", { save: { done: 40 * GB, total: 63.6 * GB } }), "40.0 of 63.6 GB is saved and stays saved. Free up about 26 GB, then Try again continues from there.");
  eq(stoppedText("pc couldn't load its part of the model (boom)", { me: "mac" }), "pc couldn't load its part.");
  eq(stoppedText("Unexpected token 'm', \"maximum qu\"... is not valid JSON"), "This computer couldn't load its part.", "the technical why stays in the tooltip");
  eq(stoppedText("This room is 4 GB short for Qwen3.8 27B", { plain: true }), "This room is 4 GB short for Qwen3.8 27B.");
    eq(stopWho("pc couldn't load its part of the model (x)"), "pc"); ok(stopIsDisk("disk full") && !stopIsDisk("lost"));
});

Deno.test("no technical words in anything the card says", () => {
  const BAD = /\b(OPFS|experts?|n-?gram|kernels?|MoE|KV|tok\/s)\b/i;
  const P = (s, w, g) => [{ id: "save", state: s, done: 5, total: 10 }, { id: "words", state: w, done: 5, total: 10 }, { id: "gpu", state: g, done: 5, total: 10 }];
  const S = ["wait", "run", "done", "skip", "fail"], out = [...WORDS_NOTE, stepLine("save"), stepLine("words"), stepLine("gpu")];
  for (const a of S) for (const b of S) for (const c of S) {
    const parts = P(a, b, c);
    for (const ph of ["pressed", "run", "ready", "stopped"]) {
      out.push(sentence({ left: null, slowest: "pc" }, ph, [{ name: "mac", you: true, parts, done: 5 }]));
      out.push(sentence({ left: 30, slowest: "pc" }, ph, [{ name: "mac", you: true, parts }, { name: "pc", parts }], { paced: a === "run" }));
      out.push(headline({ left: 100 }, ph, { model: "Qwen3.8 Flash", why: "x" }), eyebrow(ph, { model: "Qwen3.8 Flash", first: firstStart(parts) }));
      out.push(footer(ph, { solo: a === "run", first: true, took: 60 }));
    }
    out.push(deviceVerb(parts, { slowest: "pc" }), ...parts.flatMap((p) => [partRow(p).label, partRow(p).value, partRow(p, false).label, partRow(p, false).value]));
  }
  out.push(stoppedText("QuotaExceededError", { save: { done: 1, total: 2 * GB } }), stoppedText("x"));
  for (const s of out) ok(!BAD.test(s), `jargon in "${s}"`);
});
