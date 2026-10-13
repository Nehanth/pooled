// room/loadsteps.js: one device's load as named steps on one bar (Qwen3.8 Flash: save to this computer, save the word
// list, then the GPU). The bar must only go forward, reach 100% only at the end of the last step, and say it in plain
// words (no "experts", "OPFS", "n-gram", "kernels" on the screen).
import { planSteps, overall, drop, loadTracker, stepLabel, stepLine, rowVerb, STEP_ORDER } from "../../room/loadsteps.js";

const eq = (a, b, m = "") => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error(`${m} ${ja} != ${jb}`); };
const ok = (c, m = "assertion failed") => { if (!c) throw new Error(m); };
const GB = 2 ** 30;

Deno.test("planSteps: the steps in order, without the ones with nothing to do", () => {
  eq(planSteps({ saveBytes: 63 * GB, wordsBytes: 28 * GB, gpuBytes: 12 * GB }).map((s) => s.id), ["save", "words", "gpu"]);
  eq(planSteps({ saveBytes: 63 * GB, gpuBytes: 12 * GB }).map((s) => s.id), ["save", "gpu"]);
  eq(planSteps({ gpuBytes: 5 }), [{ id: "gpu", bytes: 5 }]);
  eq(planSteps({}), []);
  eq(planSteps({ saveBytes: NaN, wordsBytes: -1, gpuBytes: "3" }), [{ id: "gpu", bytes: 3 }]);
});

Deno.test("overall: one total across the steps, numbered, and 100% only at the end of the last", () => {
  const plan = planSteps({ saveBytes: 60, wordsBytes: 30, gpuBytes: 10 });
  eq(overall(plan, "save", 0), { done: 0, total: 100, pct: 0, stepNo: 1, steps: 3 });
  eq(overall(plan, "save", 60), { done: 60, total: 100, pct: 60, stepNo: 1, steps: 3 });
  eq(overall(plan, "words", 15), { done: 75, total: 100, pct: 75, stepNo: 2, steps: 3 });
  eq(overall(plan, "gpu", 9.99).pct, 99, "not 100 before the last byte");
  eq(overall(plan, "gpu", 10), { done: 100, total: 100, pct: 100, stepNo: 3, steps: 3 });
  eq(overall(plan, "gpu", 1e9).done, 100, "clamped to the step");
  // steps renumber when one is gone
  eq(overall(drop(plan, "words"), "gpu", 0), { done: 60, total: 70, pct: 85, stepNo: 2, steps: 2 });
  eq(overall(planSteps({ gpuBytes: 10 }), "gpu", 5), { done: 5, total: 10, pct: 50, stepNo: 1, steps: 1 });
});

Deno.test("loadTracker: monotonic across save -> words -> gpu, a dropped step never lowers the bar", () => {
  const t = loadTracker(planSteps({ saveBytes: 60, wordsBytes: 30, gpuBytes: 10 }));
  let last = -1, lastDone = -1;
  const seen = [];
  const step = (id, f) => {
    const o = t.at(id, f);
    ok(o.pct >= last && o.done >= lastDone, `went back at ${id} ${f}: ${o.pct} < ${last}`);
    last = o.pct; lastDone = o.done; seen.push(o.pct);
    return o;
  };
  for (const f of [0, 0.1, 0.5, 1]) step("save", f);
  step("words", 0);
  t.drop("words");   // the word list was on this computer already
  const after = step("gpu", 0);
  eq([after.stepNo, after.steps, after.total], [2, 2, 70]);
  ok(after.pct >= 60, "the drop moves the bar forward");
  for (const f of [0.3, 0.99]) ok(step("gpu", f).pct < 100, "not ready before the last GPU byte");
  eq(step("gpu", 1).pct, 100);
  // a progress tick that reports less than before (a retried read) does not move it back
  eq(t.at("gpu", 0.2).pct, 100);
});

Deno.test("loadTracker: a save step found already on disk is dropped before it starts: the bar starts at 0 of the rest", () => {
  const t = loadTracker(planSteps({ saveBytes: 60, gpuBytes: 10 }));
  t.drop("save");
  eq(t.at("gpu", 0), { done: 0, total: 10, pct: 0, stepNo: 1, steps: 1 });
  eq(t.at("gpu", 0.5).pct, 50);
});

Deno.test("the words on the card: plain, one per step, the count only with more than one step", () => {
  eq(stepLabel("save"), "Saving to this computer");
  eq(stepLabel("words"), "Saving to this computer");
  eq(stepLabel("gpu"), "Loading onto the graphics chip");
  eq(stepLine("save", { stepNo: 1, steps: 3 }), "Step 1 of 3 · Saving the model to this computer (first time only)");
  eq(stepLine("words", { stepNo: 2, steps: 3 }), "Step 2 of 3 · Saving the model's word list to this computer (first time only)");
  eq(stepLine("gpu", { stepNo: 3, steps: 3 }), "Step 3 of 3 · Putting the model on the graphics chip");
  eq(stepLine("gpu", { stepNo: 1, steps: 1 }), "Putting the model on the graphics chip");
  eq([rowVerb("save"), rowVerb("words"), rowVerb("gpu"), rowVerb(null)], ["saving", "saving", "loading", "loading"]);
});

// what a person reads while the model loads: none of the developer words
export const JARGON = /\b(experts?|OPFS|n-?gram|kernels?|shaders?|MoE|tok\/s|T1|pledges?|GPU pipelines?|re-?deal\w*)\b/i;
Deno.test("no jargon in any step's label, line or verb", () => {
  for (const id of STEP_ORDER) {
    for (const s of [stepLabel(id), stepLine(id, { stepNo: 1, steps: 3 }), rowVerb(id)]) ok(!JARGON.test(s), `jargon in "${s}"`);
  }
});

// room.js's loading strings, read from its source (it can't be imported without a DOM): the status lines a load
// shows on the card, the card's note, and the GPU check's message
Deno.test("room.js and the loader: the load's status lines are plain words", async () => {
  const room = await Deno.readTextFile(new URL("../../room.js", import.meta.url));
  const loader = await Deno.readTextFile(new URL("../../room/model-loader.js", import.meta.url));
  const shown = [...(room + loader).matchAll(/aiStatus\("([^"]+)"\)/g)].map((m) => m[1]);
  for (const s of ["Getting this computer ready\\u2026", "Checking the graphics chip\\u2026", "Looking up the model\\u2026", "Almost ready\\u2026",
    "Checking what's already saved on this computer\\u2026", "Waiting for you to say if the model may be saved on this computer",
    "Waiting a moment so every device finishes together"]) ok(shown.includes(s) || room.includes(`"${s}"`), `missing status "${s}"`);
  for (const s of ["requesting GPU", "testing GPU kernels", "tuning kernels", "reading model index", "building GPU pipelines", "please screenshot this", "please send me this line"])
    ok(!room.includes(s) && !loader.includes(s), `still says "${s}"`);
  const gpuFail = /const GPU_CHECK_FAILED = "([^"]+)"/.exec(room)?.[1];
  ok(gpuFail && !JARGON.test(gpuFail) && !/FAILED/.test(gpuFail), String(gpuFail));
});
