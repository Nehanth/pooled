// One device's load as named steps on one bar (the load card, room.js lcStatus / loadCardRender). Qwen3.8 Flash on a
// tab whose user allowed its disk does three things in a row: saves its offloaded layers' routed experts to this
// computer (room/flashdisk.js tabExpertTier), saves the n-gram table there (room/model-loader.js saveNgramEarly), then
// puts its layers on the GPU. Each one used to fill the bar to 100% and start over, so the card said "ready" twice and
// then sat on "Starting" while 28.8 GB were still being written. Here the three share one total, sized before any of
// them starts, so the bar only goes forward and reaches 100% when the last byte is on the GPU.
// Pure (no DOM, no clock): tests/unit/loadsteps_test.js. The visible words are plain on purpose (no "experts",
// "OPFS", "n-gram", "kernels"): the detail goes to the console and tooltips.

// the steps in the order a load runs them
export const STEP_ORDER = ["save", "words", "gpu"];

// -> the plan: [{ id, bytes }] in STEP_ORDER, without the steps that have nothing to do (0 bytes)
export function planSteps({ saveBytes = 0, wordsBytes = 0, gpuBytes = 0 } = {}) {
  const by = { save: +saveBytes || 0, words: +wordsBytes || 0, gpu: +gpuBytes || 0 };
  return STEP_ORDER.filter((id) => by[id] > 0).map((id) => ({ id, bytes: by[id] }));
}

// Where the whole load is when step `stepId` has done `stepDone` of its bytes: { done, total, pct (0..100, whole),
// stepNo (1-based), steps (how many) }. The steps before it count as done, the ones after as not started; a step the
// plan does not have (dropped) counts as the end of the step before its place.
export function overall(plan, stepId, stepDone = 0) {
  const total = plan.reduce((a, s) => a + s.bytes, 0);
  let done = 0, stepNo = 0;
  const at = STEP_ORDER.indexOf(stepId);
  for (let i = 0; i < plan.length; i++) {
    const s = plan[i], k = STEP_ORDER.indexOf(s.id);
    if (k < at) { done += s.bytes; stepNo = i + 1; }
    else if (k === at) { done += Math.max(0, Math.min(s.bytes, +stepDone || 0)); stepNo = i + 1; break; }
    else break;
  }
  if (!stepNo && plan.length) stepNo = 1;
  return { done, total, pct: total ? Math.min(100, Math.floor(done / total * 100)) : 0, stepNo, steps: plan.length };
}

// the plan without step `id` (it turned out to have nothing to do: the copy was on this computer already)
export const drop = (plan, id) => plan.filter((s) => s.id !== id);

// A load's progress as it goes: at(stepId, fraction 0..1 of that step) -> overall(...) whose pct and done never go
// back (a dropped step shrinks the total; the bar can jump forward, never back). drop(id) as above.
export function loadTracker(plan) {
  let p = plan.slice(), best = 0, bestDone = 0;
  return {
    get plan() { return p; },
    at(stepId, frac) {
      const s = p.find((x) => x.id === stepId);
      const o = overall(p, stepId, s ? Math.max(0, Math.min(1, +frac || 0)) * s.bytes : 0);
      best = Math.max(best, o.pct); bestDone = Math.max(bestDone, Math.min(o.done, o.total));
      return { ...o, pct: best, done: Math.max(o.done, Math.min(bestDone, o.total)) };
    },
    drop(id) { p = drop(p, id); },
  };
}

// The pill on the load card for a step ("Copying from the room" / "Loading from cache" replace the GPU step's when
// that is where its bytes come from: room.js lcStatus)
export function stepLabel(id) {
  return id === "save" || id === "words" ? "Saving to this computer" : "Loading onto the graphics chip";
}
const LINES = {
  save: "Saving the model to this computer (first time only)",
  words: "Saving the model's word list to this computer (first time only)",
  gpu: "Putting the model on the graphics chip",
};
// the line under the bar: "Step 1 of 3 · Saving the model to this computer (first time only)"; without a count when
// there is only one step
export function stepLine(id, { stepNo = 1, steps = 1 } = {}) {
  const t = LINES[id] || LINES.gpu;
  return steps > 1 ? `Step ${stepNo} of ${steps} · ${t}` : t;
}
// a device's row on the load card and its card in the room: "saving" while it writes to its disk, "loading" after
export const rowVerb = (id) => (id === "save" || id === "words" ? "saving" : "loading");
