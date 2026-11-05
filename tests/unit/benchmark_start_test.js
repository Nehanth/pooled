import { browserReady, configureBrowser } from "../../benchmarks/bench_room_browser.js";

Deno.test("benchmark startup rejects unknown selections before any setting changes", () => {
  const saved = Object.getOwnPropertyDescriptor(globalThis, "document"), events = [];
  const select = (value, options) => ({ value, options: options.map((value) => ({ value })), dispatchEvent: (e) => events.push(e.type) });
  const model = select("small", ["small", "qwen3-1.7b"]), split = select("speed", ["speed", "memory"]);
  Object.defineProperty(globalThis, "document", { configurable: true, value: { getElementById: (id) => ({ "ai-model": model, "ai-split": split })[id] } });
  try {
    for (const invalid of ["auto", ""]) {
      let error; try { configureBrowser({ model: "qwen3-1.7b", split: invalid }); } catch (e) { error = e; }
      if (!error?.message.includes("unsupported ai-split") || model.value !== "small" || split.value !== "speed" || events.length) throw Error("invalid configuration changed controls");
    }
    configureBrowser({ model: "qwen3-1.7b", split: "memory" });
    if (model.value !== "qwen3-1.7b" || split.value !== "memory" || events.join() !== "change,change") throw Error("valid controls not applied");
    configureBrowser({});
    if (events.length !== 2) throw Error("omitted settings must stay unchanged");
  } finally { if (saved) Object.defineProperty(globalThis, "document", saved); else delete globalThis.document; }
});

Deno.test("benchmark startup rejects GPU and app failures instead of waiting for readiness", () => {
  const savedWindow = Object.getOwnPropertyDescriptor(globalThis, "window"), savedDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  let state = { ready: false, busy: false }, status = "downloading";
  Object.defineProperty(globalThis, "window", { configurable: true, value: { __benchGpuErrors: [], __pooledBench: { snapshot: () => state } } });
  Object.defineProperty(globalThis, "document", { configurable: true, value: { getElementById: () => ({ textContent: status }) } });
  const eq = (a, b) => { if (a !== b) throw Error(`${a} != ${b}`); };
  const fails = (text) => { let error; try { browserReady(); } catch (e) { error = e; } if (!error?.message.includes(text)) throw Error("expected " + text); };
  try {
    eq(browserReady(), false);
    state = { ready: true, busy: true }; eq(browserReady(), false);
    state.busy = false; eq(browserReady(), true);
    window.__benchGpuErrors.push("A valid external Instance reference no longer exists.");
    fails("GPU failure during startup");
    window.__benchGpuErrors.length = 0; state.ready = false; status = "failed: GPU self-test FAILED";
    fails("GPU self-test FAILED");
  } finally {
    if (savedWindow) Object.defineProperty(globalThis, "window", savedWindow); else delete globalThis.window;
    if (savedDocument) Object.defineProperty(globalThis, "document", savedDocument); else delete globalThis.document;
  }
});
