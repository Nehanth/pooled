// room/compute.js: the full-screen "this device is lending its GPU" view, over a small fake DOM.
// The clock, animation frames, intervals and window listeners are all fakes, so nothing here
// depends on real time.
import { computeScreen } from "../../room/compute.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const IDS = ["compute-screen", "cs-flow", "cs-logo", "cs-code", "cs-devs", "cs-title", "cs-sub", "cs-live", "cs-tok", "cs-rate", "cs-ms", "cs-strip", "compute-exit", "compute-open"];

function fakeEl(id, dom) {
  const attrs = new Map(), props = new Map(), on = {}, classes = new Set();
  return {
    id, hidden: false, textContent: "", innerHTML: "", dataset: {}, offsetWidth: 0, width: 0, height: 0,
    parentNode: { hidden: false },
    style: { setProperty: (k, v) => props.set(k, String(v)), getPropertyValue: (k) => props.get(k) ?? "" },
    classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c), toggle: (c, v) => (v ? classes.add(c) : classes.delete(c)) },
    toggleAttribute(n, v) { v ? attrs.set(n, "") : attrs.delete(n); }, hasAttribute: (n) => attrs.has(n),
    setAttribute: (n, v) => attrs.set(n, v), getAttribute: (n) => attrs.get(n) ?? null,
    addEventListener(t, f) { (on[t] ||= []).push(f); }, fire(t, e = {}) { for (const f of on[t] || []) f(e); },
    focus() { dom.focused = id; },
    getBoundingClientRect: () => ({ top: 0, left: 100, width: 300, height: 600 }),
    querySelector: () => null, querySelectorAll: () => [], animate() {},
    getContext: () => dom.ctx,
  };
}

// Installs the fakes, runs fn(dom), and puts every global back even when fn throws.
async function withDom(fn, { missing = [], reduced = true } = {}) {
  const dom = { els: {}, focused: null, now: 10_000, rafs: [], intervals: new Map(), winOn: {}, docOn: {}, calls: 0 };
  for (const id of IDS) if (!missing.includes(id)) dom.els[id] = fakeEl(id, dom);
  const meta = fakeEl("meta", dom), root = fakeEl("html", dom);
  dom.meta = meta; dom.html = root;
  dom.ctx = new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => { dom.calls++; return { addColorStop() {} }; }), set: (t, k, v) => { t[k] = v; return true; } });
  const G = globalThis;
  const keys = ["document", "matchMedia", "requestAnimationFrame", "cancelAnimationFrame", "devicePixelRatio", "addEventListener", "setInterval", "clearInterval"];
  const saved = Object.fromEntries(keys.map((k) => [k, Object.getOwnPropertyDescriptor(G, k)]));
  const savedNow = performance.now;
  const set = (k, v) => Object.defineProperty(G, k, { configurable: true, writable: true, value: v });
  let seq = 0;
  set("document", {
    hidden: false,
    getElementById: (id) => dom.els[id] || null,
    querySelector: (s) => (s === 'meta[name="theme-color"]' ? meta : null),
    documentElement: root,
    addEventListener: (t, f) => (dom.docOn[t] ||= []).push(f),
  });
  set("matchMedia", () => ({ matches: reduced }));
  set("requestAnimationFrame", (f) => { dom.rafs.push(f); return dom.rafs.length; });
  set("cancelAnimationFrame", () => {});
  set("devicePixelRatio", 3);
  set("addEventListener", (t, f) => (dom.winOn[t] ||= []).push(f));
  set("setInterval", (f, ms) => { dom.intervals.set(++seq, { f, ms }); return seq; });
  set("clearInterval", (id) => dom.intervals.delete(id));
  performance.now = () => dom.now;
  dom.flushFrames = () => { const fs = dom.rafs.splice(0); for (const f of fs) f(dom.now); };
  try { await fn(dom); } finally {
    for (const k of keys) saved[k] ? Object.defineProperty(G, k, saved[k]) : delete G[k];
    performance.now = savedNow;
  }
}
const $ = (dom, id) => dom.els[id];

Deno.test("compute screen: without its elements it is a harmless stub", () => withDom(() => {
  const s = computeScreen({ state: () => { throw new Error("must not read state"); } });
  s.open(); s.pass(3, 10); s.refresh(); s.close();
  eq(s.isOpen, false);
}, { missing: ["cs-logo"] }));

Deno.test("compute screen: open and close toggle the screen, the browser bar colour and focus; Escape closes", () => withDom((dom) => {
  let awake = 0;
  const s = computeScreen({ state: () => ({ phase: "idle", devices: 1 }), keepAwake: () => awake++ });
  eq($(dom, "cs-logo").innerHTML.match(/<circle/g).length, 9, "the nine-dot mark");
  ok($(dom, "cs-logo").innerHTML.includes('class="lit"'), "the blue dot");
  s.open(); s.open();
  eq([s.isOpen, awake, $(dom, "compute-screen").hidden], [true, 1, false], "opening twice keeps the screen awake once");
  eq(dom.meta.getAttribute("content"), "#000000");
  ok(dom.html.classList.contains("computing"));
  eq(dom.focused, "compute-exit");
  eq([...dom.intervals.values()].map((t) => t.ms), [1000], "one refresh timer");
  // the first animation frame sizes the canvas (devicePixelRatio 3 is capped at 2)
  dom.flushFrames();
  eq([$(dom, "cs-flow").width, $(dom, "cs-flow").height], [600, 1200]);
  dom.flushFrames();
  ok(dom.calls > 0, "the lane was drawn");
  for (const f of dom.winOn.keydown) f({ key: "Enter" });
  ok(s.isOpen, "only Escape closes");
  for (const f of dom.winOn.keydown) f({ key: "Escape" });
  eq([s.isOpen, $(dom, "compute-screen").hidden, dom.meta.getAttribute("content"), dom.focused], [false, true, "#F6F5F1", "compute-open"]);
  ok(!dom.html.classList.contains("computing"));
  eq(dom.intervals.size, 0, "the refresh timer is cleared");
  s.open(); $(dom, "compute-exit").fire("click");
  eq(s.isOpen, false, "the exit button closes");
}));

Deno.test("compute screen (Serve API): onShow follows open and close; Tab cycles through every control on screen", () => withDom((dom) => {
  const shown = [];
  const s = computeScreen({ state: () => ({ phase: "idle", devices: 1 }), onShow: (on) => shown.push(on) });
  s.open(); s.open(); s.close(); s.close();
  eq(shown, [true, false], "once per change");
  // the API half: a button, the switch, a folded summary, a code block, a hidden button, a -1 tab and one off screen (in a closed <details>)
  const el = (id, o = {}) => ({ id, hidden: false, disabled: false, tabIndex: 0, getClientRects: () => [1], focus() { dom.focused = id; }, ...o });
  const f = [el("compute-exit"), el("api-copy", { getClientRects: () => [] }), el("summary"), el("api-t-py", { tabIndex: -1 }), el("api-code"), el("cs-new", { hidden: true }), el("api-allow")];
  let sel = "";
  $(dom, "compute-screen").querySelectorAll = (q) => { sel = q; return f; };
  s.open();
  const tab = (from, shiftKey = false) => { document.activeElement = f.find((x) => x.id === from); let prevented = false; for (const h of dom.winOn.keydown) h({ key: "Tab", shiftKey, preventDefault: () => { prevented = true; } }); return prevented; };
  dom.focused = null;
  ok(tab("api-allow"), "Tab on the last control wraps"); eq(dom.focused, "compute-exit");
  ok(["button", "input", "summary", "[tabindex]"].every((k) => sel.includes(k)), "the trap looks past buttons: " + sel);
  dom.focused = null;
  ok(tab("compute-exit", true), "Shift+Tab on the first wraps back"); eq(dom.focused, "api-allow");
  ok(!tab("summary"), "in the middle the browser moves focus");
  delete document.activeElement;
  s.close();
}));

Deno.test("compute screen: the status line for each phase, table-driven", () => withDom((dom) => {
  let st = {};
  const s = computeScreen({ state: () => st });
  s.open();
  const cases = [
    [{ phase: "serving", lo: 2, hi: 6, model: "Qwen", devices: 3, code: "ABCD" }, "Serving", "Layers 3–6 · Qwen", false],
    // an empty range is no layers (it used to read "Layers 1–0")
    [{ phase: "serving", lo: 0, hi: 0, devices: 2 }, "Not holding layers", "The other devices run the model", true],
    [{ phase: "serving", lo: 0, hi: 1, devices: 2 }, "Serving", "Layers 1–1 · the model", false],
    [{ phase: "loading", pct: 42.4, lo: 0, hi: 10, model: "Qwen", devices: 2 }, "Loading 42%", "Layers 1–10 · Qwen", true],
    [{ phase: "loading", model: "Qwen", devices: 2 }, "Loading", "Qwen", true],
    [{ phase: "serving", devices: 2, model: "Qwen" }, "Not holding layers", "The other devices run Qwen", true],
    [{ phase: "serving", devices: 2 }, "Not holding layers", "The other devices run the model", true],
    [{ phase: "idle", devices: 1 }, "Standing by", "Waiting for the room to start a model", true],
  ];
  for (const [state, title, sub, liveHidden] of cases) {
    st = state;
    s.refresh();
    const tag = JSON.stringify(state);
    eq([$(dom, "cs-title").textContent, $(dom, "cs-sub").textContent], [title, sub], tag);
    eq($(dom, "cs-live").hidden, liveHidden, "live numbers only while serving layers: " + tag);
    eq($(dom, "compute-screen").dataset.phase, state.phase);
  }
  st = { phase: "idle", devices: 1 }; s.refresh();
  eq([$(dom, "cs-code").textContent, $(dom, "cs-devs").textContent], ["----", "1 device"]);
  st = { phase: "idle", devices: 4, code: "WXYZ" }; s.refresh();
  eq([$(dom, "cs-code").textContent, $(dom, "cs-devs").textContent], ["WXYZ", "4 devices"]);
  s.close();
}));

Deno.test("compute screen: the device colour stays readable on the black screen", () => withDom((dom) => {
  let color = "";
  const s = computeScreen({ state: () => ({ phase: "idle", devices: 1, color }) });
  s.open();
  const cases = [
    ["#2A45E0", "#2A45E0"],             // blue keeps its colour
    ["#14161D", "#EEF0F6"],             // near-black neutral shows light
    ["#5E616B", "#EEF0F6"],             // dark grey shows light
    ["#000000", "#EEF0F6"],
    ["#E4E2DA", "#E4E2DA"],             // light grey is already visible
    ["hsl(230 75% 52%)", "hsl(230 75% 52%)"],
    ["hsl(0 0% 10%)", "#EEF0F6"],
    ["hsl(230 80% 8%)", "#EEF0F6"],      // saturated but almost black
    ["red", "red"],                      // a colour it can't parse is passed through
  ];
  for (const [c, want] of cases) { color = c; s.refresh(); eq($(dom, "compute-screen").style.getPropertyValue("--me"), want, c); }
  color = ""; s.refresh();
  eq($(dom, "compute-screen").style.getPropertyValue("--me"), "red", "no colour leaves the last one");
  s.close();
}));

Deno.test("compute screen: the layer strip marks this device's layers, and the ones still loading", () => withDom((dom) => {
  let st = { phase: "idle", devices: 1 };
  const s = computeScreen({ state: () => st });
  s.open();
  const strip = $(dom, "cs-strip");
  const cells = () => [...strip.innerHTML.matchAll(/<i class="([^"]*)"/g)].map((m) => m[1]);
  st = { phase: "serving", lo: 2, hi: 6, total: 8, devices: 2 }; s.refresh();
  eq(cells(), ["", "", "mine", "mine", "mine", "mine", "", ""]);
  eq([strip.hidden, strip.style.getPropertyValue("--n")], [false, "8"]);
  st = { phase: "loading", lo: 2, hi: 6, total: 8, pct: 50, devices: 2 }; s.refresh();
  eq(cells(), ["", "", "mine", "mine", "mine wait", "mine wait", "", ""], "half loaded");
  const before = strip.innerHTML;
  strip.innerHTML = "untouched"; s.refresh();
  eq(strip.innerHTML, "untouched", "the same state does not rebuild the strip");
  strip.innerHTML = before;
  st = { phase: "serving", lo: 0, hi: 50, total: 200, devices: 2 }; s.refresh();
  eq(cells().length, 64, "at most 64 cells");
  eq(cells().filter((c) => c === "mine").length, 16, "a quarter of them");
  st = { phase: "idle", devices: 1 }; s.refresh();
  eq(strip.hidden, true, "no model, no strip");
  s.close();
}));

Deno.test("compute screen: passes count tokens, give a rate and a ms per pass, and go quiet after 2.5 s", () => withDom((dom) => {
  const s = computeScreen({ state: () => ({ phase: "serving", lo: 0, hi: 4, total: 8, devices: 2 }) });
  s.open();
  const root = $(dom, "compute-screen");
  ok(root.hasAttribute("data-quiet"), "no pass yet: quiet");
  eq([$(dom, "cs-rate").textContent, $(dom, "cs-rate").parentNode.hidden], ["0", true], "no rate it never had");
  eq([$(dom, "cs-ms").textContent, $(dom, "cs-ms").parentNode.hidden], ["", true]);
  s.pass(1, 37.6);
  ok(!root.hasAttribute("data-quiet"), "a pass wakes it");
  dom.now += 100; s.pass(1);
  dom.now += 100; s.pass(1);
  s.refresh();
  eq($(dom, "cs-tok").textContent, "3");
  eq([$(dom, "cs-rate").textContent, $(dom, "cs-rate").parentNode.hidden], ["10.0", false], "2 tokens over 0.2 s");
  eq([$(dom, "cs-ms").textContent, $(dom, "cs-ms").parentNode.hidden], ["38 ms", false]);
  dom.now += 3000; s.refresh();
  ok(root.hasAttribute("data-quiet"), "quiet after 2.5 s without a pass");
  eq($(dom, "cs-rate").textContent, "10.0", "between answers the last rate stays");
  s.pass(12_342); s.refresh();
  eq($(dom, "cs-tok").textContent, "12.3k");
  s.pass(2_500_000); s.refresh();
  eq($(dom, "cs-tok").textContent, "2.5M");
  s.close();
  s.pass(1);   // closed: counted, not drawn
  s.open();
  eq($(dom, "cs-tok").textContent, "2.5M");
  s.close();
}));

Deno.test("compute screen: with motion on, a pass sweeps the strip once per burst", () => withDom(async (dom) => {
  const timers = [];
  const savedST = globalThis.setTimeout;
  globalThis.setTimeout = (f, ms) => { timers.push({ f, ms }); return timers.length; };
  try {
    const s = computeScreen({ state: () => ({ phase: "serving", lo: 0, hi: 4, total: 8, devices: 2 }) });
    s.open();
    s.pass(1);
    eq(timers.length, 1, "one sweep scheduled");
    dom.now += 50; s.pass(1);
    eq(timers.length, 2, "a pass inside 140 ms is owed, not dropped");
    eq(timers[1].ms, 90);
    dom.now += 10; s.pass(1);
    eq(timers.length, 2, "only one owed spawn at a time");
    timers[0].f();
    ok($(dom, "cs-strip").classList.contains("sweep"));
    s.close();
  } finally { globalThis.setTimeout = savedST; }
}, { reduced: false }));
