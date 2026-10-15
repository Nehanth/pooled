// room/autoadd.js: a device that joins while the model runs is added on its own (a re-deal), only when it can hold part
// of the model, a few seconds after the newest join, never during an answer, and never in a loop.
import { addable, addWhen, wouldHelp, GRACE_MS, GAP_MS, POLL_MS, MAX_PER_NAME } from "../../room/autoadd.js";

const eq = (a, b, m = "") => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error(`${m} ${ja} != ${jb}`); };

const pc = { name: "pc", webgpu: true, api: false, phone: false, gb: 10 };
Deno.test("addable: devices that can hold a layer; not a phone, an API app, a browser without WebGPU or a pledge under one layer", () => {
  eq(addable([pc]), ["pc"]);
  eq(addable([{ ...pc, name: "phone", phone: true }]), [], "a phone only asks while computers hold the model");
  eq(addable([{ ...pc, name: "app", api: true }]), []);
  eq(addable([{ ...pc, name: "old", webgpu: false }]), []);
  eq(addable([{ ...pc, name: "tiny", gb: 0.3 }], { layerGB: 0.5 }), [], "under one layer");
  eq(addable([{ ...pc, gb: 0.6 }], { layerGB: 0.5 }), ["pc"]);
  eq(addable([pc], { skip: new Set(["pc"]) }), [], "Not now");
  eq(addable([pc], { added: new Map([["pc", MAX_PER_NAME]]) }), [], "added twice already: the button only");
  eq(addable([pc], { added: new Map([["pc", MAX_PER_NAME - 1]]) }), ["pc"]);
  eq(addable([null, {}, pc]), ["pc"]);
});
Deno.test("wouldHelp: by memory always; for speed only when the room is short or uses every device", () => {
  eq(wouldHelp({ split: "memory" }), true);
  eq(wouldHelp({ split: "speed", offload: true }), true, "the Flash demo: the Mac alone reads experts from disk");
  eq(wouldHelp({ split: "speed", chain: 0 }), false, "the host holds it all: a newcomer would be left out");
  eq(wouldHelp({ split: "speed", chain: 1, unneeded: true }), false, "a device is already left out as not needed");
  eq(wouldHelp({ split: "speed", chain: 2 }), true);
});
Deno.test("addWhen: a few seconds after the newest join, once idle, at most one re-deal per GAP_MS", () => {
  const base = { on: true, host: true, online: true, names: ["pc"], joinedAt: 1000, lastAt: 0 };
  eq(addWhen({ ...base, now: 1000 }), { go: false, wait: GRACE_MS }, "the grace: several joiners, one re-deal");
  eq(addWhen({ ...base, now: 1000 + GRACE_MS }), { go: true, wait: 0 });
  eq(addWhen({ ...base, now: 1000 + GRACE_MS, busy: true }), { go: false, wait: POLL_MS }, "never during an answer");
  eq(addWhen({ ...base, now: 1000 + GRACE_MS, queued: 2 }), { go: false, wait: POLL_MS }, "nor with questions waiting");
  eq(addWhen({ ...base, now: 1000 + GRACE_MS, online: false }), { go: false, wait: POLL_MS }, "nor while it loads");
  eq(addWhen({ ...base, now: 1000 + GRACE_MS, loading: true }), { go: false, wait: POLL_MS });
  // a device that drops and comes back: the last re-deal was 10 s ago, so it waits out the gap
  eq(addWhen({ ...base, joinedAt: 0, lastAt: 50000, now: 60000 }), { go: false, wait: GAP_MS - 10000 });
  eq(addWhen({ ...base, joinedAt: 0, lastAt: 50000, now: 50000 + GAP_MS }), { go: true, wait: 0 });
  for (const off of [{ on: false }, { host: false }, { degraded: true }, { names: [] }]) eq(addWhen({ ...base, now: 99999, ...off }), { go: false, wait: null }, JSON.stringify(off));
});
Deno.test("addWhen: a device that keeps dropping and coming back is added at most MAX_PER_NAME times, once per GAP_MS", () => {
  const added = new Map(); let lastAt = 0, deals = 0;
  for (let now = 0; now < 30 * 60000; now += 1000) {
    // it rejoins every 20 s (a flaky link)
    const joinedAt = now - (now % 20000);
    const names = addable([pc], { added });
    const w = addWhen({ on: true, host: true, online: true, names, joinedAt, lastAt, now });
    if (w.go) { deals++; lastAt = now; for (const n of names) added.set(n, (added.get(n) || 0) + 1); }
  }
  eq(deals, MAX_PER_NAME);
});
