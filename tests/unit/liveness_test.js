// room/liveness.js: the ping loop's silent-link drop (#124) and the host's duplicate-name probe. No GPU.
import { PING_MS, SILENT_MS, PHONE_SILENT_MS, HOST_SILENT_MS, lastHeard, silentLimit, isSilentGone, midLoad,
  uniqueName, quietNamesake, staleNamesakes, renameTo, QUIET_MS } from "../../room/liveness.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

Deno.test("silent drop: a computer that vanished is dropped after SILENT_MS", () => {
  ok(!isSilentGone({ now: SILENT_MS, heard: 0 }), "at the limit it stays");
  ok(isSilentGone({ now: SILENT_MS + 1, heard: 0 }), "past the limit it goes");
  ok(SILENT_MS >= 4 * PING_MS, "several pings fit in the limit");
});

Deno.test("silent drop: a phone with its screen locked for a while keeps its link", () => {
  ok(!isSilentGone({ now: 30000, heard: 0, phone: true }), "30 s locked: kept");
  ok(isSilentGone({ now: PHONE_SILENT_MS + 1, heard: 0, phone: true }), "gone for over a minute: dropped");
  ok(PHONE_SILENT_MS > SILENT_MS && PHONE_SILENT_MS >= 60000);
});

Deno.test("silent drop: a guest gives the host longer", () => {
  ok(!isSilentGone({ now: 20000, heard: 0, toHost: true }));
  ok(isSilentGone({ now: HOST_SILENT_MS + 1, heard: 0, toHost: true }));
  eq(silentLimit({ toHost: true, phone: true }), HOST_SILENT_MS, "the host link's limit does not depend on the phone flag");
});

Deno.test("silent drop: never while a device loads its layers", () => {
  eq(silentLimit({ loading: true }), Infinity);
  ok(!isSilentGone({ now: 10 * 60000, heard: 0, loading: true }), "a worker blocked on a shard for 10 min is kept");
  ok(!isSilentGone({ now: 10 * 60000, heard: 0, loading: true, toHost: true }));
  ok(!isSilentGone({ now: 10 * 60000, heard: 0, loading: true, phone: true }));
});

Deno.test("silent drop: a late tick (this tab stalled) judges nobody", () => {
  ok(!isSilentGone({ now: 10 * 60000, heard: 0, late: true }));
});

Deno.test("lastHeard: a message, a wire frame or a keep-alive byte all count", () => {
  eq(lastHeard({ seen: 5, link: { rxAt: 9 } }), 9);
  eq(lastHeard({ seen: 12, link: { rxAt: 9 } }), 12);
  eq(lastHeard({ seen: 5 }), 5);
  eq(lastHeard(undefined), -Infinity);
  // a phone streaming hidden states answers no ping while busy, but its frames keep it alive
  const e = { seen: 0, link: { rxAt: 58000 } };
  ok(!isSilentGone({ now: 60000, heard: lastHeard(e) }));
});

Deno.test("midLoad: only a chain device that has not reported ready, while the start runs", () => {
  ok(midLoad({ starting: true, inChain: true, ready: false }));
  ok(!midLoad({ starting: true, inChain: true, ready: true }));
  ok(!midLoad({ starting: true, inChain: false, ready: false }), "a device not in the chain loads nothing");
  ok(!midLoad({ starting: false, inChain: true, ready: false }), "after the start (online) it is not loading");
});

Deno.test("uniqueName: a free name is kept, a taken one gets the next number", () => {
  const roster = new Map([["a", { name: "laptop" }], ["b", { name: "laptop 2" }], ["c", { name: "phone" }]]);
  eq(uniqueName("desk", "z", "host", roster), "desk");
  eq(uniqueName("laptop", "z", "host", roster), "laptop 3");
  eq(uniqueName("laptop", "a", "host", roster), "laptop", "a device's own entry does not count against it");
  eq(uniqueName("host", "z", "host", roster), "host 2", "the host's own name is taken");
  const long = "x".repeat(40);
  eq(uniqueName(long, "z", long, new Map()), "x".repeat(36) + " 2", "the number fits in the 40-char name");
});

Deno.test("quietNamesake: probe a namesake only when it has been quiet", () => {
  const roster = new Map([["a", { name: "laptop" }], ["b", { name: "phone" }]]);
  const heard = { a: 1000, b: 1000 };
  const heardOf = (id) => heard[id] ?? -Infinity;
  eq(quietNamesake("laptop", "z", roster, heardOf, 1000 + QUIET_MS - 1), null, "heard from just now: no probe");
  eq(quietNamesake("laptop", "z", roster, heardOf, 1000 + QUIET_MS), "a");
  eq(quietNamesake("desk", "z", roster, heardOf, 99999), null, "no namesake");
  eq(quietNamesake("laptop", "a", roster, heardOf, 99999), null, "a device is not its own namesake");
  eq(quietNamesake("laptop", "z", new Map([["q", { name: "laptop" }]]), heardOf, 0), "q", "a namesake with no link is quiet");
});

Deno.test("staleNamesakes: after the probe, drop only those that did not answer", () => {
  const roster = new Map([["a", { name: "laptop" }], ["b", { name: "laptop" }], ["c", { name: "phone" }]]);
  const heard = { a: 500, b: 2100, c: 0 };
  const heardOf = (id) => heard[id] ?? -Infinity;
  eq(staleNamesakes("laptop", "z", roster, heardOf, 2000), ["a"], "b answered the probe");
  eq(staleNamesakes("laptop", "a", roster, heardOf, 2000), [], "the newcomer's own entry is never dropped");
  eq(staleNamesakes("phone", "z", roster, heardOf, 2000), ["c"]);
  eq(staleNamesakes("desk", "z", roster, heardOf, 2000), []);
});

Deno.test("renameTo: a device takes the name the host's roster gives it", () => {
  const members = [{ id: "h", name: "host" }, { id: "me", name: "laptop 2" }];
  eq(renameTo(members, "me", "laptop", true), "laptop 2");
  eq(renameTo(members, "me", "laptop 2", true), null, "same name: nothing to do");
  eq(renameTo(members, "me", "laptop", false), null, "only the host's roster counts");
  eq(renameTo(members, "other", "laptop", true), null, "not listed yet");
  eq(renameTo([{ id: "me", name: "" }], "me", "laptop", true), null, "an empty name is ignored");
  eq(renameTo(undefined, "me", "laptop", true), null);
});
