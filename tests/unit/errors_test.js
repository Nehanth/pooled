// room/errors.js: every failure reads as a sentence with a next step; join waits say "still
// connecting" and give ICE longer; a protocol mismatch names the older side and who reloads.
import { peerErrorText, peerErrorLoud, FetchError, fetchErrorText, joinStep, versionMismatch,
  JOIN_QUIET_MS, JOIN_TIMEOUT_MS, JOIN_ICE_MS } from "../../room/errors.js";
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
// one or more sentences ending in a full stop (a device name may start it lowercase): never a bare
// code like "error: socket-error"
const sentence = (s) => ok(typeof s === "string" && /^\S.* .*\.$/.test(s) && !/^error:/i.test(s), "not a sentence: " + s);

const PEER_TYPES = ["browser-incompatible", "disconnected", "invalid-id", "invalid-key", "network", "peer-unavailable",
  "ssl-unavailable", "server-error", "socket-error", "socket-closed", "unavailable-id", "webrtc"];

Deno.test("errors: every PeerJS error type is a sentence, on the join screen and in a room", () => {
  for (const t of PEER_TYPES) { sentence(peerErrorText(t)); sentence(peerErrorText(t, { inRoom: true })); }
  sentence(peerErrorText("something-new"));
  ok(peerErrorText("something-new").includes("something-new"), "an unknown type is named");
  sentence(peerErrorText(undefined));
});
Deno.test("errors: join-screen wording for the common cases", () => {
  ok(/No room with that code/.test(peerErrorText("peer-unavailable")));
  ok(/Press Join/.test(peerErrorText("unavailable-id")));
  ok(/internet connection/.test(peerErrorText("network")));
  ok(/keep working/.test(peerErrorText("disconnected", { inRoom: true })), "in a room a lost broker leaves the room running");
});
Deno.test("errors: only signaling and WebRTC failures toast inside a room", () => {
  ok(peerErrorLoud("disconnected") && peerErrorLoud("network") && peerErrorLoud("webrtc"));
  ok(!peerErrorLoud("peer-unavailable") && !peerErrorLoud("unavailable-id"));
});

Deno.test("errors: HTTP statuses map to what to do", () => {
  const hf = "https://huggingface.co/x/resolve/main/m.gguf";
  ok(/rate-limiting downloads \(429\)\. Wait a minute/.test(fetchErrorText(429, hf)));
  ok(/^Hugging Face/.test(fetchErrorText(429, hf)));
  ok(/^The model host/.test(fetchErrorText(429, "https://example.com/m.bin")), "another host is not called Hugging Face");
  for (const s of [502, 503, 504]) ok(/busy or down/.test(fetchErrorText(s, hf)), String(s));
  ok(/whole file/.test(fetchErrorText(200, hf)), "a 200 means ranges are refused");
  ok(/Can't reach Hugging Face/.test(fetchErrorText(0, hf)), "no status: offline or blocked");
  ok(/opened|open an issue/.test(fetchErrorText(404, hf)));
  for (const s of [0, 200, 401, 403, 404, 410, 418, 429, 500, 502, 503, 504]) sentence(fetchErrorText(s, hf));
  for (const s of [0, 429, 503]) ok(/cached layers are kept/.test(fetchErrorText(s, hf)), "retryable: " + s);
});
Deno.test("errors: FetchError carries the status and the sentence", () => {
  const e = new FetchError(429, "https://huggingface.co/a");
  ok(e instanceof Error); eq(e.status, 429); eq(e.message, fetchErrorText(429, "https://huggingface.co/a")); eq(e.name, "FetchError");
});

Deno.test("errors: join waits say still connecting, then give ICE longer, then fail with a reason", () => {
  eq(joinStep(1000, undefined), {});
  eq(joinStep(JOIN_QUIET_MS, undefined).status, "Still connecting…");
  ok(!joinStep(JOIN_TIMEOUT_MS - 1, "checking").fail);
  // no answer from signaling by the timeout: no such room
  ok(/No room with that code/.test(joinStep(JOIN_TIMEOUT_MS, undefined).fail));
  ok(/No room with that code/.test(joinStep(JOIN_TIMEOUT_MS, "new").fail));
  // the room answered and the two devices are still looking for a path: wait, up to JOIN_ICE_MS
  ok(joinStep(JOIN_TIMEOUT_MS, "checking").status && !joinStep(JOIN_TIMEOUT_MS, "checking").fail);
  ok(joinStep(JOIN_ICE_MS - 1, "checking").status);
  ok(/can't reach each other/.test(joinStep(JOIN_ICE_MS, "checking").fail));
  // a path that failed outright does not get the extension
  ok(/can't reach each other/.test(joinStep(JOIN_TIMEOUT_MS, "failed").fail));
  ok(/can't reach each other/.test(joinStep(JOIN_TIMEOUT_MS, "disconnected").fail));
  for (const ice of [undefined, "new", "checking", "failed"]) sentence(joinStep(JOIN_ICE_MS, ice).fail);
});

Deno.test("errors: host sees an older joiner; the joiner is told to reload", () => {
  const m = versionMismatch({ mine: 4, theirs: 3, name: "otter", me: "host", iAmHost: true });
  eq(m.local, "otter is on an older version of Pooled (protocol 3, this tab 4). Ask them to reload the page.");
  eq(m.remote, "This room's host is on a newer version of Pooled (protocol 4, this tab 3). Reload this page to update, then join again.");
});
Deno.test("errors: host sees a newer joiner; the joiner is told to ask the host", () => {
  const m = versionMismatch({ mine: 4, theirs: 5, name: "otter", me: "host", iAmHost: true });
  ok(/^otter is on a newer version .*Reload this page to update\.$/.test(m.local), m.local);
  ok(/^This room's host is on an older version .*Ask the host to reload the page, then join again\.$/.test(m.remote), m.remote);
});
Deno.test("errors: a joiner reads its host's hello; worker to worker names both devices", () => {
  const j = versionMismatch({ mine: 5, theirs: 4, name: "host", theyHost: true, me: "otter" });
  ok(/^This room's host is on an older version .*Ask the host to reload the page, then join again\.$/.test(j.local), j.local);
  ok(/^otter is on a newer version .*Reload this page to update\.$/.test(j.remote), j.remote);
  const w = versionMismatch({ mine: 4, theirs: 3, name: "fox", me: "owl" });
  ok(w.local.startsWith("fox is on an older version") && w.remote.startsWith("owl is on a newer version"));
});
Deno.test("errors: a peer from before versioning counts as protocol 1", () => {
  const m = versionMismatch({ mine: 4, theirs: undefined, name: "otter", me: "host", iAmHost: true });
  ok(/protocol 1, this tab 4/.test(m.local), m.local);
  sentence(m.local); sentence(m.remote);
  ok(/A device is on/.test(versionMismatch({ mine: 4, theirs: 3 }).local), "no name");
});
Deno.test("errors: a version that is not a small whole number shows as ? and asks both to reload", () => {
  for (const v of ["4<b>", { x: 1 }, 4.5, -1, 1e9, "x".repeat(500)]) {
    const m = versionMismatch({ mine: 4, theirs: v, name: "otter", me: "host", iAmHost: true });
    ok(/protocol \?, this tab 4/.test(m.local) && /protocol 4, this tab \?/.test(m.remote), m.local + " | " + m.remote);
    ok(/different version .*Reload both pages/.test(m.local) && /different version .*then join again\.$/.test(m.remote), m.remote);
    sentence(m.local); sentence(m.remote);
  }
});
