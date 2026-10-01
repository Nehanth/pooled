// Onboarding's knock against a real gated host, over real WebRTC on this machine, with no GPU: a room
// node whose Dawn is a stand-in with no adapter hosts room <code> on a local PeerJS server; the
// plugin's knock (the `pooled serve` bridge) waits in its lobby until allowJoin, keeps the pass, comes
// back in with it without asking, walks in with the invite key, and is told when the host says no.
// Skipped without packages/room-node's dev dependencies (the "peer" server): cd packages/room-node && npm install
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { setStateDir, joinState } from "../src/state.js";

const NN = path.resolve(new URL("../../room-node/node_modules", import.meta.url).pathname);
const skip = !fs.existsSync(path.join(NN, ".bin/peerjs")) && "no local PeerJS server (cd packages/room-node && npm install)";

test("knock: lobby → Allow → pass; the pass and the invite key get in without asking; Deny is an answer", { skip, timeout: 120000 }, async () => {
  setStateDir(fs.mkdtempSync(path.join(os.tmpdir(), "pooled-oc-knock-")));
  const port = 19400 + Math.floor(Math.random() * 400);
  const peer = spawn(path.join(NN, ".bin/peerjs"), ["--port", String(port), "--path", "/", "--host", "127.0.0.1"], { stdio: "ignore" });
  const signal = `127.0.0.1:${port}`;
  let host = null;
  try {
    await new Promise((r) => setTimeout(r, 1500));
    const { createRoom } = await import("../../room-node/roomnode.js");
    const { knock } = await import("../src/setup.js");
    const noGpu = async () => ({ create: () => ({ requestAdapter: async () => null }), globals: {} });
    const code = "K7QXAB";
    host = await createRoom({ code, gate: true, ask: true, signal, name: "mac", setup: { webgpu: noGpu }, log: () => {} });
    const requests = [];
    host.on("joinrequest", (q) => { requests.push(q); setTimeout(() => (requests.length === 3 ? host.denyJoin(q.id) : host.allowJoin(q.id)), 300); });
    const lobby = [];
    // 1. the code alone: waits in the lobby, the host allows it, the pass comes back
    const a = await knock(code, null, { signal, name: "pc (OpenClaw)", waitMs: 30000, onLobby: (b) => lobby.push(b.hostName) });
    assert.equal(a.ok, true, JSON.stringify(a));
    assert.equal(requests.length, 1); assert.match(requests[0].line, /pc \(OpenClaw\) wants to join \(API client\)/);
    assert.deepEqual(lobby, ["mac"], "onboarding saw the lobby, with the host's name");
    assert.ok(a.pass && a.pass.length >= 22); assert.equal(a.host, "mac");
    // 2. back with that pass (onboarding keeps it for the gateway): in without a request
    const { saveJoinState } = await import("../src/state.js");
    saveJoinState(code, { pass: a.pass });
    const b = await knock(code, null, { signal, waitMs: 30000 });
    assert.equal(b.ok, true); assert.equal(requests.length, 1, "no new request");
    // 3. a fresh device with the invite key: in at once
    saveJoinState(code, { pass: "x".repeat(22) });   // (a pass the host never gave: ignored)
    const c = await knock(code, host.gate.key, { signal, waitMs: 30000 });
    assert.equal(c.ok, true); assert.equal(requests.length, 1);
    assert.equal(joinState(code).pass, "x".repeat(22), "knock() only reports; runSetup keeps the new pass");
    // 4. the code alone, and the host says no
    const d = await knock(code, null, { signal, name: "stranger", waitMs: 30000 });
    assert.equal(requests.length, 2);
    assert.equal(d.ok, true, "the second request was allowed too");
    const e = await knock(code, null, { signal, name: "stranger 2", waitMs: 30000 });
    assert.equal(requests.length, 3);
    assert.equal(e.ok, false); assert.match(e.refused, /didn't let this device in/);
  } finally {
    try { await host?.close(); } catch {}
    peer.kill();
  }
});
