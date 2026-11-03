// A host can disappear while a worker waits for slower devices in the load pacer.
import { roomFns } from "./room_src.js";

Deno.test("host departure: cancelling a load interrupts its pacing wait", async () => {
  const ai = { gpuPct: 20, startFailed: null };
  const lc = {};
  let release, waits = 0;
  const { aiLoadPace } = roomFns(["aiLoadPace"], {
    ai, lc, slowestLoadProgress: () => 0, aiStatus: () => {}, lcPaint: () => {},
    setTimeout: (fn) => {
      if (++waits === 1) release = fn;
      // Let a broken pacer finish instead of hanging the test: the slow peer catches up later.
      else { ai.gpuPct = 100; fn(); }
    },
  });
  const loading = aiLoadPace();
  ai.startFailed = "the model's host left the room";
  release();
  let stopped = "";
  try { await loading; } catch (err) { stopped = err.message; }
  if (stopped !== ai.startFailed) throw new Error("the cancelled load resumed after its pacing wait");
});

Deno.test("host departure: a creator's loss retains the existing room recovery path", () => {
  const engine = {};
  const ai = { dealer: "pooled-room-ABCD", role: "worker", engine, pressedHere: true };
  const { followedHostLeft } = roomFns(["followedHostLeft"], {
    ai, PREFIX: "pooled-room-", roomCode: "ABCD", clearTimeout: () => {},
  });
  followedHostLeft();
  if (ai.engine !== engine || ai.role !== "worker") throw new Error("creator recovery discarded the worker's layers");
  if (ai.dealer !== null) throw new Error("the departed creator is still followed");
});
