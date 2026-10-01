// Diagnostics must follow the extracted functions; stale anchors fail before a GPU run.
import { patchGenerator, patchPipeline, patchTransport } from "../e2e/room_trace.mjs";

Deno.test("runtime trace: generation, pipeline and transport anchors remain usable", async () => {
  for (const [file, patch, marks] of [
    ["engine/generate.js", patchGenerator, ["h.fuse0", "h.step0", "h.ret", "window.__nospec"]],
    ["room/pipeline.js", patchPipeline, ["h.lap0", "h.unp0", "w.enq", "w.start", "w.pack1"]],
    ["room/transport.js", patchTransport, ["send0", "rx0", "dlv"]],
  ]) {
    const src = await Deno.readTextFile(new URL("../../" + file, import.meta.url));
    const result = patch(src);
    for (const mark of marks) if (!result.includes(mark)) throw new Error(file + " missing " + mark);
  }
});

Deno.test("runtime profilers: untraced plain-decode switches match the generator", async () => {
  const generator = await Deno.readTextFile(new URL("../../engine/generate.js", import.meta.url));
  const room = await Deno.readTextFile(new URL("../../room.js", import.meta.url));
  const extract = (src, name) => {
    const start = src.indexOf("function " + name + "(");
    if (start < 0) throw new Error("missing " + name);
    return src.slice(start, src.indexOf("\n}", start) + 2);
  };
  const latency = await Deno.readTextFile(new URL("../e2e/room_latency.mjs", import.meta.url));
  const patchRoom = new Function(extract(latency, "patchRoom") + "; return patchRoom;")();
  if (!patchRoom(room).includes("window.__netlag")) throw new Error("missing live latency switch");
  const xroom = await Deno.readTextFile(new URL("../e2e/xroom.mjs", import.meta.url));
  const rep = (src, a, b) => { if (!src.includes(a)) throw new Error("missing anchor: " + a); return src.replace(a, b); };
  const serveGenerator = new Function("rep", "FIXK", "TRACE", extract(xroom, "serveGenerator") + "; return serveGenerator;")(rep, 0, false);
  for (const patched of [patchRoom(generator, true), serveGenerator(generator)]) {
    if ((patched.match(/!window.__nospec/g) || []).length !== 2) throw new Error("both speculation branches need the switch");
  }
});
