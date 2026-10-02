// The first question after a gateway start is the slow one: the room has no checkpoints yet, so it
// prefills OpenClaw's whole system prompt and tool list (~17k tokens, about a minute on a 1.7B split
// over two devices) before it answers. Every later call resumes from the pinned prefix of that
// prompt (room-node ckpt.js). So the host keeps the last request's system prompt and tools
// (prewarm.json, 0600, next to room.json) and, when the room comes online, asks once with them and a
// one-word question for a one-token answer: that fills the pinned checkpoints before anyone asks.
// Host mode only; POOLED_PREWARM=0 (or config prewarm: false) turns it off.
// After a gateway restart the room node usually has those checkpoints on disk already (room-node
// ckptdisk.js: read back when the room comes online, when the room is exactly the same), so this ask
// then reads only its last few tokens; it still fills them when the disk copies were not usable.
import path from "node:path";
import { pooledDir, readJson, writeJson } from "./state.js";

export const prewarmFile = () => path.join(pooledDir(), "prewarm.json");
let last = null;

// what is kept of an ask body: the parts that make the prompt's fixed start (a v2 ask: tools)
export function fixedPart(body) {
  if (!body || body.api !== 2 || typeof body.system !== "string" || !body.system) return null;
  const params = { ...(body.params || {}) };
  delete params.maxTokens; delete params.stop; delete params.format; delete params.allowed; delete params.maxCalls;
  return { api: 2, system: body.system, ...(body.tools?.length ? { tools: body.tools } : {}), params };
}

// after each ask of this host's own OpenClaw: keep its fixed part when it changed
export function remember(body, model) {
  const f = fixedPart(body);
  if (!f) return false;
  const s = JSON.stringify([model, f]);
  if (s === last) return false;
  last = s;
  try { writeJson(prewarmFile(), { at: new Date().toISOString(), model, body: f }); return true; } catch { return false; }
}

// the ask that fills the checkpoints: the saved fixed part, one short user turn, one token out
export function replayBody(saved) {
  const b = saved?.body;
  if (!b?.system) return null;
  // (the saved tool choice stays: "none" would leave the tools out of the prompt, and the pin with them)
  return { ...b, messages: [{ role: "user", text: "Hi" }], params: { ...(b.params || {}), maxTokens: 1, temperature: 0, client: "OpenClaw (warm-up)" } };
}

// r: the room handle (host). -> { promptTokens, ms } | null (nothing saved, or it failed)
export async function prewarm(r, { log = () => {}, file = prewarmFile() } = {}) {
  const saved = readJson(file);
  const body = replayBody(saved);
  if (!body) return null;
  const t0 = Date.now();
  const d = await new Promise((res) => {
    let pt = 0;
    r.node.request(body, (m) => {
      if (m.t === "ai-genstart") pt = m.promptTokens || 0;
      if (m.t === "ai-gendone" || m.t === "ai-busy") res({ ...m, promptTokens: pt });
    }, { rid: "prewarm" + Date.now().toString(36) });
  });
  if (d.t === "ai-busy") { log(`warm-up skipped: ${d.why || d.code}`); return null; }
  const ms = Date.now() - t0;
  log(`warmed up OpenClaw's system prompt and tools: ${d.promptTokens} tokens in ${(ms / 1000).toFixed(1)} s, so the first question starts from them`);
  return { promptTokens: d.promptTokens, ms };
}
