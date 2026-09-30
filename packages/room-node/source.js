// Where a room node reads a model's bytes: a local copy of the room's GGUF when it has one (in
// modelDir: <model key>/<file name> as `pooled pull` keeps it, or the LOCAL layout below), else HTTP range requests to that URL,
// the same requests the room page makes. The room page's weight caches, peer-to-peer weight
// transfer and download pacing are browser features left out of this prototype.
import fs from "node:fs";
import path from "node:path";
import { MODELS } from "../../room/models.js";
import { parseGGUFHeader } from "../../engine/gguf.js";

// file name in the room's URL -> path under modelDir (the layout of tests/e2e/xroom.mjs LOCAL)
export const LOCAL = {
  "Qwen3.8-27B-Q4_0.gguf": "q38/model.gguf",
  "Qwen3-0.6B-Q8_0.gguf": "qwen/model.gguf",
  "Qwen3-1.7B-Q8_0.gguf": "qwen17/model.gguf",
  "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf": "q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf",
};

export function openModel(modelKey, { modelDir = process.env.POOLED_MODELS || null } = {}) {
  const M = MODELS[modelKey];
  if (!M || (M.kind !== "gguf" && M.kind !== "qwen35")) throw new Error(`unsupported model ${modelKey}`);
  const base = M.gguf.split("/").pop();
  // `pooled pull`'s layout (<dir>/<model key>/<file>, cli/lib/cache.js) first, then the test layout
  const local = modelDir ? [path.join(modelDir, modelKey, base), LOCAL[base] && path.join(modelDir, LOCAL[base])].find((f) => f && fs.existsSync(f)) || null : null;
  let fh = null;
  const readAt = local
    ? async (off, len) => {
      fh ||= await fs.promises.open(local, "r");
      const out = new Uint8Array(len);
      let o = 0;
      while (o < len) { const { bytesRead } = await fh.read(out, o, Math.min(len - o, 1 << 30), off + o); if (bytesRead <= 0) break; o += bytesRead; }
      if (o !== len) throw new Error(`short read in ${local} at ${off}`);
      return out;
    }
    : async (off, len) => {
      const r = await fetch(M.gguf, { headers: { range: `bytes=${off}-${off + len - 1}` } });
      if (r.status !== 206 && !(r.status === 200 && off === 0)) throw new Error(`range fetch ${r.status} for ${M.gguf}`);
      const b = new Uint8Array(await r.arrayBuffer());
      return b.length > len ? b.subarray(0, len) : b;
    };
  const sideFile = (url) => {   // config.json / tokenizer.json for the dense models: next to the local GGUF, else the URL
    const f = local && path.join(path.dirname(local), url.split("/").pop());
    return f && fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : fetch(url).then((r) => r.json());
  };
  return {
    M, local, url: M.gguf,
    readAt,
    bytesOf: (info) => readAt(info.byteOffset, info.byteLength),
    // the GGUF index (and the tokenizer when asked): 12 MB first, doubling, as room.js fetchGGUFHeader
    async header(needTokenizer = true) {
      for (let size = 12 * 2 ** 20; ; size *= 2) {
        const buf = await readAt(0, size);
        try { return parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), { skipTokenizer: !needTokenizer }); }
        catch (e) { if (size > 256 * 2 ** 20) throw e; }
      }
    },
    cfg: () => (M.cfg ? sideFile(M.cfg) : null),
    tokJson: () => (M.tok ? sideFile(M.tok) : null),
    async close() { try { await fh?.close(); } catch {} fh = null; },
  };
}
