// Downloading a model into the shared cache (~/.pooled/models, as `pooled pull`): resumable, checked
// against its size and SHA-256 (cli/lib/cache.js pullModel), one writer at a time (pulllock.js).
// A download's state is a plain object the gateway shows in /pooled and in the chat:
//   { key, state: "waiting" | "running" | "checking" | "done" | "error", done, total, bps, error }
import { pullModel, rateMeter, fmtBytes, fmtEta } from "../../../cli/lib/cache.js";
import { MODELS, FILES, diskState, modelInfo } from "./models.js";
import { lock } from "./pulllock.js";

export function pullState(key) { return { key, state: "waiting", done: 0, total: FILES[key]?.bytes || 0, bps: null, error: null }; }

// -> st (state "done" or "error"); onChange(st) as it goes (at most a few times a second)
// (models: { MODELS, FILES } other than room/models.js, for tests)
export async function download(dir, key, { signal, onChange = () => {}, fetch, st = pullState(key), models = { MODELS, FILES } } = {}) {
  const rate = rateMeter();
  let last = 0;
  const change = (force) => { const t = Date.now(); if (force || t - last > 250) { last = t; onChange(st); } };
  let release = null;
  try {
    release = await lock(dir, key, { signal, onWait: () => { st.state = "waiting"; change(true); } });
    st.done = models.MODELS === MODELS ? diskState(dir, key).partBytes || 0 : 0;
    st.state = "running"; change(true);
    await pullModel(key, { dir, MODELS: models.MODELS, FILES: models.FILES, signal, ...(fetch ? { fetch } : {}),
      onProgress: (p) => { st.done = p.done; st.total = p.total || st.total; st.bps = rate(p.done) ?? st.bps; change(); },
      onVerify: (d, t) => { st.state = "checking"; st.done = d; st.total = t; change(); } });
    st.state = "done"; st.done = st.total; change(true);
  } catch (e) {
    st.state = "error"; st.error = e.type === "aborted" ? "stopped" : e.message; change(true);
  } finally { release?.(); }
  return st;
}

// "42% (0.8 GB of 1.8 GB), about 18s left" / "checking the download"
export function pullLine(st) {
  if (!st) return "";
  if (st.state === "waiting") return "waiting for another download of it to finish";
  if (st.state === "checking") return `checking the download (SHA-256 ${st.total ? Math.floor((st.done / st.total) * 100) : 0}%)`;
  if (st.state === "done") return `downloaded (${fmtBytes(st.total)})`;
  if (st.state === "error") return `download failed: ${st.error}`;
  const pct = st.total ? Math.floor((st.done / st.total) * 100) : null;
  return `${pct != null ? `${pct}% ` : ""}(${fmtBytes(st.done)}${st.total ? ` of ${fmtBytes(st.total)}` : ""})${st.bps && st.total ? `, about ${fmtEta((st.total - st.done) / st.bps)} left` : ""}`;
}
export const downloadingMessage = (st, code) =>
  `Pooled is downloading ${modelInfo(st.key).name} for room ${code}: ${pullLine(st)}. Ask again when it's done; devices can join the room meanwhile`;
