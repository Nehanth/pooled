// What the plugin keeps between gateway runs, next to OpenClaw's own state (not in openclaw.json):
//   <state dir>/pooled/room.json   (mode 0600: it holds secrets)
//     host:  { code, gate }         the room this machine hosts: its code and its saved gate (the
//                                   invite key, and the hashes of the passes it gave out), so a
//                                   restart keeps the invite link working and lets devices back in
//     join:  { <CODE>: { key, pass, host, model, ctx, at } }
//                                   a room this machine joins: the invite key from a pasted link, the
//                                   pass the host gave it (in again without a new Allow), and what it
//                                   learned of the host (its name, model and context)
//   <state dir>/pooled/status.json  the room as the gateway sees it (for `cat`, and /pooled)
//   <state dir>/pooled/prewarm.json the last request's system prompt and tools (prewarm.js)
// The state dir: OpenClaw's (plugin-sdk/state-paths resolveStateDir), else $OPENCLAW_STATE_DIR,
// else ~/.openclaw.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

let base = null;
export function setStateDir(dir) { base = dir || null; }
export const stateDir = (env = process.env) => base || env.OPENCLAW_STATE_DIR || path.join(os.homedir(), ".openclaw");
export const pooledDir = () => path.join(stateDir(), "pooled");
export const roomFile = () => path.join(pooledDir(), "room.json");

// OpenClaw's own answer for the state dir (profiles, --dev): best effort, the env fallback otherwise
export async function initStateDir() {
  if (base) return base;
  try {
    const m = await import("openclaw/plugin-sdk/state-paths");
    const d = m.resolveStateDir?.();
    if (typeof d === "string" && d) base = d;
  } catch {}
  return stateDir();
}

export function readJson(file) {
  try { const v = JSON.parse(fs.readFileSync(file, "utf8")); return v && typeof v === "object" ? v : null; } catch { return null; }
}
// written whole (temp file + rename) with mode 0600
export function writeJson(file, v, mode = 0o600) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(v, null, 1) + "\n", { mode });
  try { fs.chmodSync(tmp, mode); } catch {}
  fs.renameSync(tmp, file);
}

export const loadRoomState = () => { const v = readJson(roomFile()) || {}; return { host: v.host || null, join: v.join && typeof v.join === "object" ? v.join : {} }; };
export function saveRoomState(st) { writeJson(roomFile(), { host: st.host || null, join: st.join || {} }); }

// the saved gate of the room this machine hosts under `code` (another code: none)
export function savedGate(code) { const h = loadRoomState().host; return h && h.code === code && h.gate ? h.gate : null; }
// -> true when it changed what was on disk
export function saveHostGate(code, gate) {
  const st = loadRoomState();
  const next = { code, gate };
  if (JSON.stringify(st.host) === JSON.stringify(next)) return false;
  st.host = next;
  saveRoomState(st);
  return true;
}
export const joinState = (code) => loadRoomState().join[code] || {};
// merge fields into what is kept for a joined room (null / undefined fields are left as they were)
export function saveJoinState(code, fields) {
  const st = loadRoomState();
  const cur = st.join[code] || {};
  const next = { ...cur };
  for (const [k, v] of Object.entries(fields)) if (v != null && v !== "") next[k] = v;
  if (JSON.stringify(next) === JSON.stringify(cur)) return false;
  next.at = new Date().toISOString();
  st.join[code] = next;
  // keep the few most recent rooms
  const keys = Object.keys(st.join).sort((a, b) => String(st.join[b].at || "").localeCompare(String(st.join[a].at || "")));
  for (const k of keys.slice(8)) delete st.join[k];
  saveRoomState(st);
  return true;
}
