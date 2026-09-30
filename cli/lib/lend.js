// pooled join / pooled host: the parts with no GPU and no network, so they can be unit tested:
// argument parsing, how much memory to lend (the memory rule), the status line, and what an error
// tells the person at the terminal. cli/lib/lendrun.js runs the room node with them.
import { parseArgs } from "node:util";
import { roomCodeFrom, roomKeyFrom } from "./room.js";
import { cleanText } from "./common.js";

export const DESK_MAX_GB = 64;          // room/pledge.js: the most one computer lends a room
export const DISCRETE_RESERVE_GB = 1.5; // kept free on a discrete GPU (the desktop, other apps)
export const UNIFIED_KEEP_GB = 8;       // kept for the system on unified memory: at least this much
export const UNIFIED_KEEP_FRAC = 0.35;  // ... or this share of it, whichever is more
export const LINUX_AVAIL_SLACK_GB = 2;  // unified memory on Linux: never more than what is free now, less this
export const UPDATE_HINT = "npx @pooled/cli@latest";

export const HELP_JOIN = `Usage
  pooled join <ROOM CODE | "room link"> [options]

  Lends this computer's GPU to a Pooled room: the room's host deals this device some of the
  model's layers, and they run here until you press Ctrl-C (which leaves the room and frees the GPU).
  Needs a GPU that Dawn can use (Metal on macOS 26+, Vulkan on Linux, D3D12 on Windows).

  Getting in: with the room's invite link (in quotes: "https://pooled.run/r/4TKG9P#k=..."), the
  host lets this computer in at once. With the code alone (4TK-G9P), a host that asks before new
  devices join sees "<name> wants to join" and this waits until it presses Allow.

Options
  --gb <n|max>      how much memory to lend, in GB. Default: the memory rule, printed at start:
                    a discrete GPU lends its free memory less ${DISCRETE_RESERVE_GB} GB; unified memory (Apple
                    silicon, GB10) lends the total less max(${UNIFIED_KEEP_GB} GB, ${Math.round(UNIFIED_KEEP_FRAC * 100)}%). "max" keeps only a
                    small margin. At most ${DESK_MAX_GB} GB per device.
  --name <s>        how the room shows this device (default: "node-" and 3 letters made from this
                    computer's hostname, the same each run, so a restart takes back its slot)
  --signal <spec>   PeerJS signaling server(s), as the room page's ?signal= (comma list;
                    default: the PeerJS cloud pooled.run uses)
  --models <dir>    read the weights from local files instead of downloading them
  --no-check        skip the test allocation that confirms the memory is there
  --wait <min>      after the host has been gone a minute, keep trying to rejoin for this long
                    (default 10); only a host of the same name: a room code can be reused by a
                    new room, and pooled join never joins a room you did not name

  --json-log        one JSON object per log line (no status line)
  --quiet           only errors and the status line
  -h, --help        this help

  Every device that holds layers computes on every prompt in the room: this computer sees the
  hidden states of what is asked there (they carry the prompts and answers), and whoever the host
  lets in can use what it lends. Lend to rooms you trust.
`;

export const HELP_HOST = `Usage
  pooled host [options]

  Opens a new Pooled room on this computer, which holds the embedding, the head and its share of the
  layers. Other devices join with the invite link it prints (a browser tab, a phone, pooled join) or
  with the code. In a terminal, press Enter to deal the layers over the devices in the room, and
  Enter again to re-deal after more join. Ask from the room page, from the terminal with
  pooled chat, or from your own tools with pooled serve.

  Who gets in: a device with the invite link (its #k= key) comes in at once. One with the code
  alone waits until you let it in: in a terminal, press a to allow the oldest request, d to deny
  it. Without a terminal it waits for good (give it the link), unless you pass --allow-all.

Options
  --model <key>     the model (default qwen3-1.7b; --model list prints them)
  --gb <n|max>      how much memory to lend, as pooled join (default: the memory rule)
  --devices <n>     deal the layers as soon as n devices (this one included) are in the room,
                    and again when the room went on without one and n are back; the default
                    without a terminal is 1 (start at once)
  --code <CODE>     the room code to use (default: a random one of six characters)
  --allow-all       let in anyone with the room code, without asking (the invite link is not
                    needed; also lets in room pages from before the gate)
  --ctx <n>         the context to ask for, in tokens (default: the model's room default)
  --name <s>        how the room shows this device
  --signal <spec>   PeerJS signaling server(s), as pooled join
  --models <dir>    read the weights from local files instead of downloading them
  --no-check        skip the test allocation that confirms the memory is there
  --json-log        one JSON object per log line (no status line)
  --quiet           only errors and the status line
  -h, --help        this help

  Every device that holds layers sees the hidden states of what is asked in the room (they carry
  the prompts and answers), and whoever is in can ask. Share the invite link only with people you
  trust, and let in only devices you know.
`;

export class UsageError extends Error {}

// --gb: a positive number of GB, or "max" -> { gb } | { max: true } | null (not given)
export function parseGb(v) {
  if (v == null) return null;
  if (String(v).trim().toLowerCase() === "max") return { max: true };
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) throw new UsageError(`--gb must be a number of GB (like 12) or "max", not "${v}"`);
  if (n < 1) throw new UsageError("--gb must be at least 1 (GB)");
  return { gb: Math.min(n, DESK_MAX_GB) };
}

const COMMON = {
  gb: { type: "string" }, name: { type: "string" }, signal: { type: "string" }, models: { type: "string" },
  "no-check": { type: "boolean" }, "json-log": { type: "boolean" }, quiet: { type: "boolean" }, help: { type: "boolean", short: "h" },
};

// argv after the command word -> options, or throws UsageError. models: MODELS (room/models.js) for
// checking --model; left out, any key passes (the runner checks it again)
export function parseLendArgs(cmd, argv, { models = null } = {}) {
  const options = cmd === "join"
    ? { ...COMMON, wait: { type: "string" } }
    : { ...COMMON, model: { type: "string" }, devices: { type: "string" }, code: { type: "string" }, ctx: { type: "string" }, "allow-all": { type: "boolean" } };
  let r;
  try { r = parseArgs({ args: argv, options, allowPositionals: true, strict: true }); }
  catch (e) { throw new UsageError(e.message.replace(/^.*?: /, "")); }
  const o = r.values, pos = r.positionals;
  if (o.help) return { cmd, help: true };
  const out = { cmd, gb: parseGb(o.gb), name: o.name ? cleanText(o.name, 40) : undefined, signal: o.signal || null,
    modelDir: o.models || null, check: !o["no-check"], jsonLog: !!o["json-log"], quiet: !!o.quiet };
  if (o.name != null && !out.name) throw new UsageError("--name must not be empty");
  if (cmd === "join") {
    if (pos.length > 1) throw new UsageError(`one room code, not ${pos.length}: ${pos.join(" ")}`);
    const code = roomCodeFrom(pos[0]);
    if (!code) throw new UsageError(`give a room code (six letters and digits like 4TK-G9P; older rooms have four) or a room link${pos[0] ? `, not "${pos[0]}"` : ""}`);
    out.code = code;
    out.key = roomKeyFrom(pos[0]);   // the invite link's #k=: in without the host's Allow
    const wait = o.wait == null ? 10 : Number(o.wait);
    if (!Number.isFinite(wait) || wait < 0) throw new UsageError("--wait must be a number of minutes");
    out.waitMs = wait * 60000;
  } else {
    if (pos.length) throw new UsageError(`pooled host takes no room code (it makes one; --code picks it): "${pos[0]}"`);
    out.model = o.model || "qwen3-1.7b";
    out.allowAll = !!o["allow-all"];
    if (out.model !== "list" && models && !hostable(models).includes(out.model))
      throw new UsageError(`unknown model "${out.model}"; one of: ${hostable(models).join(", ")}`);
    if (o.devices != null) {
      const n = Number(o.devices);
      if (!Number.isInteger(n) || n < 1 || n > 64) throw new UsageError("--devices must be a whole number from 1 to 64");
      out.devices = n;
    }
    if (o.code != null) {
      const c = String(o.code).toUpperCase().replace(/[\s-]/g, "");
      // room/joingate.js's alphabet: no I, L, O, U, 0, 1 (they read alike); six characters, or four as older rooms
      if (!/^[ABCDEFGHJKMNPQRSTVWXYZ2-9]{6}$|^[ABCDEFGHJKMNPQRSTVWXYZ2-9]{4}$/.test(c)) throw new UsageError("--code must be 6 (or 4) letters and digits, without I, L, O, U, 0 or 1");
      out.roomCode = c;
    }
    if (o.ctx != null) {
      const n = Number(o.ctx);
      if (!Number.isInteger(n) || n < 256) throw new UsageError("--ctx must be a whole number of tokens (256 or more)");
      out.ctx = n;
    }
  }
  return out;
}
// the models a node can host (the loaders in packages/room-node/source.js)
export const hostable = (models) => Object.keys(models).filter((k) => models[k].kind === "gguf" || models[k].kind === "qwen35");

// ---------------- the memory rule ----------------
// What the GPU has, from the OS (WebGPU does not say how much memory a GPU has):
//   { kind: "discrete", name, totalGB, freeGB, source } | { kind: "unified", name, totalGB, availGB?, source }
//   | { kind: "unknown", why }
// run(cmd, args) -> stdout or throws; read(path) -> text or throws. Injected so tests need no GPU.
export function detectMemory({ platform = process.platform, arch = process.arch, run, read, totalmem, freemem } = {}) {
  const GB = 2 ** 30;
  if (platform === "darwin") {
    if (arch === "arm64") return { kind: "unified", name: "Apple silicon", totalGB: totalmem() / GB, source: "unified memory (macOS)" };
    return { kind: "unknown", why: "an Intel Mac's GPU memory can't be read from here" };
  }
  // NVIDIA (Linux, Windows): nvidia-smi; a GB10 / Jetson reports [N/A]: its GPU shares the system's memory
  try {
    const out = run("nvidia-smi", ["--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]);
    const rows = String(out).trim().split(/\r?\n/).map((l) => l.split(",").map((x) => x.trim())).filter((r) => r.length >= 3 && r[0]);
    if (rows.length) {
      const known = rows.filter((r) => Number.isFinite(+r[1]) && Number.isFinite(+r[2]) && +r[1] > 0);
      if (known.length) {
        // several GPUs: Dawn takes the high-performance adapter; lend by the one with the most free memory
        const best = known.reduce((a, b) => (+b[2] > +a[2] ? b : a));
        return { kind: "discrete", name: best[0], totalGB: +best[1] / 1024, freeGB: +best[2] / 1024, source: "nvidia-smi", gpus: rows.length };
      }
      if (platform === "linux") return { kind: "unified", name: rows[0][0], totalGB: totalmem() / GB, availGB: freemem() / GB, source: "unified memory (nvidia-smi reports none of its own)" };
    }
  } catch {}
  if (platform === "linux") {
    // AMD (amdgpu): sysfs
    for (let i = 0; i < 8; i++) {
      try {
        const dir = `/sys/class/drm/card${i}/device/`;
        const total = +read(dir + "mem_info_vram_total"), used = +read(dir + "mem_info_vram_used");
        if (total > 2 * GB && Number.isFinite(used)) return { kind: "discrete", name: `AMD GPU (card${i})`, totalGB: total / GB, freeGB: (total - used) / GB, source: "amdgpu sysfs" };
      } catch {}
    }
  }
  return { kind: "unknown", why: platform === "win32" ? "only NVIDIA GPUs can be read on Windows so far" : "no nvidia-smi or amdgpu memory counters here" };
}

const floorGb = (x) => Math.floor(x * 2) / 2;   // half GB steps

// The rule: how much to lend, and one line on why. mem: detectMemory(); want: parseGb(--gb) or null;
// maxBufGB: the adapter's largest buffer (the fallback when the OS says nothing: half of it, as
// room/pledge.js does for a device that pledges nothing). -> { gb, why, low, over }
//   low: less than 1 GB is free (not enough to lend); over: --gb asks for more than looks free
export function memoryRule(mem, want = null, { maxBufGB = 0 } = {}) {
  let avail = null, why;
  if (mem.kind === "discrete") {
    const keep = want?.max ? 0.5 : DISCRETE_RESERVE_GB;
    avail = Math.max(0, mem.freeGB - keep);
    why = `${fmtGb(mem.freeGB)} GB free on ${mem.name}, less ${keep} GB kept free`;
  } else if (mem.kind === "unified") {
    const keep = want?.max ? Math.min(4, UNIFIED_KEEP_GB) : Math.max(UNIFIED_KEEP_GB, mem.totalGB * UNIFIED_KEEP_FRAC);
    avail = Math.max(0, mem.totalGB - keep);
    why = `${fmtGb(mem.totalGB)} GB of unified memory, less ${fmtGb(keep)} GB kept for the system`;
    if (mem.availGB != null && mem.availGB - LINUX_AVAIL_SLACK_GB < avail) {
      avail = Math.max(0, mem.availGB - LINUX_AVAIL_SLACK_GB);
      why += `; ${fmtGb(mem.availGB)} GB is free right now, so ${fmtGb(avail)} GB`;
    }
  }
  if (want?.gb) {
    const over = avail != null && want.gb > avail + 0.01;
    return { gb: want.gb, why: `--gb ${want.gb}` + (over ? ` (more than the ${fmtGb(avail)} GB that looks free: the load may fail)` : ""), over, low: false };
  }
  if (avail == null) {
    const gb = Math.min(DESK_MAX_GB, Math.max(1, Math.round((maxBufGB || 2) * 0.5)));
    return { gb, why: `couldn't read this GPU's memory (${mem.why}): lending ${gb} GB; --gb sets it`, low: false, over: false };
  }
  if (avail < 1) return { gb: 0, why, low: true, over: false };
  let gb = floorGb(avail);
  if (gb > DESK_MAX_GB) { gb = DESK_MAX_GB; why += `, capped at ${DESK_MAX_GB} GB per device`; }
  return { gb, why, low: false, over: false };
}

// after the test allocation: the GB that were really there (null = not checked) -> the pledge
export function afterCheck(rule, gotGB) {
  if (gotGB == null || gotGB >= rule.gb - 0.01) return rule;
  const gb = floorGb(Math.max(0, gotGB - 1));   // what the GPU gave, less 1 GB for the engine's own buffers
  if (gb < 1) return { ...rule, gb: 0, low: true, why: `${rule.why}; the GPU gave only ${fmtGb(gotGB)} GB in a test allocation` };
  return { ...rule, gb, why: `${rule.why}; the GPU gave only ${fmtGb(gotGB)} GB in a test allocation, so ${gb} GB` };
}

// a room code as people read it: six in two groups of three ("4TK-G9P"); four as they were
export const fmtCode = (c) => (String(c || "").length === 6 ? `${c.slice(0, 3)}-${c.slice(3)}` : String(c || ""));
export const fmtGb = (x) => (Math.round(x * 10) / 10).toString();

// ---------------- the status line ----------------
// s: { code, phase, devices, range: [lo, hi) | null, model, tps, passes, pct, tries, host, signaling, hosting }
// phases: connecting | waiting | ready | loading | online | answering | degraded | hostgone | rejoining | leaving
export function formatStatus(s, width = 0) {
  const phase = {
    connecting: "connecting", lobby: "waiting for the host to let you in", waiting: s.hosting ? "waiting for devices" : "waiting for the host to deal layers",
    ready: "layers loaded: waiting for the rest of the room",
    guest: "in the room without layers (the host re-deals to include this device)",
    loading: `loading layers${s.pct != null ? ` ${s.pct}%` : ""}`, online: "online", answering: "answering",
    degraded: "a device left: waiting for it", hostgone: "lost the host: knocking", rejoining: `rejoining${s.tries ? ` (try ${s.tries})` : ""}`,
    leaving: "leaving",
  }[s.phase] || s.phase;
  const parts = [`room ${fmtCode(s.code)}`, phase];
  if (s.devices != null) parts.push(`${s.devices} device${s.devices === 1 ? "" : "s"}`);
  if (s.range) parts.push(`layers ${s.range[0]}-${s.range[1] - 1}${s.embed ? " + embed/head" : ""}${s.model ? ` of ${s.model}` : ""}`);
  else if (s.model && s.hosting) parts.push(s.model);
  if (s.tps != null) parts.push(`${s.tps.toFixed(1)} tok/s`);
  if (s.lobby > 0) parts.push(`${s.lobby} waiting to join`);
  parts.push(`${s.passes || 0} pass${s.passes === 1 ? "" : "es"}`);
  if (s.signaling === false) parts.push("signaling down (links still up)");
  const line = parts.join(" · ");
  return width > 0 && line.length > width ? line.slice(0, Math.max(1, width - 1)) + "…" : line;
}

// pooled host --devices N: deal again by itself? When the room runs on fewer than N devices (one
// stayed away past the rejoin grace and the room re-dealt without it) and N are in the room again,
// counting only devices the last deal did not see (a phone that deal left out, or a device it
// dropped after a load death, must not make it re-deal over and over). room: { online, busy,
// starting, chain: [id], peers: [id] (GPU devices other than the host), dealt: Set of ids }
export function autoRedeal(devices, { online, busy, starting, chain, peers, dealt }) {
  if (!(devices > 1) || !online || busy || starting) return false;
  if (chain.length + 1 >= devices || peers.length + 1 < devices) return false;
  return peers.some((id) => !chain.includes(id) && !dealt?.has(id));
}

// passes run here since the start: the node's own counters restart with a re-deal or a new node
export function passCounter() {
  let base = 0, last = 0, of = null;
  return (node, n) => {
    if (node !== of || n < last) { base += last; last = 0; of = node; }
    last = n;
    return base + n;
  };
}

// tok/s from a host's answer stats ("48 tok · 21.3 tok/s · 2 devices · ...") -> number | null
export function tpsFromStats(stats) {
  const m = /(\d+(?:\.\d+)?) tok\/s/.exec(String(stats || ""));
  return m ? +m[1] : null;
}

// ---------------- errors ----------------
// pooled join's default name: "node-" and 3 letters from the hostname (hashed: the hostname itself,
// often a person's name, is not shown to the room), the same on every run, so a join started again
// after a crash is re-seated in its old slot (the host knows a device by its name)
export function deviceName(hostname = "") {
  let h = 2166136261;   // FNV-1a
  for (const ch of String(hostname)) { h ^= ch.codePointAt(0); h = Math.imul(h, 16777619) >>> 0; }
  const A = "abcdefghjkmnpqrstvwxyz";   // no i, l, o, u (they read alike)
  let s = "";
  for (let i = 0; i < 3; i++) { s += A[h % A.length]; h = Math.floor(h / A.length); }
  return "node-" + s;
}

// the host's protocol vs this one -> what to do about it
export function versionAdvice({ mine, theirs, theyHost = true, code = "" }) {
  const who = theyHost ? "This room's host" : "A device in this room";
  if (!Number.isInteger(theirs)) return `${who} is on a different version of Pooled (protocol ${theirs ?? "?"}, this pooled ${mine}). Update this one (${UPDATE_HINT}${code ? ` join ${code}` : ""}) and ask the host to reload the room page.`;
  if (theirs > mine) return `${who} is on a newer version of Pooled (protocol ${theirs}, this pooled ${mine}). Update: ${UPDATE_HINT}${code ? ` join ${code}` : ""}`;
  return `${who} is on an older version of Pooled (protocol ${theirs}, this pooled ${mine}). Ask ${theyHost ? "the host" : "them"} to reload the room page, then join again.`;
}

// the host's bye (it refused this device): a version mismatch in its wording -> { theirs } | null
export function versionFromBye(reason, mine) {
  const r = String(reason || "");
  let m = /protocol (\d+), this (?:tab|device|pooled) (\d+)/.exec(r);        // room/errors.js versionLine
  if (m) { const [a, b] = [+m[1], +m[2]]; return { theirs: a === mine ? b : a }; }
  m = /speaks room protocol (\d+), this device (\d+)/.exec(r);            // a room node host
  if (m) return { theirs: +m[1] };
  return null;
}

// any error from joining / hosting -> { message, hint?, code }: code is the exit status
export function explainError(err, { code = "", cmd = "join", mine = 4 } = {}) {
  const msg = String(err?.message || err || "unknown error");
  const t = err?.type || err?.code || "";
  if (t === "dawn-missing" || t === "dawn-broken" || t === "room-node-missing") return { message: msg, hint: err.hint, code: 1 };
  if (t === "no-adapter" || /no WebGPU adapter/i.test(msg))
    return { message: "No WebGPU adapter: Dawn found no GPU it can use on this computer.",
      hint: "It needs Metal (macOS 26 or newer), Vulkan (Linux: a Vulkan driver, e.g. the NVIDIA or Mesa one) or D3D12 (Windows). pooled serve works without a GPU.", code: 1 };
  if (t === "low-memory")
    return { message: `Not enough GPU memory to lend: ${msg}.`, hint: "Close other GPU apps, or lend a set amount with --gb N (at least 1).", code: 1 };
  if (t === "room-not-found")
    return { message: `No room ${code}.`, hint: "Check the code, and that the host's page (or pooled host) is still open.", code: 1 };
  if (t === "signaling-down")
    return { message: `Can't reach the signaling server${err.tried?.length > 1 ? "s" : ""} (${(err.tried || []).join(", ") || "PeerJS cloud"}).`,
      hint: "Devices use it only to find each other; it may be down or blocked on this network. Try again in a minute, or run your own (docs/self-host-signaling.md) and pass --signal.", code: 1 };
  if (t === "unavailable-id")
    return { message: `Room code ${code} is taken.`, hint: "Pick another with --code, or leave it out for a random one.", code: 1 };
  if (t === "version") return { message: versionAdvice({ mine, theirs: err.theirs, theyHost: err.theyHost !== false, code }), code: 1 };
  if (t === "kicked") return { message: `The host refused this device: ${cleanText(msg, 300)}`, code: 1 };
  if (/out of memory|OOM|allocation failed|Failed to allocate|createBuffer/i.test(msg))
    return { message: `The GPU ran out of memory: ${cleanText(msg, 200)}`, hint: "Lend less with --gb N, or close other GPU apps.", code: 1 };
  if (t === "other-host")
    return { message: `Room ${code} now has another host (${cleanText(err.now, 40)}, not ${cleanText(err.was, 40)}): a new room under the same code, so pooled left it.`,
      hint: `If you trust it, join it on purpose: pooled join ${code}`, code: 1 };
  if (t === "room-over") return { message: `Room ${code} is over: the host left and did not come back.`, code: 1 };
  return { message: cleanText(msg, 400), code: 1 };
}
