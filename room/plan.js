// Layer placement and the model ladder. DOM-free so it can be unit tested.

// Deal L layers over devices in proportion to what each can hold. caps: bytes each device offers
// for layers (host first; the host's cap already has the embedding, head and draft block taken
// out). Every device gets at least one layer; rounding leftovers go to the largest remainders.
// Returns { assigned: [count per device], ranges: [[lo, hi) per device] }.
export function planSplit(L, caps) {
  const totalCap = caps.reduce((s, c) => s + c, 0);
  const assigned = caps.map((c) => Math.floor(L * c / totalCap));
  const fracs = caps.map((c, i) => ({ i, f: L * c / totalCap - assigned[i] })).sort((a, b) => b.f - a.f);
  const rem = L - assigned.reduce((a, b) => a + b, 0);
  for (let k = 0; k < rem; k++) assigned[fracs[k % fracs.length].i]++;
  for (let i = 1; i < assigned.length; i++)
    if (assigned[i] === 0) { const j = assigned.indexOf(Math.max(...assigned)); assigned[j]--; assigned[i]++; }
  const ranges = [];
  let acc = 0;
  for (const a of assigned) { ranges.push([acc, acc + a]); acc += a; }
  return { assigned, ranges };
}

// What this room can run, smallest model first: [{ key, need, ok, short }]. short is how many
// more GB the room needs for that model (0 when it fits).
export function ladder(needGB, pledgedGB) {
  return Object.entries(needGB)
    .sort((a, b) => a[1] - b[1])
    .map(([key, need]) => ({ key, need, ok: pledgedGB >= need, short: Math.max(0, +(need - pledgedGB).toFixed(1)) }));
}

// The largest model that fits the room's pledges, or the smallest one when nothing fits yet.
export function bestFit(needGB, pledgedGB) {
  const l = ladder(needGB, pledgedGB);
  const fits = l.filter((x) => x.ok);
  return (fits.length ? fits[fits.length - 1] : l[0]).key;
}

// Room code from a join link: /r/ABCD, /room/ABCD, ?code=ABCD or #ABCD. Codes use the room's
// 30-letter alphabet (no I, L, O, U, 0, 1); anything else is ignored. Returns "" when absent.
export function codeFromLocation(pathname = "", search = "", hash = "") {
  const ok = (c) => /^[A-HJKMNP-TV-Z2-9]{4,6}$/.test(c) ? c : "";
  const m = /\/(?:r|room)\/([A-Za-z0-9]{4,6})\/?$/.exec(pathname);
  if (m) return ok(m[1].toUpperCase());
  const q = new URLSearchParams(search).get("code");
  if (q) return ok(q.trim().toUpperCase());
  const h = hash.replace(/^#/, "");
  if (h && h !== "debug") return ok(h.toUpperCase());
  return "";
}

// Deal L layers for speed instead of by memory. A token's lap costs every device's compute for
// its layers plus one network hop per device in the chain, so: fill the fastest devices first,
// each up to what it can hold, and leave out devices that are not needed. caps: layers each
// device can hold (host first); msPerLayer: measured compute per layer (null or missing = not
// measured yet: then fewest hops wins, biggest devices first). The host always keeps at least
// one layer (it holds the embedding and the head anyway). Returns planSplit's shape plus
// `used`: the device indices that hold layers, in the original order.
export function planForSpeed(L, caps, msPerLayer = []) {
  const n = caps.length;
  const known = msPerLayer.filter((x) => x > 0);
  const fallback = known.length ? Math.max(...known) * 1.5 : 1;   // unmeasured: assume slower than any measured device
  const cost = caps.map((_, i) => msPerLayer[i] > 0 ? msPerLayer[i] : fallback);
  const order = [...caps.keys()].sort((a, b) => cost[a] - cost[b] || (a === 0) * -1 + (b === 0) || caps[b] - caps[a] || a - b);
  const assigned = new Array(n).fill(0);
  assigned[0] = 1;
  let left = L - 1;
  for (const i of order) {
    if (left <= 0) break;
    const take = Math.min(left, Math.max(0, Math.floor(caps[i]) - assigned[i]));
    assigned[i] += take; left -= take;
  }
  // nobody can hold the rest: spread it over everyone in proportion to capacity, as planSplit does
  if (left > 0) {
    const tot = caps.reduce((s, c) => s + Math.max(c, 1), 0);
    const extra = caps.map((c) => Math.floor(left * Math.max(c, 1) / tot));
    let r = left - extra.reduce((a, b) => a + b, 0);
    for (let k = 0; r > 0; k = (k + 1) % n, r--) extra[order[k]]++;
    extra.forEach((x, i) => { assigned[i] += x; });
  }
  const ranges = [];
  let acc = 0;
  for (const a of assigned) { ranges.push([acc, acc + a]); acc += a; }
  return { assigned, ranges, used: assigned.map((a, i) => (a > 0 ? i : -1)).filter((i) => i >= 0) };
}

// The device that becomes the model host when someone presses Start. The model host holds the
// embedding, the head and the draft block, samples every token and runs the Code agent, so it is
// the strongest device, whoever pressed Start: a device with WebGPU first, then a computer before
// a phone (a phone's lent memory says little about its GPU, and a phone tab sleeps with its
// screen), then the most memory lent, then the lowest id so every screen picks the same one.
// devices: [{ id, meta: { webgpu, contribGB, phone, ua } }]. Returns an id (null for none).
export function pickModelHost(devices) {
  const isPhone = (m) => !!(m?.phone || /^(iPhone|Android)$/.test(m?.ua || ""));
  const gb = (m) => (m?.webgpu ? +m?.contribGB || 0 : 0);
  let best = null;
  for (const d of devices) {
    if (!d?.id) continue;
    const k = [d.meta?.webgpu ? 1 : 0, isPhone(d.meta) ? 0 : 1, gb(d.meta)];
    if (!best) { best = { id: d.id, k }; continue; }
    const c = k[0] - best.k[0] || k[1] - best.k[1] || k[2] - best.k[2] || (d.id < best.id ? 1 : -1);
    if (c > 0) best = { id: d.id, k };
  }
  return best?.id ?? null;
}

// ---- spare copies (experimental, off by default; docs/protocol.md "Spare copies") ----
// A spare is an otherwise idle device that holds a copy of one worker's layers, loaded and ready.
// When that worker drops, the spare takes its place in the chain at once instead of the whole room
// re-dealing. The model host is never copied (it holds the conversation, the embedding and head).

// How risky a device is to hold layers: a phone 4 (tabs freeze with the screen, batteries run
// out), a laptop on battery 2, anything else 1.
export function deviceRisk(m = {}) {
  if (m.phone || /^(iPhone|Android)$/.test(m.ua || "")) return 4;
  if (m.battery === "discharging" || m.onBattery) return 2;
  return 1;
}
// Can device `m` be a spare at all: a computer with WebGPU that speaks the spare protocol (rep).
export const spareCapable = (m = {}) => !!m.webgpu && !!m.rep && deviceRisk(m) < 4;

// Pick a spare for each segment. segments: [{ idx, layers, meta, ms }] (ms = measured ms per layer,
// or null), candidates: [{ idx, cap, meta, ms }] (cap in layers, already net of the KV cache at the
// room's context). Riskier and bigger segments are covered first: risk x (1 + share of the model).
// A candidate covers a segment when it holds all of it with `headroom` to spare, is not more than
// `slowdown` times slower per layer when both are measured, and among those it prefers the same
// GPU (vendor and architecture, so the kernels match) and then the smallest cap, which keeps the
// big spares for the big segments. One spare per segment, one segment per spare.
// Returns Map(segment idx -> candidate idx).
export function coverSegments(segments, candidates, { headroom = 1.1, slowdown = 1.5 } = {}) {
  const total = segments.reduce((s, x) => s + x.layers, 0) || 1;
  const order = [...segments].sort((a, b) =>
    deviceRisk(b.meta) * (1 + b.layers / total) - deviceRisk(a.meta) * (1 + a.layers / total) || b.layers - a.layers || a.idx - b.idx);
  const free = candidates.filter((c) => spareCapable(c.meta));
  const out = new Map();
  for (const seg of order) {
    const fits = free.filter((c) => c.cap >= seg.layers * headroom && !(c.ms > 0 && seg.ms > 0 && c.ms > seg.ms * slowdown));
    if (!fits.length) continue;
    const same = (c) => (c.meta?.gpu && c.meta.gpu === seg.meta?.gpu ? 0 : 1);
    fits.sort((a, b) => same(a) - same(b) || a.cap - b.cap || a.idx - b.idx);
    const pick = fits[0];
    out.set(seg.idx, pick.idx);
    free.splice(free.indexOf(pick), 1);
  }
  return out;
}

// Deal the layers so that spares are possible: pack the model onto the fewest, most reliable
// devices (a computer before a laptop on battery before a phone, then the fastest measured, then
// the most memory; the host always holds at least one layer), and keep the rest free to be spares.
// caps: layers each device can hold (host first), meta: each device's meta, msPerLayer: measured
// (optional). Returns planForSpeed's shape plus spareOf: Map(primary device idx -> spare device idx)
// and idle: devices that hold nothing and cover nothing. When not one segment can be covered, it
// returns null: the caller deals as it would with spares off (nothing changes for that room).
export function planReplicas(L, caps, meta = [], msPerLayer = [], opts = {}) {
  const n = caps.length;
  const ms = (i) => (msPerLayer[i] > 0 ? msPerLayer[i] : Infinity);
  const order = [...caps.keys()].filter((i) => i > 0)
    .sort((a, b) => deviceRisk(meta[a]) - deviceRisk(meta[b]) || ms(a) - ms(b) || caps[b] - caps[a] || a - b);
  const assigned = new Array(n).fill(0);
  assigned[0] = Math.max(1, Math.min(L, Math.floor(caps[0])));
  let left = L - assigned[0];
  for (const i of order) {
    if (left <= 0) break;
    const take = Math.min(left, Math.max(0, Math.floor(caps[i])));
    assigned[i] = take; left -= take;
  }
  if (left > 0) return null;   // the model needs every device: no room for a spare
  const used = assigned.map((a, i) => (a > 0 ? i : -1)).filter((i) => i >= 0);
  const segments = used.filter((i) => i > 0).map((i) => ({ idx: i, layers: assigned[i], meta: meta[i], ms: msPerLayer[i] }));
  const candidates = [...caps.keys()].filter((i) => i > 0 && !assigned[i]).map((i) => ({ idx: i, cap: caps[i], meta: meta[i], ms: msPerLayer[i] }));
  const spareOf = coverSegments(segments, candidates, opts);
  if (!spareOf.size) return null;
  const ranges = [];
  let acc = 0;
  for (const a of assigned) { ranges.push([acc, acc + a]); acc += a; }
  const spares = new Set(spareOf.values());
  return { assigned, ranges, used, spareOf, idle: candidates.map((c) => c.idx).filter((i) => !spares.has(i)) };
}
