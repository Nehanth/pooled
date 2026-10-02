// Layer placement and the model ladder. DOM-free so it can be unit tested.

// Deal L layers over devices in proportion to what each can hold. caps: bytes each device offers
// for layers (host first; the host's cap already has the embedding, head and draft block taken
// out). Every device gets at least one layer; rounding leftovers go to the largest remainders.
// max (optional): the most layers each device may hold (its pledge in whole layers, room/plan.js
// layerCaps). With it no device ever gets more than its max: what a full device cannot take goes to
// the others with room, a device with max 0 holds nothing, and when the maxes cannot hold all L
// layers the deal is not made (`short` says how many layers are missing; the room must not start).
// Returns { assigned: [count per device], ranges: [[lo, hi) per device], short }.
export function planSplit(L, caps, max = null) {
  // a cap that is not a positive number (a malformed pledge) counts as nothing; when nobody
  // offers anything the layers are dealt evenly instead of dividing by zero
  caps = caps.map((c) => (Number.isFinite(c) && c > 0 ? c : 0));
  if (max) return splitWithin(L, caps, max.map((m) => (Number.isFinite(m) && m > 0 ? Math.floor(m) : 0)));
  const top = Math.max(0, ...caps);
  if (top > 0) caps = caps.map((c) => c / top);   // only the ratios matter; keeps L * c finite
  let totalCap = caps.reduce((s, c) => s + c, 0);
  if (!(totalCap > 0)) { caps = caps.map(() => 1); totalCap = caps.length; }
  const assigned = caps.map((c) => Math.floor(L * c / totalCap));
  const fracs = caps.map((c, i) => ({ i, f: L * c / totalCap - assigned[i] })).sort((a, b) => b.f - a.f);
  const rem = L - assigned.reduce((a, b) => a + b, 0);
  for (let k = 0; k < rem; k++) assigned[fracs[k % fracs.length].i]++;
  // the host first (it runs the embedding and the head, so it must hold a layer too), and only
  // take a layer from a device that keeps one: with fewer layers than devices the last ones wait
  for (let i = 0; i < assigned.length; i++)
    if (assigned[i] === 0) { const j = assigned.indexOf(Math.max(...assigned)); if (assigned[j] > 1 || (i === 0 && L > 0)) { assigned[j]--; assigned[i]++; } }
  return { assigned, ranges: rangesOf(assigned), short: 0 };
}
const rangesOf = (assigned) => { const r = []; let acc = 0; for (const a of assigned) { r.push([acc, acc + a]); acc += a; } return r; };

// planSplit with a most-layers-per-device: in proportion to caps, each device held to its max, the
// excess dealt again over the devices with room (in proportion too), rounding leftovers to the
// largest remainders among devices with room. The host keeps at least one layer; a device whose
// max is 0 holds none; the others get one each while there are layers enough.
function splitWithin(L, caps, max) {
  const n = caps.length, assigned = new Array(n).fill(0);
  if (!n || L <= 0) return { assigned, ranges: rangesOf(assigned), short: Math.max(0, L) };
  const room = max.reduce((s, m) => s + m, 0);
  if (max[0] < 1 || room < L) return { assigned, ranges: rangesOf(assigned), short: Math.max(L - room, max[0] < 1 ? 1 : 0) };
  // weights: the byte caps' ratios, or even when none is given; a device with room but no weight
  // (a malformed cap next to a whole max) still counts a little so it can take an overflow
  const top = Math.max(0, ...caps);
  const w = caps.map((c, i) => (max[i] > 0 ? (top > 0 ? Math.max(c / top, 1e-9) : 1) : 0));
  let left = L;
  // water-fill by weight: every round deals what is left over the devices still below their max
  for (let guard = 0; left > 0 && guard < n + 2; guard++) {
    const open = [...Array(n).keys()].filter((i) => assigned[i] < max[i]);
    const tw = open.reduce((s, i) => s + w[i], 0);
    const want = open.map((i) => ({ i, x: left * w[i] / tw }));
    let dealt = 0;
    for (const { i, x } of want) { const t = Math.min(Math.floor(x), max[i] - assigned[i]); assigned[i] += t; dealt += t; }
    left -= dealt;
    if (left > 0) {
      // rounding leftovers: largest remainders among devices that still have room
      const fr = want.filter(({ i }) => assigned[i] < max[i]).map(({ i, x }) => ({ i, f: x - Math.floor(x) })).sort((a, b) => b.f - a.f || a.i - b.i);
      for (const { i } of fr) { if (left <= 0) break; assigned[i]++; left--; }
    }
  }
  // then, as planSplit does, every device with room holds a layer (the host first), taken from the
  // device holding the most while it keeps one: with fewer layers than devices the last ones wait
  for (let i = 0; i < n; i++)
    if (assigned[i] === 0 && max[i] > 0) { const j = assigned.indexOf(Math.max(...assigned)); if (assigned[j] > 1 || (i === 0 && j !== 0)) { assigned[j]--; assigned[i]++; } }
  return { assigned, ranges: rangesOf(assigned), short: 0 };
}

// How many whole layers each device can hold within its pledge. pledges: bytes each device lends
// (host first); layerBytes: one layer with its share of the KV cache at the room's context;
// hostBytes: what the model host holds besides its layers (the embedding, the output head, the
// draft block: room/models.js hostHeldBytes). A pledge is a promise: nobody is dealt more than this.
export function layerCaps(pledges, layerBytes, hostBytes = 0) {
  return pledges.map((p, i) => {
    const b = (Number.isFinite(p) && p > 0 ? p : 0) - (i === 0 ? hostBytes : 0);
    return layerBytes > 0 && b > 0 ? Math.floor(b / layerBytes + 1e-9) : 0;
  });
}

// Whether a room's pledges hold a model, and if not by how much they fall short. Same arguments as
// layerCaps plus L. Returns { fits, caps, missing, short, raise }: caps in layers per device,
// missing layers, short = the fewest extra bytes one device could add to make it fit (0 when it
// fits), raise[i] = the extra bytes device i alone would have to add (Infinity when it alone can't:
// the host must hold the embedding and head plus a layer before anyone else's layers count).
// off (optional, expert offload): { expertBytes, ram: [bytes per device, host first] }. When the pledges
// alone fall short, a device with RAM to park experts in counts with what it holds that way (offloadCap);
// the result then fits with offload: true, caps in layers with offload and residentCaps the pledge-only ones.
export function roomFit(L, pledges, layerBytes, hostBytes = 0, off = null) {
  const fit = residentFit(L, pledges, layerBytes, hostBytes);
  if (fit.fits || !offloadOn(off)) return fit;
  const caps = offloadCaps(L, pledges, layerBytes, hostBytes, off);
  if (caps[0] < 1 || caps.reduce((s, c) => s + c, 0) < L) return fit;
  return { fits: true, caps, missing: 0, short: 0, raise: caps.map(() => 0), offload: true, residentCaps: fit.caps };
}
function residentFit(L, pledges, layerBytes, hostBytes) {
  const caps = layerCaps(pledges, layerBytes, hostBytes);
  const has = caps.reduce((s, c) => s + c, 0);
  const hostOk = caps[0] >= 1;
  const missing = Math.max(0, L - has, hostOk ? 0 : 1);
  if (!missing) return { fits: true, caps, missing: 0, short: 0, raise: caps.map(() => 0) };
  const p = (i) => (Number.isFinite(pledges[i]) && pledges[i] > 0 ? pledges[i] : 0);
  const others = has - caps[0];
  const raise = caps.map((c, i) => {
    if (i === 0) return hostBytes + Math.max(1, L - others) * layerBytes - p(0);
    if (!hostOk) return Infinity;
    return (c + (L - has)) * layerBytes - p(i);
  }).map((b) => (b === Infinity ? b : Math.max(1, Math.ceil(b))));
  return { fits: false, caps, missing, short: Math.min(...raise), raise };
}

// ---------------- expert offload (engine/expert_store.js; room-node devices with a discrete GPU) ----------------
// A MoE layer's routed experts are most of its bytes (35B: 470 of 498 MB; 122B: 1.41 of 1.49 GB). A device that
// offloads keeps the experts of some of its layers in system RAM ("parked") and copies the ones a token picks into
// a VRAM cache. Its pledge (VRAM) then pays for: its layers less the parked experts, plus the cache: one whole
// layer's experts for prefill (the region) and at least OFFLOAD_MIN_SLOTS experts per offloaded layer. Its RAM
// (meta.ramGB) pays for the parked experts. Neither is ever exceeded: a pledge is a promise (#291), and so is ramGB.
// A device offloads only what its pledge can't hold, the fewest layers it can (each costs a GPU round trip per
// token), and the rest of its pledge becomes the cache.
export const OFFLOAD_MIN_SLOTS = 16;
export const OFFLOAD_EXPERTS = 256;   // routed experts per layer (Qwen3.5 / 3.6 MoE): the cache's unit is 1/256 layer
const offloadOn = (off) => !!off && off.expertBytes > 0 && (off.ram || []).some((r) => r > 0);
// The fewest of n layers to offload so they fit `bytes` of VRAM (the pledge, less the host's own tensors) and
// `ram` bytes of RAM: 0 when they fit without, -1 when they can't fit at all.
export function offloadNeed(n, bytes, ram, layerBytes, expertBytes, minSlots = OFFLOAD_MIN_SLOTS, nExp = OFFLOAD_EXPERTS) {
  if (!(n > 0)) return 0;
  if (n * layerBytes <= bytes + 1e-6) return 0;
  const E = expertBytes;
  if (!(E > 0) || !(E < layerBytes) || !(ram >= E)) return -1;
  // n layers, m offloaded: n*layer - m*E (parked) + E (region) + m*E*minSlots/nExp (slots) <= bytes
  const save = E * (1 - minSlots / nExp);
  const m = Math.max(1, Math.ceil((n * layerBytes + E - bytes) / save - 1e-9));
  return m <= n && m * E <= ram + 1e-6 ? m : -1;
}
// The most layers (up to L) a device holds with offload. bytes: VRAM for layers (pledge less host tensors).
export function offloadCap(L, bytes, ram, layerBytes, expertBytes, minSlots = OFFLOAD_MIN_SLOTS) {
  let n = layerBytes > 0 && bytes > 0 ? Math.min(L, Math.floor(bytes / layerBytes + 1e-9)) : 0;
  while (n < L && offloadNeed(n + 1, bytes, ram, layerBytes, expertBytes, minSlots) >= 0) n++;
  return n;
}
const vramFor = (pledges, hostBytes, i) => (Number.isFinite(pledges[i]) && pledges[i] > 0 ? pledges[i] : 0) - (i === 0 ? hostBytes : 0);
// layers each device holds with offload where it can (off.ram[i] > 0), within its pledge otherwise
export function offloadCaps(L, pledges, layerBytes, hostBytes, off) {
  const rc = layerCaps(pledges, layerBytes, hostBytes);
  return pledges.map((_, i) => (off.ram?.[i] > 0 ? Math.max(rc[i], offloadCap(L, vramFor(pledges, hostBytes, i), off.ram[i], layerBytes, off.expertBytes, off.minSlots)) : rc[i]));
}
// What one device's n layers look like with offload: null when they fit its pledge as they are, else
// { layers: m (the last m of its range are offloaded), vramBytes: its expert cache (region + slots: the rest of the
// pledge), ramBytes: the experts parked, slots: experts cached per offloaded layer (ExpertStore sizes it the same) }.
// m: the fewest layers that leave each offloaded layer OFFLOAD_GOOD_SLOTS cached experts, else as many as RAM
// allows. Offloading one more layer frees its experts but adds a GPU round trip per token (~0.4 ms on the RTX 5070);
// the cache's hit rate falls fast below a quarter of the experts (35B on the 5070: 109 slots 90% hits, 50 slots 70%,
// docs/bench-log.md 2026-10-02), so the cache's size wins until then.
export const OFFLOAD_GOOD_SLOTS = 64;
export function offloadPlan(n, bytes, ram, layerBytes, expertBytes, minSlots = OFFLOAD_MIN_SLOTS, nExp = OFFLOAD_EXPERTS) {
  const m0 = offloadNeed(n, bytes, ram, layerBytes, expertBytes, minSlots, nExp);
  if (m0 <= 0) return null;
  const E = expertBytes, cache = (m) => bytes - (n * layerBytes - m * E), slots = (m) => Math.floor((cache(m) - E) / (m * E / nExp));
  const top = Math.min(n, Math.floor(ram / E + 1e-9));
  let m = m0;
  while (m < top && slots(m) < OFFLOAD_GOOD_SLOTS) m++;
  return { layers: m, vramBytes: Math.floor(cache(m)), ramBytes: m * E, slots: Math.min(nExp, slots(m)) };
}

// The deal when the pledges alone can't hold the model and offload can (roomFit's offload: true): every device
// that does not offload holds what its pledge holds (all of it: the room is short without offload), and the
// devices that offload take the rest, each from its pledge-only layers up to its cap with offload, in proportion
// to the extra each can take. Phones are left out when the computers hold the model (as dealRoom does).
function dealOffload({ L, layerBytes, hostBytes, pledges, phone, phoneLayers, off, fit }) {
  const n = pledges.length, rc = fit.residentCaps, oc = fit.caps, out = {};
  const canOff = (i) => off.ram?.[i] > 0 && oc[i] > rc[i];
  let idx = [...Array(n).keys()].filter((i) => i === 0 || oc[i] >= 1);
  for (let i = 1; i < n; i++) if (oc[i] < 1) out[i] = "small";
  if (!phoneLayers) {
    const comp = idx.reduce((s, i) => s + (i === 0 || !phone?.[i] ? oc[i] : 0), 0);
    if (comp >= L) { idx.filter((i) => i > 0 && phone?.[i]).forEach((i) => { out[i] = "unneeded"; }); idx = idx.filter((i) => i === 0 || !phone?.[i]); }
  }
  const a = new Map(idx.map((i) => [i, rc[i]]));
  let left = L - [...a.values()].reduce((s, x) => s + x, 0);
  const offIdx = idx.filter(canOff), extra = new Map(offIdx.map((i) => [i, oc[i] - rc[i]]));
  const tot = offIdx.reduce((s, i) => s + extra.get(i), 0);
  if (left <= 0 || left > tot) return null;   // (left <= 0: the pledges hold it after all; roomFit says so first)
  // in proportion to the extra each can take, largest remainders first, never past a cap
  const want = offIdx.map((i) => ({ i, x: left * extra.get(i) / tot }));
  for (const { i, x } of want) { const t = Math.min(Math.floor(x), extra.get(i)); a.set(i, a.get(i) + t); left -= t; }
  want.sort((p, q) => (q.x - Math.floor(q.x)) - (p.x - Math.floor(p.x)) || p.i - q.i);
  for (let guard = 0; left > 0 && guard < 4 * n + 4; guard++) for (const { i } of want) { if (left <= 0) break; if (a.get(i) < oc[i]) { a.set(i, a.get(i) + 1); left--; } }
  if (left > 0) return null;
  const used = idx.filter((i) => i === 0 || a.get(i) > 0);
  idx.forEach((i) => { if (!used.includes(i)) out[i] = "unneeded"; });
  const assigned = used.map((i) => a.get(i)), ranges = rangesOf(assigned);
  const offload = used.map((i, k) => {
    if (!(off.ram?.[i] > 0)) return null;
    const p = offloadPlan(assigned[k], vramFor(pledges, hostBytes, i), off.ram[i], layerBytes, off.expertBytes, off.minSlots);
    return p && { ...p, lo: ranges[k][1] - p.layers, hi: ranges[k][1] };
  });
  // a device that offloads holds its whole pledge (its layers' resident part, then the cache); its RAM apart
  const held = used.map((i, k) => (offload[k] ? vramFor(pledges, hostBytes, i) : assigned[k] * layerBytes) + (i === 0 ? hostBytes : 0));
  return { fit, used, assigned, ranges, held, out, offload };
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
// measured yet: then fewest hops wins, biggest devices first). phone: devices that are phones;
// until a phone is measured it counts as PHONE_COST times slower than an unmeasured computer, so
// computers fill first. The host always keeps at least one layer (it holds the embedding and the
// head anyway). No device gets more than its cap: when the caps cannot hold all L layers (or the
// host's cap is under one layer) nothing is dealt and `short` says how many layers are missing.
// Returns planSplit's shape plus `used`: the device indices that hold layers, in the original order.
export const PHONE_COST = 20;   // one iPhone layer ~ 15-25 GB10 layers (docs/bench-log.md, device matrix)
export function planForSpeed(L, caps, msPerLayer = [], phone = []) {
  caps = caps.map((c) => (Number.isFinite(c) && c > 0 ? c : 0));   // a malformed cap holds nothing
  msPerLayer = msPerLayer || [];
  const n = caps.length;
  const known = msPerLayer.filter((x) => x > 0);
  const fallback = known.length ? Math.max(...known) * 1.5 : 1;   // unmeasured: assume slower than any measured device
  const cost = caps.map((_, i) => msPerLayer[i] > 0 ? msPerLayer[i] : fallback * (phone?.[i] ? PHONE_COST : 1));
  const order = [...caps.keys()].sort((a, b) => cost[a] - cost[b] || (a === 0) * -1 + (b === 0) || caps[b] - caps[a] || a - b);
  const assigned = new Array(n).fill(0);
  assigned[0] = 1;
  let left = L - 1;
  for (const i of order) {
    if (left <= 0) break;
    const take = Math.min(left, Math.max(0, Math.floor(caps[i]) - assigned[i]));
    assigned[i] += take; left -= take;
  }
  // nobody can hold the rest: the room is short. A pledge is a promise, so nothing is spread past
  // anyone's cap (this used to spill the rest over everyone by capacity): the caller must not start
  // and says how short the room is (roomFit). The host needs its one layer within its cap too.
  const short = Math.max(0, left) + (caps[0] < 1 && L > 0 ? 1 : 0);
  if (short || L <= 0) assigned.fill(0);
  return { assigned, ranges: rangesOf(assigned), short, used: assigned.map((a, i) => (a > 0 ? i : -1)).filter((i) => i >= 0) };
}

// A phone: its own flag, or a phone user agent (the meta a device sends when it joins).
export function isPhoneMeta(m) {
  return !!(m?.phone || /^(iPhone|Android)$/.test(m?.ua || ""));
}

// Phones hold layers only when the computers cannot. One layer on a phone costs as much as 15-25
// layers on a desktop GPU plus two Wi-Fi hops per token (docs/bench-log.md, device matrix), so a
// room whose computers can hold the whole model runs faster with its phones as ask-only guests.
// capsLayers: layers each device can hold (host first); phone: which devices are phones. The host
// always counts as a computer (it holds the embedding and the head either way). Returns the
// device indices (never 0) to leave out: every phone when the others hold all L layers, else none.
export function phonesToLeaveOut(L, capsLayers, phone) {
  const others = capsLayers.reduce((s, c, i) => s + (i === 0 || !phone[i] ? Math.max(0, Math.floor(c)) : 0), 0);
  return others >= L ? capsLayers.map((_, i) => i).filter((i) => i > 0 && phone[i]) : [];
}

// The device that becomes the model host when someone presses Start. The model host holds the
// embedding, the head and the draft block, samples every token and runs the Code agent, so it is
// the strongest device, whoever pressed Start: a device with WebGPU first, then a computer before
// a phone (a phone's lent memory says little about its GPU, and a phone tab sleeps with its
// screen), then the most memory lent, then the lowest id so every screen picks the same one.
// Then speed: decode is bound by memory bandwidth and the host's extra work sits on every token's
// path, so a device of the same kind whose GPU copies memory clearly faster (meta.gbps, measured at
// load by room/gpuspeed.js, at least SPEED_EDGE times the memory pick's) and that lends at least
// half as much memory hosts instead: the fastest such device, then memory, then id. A device that
// did not measure (an older tab, or a probe that failed) leaves the pick by memory as it is, and so
// does a device with the same GPU as the memory pick (meta.gpu, e.g. two tabs on one machine or two
// identical machines): the same GPU copies at the same speed, so a gap there is load at probe time.
// devices: [{ id, meta: { webgpu, contribGB, phone, ua, gbps, gpu } }]. Returns an id (null for none).
export const SPEED_EDGE = 1.5;
export function pickModelHost(devices) {
  const gb = (m) => (m?.webgpu ? +m?.contribGB || 0 : 0);
  const key = (d) => [d.meta?.webgpu ? 1 : 0, isPhoneMeta(d.meta) ? 0 : 1, gb(d.meta)];
  // > 0 when a is the better pick by kind, then memory, then the lower id
  const cmp = (a, b) => { const x = key(a), y = key(b); return x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || (a.id < b.id ? 1 : -1); };
  const ds = devices.filter((d) => d?.id);
  let best = null;
  for (const d of ds) if (!best || cmp(d, best) > 0) best = d;
  if (!best) return null;
  const bps = (d) => +d.meta?.gbps || 0, kb = key(best);
  if (!kb[0] || !(bps(best) > 0)) return best.id;
  let fast = null;
  for (const d of ds) {
    const k = key(d);
    if (k[0] !== kb[0] || k[1] !== kb[1] || k[2] < 0.5 * kb[2] || bps(d) < SPEED_EDGE * bps(best)) continue;
    if (d !== best && d.meta?.gpu && d.meta.gpu === best.meta?.gpu) continue;   // same GPU: the gap is noise
    if (!fast || bps(d) > bps(fast) || (bps(d) === bps(fast) && cmp(d, fast) > 0)) fast = d;
  }
  return (fast || best).id;
}

// GB for the screen, rounded up to a tenth so "short by" never undersells what is missing
export const gbUp = (bytes) => Math.ceil(bytes / 2 ** 30 * 10 - 1e-9) / 10;

// The one sentence a room that is short for a model shows under Start (and the reason a start stops).
// label: the model's short name; fit: roomFit's result (host first); names: the devices in the same
// order; spareGB: how much more each device could still lend (0 for a phone at its cap). Names up to
// two devices that could close the gap alone, the host's raise first when it is the smaller one.
export function shortNote(label, fit, names = [], spareGB = []) {
  if (fit.fits) return "";
  const who = givers(fit, spareGB).slice(0, 2).map(({ i, b }) => `${names[i] || "a device"} could give ${gbUp(b)} GB more`);
  const head = `This room is ${gbUp(shortBy(fit, spareGB))} GB short for ${label}`;
  const host = fit.caps[0] < 1 ? `; the model host (${names[0] || "the biggest device"}) needs room for the embedding, the output head and a layer` : "";
  return `${head}${host}. Add a device or raise a pledge${who.length ? `: ${who.join(", or ")}` : ""}.`;
}
// the devices that could close the gap alone, fewest bytes first: [{ i, b }]
const givers = (fit, spareGB) => fit.raise.map((b, i) => ({ i, b })).filter(({ i, b }) => b < Infinity && gbUp(b) <= (spareGB[i] || 0) + 1e-9)
  .sort((a, b) => a.b - b.b || a.i - b.i);
// How short a room is, in bytes, for the screen: what the device that can give the least more would
// have to add (so "0.8 GB short" and "Laptop could give 0.8 GB more" agree), or when no device can,
// the smallest raise anyone would need. 0 when it fits.
export function shortBy(fit, spareGB = []) {
  if (fit.fits) return 0;
  const g = givers(fit, spareGB);
  return g.length ? g[0].b : fit.short;
}

// The whole deal the model host makes at Start, from bytes: which devices hold which layers, within
// every pledge. L layers of layerBytes each (with their KV cache at the room's context); hostBytes
// what the host holds besides its layers (room/models.js hostHeldBytes); pledges: bytes each device
// lends, host first; mode: "speed" (planForSpeed: fastest first, ms per layer where measured) or
// "memory" (planSplit in proportion to pledges, phones left out while the computers can hold the
// model unless phoneLayers); phone: which devices are phones.
// Returns { fit, used: device indices that hold layers (host first, in order), assigned and ranges
// for those devices, held: bytes each device in `used` holds, out: { index: "small" | "unneeded" } }.
// When the pledges cannot hold the model nothing is dealt: fit.fits is false (fit.short, fit.raise).
// off (optional): roomFit's expert offload ({ expertBytes, ram }). Only when the pledges alone can't hold the
// model does a device offload (dealOffload); the result then also has `offload`: per device in `used`, null or
// { layers, lo, hi, vramBytes, ramBytes } (offloadPlan, with the range of its layers to offload).
export function dealRoom({ L, layerBytes, hostBytes = 0, pledges, mode = "memory", ms = [], phone = [], phoneLayers = false, off = null }) {
  const fit = roomFit(L, pledges, layerBytes, hostBytes, off);
  const none = { fit, used: [], assigned: [], ranges: [], held: [], out: {} };
  if (!fit.fits) return none;
  if (fit.offload) return dealOffload({ L, layerBytes, hostBytes, pledges, phone, phoneLayers, off, fit }) || { ...none, fit: { ...fit, fits: false } };
  const out = {};
  // a device whose pledge is under one layer holds none (the host always has room: fit says so)
  let idx = fit.caps.map((c, i) => i).filter((i) => i === 0 || fit.caps[i] >= 1);
  fit.caps.forEach((c, i) => { if (i > 0 && c < 1) out[i] = "small"; });
  let assigned, ranges;
  if (mode === "speed") {
    const sp = planForSpeed(L, idx.map((i) => fit.caps[i]), idx.map((i) => ms?.[i]), idx.map((i) => !!phone?.[i]));
    if (sp.short) return none;
    idx.forEach((i, k) => { if (!sp.used.includes(k)) out[i] = "unneeded"; });
    assigned = sp.used.map((k) => sp.assigned[k]); ranges = sp.used.map((k) => sp.ranges[k]);
    idx = sp.used.map((k) => idx[k]);
  } else {
    const off = phoneLayers ? [] : phonesToLeaveOut(L, idx.map((i) => fit.caps[i]), idx.map((i, k) => k > 0 && !!phone?.[i]));
    off.forEach((k) => { out[idx[k]] = "unneeded"; });
    idx = idx.filter((_, k) => !off.includes(k));
    const sp = planSplit(L, idx.map((i) => pledges[i] - (i === 0 ? hostBytes : 0)), idx.map((i) => fit.caps[i]));
    if (sp.short) return none;
    // fewer layers than devices: the ones dealt none are not needed
    const keep = sp.assigned.map((a, k) => (k === 0 || a > 0 ? k : -1)).filter((k) => k >= 0);
    idx.forEach((i, k) => { if (!keep.includes(k)) out[i] = "unneeded"; });
    assigned = keep.map((k) => sp.assigned[k]); ranges = keep.map((k) => sp.ranges[k]);
    idx = keep.map((k) => idx[k]);
  }
  const held = idx.map((i, k) => assigned[k] * layerBytes + (i === 0 ? hostBytes : 0));
  return { fit, used: idx, assigned, ranges, held, out };
}
