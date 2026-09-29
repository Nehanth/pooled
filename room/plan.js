// Layer placement and the model ladder. DOM-free so it can be unit tested.

// Deal L layers over devices in proportion to what each can hold. caps: bytes each device offers
// for layers (host first; the host's cap already has the embedding, head and draft block taken
// out). Every device gets at least one layer; rounding leftovers go to the largest remainders.
// Returns { assigned: [count per device], ranges: [[lo, hi) per device] }.
export function planSplit(L, caps) {
  // a cap that is not a positive number (a malformed pledge) counts as nothing; when nobody
  // offers anything the layers are dealt evenly instead of dividing by zero
  caps = caps.map((c) => (Number.isFinite(c) && c > 0 ? c : 0));
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
// measured yet: then fewest hops wins, biggest devices first). phone: devices that are phones;
// until a phone is measured it counts as PHONE_COST times slower than an unmeasured computer, so
// computers fill first. The host always keeps at least one layer (it holds the embedding and the
// head anyway). Returns planSplit's shape plus `used`: the device indices that hold layers, in
// the original order.
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
