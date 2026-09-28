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

// A device's layers as the room shows them: "0–19", or with the host's tail "0–37, 39–39".
// Returns [[lo, hi) per span]; a string without a range gives [].
export function layerSpans(s) {
  return String(s ?? "").split(",").map((p) => /^\s*(\d+)\D+(\d+)\s*$/.exec(p)).filter(Boolean).map((m) => [+m[1], +m[2] + 1]);
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

// Where each device's layers sit: assigned[i] layers per device in chain order, host first. A
// hybrid model (Qwen3.5/3.6: full attention every few layers, DeltaNet in between) ends on a full
// attention layer, and the chain's last device holds it. On a phone that is the worst layer to
// hold: attention without subgroups grows with the context (+3 ms from position 250 to 600 on an
// iPhone 14 Pro Max) and its KV cache sits on the smallest device, while a DeltaNet layer costs the
// same at every position with a fixed-size state. So when the chain ends on a phone and that moves
// a full-attention layer off the phones, every device's slice moves one layer earlier and the host
// keeps the model's last layer as a tail: it runs it on the hidden state the chain returns, before
// the head. No extra hop and nothing changes on the other devices (each still gets one range).
// isFull(i): layer i is full attention. Returns { ranges: [[lo, hi) per device], tail: [lo, hi)
// the host runs after the chain, or null }.
export function placeLayers(L, assigned, { isFull = () => false, phone = [] } = {}) {
  const ranges = [];
  let acc = 0;
  for (const a of assigned) { ranges.push([acc, acc + a]); acc += a; }
  const n = assigned.length;
  if (n < 2 || acc !== L || !phone[n - 1] || assigned[0] < 2) return { ranges, tail: null };
  let p = n - 1;                                   // the phones at the end of the chain
  while (p > 1 && phone[p - 1]) p--;
  const fulls = (a, b) => { let k = 0; for (let i = a; i < b; i++) k += isFull(i) ? 1 : 0; return k; };
  if (fulls(ranges[p][0] - 1, L - 1) >= fulls(ranges[p][0], L)) return { ranges, tail: null };
  return { ranges: ranges.map(([a, b], i) => (i === 0 ? [a, b - 1] : [a - 1, b - 1])), tail: [L - 1, L] };
}

// The device that becomes the model host when someone presses Start. The model host holds the
// embedding, the head and the draft block, samples every token and runs the Code agent, so it is
// the strongest device, whoever pressed Start: a device with WebGPU first, then a computer before
// a phone (a phone's lent memory says little about its GPU, and a phone tab sleeps with its
// screen), then the most memory lent, then the lowest id so every screen picks the same one.
// devices: [{ id, meta: { webgpu, contribGB, phone, ua } }]. Returns an id (null for none).
export function pickModelHost(devices) {
  const gb = (m) => (m?.webgpu ? +m?.contribGB || 0 : 0);
  let best = null;
  for (const d of devices) {
    if (!d?.id) continue;
    const k = [d.meta?.webgpu ? 1 : 0, isPhoneMeta(d.meta) ? 0 : 1, gb(d.meta)];
    if (!best) { best = { id: d.id, k }; continue; }
    const c = k[0] - best.k[0] || k[1] - best.k[1] || k[2] - best.k[2] || (d.id < best.id ? 1 : -1);
    if (c > 0) best = { id: d.id, k };
  }
  return best?.id ?? null;
}
