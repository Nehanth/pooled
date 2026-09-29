// Per-device split of every traced lap in a room of any number of devices (tests/e2e/xroom_cluster.sh):
//   node tests/e2e/xroom_chain_report.mjs host.json [worker traces...] [--clock clock.txt] [--bin 30] [--json]
// Worker traces: a computer guest's (xroom.mjs --trace-out: {role: "guest", trace}) or the phone's
// (xroom_phone.mjs --trace-out: {role: "phone", name, trace}); both carry wire marks
// [event, kind, pos, ms, ...] (send0 / send1 / rx0 / rx1), GPU submits {t, tEnc, gpu} and mapAsyncs.
//
// No clock alignment is needed for the split itself: every device's time in a lap is measured on its
// own clock (its "residence": last slice of the frame in -> last slice of its frame out), the host's
// wait is measured on the host's, and the wire (every hop of the lap together) is the host's wait
// minus every worker's residence. The clocks are only used to put each worker's laps into the host's
// traced rounds (a window of the round +- 400 ms): a computer guest's trace is on its machine's system
// clock; the phone's is on the phone's, shifted by its own WebDriver clock samples (phone - the Mac it
// is attached to) and, with --clock (the cluster script's "mac_minus_here <ms>" line), the Mac's
// offset from the host machine's.
// Residence split: pre = frame in -> first command encoder (unpack, upload, JS/event loop); gpu = the
// timestamped GPU time of its command buffers; readback = last mapAsync resolved - first submit - gpu
// (submit, scheduling and the map's completion latency beyond the GPU work); post = map resolved ->
// first slice out (f32 -> f16 pack, JS); send = first -> last slice handed to the channel.
// Worker laps over the whole session (not only traced rounds) also go into a timeline per --bin
// seconds (a phone tracing a long run: does its GPU time per lap grow as it warms up?).
import fs from "fs";
import { table, r2 } from "./room_prof_report.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const files = argv.filter((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--") && argv[i - 1] !== "--json"));
const H = JSON.parse(fs.readFileSync(files[0], "utf8"));
const BIN = +arg("bin", 30) * 1000;
let macMinusHere = 0;
if (arg("clock") && fs.existsSync(arg("clock"))) { const m = /mac_minus_here (-?[\d.]+)/.exec(fs.readFileSync(arg("clock"), "utf8")); if (m) macMinusHere = +m[1]; }
const hostOnMac = H.platform === "darwin";
const q = (a, f) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : null; };
const med = (a) => q(a, 0.5), p95 = (a) => q(a, 0.95);
const FWD = new Set(["ai-hidden", "ai-hidden-b"]);
const fam = (k) => (/-b$/.test(k) ? "b" : "1");

// ---- workers ----
const workers = files.slice(1).map((f) => {
  const J = JSON.parse(fs.readFileSync(f, "utf8"));
  const T = J.trace;
  const isPhone = J.role === "phone";
  // shift onto the host's clock (coarse)
  let off = 0;
  if (isPhone) {
    const c = (T.clock || []).filter((x) => x && Number.isFinite(x.off)).sort((a, b) => a.rtt - b.rtt);
    const phoneMinusMac = c.length ? med(c.slice(0, Math.max(1, Math.ceil(c.length / 2))).map((x) => x.off)) : 0;
    off = phoneMinusMac + (hostOnMac ? 0 : macMinusHere);
  } else {
    const onMac = J.platform === "darwin";
    off = onMac && !hostOnMac ? macMinusHere : !onMac && hostOnMac ? -macMinusHere : 0;
  }
  const hp = (T.hp || []).map((e) => [e[0], e[1], e[2], e[3] - off, e[4], e[5]]).sort((a, b) => a[3] - b[3]);
  const subs = (T.gp?.subs || []).map((x) => ({ ...x, t: x.t - off, tEnc: x.tEnc - off })).sort((a, b) => a.t - b.t);
  const maps = (T.gp?.maps || []).filter((m) => m.t1).map((m) => ({ ...m, t0: m.t0 - off, t1: m.t1 - off })).sort((a, b) => a.t0 - b.t0);
  const lag = (T.lag || []).map(([t, l]) => [t - off, l]);
  const name = isPhone ? J.name || "phone" : (J.platform === "darwin" ? "m5max" : "gb10");
  // laps: forward frame in (rx1) -> this device's frame out for the same position (send0, send1)
  const laps = [], pend = new Map();
  for (const e of hp) {
    const [ev, kind, pos, t] = e;
    if (ev === "rx1" && FWD.has(kind)) pend.set(fam(kind) + "|" + pos, { kind, pos, rx1: t });
    else if (ev === "send0" && pend.has(fam(kind) + "|" + pos) && kind !== undefined) { const L = pend.get(fam(kind) + "|" + pos); if (!L.send0) { L.send0 = t; L.bytesOut = e[4]; L.out = kind; } }
    else if (ev === "send1" && pend.has(fam(kind) + "|" + pos)) { const L = pend.get(fam(kind) + "|" + pos); if (L.send0) { L.send1 = t; pend.delete(fam(kind) + "|" + pos); laps.push(L); } }
  }
  // split each lap with the submits and maps inside it
  let si = 0, mi = 0;
  for (const L of laps) {
    while (si < subs.length && subs[si].t < L.rx1) si++;
    const S = []; for (let j = si; j < subs.length && subs[j].t <= L.send0; j++) S.push(subs[j]);
    while (mi < maps.length && maps[mi].t0 < L.rx1) mi++;
    const M = []; for (let j = mi; j < maps.length && maps[j].t0 <= L.send0; j++) M.push(maps[j]);
    L.res = L.send1 - L.rx1;
    L.nsub = S.length; L.nmap = M.length;
    L.gpu = S.some((x) => x.gpu >= 0) ? S.reduce((a, x) => a + Math.max(0, x.gpu || 0), 0) : null;
    if (S.length) { L.pre = Math.max(0, Math.min(...S.map((x) => x.tEnc)) - L.rx1); L.enc = S.reduce((a, x) => a + (x.t - x.tEnc), 0); }
    const lastMap = M.length ? M[M.length - 1] : null;
    if (lastMap && S.length) { L.gpuWall = lastMap.t1 - S[0].t; L.readback = L.gpu != null ? L.gpuWall - L.gpu : null; L.post = L.send0 - lastMap.t1; }
    L.send = L.send1 - L.send0;
    L.lag = lag.filter(([t]) => t >= L.rx1 && t <= L.send1).reduce((a, [, l]) => Math.max(a, l), 0);
  }
  return { file: f, name, isPhone, off, laps, links: T.links || [], notes: T.notes || [], devs: T.devs, subsN: subs.length };
});

// ---- host laps per traced round ----
function hostLaps(tr) {
  const hp = [...tr.hp].sort((a, b) => a[3] - b[3]);
  const subs = [...(tr.gp?.subs || [])].sort((a, b) => a.t - b.t), maps = (tr.gp?.maps || []).filter((m) => m.t1);
  const laps = [], steps = [], byKey = new Map();
  let step = null;
  for (const e of hp) {
    const [ev, kind, pos, t, a, b] = e;
    const key = fam(kind) + "|" + pos;
    if (ev === "h.step0") { step = { t0: t, via: kind, K: b ?? a, pos }; step.K = a ?? step.K; steps.push(step); }
    else if (ev === "h.step1" && step) { step.t1 = t; step.tokens = a; step = null; }
    else if (ev === "h.lap0") { const L = { kind, pos, lap0: t, n: a || 1, step }; laps.push(L); byKey.set(key, L); }
    else if (ev === "h.pack1" && byKey.has(key)) byKey.get(key).pack1 = t;
    else if (ev === "send1" && FWD.has(kind) && byKey.has(key)) { const L = byKey.get(key); if (!L.send1) L.send1 = t; }
    else if (ev === "rx1" && !FWD.has(kind) && byKey.has(key)) { const L = byKey.get(key); if (!L.rx1) L.rx1 = t; }
    else if (ev === "h.ret" && byKey.has(key)) byKey.get(key).ret = t;
    else if (ev === "h.head0" && byKey.has(key)) byKey.get(key).head0 = t;
    else if (ev === "h.head1" && byKey.has(key)) byKey.get(key).head1 = t;
  }
  for (const L of laps) {
    const S = subs.filter((x) => x.t >= L.lap0 && x.t <= (L.pack1 ?? L.lap0));
    L.hostPre = L.pack1 - L.lap0; L.hostGpu = S.reduce((a, x) => a + (x.gpu || 0), 0);
    L.wait = L.rx1 - L.send1;
    L.head = L.head1 != null ? L.head1 - L.head0 : null;
  }
  laps.forEach((L, i) => { const nx = laps[i + 1]; L.lapMs = nx && nx.lap0 - L.lap0 < 2000 ? nx.lap0 - L.lap0 : null; });
  // prefill laps are the first frames of the round (batched b-frames before any step); decode laps: plain = "ai-hidden",
  // spec = b-frames inside a step
  const dec = laps.filter((L) => L.kind === "ai-hidden" || L.step);
  return { laps: dec, prefill: laps.filter((L) => !(L.kind === "ai-hidden" || L.step)), steps: steps.filter((s) => s.t1) };
}

const out = { model: H.model, split: H.split, rounds: [], workers: workers.map((w) => ({ name: w.name, laps: w.laps.length, off: r2(w.off), notes: w.notes, devs: w.devs })) };
const lines = [`# ${H.model} · ${H.split} · host ${hostOnMac ? "m5max" : "gb10"} · workers: ${workers.map((w) => `${w.name} (${w.laps.length} laps)`).join(", ")}`];
const rows = [], wrows = [], srows = [];
for (const tr of H.traces || []) {
  const R = hostLaps(tr.host);
  const label = `${tr.mode} ${tr.prompt} r${tr.round}`;
  const win = [tr.t0 - 400, tr.t1 + 400];
  // attach each worker's lap for the same frame (family + position) inside the round's window
  const wl = workers.map((w) => { const m = new Map(); for (const L of w.laps) if (L.rx1 >= win[0] && L.rx1 <= win[1]) m.set(fam(L.kind) + "|" + L.pos, L); return m; });
  const full = [];
  for (const L of R.laps) {
    const key = fam(L.kind) + "|" + L.pos;
    const ws = wl.map((m) => m.get(key));
    if (!Number.isFinite(L.wait) || ws.some((x) => !x)) continue;
    const resSum = ws.reduce((a, x) => a + x.res, 0);
    full.push({ L, ws, wire: L.wait - resSum });
  }
  const row = { round: label, "tok/s": (H.rows || []).find((r) => r.idx === tr.idx)?.tps, laps: full.length, "lap p50": r2(med(full.map((x) => x.L.lapMs))), "lap p95": r2(p95(full.map((x) => x.L.lapMs))),
    "host pre": r2(med(full.map((x) => x.L.hostPre))), "host gpu": r2(med(full.map((x) => x.L.hostGpu))), "wait p50": r2(med(full.map((x) => x.L.wait))), "wait p95": r2(p95(full.map((x) => x.L.wait))),
    "head": r2(med(full.map((x) => x.L.head))), "wire (all hops) p50": r2(med(full.map((x) => x.wire))), "wire p95": r2(p95(full.map((x) => x.wire))) };
  workers.forEach((w, i) => { row[w.name + " res"] = r2(med(full.map((x) => x.ws[i].res))); });
  rows.push(row);
  workers.forEach((w, i) => {
    const X = full.map((x) => x.ws[i]);
    wrows.push({ round: label, device: w.name, laps: X.length, "res p50": r2(med(X.map((x) => x.res))), "res p95": r2(p95(X.map((x) => x.res))), pre: r2(med(X.map((x) => x.pre))), enc: r2(med(X.map((x) => x.enc))),
      gpu: r2(med(X.map((x) => x.gpu))), "gpu p95": r2(p95(X.map((x) => x.gpu))), readback: r2(med(X.map((x) => x.readback))), post: r2(med(X.map((x) => x.post))), send: r2(med(X.map((x) => x.send))),
      "subs/maps": `${med(X.map((x) => x.nsub))}/${med(X.map((x) => x.nmap))}`, "lag max": r2(Math.max(0, ...X.map((x) => x.lag || 0))), "KB out": X[0] ? r2((X[0].bytesOut || 0) / 1024) : null });
  });
  if (R.steps.length) {
    const byK = {};
    for (const s of R.steps) { const k = `${s.via} K=${s.K}`; (byK[k] ||= []).push(s); }
    for (const [k, S] of Object.entries(byK)) srows.push({ round: label, drafts: k, steps: S.length, "tok/step": r2(S.reduce((a, s) => a + s.tokens, 0) / S.length), "accepted": r2(S.reduce((a, s) => a + Math.max(0, s.tokens - 1), 0) / S.reduce((a, s) => a + (s.K || 0), 0)), "step p50": r2(med(S.map((s) => s.t1 - s.t0))), "step p95": r2(p95(S.map((s) => s.t1 - s.t0))) });
  }
  out.rounds.push({ label, row, steps: srows.filter((r) => r.round === label) });
}
if (rows.length) {
  lines.push("\n## laps (ms, medians unless p95; decode laps of traced rounds; wire = host wait - every worker's residence)");
  lines.push(table(rows, Object.keys(rows[0])));
  lines.push("\n## worker residence split (ms per lap)");
  lines.push(table(wrows, Object.keys(wrows[0])));
}
if (srows.length) { lines.push("\n## speculative steps per draft depth (accepted = (tokens - 1) / drafts)"); lines.push(table(srows, Object.keys(srows[0]))); }
// session-wide timeline per worker (every lap it served, traced round or not)
for (const w of workers) {
  if (!w.laps.length) continue;
  const t0 = w.laps[0].rx1, bins = new Map();
  for (const L of w.laps) { const b = Math.floor((L.rx1 - t0) / BIN); (bins.get(b) || bins.set(b, []).get(b)).push(L); }
  const tl = [...bins].map(([b, X]) => {
    const one = X.filter((x) => x.kind === "ai-hidden"), bb = X.filter((x) => x.kind === "ai-hidden-b");
    return { "t s": (b * BIN) / 1000, laps: X.length, "1-col laps": one.length, "1-col gpu p50": r2(med(one.map((x) => x.gpu))), "1-col res p50": r2(med(one.map((x) => x.res))), "1-col res p95": r2(p95(one.map((x) => x.res))),
      "block laps": bb.length, "block gpu p50": r2(med(bb.map((x) => x.gpu))), "block res p50": r2(med(bb.map((x) => x.res))), "lag max": r2(Math.max(0, ...X.map((x) => x.lag || 0))) };
  });
  out[w.name + "Timeline"] = tl;
  lines.push(`\n## ${w.name}: every lap it served, per ${BIN / 1000} s (1-col = plain token, block = speculative verify / prefill)`);
  lines.push(table(tl, Object.keys(tl[0])));
  const lk = w.links.map((x) => x.links.map((l) => `${l.remote} ${l.rtt}`).join(", ")).filter(Boolean);
  if (lk.length) lines.push(`${w.name} links (remote, STUN rtt ms), first / last: ${lk[0]} / ${lk[lk.length - 1]}`);
  if (w.notes.length) lines.push(`${w.name} notes: ${w.notes.join("; ")}`);
}
if (argv.includes("--json")) console.log(JSON.stringify({ ...out, rows, wrows, srows }, null, 1));
else console.log(lines.join("\n"));
