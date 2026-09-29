// Tables from a two-machine run (tests/e2e/xroom.mjs, usually via xroom_pair.sh):
//   node tests/e2e/xroom_report.mjs host.json [guest-trace.json] [--ping ping.txt] [--ping-log ping_during.txt]
//        [--merged merged.json] [--json]
//
// Always: one row per round (mode, prompt, tok/s, acceptance, prefill, the answer's hash) with the
// ping round trip measured during that round (--ping-log: `ping -D` output; the link is shared Wi-Fi
// in some setups, and a round that met a latency spike says so here).
// With a guest trace (tracing on): the host's result carries its trace per traced round; the
// guest's file is one trace of every traced round. Both are on their own machine's system clock,
// which is not good enough for a millisecond wire, so the guest's offset is measured from the
// frames themselves: for every lap, forward = guest's last slice in - host's last slice out,
// backward = host's last slice in - guest's last slice out. Forward is (one-way delay + offset) and
// backward (one-way delay - offset); with the same delay both ways (same path, same frame size)
// offset = (p10 forward - p10 backward) / 2. forward + backward is the lap's wire time and needs no
// clock at all: that sum is the number to compare with the ping round trip.
// Then every traced round goes through room_prof_report.mjs's analysis (lap segments, hops,
// handoffs, per-device GPU, host phases, speculation), plus the wire table here.
import fs from "fs";
import { analyze, render, table, r2 } from "./room_prof_report.mjs";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const pos = process.argv.slice(2).filter((a, i, all) => !a.startsWith("--") && !(i > 0 && all[i - 1].startsWith("--") && all[i - 1] !== "--json"));
const H = JSON.parse(fs.readFileSync(pos[0], "utf8"));
const G = pos[1] ? JSON.parse(fs.readFileSync(pos[1], "utf8")).trace : null;
const q = (a, f) => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : null; };

// ping -D lines: "[1790576446.123456] 64 bytes from ...: icmp_seq=1 ttl=64 time=4.12 ms"
const pings = arg("ping-log") && fs.existsSync(arg("ping-log"))
  ? fs.readFileSync(arg("ping-log"), "utf8").split("\n").map((l) => /^\[([\d.]+)\].*time=([\d.]+) ms/.exec(l)).filter(Boolean).map((m) => ({ t: +m[1] * 1000, ms: +m[2] }))
  : [];
const pingIn = (t0, t1) => { const v = pings.filter((x) => x.t >= t0 && x.t <= t1).map((x) => x.ms); return v.length ? { p50: r2(q(v, 0.5)), p90: r2(q(v, 0.9)), max: r2(Math.max(...v)), n: v.length } : null; };
const rounds = (H.rows || []).map((r) => {
  const pg = r.t0 ? pingIn(r.t0, r.t1) : null;
  return { round: r.idx, mode: r.mode, prompt: r.prompt, traced: r.traced ? "yes" : "", "tok/s": r.tps, tokens: r.tokens, accepted: r.accepted, lookup: r.lookupTok, "prefill s": r.prefillS,
    "ping p50": pg?.p50, "ping p90": pg?.p90, "ping max": pg?.max, "card rtt": (r.rtt || []).filter((x) => x !== "-").join(" "), answer: r.answerSha };
});

// the guest records only while a round is traced: split its marks into one cluster per traced round
function clusters(hp, gapMs = 600) {
  const out = []; let cur = null;
  for (const e of [...hp].sort((a, b) => a[3] - b[3])) {
    if (!cur || e[3] - cur.t1 > gapMs) { cur = { t0: e[3], t1: e[3], hp: [] }; out.push(cur); }
    cur.hp.push(e); cur.t1 = e[3];
  }
  return out;
}
const RET = { "ai-hidden": "ai-hiddenret", "ai-hidden-b": "ai-hiddenret-b" };
function firstBy(hp, ev) { const m = new Map(); for (const e of hp) if (e[0] === ev) { const k = e[1] + "|" + e[2]; if (!m.has(k)) m.set(k, e); } return m; }
// forward / backward raw deltas of every lap in a round (guest clock - host clock included)
function pairs(hostHp, guestHp) {
  const hs = firstBy(hostHp, "send1"), hr = firstBy(hostHp, "rx1"), gs = firstBy(guestHp, "send1"), gr = firstBy(guestHp, "rx1"), gsz = firstBy(hostHp, "send0");
  const out = [];
  for (const [k, e] of hs) {
    const [kind, p] = k.split("|");
    if (!RET[kind]) continue;
    const a = gr.get(k), b = gs.get(RET[kind] + "|" + p), c = hr.get(RET[kind] + "|" + p);
    if (!a || !b || !c) continue;
    out.push({ kind, pos: +p, bytes: gsz.get(k)?.[4], fwd: a[3] - e[3], bwd: c[3] - b[3] });
  }
  return out;
}
function shift(tr, d) {
  return { ...tr, hp: tr.hp.map((e) => [e[0], e[1], e[2], e[3] - d, e[4], e[5]]), gp: { subs: tr.gp.subs.map((x) => ({ ...x, t: x.t - d, tEnc: x.tEnc - d })), maps: tr.gp.maps.map((m) => ({ ...m, t0: m.t0 - d, t1: m.t1 ? m.t1 - d : 0 })) } };
}

const notes = [];
const merged = { model: H.model, devices: 2, wire: "stripe4", split: H.split, runs: [] };
const wireRows = [];
if (G) {
  // the guest marks the start of every traced round ("x-trace", pos = the round's index); older
  // traces without them are split at gaps
  const starts = G.hp.filter((e) => e[0] === "x-trace").sort((a, b) => a[3] - b[3]);
  const cl = starts.length ? starts.map((e, i) => ({ idx: e[2], t0: e[3], t1: starts[i + 1] ? starts[i + 1][3] - 1 : Infinity })) : clusters(G.hp);
  const traced = H.traces || [];
  if (!starts.length && cl.length !== traced.length) notes.push(`guest trace has ${cl.length} clusters for ${traced.length} traced rounds: matched by time instead`);
  traced.forEach((rd, i) => {
    // the guest's marks for this round: its x-trace section, or the i-th cluster, or (clusters do
    // not line up) the marks inside the round's window on the system clocks
    const c = starts.length ? cl.find((x) => x.idx === rd.idx) || { t0: 0, t1: -1 } : cl.length === traced.length ? cl[i] : { t0: rd.t0 - 500, t1: rd.t1 + 500 };
    const inWin = (t) => t >= c.t0 - 1 && t <= c.t1 + 1;
    const g = { hp: G.hp.filter((e) => inWin(e[3])), gp: { subs: G.gp.subs.filter((x) => inWin(x.t)), maps: G.gp.maps.filter((m) => inWin(m.t0)) } };
    const P = pairs(rd.host.hp, g.hp);
    const off = P.length ? (q(P.map((x) => x.fwd), 0.1) - q(P.map((x) => x.bwd), 0.1)) / 2 : 0;
    const label = `${rd.mode} · ${rd.prompt} · round ${rd.round}`;
    merged.runs.push({ label, mode: rd.mode, status: rd.status, reply: rd.reply, crumb: rd.crumb, traces: { host: rd.host, worker1: shift(g, off) }, offset: off });
    // wire per frame size: one-way each way after the offset, and forward + backward (no clock needed)
    const bySize = {};
    for (const x of P) (bySize[x.kind + " " + (x.bytes >> 10) + " KB"] ||= []).push(x);
    const pg = pingIn(rd.t0, rd.t1);
    for (const [sz, xs] of Object.entries(bySize)) wireRows.push({ round: label, frame: sz, laps: xs.length, "fwd one-way p50": r2(q(xs.map((x) => x.fwd - off), 0.5)), "bwd one-way p50": r2(q(xs.map((x) => x.bwd + off), 0.5)),
      "fwd+bwd p10": r2(q(xs.map((x) => x.fwd + x.bwd), 0.1)), "fwd+bwd p50": r2(q(xs.map((x) => x.fwd + x.bwd), 0.5)), "fwd+bwd p90": r2(q(xs.map((x) => x.fwd + x.bwd), 0.9)), "ping p50": pg?.p50, "offset ms": r2(off) });
  });
  if (arg("merged")) fs.writeFileSync(arg("merged"), JSON.stringify(merged));
}
const rep = G ? analyze(merged) : null;
rep?.runs.forEach((R, i) => { R.label = merged.runs[i].label; });
if (process.argv.includes("--json")) { console.log(JSON.stringify({ rounds, report: rep, wire: wireRows, notes, links: H.links }, null, 1)); process.exit(0); }
const o = [];
o.push(`# ${H.model} · ${H.split}${H.solo ? " · solo" : ""}`);
o.push(table(rounds, Object.keys(rounds[0] || { round: 0 })));
if (arg("ping")) o.push("ping: " + fs.readFileSync(arg("ping"), "utf8").split("\n").filter((l) => /rtt|packets/.test(l)).join(" | "));
if (Array.isArray(H.links)) o.push("room link: " + [...new Set(H.links.map((l) => `${l.local} -> ${l.remote}`))].join("; ") + ` · STUN rtt ${H.links.map((l) => l.stunRttMs).join("/")} ms`);
for (const n of notes) o.push("note: " + n);
if (G) {
  o.push("\n## wire per lap (last slice out -> last slice in; fwd = host -> guest)");
  o.push(table(wireRows, Object.keys(wireRows[0] || { round: 0 })));
  o.push("\n" + render(rep));
}
console.log(o.join("\n"));
