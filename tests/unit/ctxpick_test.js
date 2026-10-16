// room/ctxpick.js: the picker (room.js roomFitFor) and Start (room.js aiStart) pick the same context and need the same
// memory for Qwen3.8 Flash at every context choice (the #386 lesson: the picker sized Flash from SHAPE, Start from the
// header, two different ways, and a room the picker said fits failed at Start). The picker's inputs come from SHAPE
// (roomCosts), Start's from the real header (tests/fixtures/q38flash), exactly as the two call roomCtxPlan.
//   deno test --allow-read tests/unit/ctxpick_test.js
import { parseGGUFHeader, qwen4expConfig, qwen4expShardBytes } from "../../engine/gguf.js";
import { mergeSplitHeaders, flashLayerCost, flashLongMeta, expertsOf, CTX, FLASH_CTX_OPTIONS, FLASH_CTX_PROVEN, maxSeqFor } from "../../room/models.js";
import { minFitPledge, gbUp } from "../../room/plan.js";
import { pledgeGB, dealPledgeBytes } from "../../room/pledge.js";
import { roomCtxPlan, roomCosts, hostMarginFor, ctxOptions, fitAtCtx, prefillSeconds, prefillMode, durationWords } from "../../room/ctxpick.js";
import { readFixture, ggufFromFixture } from "../helpers/gguf_fixture.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const GB = 2 ** 30, F = "qwen3.8-flash";
const FIX = readFixture("q38flash");
const G = mergeSplitHeaders(FIX.files.slice(0, 3).map((f) => parseGGUFHeader(ggufFromFixture(f), { skipTokenizer: true })));
const L = qwen4expConfig(G.meta).L;
// Start's costs, as room.js aiStart builds them from the header: layerAt(c, kv), the embedding and the head, the experts
const startCosts = (() => {
  const embed = qwen4expShardBytes(G, { lo: 0, hi: 0, hasEmbed: true, hasHead: true, qsa: false }), ex = expertsOf(G, L);
  return (c, kv) => ({ L, layers: flashLayerCost(G, c, { kv: kv === "f32" ? "f32" : "f16" }), hostBytes: embed, experts: ex });
})();

const long = flashLongMeta(128);   // a build that loads Flash with QSA in f16 (WebGPU's 128 MiB binding: 128K)
const tab = (gb, extra = {}) => ({ webgpu: true, contribGB: gb, flashLong: long, tabDisk: { ramGB: 4, models: [F] }, ...extra });
const pc = (gb, ram) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram, offloadFlash: true, native: "node-dawn", flashLong: long });
const spark = (gb) => ({ webgpu: true, contribGB: gb, ngram: true, native: "node-dawn", flashLong: flashLongMeta(4096) });
const ROOMS = {
  "Mac tab 22 + PC 12 (the demo)": [tab(22), pc(12, 24)],
  "Mac tab 22 + PC 10": [tab(22), pc(10, 24)],
  "Mac tab 22 alone": [tab(22)],
  "two Sparks 40 / 40": [spark(40), spark(40)],
  "two Sparks 34 / 34": [spark(34), spark(34)],
  "a Spark and an older laptop": [spark(64), { webgpu: true, contribGB: 16 }],
  "f32-only build": [spark(40), { ...spark(40), flashLong: { ctx: 65536, kv: ["f32"], batch: 1 } }],
};
const pledgesOf = (metas) => metas.map((m) => dealPledgeBytes(m, Infinity, F));

Deno.test("ctxpick: the picker and Start pick the same context, format and fit, for every choice (#386)", () => {
  for (const [name, metas] of Object.entries(ROOMS)) for (const choice of [0, ...FLASH_CTX_OPTIONS]) {
    const pledges = pledgesOf(metas);
    const picker = roomCtxPlan(F, { metas, pledges, choice, margin: hostMarginFor(F, metas[0]) });
    const start = roomCtxPlan(F, { metas, pledges, choice, costs: startCosts, margin: hostMarginFor(F, metas[0], G) });
    const at = `${name}, choice ${choice}`;
    eq([start.ctx, start.kv, start.cap, start.fits, start.fellBack], [picker.ctx, picker.kv, picker.cap, picker.fits, picker.fellBack], at);
    const pf = picker.fitAt(picker.ctx).fit, sf = start.fitAt(start.ctx).fit;
    eq([sf.fits, sf.caps, !!sf.offload, !!sf.disk], [pf.fits, pf.caps, !!pf.offload, !!pf.disk], `${at}: the fit`);
    // the need the picker and the pool show (minFitPledge at the picked context) is what Start's fit needs
    const need = (p) => minFitPledge(pledges, (pl) => !!p.fitAt(p.ctx, pl).fit.fits);
    const np = need(picker), ns = need(start);
    eq(np && gbUp(np.bytes), ns && gbUp(ns.bytes), `${at}: the need`);
    // and Start's own deal (its split options) deals within every pledge when the picker said it fits
    if (pf.fits) {
      const d = start.fitAt(start.ctx, pledges, { mode: "memory", ngram: null }).deal;
      ok(d.fit.fits, `${at}: Start's deal fits`);
      d.used.forEach((i, k) => ok(d.held[k] <= pledges[i] + 1, `${at}: device ${i} holds ${d.held[k] / GB} of ${pledges[i] / GB}`));
    }
  }
});

Deno.test("ctxpick: the picker's context choices say what each needs here, and why one is off", () => {
  for (const [name, metas] of Object.entries(ROOMS)) {
    const pledges = pledgesOf(metas), margin = hostMarginFor(F, metas[0]);
    const opts = ctxOptions(F, { metas, pledges, margin });
    eq(opts.map((o) => o.ctx), FLASH_CTX_OPTIONS, name);
    for (const o of opts) {
      // its need is the one Start would have at that choice
      const s = roomCtxPlan(F, { metas, pledges, choice: o.ctx, costs: startCosts, margin: hostMarginFor(F, metas[0], G) });
      if (o.capped) { ok(!o.fits && /older Pooled/.test(o.why), `${name} ${o.label}`); continue; }
      const m = minFitPledge(pledges, (pl) => !!fitAtCtx(F, { c: o.ctx, kv: s.kv, pledges: pl, metas, costs: startCosts, margin: hostMarginFor(F, metas[0], G) }).fit.fits);
      eq(o.needGB, m ? gbUp(m.bytes) : null, `${name} ${o.label}: need`);
      // an option that fits is the context Start opens at when it's chosen
      if (o.fits) eq(s.ctx, o.ctx, `${name} ${o.label}: Start opens there`);
      else ok(o.shortGB > 0 && o.why === `needs ${o.shortGB} GB more` || o.needGB == null, `${name} ${o.label}: ${o.why}`);
    }
  }
  // the demo room holds every proven length (and 128K too, by memory: the tab's checkpoint copy fits its 4 GB of RAM);
  // two Sparks lending 34 GB hold none of them
  const demo = ctxOptions(F, { metas: ROOMS["Mac tab 22 + PC 12 (the demo)"], pledges: pledgesOf(ROOMS["Mac tab 22 + PC 12 (the demo)"]), margin: hostMarginFor(F, ROOMS["Mac tab 22 + PC 12 (the demo)"][0]) });
  ok(demo.every((o) => o.fits), JSON.stringify(demo));
  const tight = ctxOptions(F, { metas: ROOMS["two Sparks 34 / 34"], pledges: pledgesOf(ROOMS["two Sparks 34 / 34"]) });
  ok(tight.every((o) => !o.fits && o.shortGB > 0), JSON.stringify(tight));
  // a device that loads no QSA (an older build) caps the room at 2048: every choice is off, and says why
  const old = ctxOptions(F, { metas: ROOMS["a Spark and an older laptop"], pledges: pledgesOf(ROOMS["a Spark and an older laptop"]) });
  ok(old.every((o) => o.capped), JSON.stringify(old));
});

Deno.test("ctxpick: no choice opens at the longest proven context the room holds; a choice is the most it opens at", () => {
  eq(maxSeqFor(F), Math.max(...FLASH_CTX_PROVEN));
  const metas = ROOMS["two Sparks 40 / 40"], pledges = pledgesOf(metas);
  const p = roomCtxPlan(F, { metas, pledges });
  ok(p.fits && FLASH_CTX_PROVEN.includes(p.ctx));
  for (const c of FLASH_CTX_OPTIONS) ok(roomCtxPlan(F, { metas, pledges, choice: c }).ctx <= c);
  // a room that grew short since the choice walks the ladder down (and says so: fellBack), never past the choice
  const top = Math.max(...FLASH_CTX_PROVEN), small = [spark(35), spark(35)], q = roomCtxPlan(F, { metas: small, pledges: pledgesOf(small), choice: top });
  ok(q.fits && q.ctx < top && q.fellBack, JSON.stringify([q.ctx, q.fits]));
  // ?ctx= (an explicit ask) is tried alone, and held to the longest proven context
  eq(roomCtxPlan(F, { metas: small, pledges: pledgesOf(small), ask: top }).ctx, top);
  eq(roomCtxPlan(F, { metas: small, pledges: pledgesOf(small), ask: 4 * top }).ctx, top);
  eq(CTX[F].ladder, FLASH_CTX_PROVEN.slice().sort((a, b) => b - a));
});

Deno.test("ctxpick: prefill estimates in words", () => {
  ok(prefillSeconds(F, 8192) > 0 && prefillSeconds(F, 131072) > prefillSeconds(F, 32768));
  // a disk room's layered passes: slower than every device batching, faster than stepping a column at a time
  ok(prefillSeconds(F, 8192, { mode: "perToken" }) > prefillSeconds(F, 8192, { mode: "disk" }) && prefillSeconds(F, 8192, { mode: "disk" }) > prefillSeconds(F, 8192));
  // the tab's measurements: 1887 tokens in 51 s, 7663 in 134 s
  // (the Mac tab alone: 7663 tokens in 45.7 s, 15718 in 99.2 s)
  ok(Math.abs(prefillSeconds(F, 7663, { mode: "disk" }) - 45.7) < 5 && Math.abs(prefillSeconds(F, 15718, { mode: "disk" }) - 99.2) < 5);
  eq([prefillMode({ fits: true }), prefillMode({ offload: true }), prefillMode({ offload: true, disk: true })], ["batch", "perToken", "disk"]);
  eq(prefillSeconds("qwen3.8-27b", 8192), null);
  eq([durationWords(40), durationWords(600), durationWords(7200)], ["about 40 s", "about 10 min", "about 2 h"]);
});

Deno.test("ctxpick: the demo room (Mac tab 22 + PC 12) holds 128K by the memory model, for when 128K is proven", () => {
  // FLASH_CTX_PROVEN stops at 64K (models.js: the 128K goldens differ from llama.cpp at one step); this keeps the memory
  // side ready: f16 K / V, the tab's 4 GB disk-tier RAM, the 2-slot checkpoint reserve
  const metas = ROOMS["Mac tab 22 + PC 12 (the demo)"], pledges = pledgesOf(metas);
  const f = fitAtCtx(F, { c: 131072, kv: "f16", pledges, metas, costs: roomCosts(F), margin: hostMarginFor(F, metas[0]) });
  ok(f.fit.fits && f.fit.disk, JSON.stringify(f.fit));
  const m = minFitPledge(pledges, (pl) => !!fitAtCtx(F, { c: 131072, kv: "f16", pledges: pl, metas, costs: roomCosts(F), margin: hostMarginFor(F, metas[0]) }).fit.fits);
  ok(m && m.bytes / GB < 22, `${m && m.bytes / GB} GB`);
  // with the old 3 GB of tab RAM it did not (its checkpoint copy of ~1.6 GB on top of the process's 1.5 GB)
  const old = [tab(22, { tabDisk: { ramGB: 3, models: [F] } }), metas[1]];
  ok(!fitAtCtx(F, { c: 131072, kv: "f16", pledges, metas: old, costs: roomCosts(F), margin: hostMarginFor(F, old[0]) }).fit.fits);
});

// ---- the 27B and the 35B MoE: a ladder too (room/models.js CTX ladder / options), the default past CTX def only at full speed
import { SHAPE, roomBytes, kvBytesPerLayerPos, hostHeldBytes, ctxTop, NEED_GB, NEED_MIN_GB, needText } from "../../room/models.js";
import { isPhoneMeta } from "../../room/plan.js";
const Q35 = ["qwen3.8-27b", "qwen3.6-35b-moe"];
// Start's costs for a Qwen 3.5 model, as room.js aiStart builds them from the header (layers 0-3's average weights plus
// the attention layers' K / V at the context, kvBytesPerLayerPos on the header's meta; the embedding, the head and the
// draft block on the host): here from SHAPE's header numbers and a meta with the model's attention shape
const qMeta = (m) => ({ "qwen35.attention.head_count_kv": m === "qwen3.8-27b" ? 4 : 2, "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4 });
const q35Start = (m) => {
  const s = SHAPE[m], host = hostHeldBytes("qwen35", { embed: s.embed, out: s.out, mtp: s.mtp }), ex = roomBytes(m, 8192).experts;
  return (c, kv) => ({ L: s.L, layers: s.layer + c * kvBytesPerLayerPos(qMeta(m), kv), hostBytes: host, experts: ex });
};
const t = (gb, x = {}) => ({ webgpu: true, contribGB: gb, ...x });
const iphone = () => ({ webgpu: true, contribGB: 1, ua: "iPhone" });
const rtx = (gb, ram) => ({ webgpu: true, contribGB: gb, native: "node-dawn", ...(ram ? { offload: true, ramGB: ram } : {}) });
const QROOMS = {
  "Mac 16 + desktop 7 + iPhone (24 GB)": [t(16), t(7), iphone()],
  "Mac 17 + desktop 6 + iPhone (24 GB)": [t(17), t(6), iphone()],
  "Mac 14 + desktop 8 + iPhone (23 GB)": [t(14), t(8), iphone()],
  "Mac 18 + desktop 4 (22 GB)": [t(18), t(4)],
  "two 16 GB laptops": [t(16), t(16)],
  "a GB10 lending 40": [rtx(40)],
  "an RTX 12 GB with 48 GB of RAM": [rtx(12, 48)],
  "an RTX 12 GB with RAM + a Mac 12": [rtx(12, 48), t(12)],
  "a 16 GB laptop alone": [t(16)],
  "a 20 GB laptop alone": [t(20)],
};
const qPledges = (metas, m) => metas.map((x) => dealPledgeBytes(x, Infinity, m));

Deno.test("ctxpick (27B, MoE): the picker and Start pick the same context and fit, for every room and every choice", () => {
  eq(CTX["qwen3.8-27b"].options, [8192, 16384, 32768, 65536]);
  eq(CTX["qwen3.6-35b-moe"].options, [8192, 32768, 65536, 131072]);
  for (const m of Q35) for (const [name, metas] of Object.entries(QROOMS)) for (const choice of [0, ...CTX[m].options]) {
    const pledges = qPledges(metas, m), at = `${m}, ${name}, choice ${choice}`;
    const picker = roomCtxPlan(m, { metas, pledges, choice }), start = roomCtxPlan(m, { metas, pledges, choice, costs: q35Start(m) });
    eq([start.ctx, start.want, start.kv, start.fits, start.fellBack], [picker.ctx, picker.want, picker.kv, picker.fits, picker.fellBack], at);
    ok(CTX[m].ladder.includes(picker.ctx), `${at}: ${picker.ctx} is on the ladder`);
    if (choice) ok(picker.ctx <= choice, `${at}: never past the choice`);
    const pf = picker.fitAt(picker.ctx).fit, sf = start.fitAt(start.ctx).fit;
    eq([sf.fits, sf.caps, !!sf.offload], [pf.fits, pf.caps, !!pf.offload], `${at}: the fit`);
    const need = (p) => minFitPledge(pledges, (pl) => !!p.fitAt(p.ctx, pl).fit.fits);
    const np = need(picker), ns = need(start);
    eq(np && gbUp(np.bytes), ns && gbUp(ns.bytes), `${at}: the need`);
    if (pf.fits) {
      const d = start.fitAt(start.ctx, pledges, { mode: "memory", phone: metas.map(isPhoneMeta) }).deal;
      ok(d.fit.fits, `${at}: Start's deal fits`);
      d.used.forEach((i, k) => ok(d.held[k] <= pledges[i] + 1, `${at}: device ${i} holds ${d.held[k] / GB} of ${pledges[i] / GB}`));
    }
  }
});

Deno.test("ctxpick (27B, MoE): each context choice's need is Start's at that choice; one that fits is where Start opens", () => {
  for (const m of Q35) for (const [name, metas] of Object.entries(QROOMS)) {
    const pledges = qPledges(metas, m), opts = ctxOptions(m, { metas, pledges });
    eq(opts.map((o) => o.ctx), CTX[m].options, name);
    for (const o of opts) {
      const s = roomCtxPlan(m, { metas, pledges, choice: o.ctx, costs: q35Start(m) });
      const mm = minFitPledge(pledges, (pl) => !!fitAtCtx(m, { c: o.ctx, kv: s.kv, pledges: pl, metas, costs: q35Start(m) }).fit.fits);
      eq(o.needGB, mm ? gbUp(mm.bytes) : null, `${m} ${name} ${o.label}: need`);
      ok(!o.capped, `${m} ${name} ${o.label}: no device caps a qwen35 room in the picker`);
      if (o.fits) eq(s.ctx, o.ctx, `${m} ${name} ${o.label}: Start opens there`);
      else ok(!s.fits || s.ctx < o.ctx, `${m} ${name} ${o.label}`);
    }
  }
});

Deno.test("ctxpick (27B, MoE): no choice opens at the longest step held at full speed, never below where rooms opened before", () => {
  const plan = (m, name) => { const metas = QROOMS[name]; return roomCtxPlan(m, { metas, pledges: qPledges(metas, m) }); };
  const M = "qwen3.6-35b-moe", D = "qwen3.8-27b";
  eq([ctxTop(M), ctxTop(D), maxSeqFor(M), maxSeqFor(D)], [131072, 65536, 32768, 16384]);
  // Mac + desktop + iPhone, 24 GB: the computers hold the MoE at 128K (the phone stays out), so it opens there
  const a = plan(M, "Mac 16 + desktop 7 + iPhone (24 GB)");
  eq([a.ctx, a.fits, a.fellBack, a.want], [131072, true, false, 32768]);
  eq(a.fitAt(a.ctx, qPledges(QROOMS["Mac 16 + desktop 7 + iPhone (24 GB)"], M), { phone: [false, false, true] }).deal.used, [0, 1]);
  // the same 24 GB split 17 / 6: 128K would need the phone to hold layers (a slow hop every token): 64K, 128K a click away
  const b = plan(M, "Mac 17 + desktop 6 + iPhone (24 GB)");
  eq([b.ctx, b.fellBack], [65536, false]);
  const bm = QROOMS["Mac 17 + desktop 6 + iPhone (24 GB)"], b128 = roomCtxPlan(M, { metas: bm, pledges: qPledges(bm, M), choice: 131072 });
  eq(b128.ctx, 131072, "chosen in the picker: 128K");
  ok(b128.fitAt(131072, qPledges(bm, M), { phone: bm.map(isPhoneMeta) }).deal.used.includes(2), "with the phone holding layers");
  // a GPU that holds the MoE only with experts in RAM keeps the default: a longer context would take expert cache
  const o = plan(M, "an RTX 12 GB with 48 GB of RAM");
  eq([o.ctx, o.fits, !!o.fitAt(o.ctx).fit.offload], [32768, true, true]);
  // the 27B: 64K wherever the same devices hold it; a 16 GB laptop alone holds no step (17 GB at 16K, 16.5 at 8K)
  eq(plan(D, "Mac 16 + desktop 7 + iPhone (24 GB)").ctx, 65536);
  eq(plan(D, "a 20 GB laptop alone").ctx, 65536);
  const lone = plan(D, "a 16 GB laptop alone");
  eq([lone.ctx, lone.fits], [8192, false], "short everywhere: short for the ladder's last step");
  // every room: where it opened before the ladder (CTX def), it opens at that or longer, the same fit
  for (const m of Q35) for (const [name, metas] of Object.entries(QROOMS)) {
    const pledges = qPledges(metas, m), p = roomCtxPlan(m, { metas, pledges });
    const before = p.fitAt(CTX[m].def).fit;
    if (before.fits) ok(p.fits && p.ctx >= CTX[m].def && !p.fellBack, `${m} ${name}: ${p.ctx}`);
    // past the default only with every expert on a GPU and no more devices than at the default
    if (p.ctx > CTX[m].def) {
      const used = (c) => p.fitAt(c, pledges, { phone: metas.map(isPhoneMeta) }).deal;
      ok(!used(p.ctx).fit.offload && used(p.ctx).used.length <= used(CTX[m].def).used.length, `${m} ${name}`);
    }
  }
  // ?ctx= stays as asked (tried alone)
  eq(roomCtxPlan(M, { metas: QROOMS["a 16 GB laptop alone"], pledges: qPledges(QROOMS["a 16 GB laptop alone"], M), ask: 131072 }).ctx, 131072);
  // the picker's need words: the default's need, and the ladder's last step's
  eq(needText(D), `${NEED_GB[D]} GB (${NEED_MIN_GB[D]} GB at 8K)`);
  ok(NEED_MIN_GB[M] < NEED_GB[M]);
});
