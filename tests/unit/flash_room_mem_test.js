// Qwen3.8-Flash-Next's memory in a room, one model for every caller (room/models.js flashRoomMem, flashLayerCost,
// flashCkptReserve; room/plan.js costOf / dealRoom layerCost): the picker's numbers (SHAPE, no header) equal the
// real header's (tests/fixtures/q38flash) and engine/qwen4exp_mem.js's; a deal on per-layer costs cuts ranges on their
// sums (QSA skew, the heavy layers 0-5) with the n-gram table and expert offload in the room; the checkpoint reserve
// is counted once on a device that offloads; the picker's Flash row at 2048 and 8K. No GPU.
//   deno test --allow-read tests/unit/flash_room_mem_test.js
import { qwen4expMemory } from "../../engine/qwen4exp_mem.js";
import { parseGGUFHeader, qwen4expConfig, qwen4expShardBytes } from "../../engine/gguf.js";
import { SHAPE, NEED_GB, CTX, roomBytes, expertBytesOf, ngramOf, mergeSplitHeaders, flashCfgOf, flashRoomMem, flashLayerCost, flashCkptReserve, flashTabHostMargin, FLASH_SCRATCH,
  ckptSlotsAt, CKPT_RESERVE_SLOTS, CKPT_TOKEN_CAP, FLASH_DENSE_CTX, FLASH_ROOM_QSA, FLASH_CKPT_TOKENS } from "../../room/models.js";
import { costOf, layersTotal, layerCaps, roomFit, dealRoom, minFitPledge, offloadSpareOf, OFFLOAD_CKPT_SLOTS, dealWithHostMargin } from "../../room/plan.js";
import { offloadFor, ngramFor, pledgeGB } from "../../room/pledge.js";
import { readFixture, ggufFromFixture } from "../helpers/gguf_fixture.js";

const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const GB = 2 ** 30, MiB = 2 ** 20, F = "qwen3.8-flash";
const FIX = readFixture("q38flash");
const header = () => mergeSplitHeaders(FIX.files.slice(0, 3).map((f) => parseGGUFHeader(ggufFromFixture(f), { skipTokenizer: true })));
const G = header(), cfg = qwen4expConfig(G.meta);
const sum = (a, lo = 0, hi = a.length) => { let s = 0; for (let i = lo; i < hi; i++) s += a[i]; return s; };

Deno.test("flash mem: SHAPE's per-layer weights and indexer are the header's (the picker sizes without one)", () => {
  const S = SHAPE[F];
  eq(S.layers.length, 48);
  for (let i = 0; i < 48; i++) {
    const o = { lo: i, hi: i + 1, hasEmbed: false, hasHead: false };
    eq(S.layers[i], qwen4expShardBytes(G, { ...o, qsa: false }), `layer ${i}`);
    eq(qwen4expShardBytes(G, { ...o, qsa: true }) - S.layers[i], cfg.qsa[i] ? S.indexer : 0, `layer ${i}'s indexer`);
    eq(flashLayerCost(F, 2048).experts[i], expertBytesOf(G, i, i + 1), `layer ${i}'s experts`);
  }
  eq(flashLayerCost(G, 2048).experts[47], 1415577600);
  // the heavy front: layers 0-5 (Q4_1 down experts), layer 1 the heaviest (the PLE block)
  ok(Math.min(...S.layers.slice(0, 6)) > Math.max(...S.layers.slice(6)), "layers 0-5 outweigh every later layer");
  eq(S.layers.indexOf(Math.max(...S.layers)), 1);
  // flashCfgOf: qwen4expConfig's fields that qwen4expMemory reads
  const c = flashCfgOf(S.flash, S.hDim);
  eq([c.L, c.hDim, c.nKVHeads, c.headDim, c.qsa, c.ssm, c.ple.layers, c.ple.ngram, c.ple.convK],
    [cfg.L, cfg.hDim, cfg.nKVHeads, cfg.headDim, cfg.qsa.slice(0, cfg.L), cfg.ssm, cfg.ple.layers, cfg.ple.ngram, cfg.ple.convK]);
});

Deno.test("flash mem: flashRoomMem is qwen4expMemory on the real config, from SHAPE or from the header", () => {
  for (const ctx of [2048, 8192, 32768, 65536]) for (const kv of ["f32", "f16"]) for (const qsa of [false, true])
    for (const range of [null, [0, 3], [3, 4], [0, 24], [24, 48], [1, 2]]) {
      const want = qwen4expMemory(cfg, { ctx, kv, qsa, layerRange: range });
      for (const src of [F, SHAPE[F], G]) {
        const m = flashRoomMem(src, ctx, { kv, qsa, range });
        eq([m.state, m.total, m.perToken, m.bytes], [want.state, want.total, want.perToken, want.bytes], `${ctx} ${kv} qsa ${qsa} ${range}`);
        eq([m.reserve, m.need], [0, want.total]);
      }
      const r = flashRoomMem(F, ctx, { kv, qsa, range, ckptSlots: 2 });
      eq([r.reserve, r.need], [2 * want.state, want.total + 2 * want.state]);
      eq(flashCkptReserve(F, range, ctx, 2, { kv, qsa }), 2 * want.state);
    }
  // the rooms' default: QSA at every context, 2048 included (FLASH_ROOM_QSA: the rooms load it there, ai-load.flash)
  ok(FLASH_ROOM_QSA && flashRoomMem(F, 2048).bytes.rawK > 0 && flashRoomMem(F, 4096).bytes.rawK > 0);
});

Deno.test("flash mem: flashLayerCost, each layer its own weights + state + reserve; SHAPE == header", () => {
  for (const ctx of [2048, 8192, 32768]) {
    const a = flashLayerCost(F, ctx), b = flashLayerCost(G, ctx);
    eq(Array.from(a), Array.from(b), `${ctx}: the picker's costs are the host's`);
    eq([a.ctx, a.slots], [ctx, ckptSlotsAt(ctx)]);
    const qsa = FLASH_ROOM_QSA;
    for (let i = 0; i < 48; i++) {
      const m = qwen4expMemory(cfg, { ctx, qsa, layerRange: [i, i + 1] });
      eq(a[i], SHAPE[F].layers[i] + (qsa && cfg.qsa[i] ? SHAPE[F].indexer : 0) + m.state + a.slots * m.state, `${ctx} layer ${i}`);
      eq([a.onceAt[i], a.once], qsa && cfg.qsa[i] ? [1, m.bytes.scores] : [0, a.once], `${ctx} layer ${i}: the score scratch, once`);
    }
    ok(Math.abs(a.total - sum(a) - a.once) < 1);
    ok(qsa ? a.once > 0 : a.once === 0);
    // a range's state and reserve are flashRoomMem's (the load check's) for it, exactly; its cost (costOf) is its
    // weights + flashRoomMem's need with the reserve: the score scratch once per device, not once per QSA layer
    const C = costOf(a);
    for (const [lo, hi] of [[0, 24], [24, 48], [0, 3], [3, 4], [1, 48], [6, 30]]) {
      eq(sum(a.state, lo, hi), flashRoomMem(F, ctx, { range: [lo, hi] }).state, `${ctx} [${lo}, ${hi})`);
      eq(a.slots * sum(a.state, lo, hi), flashCkptReserve(F, [lo, hi], ctx, a.slots));
      ok(Math.abs(C.span(lo, hi) - sum(a.weights, lo, hi) - flashRoomMem(F, ctx, { range: [lo, hi], ckptSlots: a.slots }).need) < 1, `${ctx} [${lo}, ${hi}): the planner's range is the load check's`);
    }
  }
  // no reserve at the dense 2048 (deals there as before), two slots past it
  eq([ckptSlotsAt(2048), ckptSlotsAt(4096), CKPT_RESERVE_SLOTS, CKPT_TOKEN_CAP(32768)], [0, 2, 2, 65536]);
  // QSA skew at 32K: a range of three DeltaNet layers [0, 3) vs one with the QSA layer [3, 4)
  const c32 = flashLayerCost(F, 32768);
  ok(c32.state[3] > 13 * sum(c32.state, 0, 3), `${c32.state[3] / MiB} MiB vs ${sum(c32.state, 0, 3) / MiB}`);
  ok(c32[3] - c32[4] > 350 * MiB, "the QSA layer with K / V, indexer and a 2-slot reserve is ~0.4 GB heavier at 32K");
});

Deno.test("flash plan: costOf, a number gives one size's arithmetic, per-layer bytes their sums", () => {
  const u = costOf(1.5 * GB);
  eq([u.at(10), u.span(3, 7), u.fitN(10 * GB), u.avg, u.resident(10, 4, null, GB)], [15 * GB, 6 * GB, 6, 1.5 * GB, 11 * GB]);
  const c = costOf([3, 1, 1, 1, 2, 1].map((x) => x * GB));
  eq([c.span(0, 6) / GB, c.at(2, 6) / GB, c.at(2, { lo: 0, fixed: true }) / GB, c.at(2, { lo: 1 }) / GB, c.at(2) / GB], [9, 3, 4, 3, 4]);
  eq([c.fitN(4 * GB, { lo: 0, fixed: true }), c.fitN(4 * GB, { lo: 1 }), c.fitN(100 * GB, { lo: 1 })], [2, 3, 5], "the heaviest window from lo");
  eq(layersTotal([1, 2, 3], 3), 6);
  // offloaded layers take their own experts off the GPU when the costs carry them, E each otherwise
  const e = Object.assign([3, 1, 1, 1, 2, 1].map((x) => x * GB), { experts: [2, 0.5, 0.5, 0.5, 1.5, 0.5].map((x) => x * GB) });
  const ce = costOf(e);
  eq([ce.resident(3, 2, 6) / GB, ce.resident(2, 1, { lo: 0, fixed: true }) / GB, ce.resident(2, 1, { lo: 1 }) / GB, c.resident(2, 1, 6, GB) / GB], [2, 3.5, 2.5, 2]);
  eq(layersTotal(2, 3), 6);
  // the host from layer 0 exactly, any other device the heaviest range it could be dealt (from layer 1)
  eq(layerCaps([5 * GB, 4 * GB], [3, 1, 1, 1, 2, 1].map((x) => x * GB), 0), [3, 3]);
  // models without per-layer costs are unchanged: a uniform array deals what the number does
  const pl = [40, 30, 12].map((g) => g * GB), lb = 1.48 * GB, arr = new Float64Array(48).fill(lb);
  for (const mode of ["memory", "speed"]) {
    const a = dealRoom({ L: 48, layerBytes: lb, hostBytes: 0.9 * GB, pledges: pl, mode });
    const b = dealRoom({ L: 48, layerCost: arr, hostBytes: 0.9 * GB, pledges: pl, mode });
    eq([b.ranges, b.used, b.held.map(Math.round)], [a.ranges, a.used, a.held.map(Math.round)], mode);
  }
});

// a room of nodes / a tab / a PC with expert offload, host first (room/pledge.js metas)
const node = (gb, extra = {}) => ({ webgpu: true, contribGB: gb, ngram: true, ...extra });
const pc = (gb, ram) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram, offloadFlash: true });
function deal(metas, ctx, mode = "memory") {
  const rb = roomBytes(F, ctx), pledges = metas.map((m) => pledgeGB(m, Infinity, F) * GB);
  const d = dealRoom({ L: 48, layerCost: rb.layerCost, hostBytes: rb.hostBytes, pledges, mode, off: offloadFor(metas, rb.experts, F), ngram: ngramFor(metas, ngramOf(F)) });
  return { rb, d, pledges };
}
// every device within its pledge, on the exact sums: resident ranges hold their costs; an offloading device its pledge
function checkHeld({ rb, d, pledges }) {
  d.used.forEach((i, k) => {
    const [lo, hi] = d.ranges[k], host = i === 0 ? rb.hostBytes : 0;
    if (!d.offload?.[k]) eq(Math.round(d.held[k]), Math.round(costOf(rb.layerCost).span(lo, hi) + host), `device ${i} holds its range's exact sum`);
    ok(d.held[k] <= pledges[i] + 1, `device ${i}: ${d.held[k] / GB} of ${pledges[i] / GB} GB`);
  });
}

Deno.test("flash plan: dealRoom with layerCost, the n-gram holder and an offloading device in the room", () => {
  for (const ctx of [2048, 8192, 32768]) {
    // a Spark with the table + an RTX 5070 PC that parks experts: layer 1 stays on the table holder (the host)
    const a = deal([node(40), pc(10, 64)], ctx);
    ok(a.d.fit.fits && a.d.offload?.[1] && !a.d.offload[0], `${ctx}: the PC offloads: ${JSON.stringify(a.d.fit)}`);
    eq([a.d.ngram.at, a.d.ngram.device], [0, 0]);
    checkHeld(a);
    // the host has no disk for the table, the worker has it: the boundary moves to it (layer 1 on the worker), and the
    // moved ranges are still within both pledges on their exact sums
    const b = deal([{ webgpu: true, contribGB: 40, diskGB: 1 }, node(100)], ctx);
    ok(b.d.fit.fits, `${ctx}: ${JSON.stringify(b.d.fit)}`);
    eq([b.d.ranges[0], b.d.ngram.device], [[0, 1], 1]);
    checkHeld(b);
    // three devices by memory: no device past its pledge on the exact sums
    checkHeld(deal([node(48), node(40), node(10)], ctx));
    checkHeld(deal([node(40), node(40)], ctx, "speed"));
  }
});

Deno.test("flash plan: an offloading tab's pledge holds on the exact sums: each offloaded layer frees its own experts", () => {
  for (const ctx of [2048, 8192]) for (const gb of [12, 22]) {
    const tab = { webgpu: true, contribGB: gb, tabDisk: { ramGB: 3, models: [F] } };
    const { rb, d } = deal([tab], ctx), o = d.offload[0], lc = rb.layerCost;
    ok(d.fit.fits && o?.diskOnly, `${ctx} ${gb}: ${JSON.stringify(d.fit)}`);
    // resident: every layer's cost less the experts of the offloaded ones [lo, hi) (layers 6-47: 52 MB fewer than E),
    // the host's part, then the pools and the spare: within the pledge
    const res = sum(lc) - sum(lc.experts, o.lo, o.hi) + rb.hostBytes;
    ok(res + o.vramBytes + o.vramSpare <= gb * GB + 1, `${ctx} ${gb}: ${(res + o.vramBytes + o.vramSpare) / GB} GB of ${gb}`);
    ok(res - (sum(lc) - (o.hi - o.lo) * rb.experts.E + rb.hostBytes) > 2 * GB * (o.hi - o.lo) / 48, "E for every layer would undercount by ~2.2 GB");
  }
});

Deno.test("flash plan: the heavy layers and the QSA skew decide what fits, not their average", () => {
  const rb = roomBytes(F, 32768), lc = rb.layerCost, C = costOf(lc);
  // the host's first 24 layers (0-5 heavy, layer 1 with the PLE block) cost ~0.14 GB more than 24 averages
  ok(C.span(0, 24) - 24 * rb.layerBytes > 0.1 * GB, `${(C.span(0, 24) - 24 * rb.layerBytes) / GB}`);
  // a pledge of exactly the host's part + 24 averages holds 23 layers, not 24
  eq(layerCaps([rb.hostBytes + 24 * rb.layerBytes], lc, rb.hostBytes), [23]);
  eq(layerCaps([rb.hostBytes + C.span(0, 24)], lc, rb.hostBytes), [24]);
  // a worker's cap holds for the heaviest range it could get: 3 layers somewhere from layer 1 include a QSA layer
  ok(C.at(3, { lo: 1 }) >= C.span(1, 4) && C.span(1, 4) > C.span(4, 7) - 1, "[1, 4) holds the QSA layer 3 and the heavy layers");
});

Deno.test("flash plan: the checkpoint reserve is counted once on a device that offloads", () => {
  const p = roomBytes(F, 2048).experts, n = 20, pos = 48;   // the PC's last 20 layers [28, 48)
  // at 2048: nothing reserved in the layers' costs, OFFLOAD_CKPT_SLOTS of 2048 positions in the offload spare (as
  // before: the averaged formula's number within a percent), exactly flashCkptReserve of its range
  // (a dense room's costs, qsa false, for the comparison with the averaged formula, which has no QSA)
  const c2 = flashLayerCost(F, 2048, { qsa: false }), C2 = costOf(c2);
  eq(c2.slots, 0);
  const sp = offloadSpareOf(p, n, C2, pos), old = offloadSpareOf(p, n);
  const ck = flashCkptReserve(F, [28, 48], 2048, OFFLOAD_CKPT_SLOTS, { qsa: false });
  eq(Math.round(sp.vram), Math.round(p.work + n * p.state + ck));
  ok(Math.abs(sp.vram - old.vram) / old.vram < 0.01 && Math.abs(sp.ram - old.ram) / old.ram < 0.01, `${sp.vram} vs ${old.vram}, ${sp.ram} vs ${old.ram}`);
  eq(old.vram, p.work + n * (p.state * (1 + OFFLOAD_CKPT_SLOTS) + p.kvPos * FLASH_CKPT_TOKENS), "without per-layer costs: today's formula");
  // past 2048: the two reserved slots are in the layers' costs, so the spare adds no checkpoint VRAM; its RAM keeps one
  // checkpoint's copy of the range at the room's context (CKPT_TOKEN_CAP / 2 positions)
  const c8 = flashLayerCost(F, 8192), C8 = costOf(c8), sp8 = offloadSpareOf(p, n, C8, pos);
  eq(c8.slots, 2);
  eq(Math.round(sp8.vram), Math.round(p.work + n * p.state));
  eq(Math.round(sp8.ram - 1.5 * GB), Math.round(flashRoomMem(F, 8192, { range: [28, 48] }).state));
  eq(Math.round(C8.span(28, 48) - sum(c8.weights, 28, 48) - sum(c8.state, 28, 48)), Math.round(flashCkptReserve(F, [28, 48], 8192) + flashRoomMem(F, 8192, { range: [31, 32] }).bytes.scores),
    "the layers carry their 2-slot reserve (and the range the score scratch, once)");
});

Deno.test("flash picker: the row at 2048 and 8K from the planner's costs (roomFit, minFitPledge, NEED_GB)", () => {
  // NEED_GB is the planner's bytes at the default context, rounded up to a tenth
  const def = roomBytes(F, CTX[F].def), needDef = (def.L * def.layerBytes + def.hostBytes) / GB;
  ok(NEED_GB[F] >= needDef && NEED_GB[F] - needDef < 0.1, `${NEED_GB[F]} vs ${needDef}`);
  // roomBytes is memoized per context (the picker's ~24 minFitPledge probes reuse one cost array)
  // (roomBytes' kv is the room's format: f16 by default, flashRoomLimits)
  ok(roomBytes(F, 8192) === roomBytes(F, 8192) && roomBytes(F, 8192).layerCost === flashLayerCost(F, 8192, { kv: "f16" }));
  const fitAt = (ctx, pl, metas) => { const rb = roomBytes(F, ctx); return roomFit(rb.L, pl, rb.layerCost, rb.hostBytes, offloadFor(metas, rb.experts, F)); };
  const sparks = [node(64), node(64)], pl = sparks.map((m) => pledgeGB(m, Infinity, F) * GB);
  const need = { 2048: minFitPledge(pl, (p) => fitAt(2048, p, sparks).fits).bytes / GB, 8192: minFitPledge(pl, (p) => fitAt(8192, p, sparks).fits).bytes / GB };
  // (f16 K / V, the rooms' format: 8K's state and its 2-slot reserve add ~0.8 GB)
  ok(need[8192] > need[2048] + 0.5 && need[8192] < need[2048] + 2, JSON.stringify(need));
  // two Sparks lending 34.3 GB each hold it at 2048 but not at 8K (state + its 2-slot reserve, f16)
  const tight = [34.3 * GB, 34.3 * GB];
  ok(fitAt(2048, tight, sparks).fits && !fitAt(8192, tight, sparks).fits);
  // the Mac tab with the disk tier: the GPU memory it needs stays a few GB at both
  const tab = [{ webgpu: true, contribGB: 22, tabDisk: { ramGB: 3, models: [F] } }], tpl = [22 * GB];
  for (const ctx of [2048, 8192]) {
    const m = minFitPledge(tpl, (p) => fitAt(ctx, p, tab).fits);
    ok(m && m.bytes / GB > 4 && m.bytes / GB < 8, `${ctx}: ${m && m.bytes / GB}`);
    ok(fitAt(ctx, tpl, tab).disk, `${ctx}: from disk`);
  }
});

Deno.test("flash plan: the last device is sized on the tail it gets, so rooms with an exact split at 2048 fit", () => {
  // two devices, whole GiB pledges at the margin: they fit on one size per layer and on the exact sums ([0, k) on the
  // host, [k, 48) on the other), so they must fit here too (sizing every non-host device on its heaviest window refused them)
  const rb = roomBytes(F, 2048), C = costOf(rb.layerCost);
  for (const gb of [[36.5, 31.75], [28, 40], [39, 29], [50, 18], [61, 7], [24.75, 42.75], [12, 56.5]]) {
    const pl = gb.map((g) => g * GB), f = roomFit(48, pl, rb.layerCost, rb.hostBytes);
    ok(f.fits, `${gb}: ${JSON.stringify(f)}`);
    for (const mode of ["memory", "speed"]) {
      const d = dealRoom({ L: 48, layerCost: rb.layerCost, hostBytes: rb.hostBytes, pledges: pl, mode });
      ok(d.used.length && d.ranges.at(-1)[1] === 48, `${gb} ${mode}`);
      checkHeld({ rb, d, pledges: pl });
    }
  }
  // 36.5 / 31.75: [0, 25) holds 35.44 GiB with the host's part, [25, 48) 31.58
  ok(C.span(0, 25) + rb.hostBytes <= 36.5 * GB && C.span(25, 48) <= 31.75 * GB);
  eq(layerCaps([36.5 * GB, 31.75 * GB], rb.layerCost, rb.hostBytes)[1], C.fitN(31.75 * GB, 48));
  // a device between the host and the last keeps its heaviest window (it could be dealt one past layer 1); one
  // size per layer is unchanged
  eq(layerCaps([40 * GB, 20 * GB, 20 * GB], rb.layerCost, rb.hostBytes)[1], C.fitN(20 * GB, { lo: 1 }));
  eq(layerCaps([10 * GB, 4 * GB, 4 * GB], GB, 0), [10, 4, 4]);
});

Deno.test("flash room: a tab model host sets aside its load margin; roomBytes is one frozen object per context", () => {
  const S = SHAPE[F], ex = flashLayerCost(F, 2048).experts;
  eq(flashTabHostMargin(F), Math.max(...S.layers) + FLASH_SCRATCH);
  eq(flashTabHostMargin(F, true), Math.max(...S.layers.map((b, i) => b - ex[i])) + FLASH_SCRATCH);
  eq([flashTabHostMargin(G), flashTabHostMargin(G, true)], [flashTabHostMargin(F), flashTabHostMargin(F, true)], "the header's and the picker's");
  ok(flashTabHostMargin(F) > 1.4 * GB && flashTabHostMargin(F) < 1.6 * GB && flashTabHostMargin(F, true) < 0.2 * GB);
  // the room page's deal (room.js aiStart): a tab host with resident layers keeps the full margin (two Sparks lending
  // 40 GB, speed: [0, 27) as before the exact costs); one that offloads every layer it holds keeps the small one
  const margin = [flashTabHostMargin(F), flashTabHostMargin(F, true)];
  const tabDeal = (metas, mode = "memory") => {
    const rb = roomBytes(F, 2048), pledges = metas.map((m) => pledgeGB(m, Infinity, F) * GB);
    return dealWithHostMargin((x) => dealRoom({ L: 48, layerCost: rb.layerCost, hostBytes: rb.hostBytes + x, pledges, mode, off: offloadFor(metas, rb.experts, F) }), ...margin);
  };
  const two = tabDeal([node(40), node(40)], "speed"), rb = roomBytes(F, 2048), C = costOf(rb.layerCost);
  eq(two.ranges[0], [0, 27]);
  ok(Math.abs(two.held[0] - C.span(0, 27) - rb.hostBytes - margin[0]) < 1 && two.held[0] <= 40 * GB);
  const tab22 = [{ webgpu: true, contribGB: 22, tabDisk: { ramGB: 3, models: [F] } }];
  const t = tabDeal(tab22), full = dealRoom({ L: 48, layerCost: rb.layerCost, hostBytes: rb.hostBytes + margin[0], pledges: [22 * GB], off: offloadFor(tab22, rb.experts, F) });
  ok(t.offload[0].lo === 0 && t.offload[0].slots > full.offload[0].slots, `${t.offload[0].slots} vs ${full.offload[0].slots}`);
  // with nothing to choose between, the full margin's deal
  let calls = 0;
  eq(dealWithHostMargin(() => { calls++; return { fit: { fits: false }, used: [] }; }, 5, 5).fit.fits, false);
  eq(calls, 1);
  ok(Object.isFrozen(rb) && Object.isFrozen(rb.experts));
  let threw = false;
  try { rb.hostBytes = 0; } catch { threw = true; }
  ok(threw && rb.hostBytes > 0, "a caller can't change another's roomBytes");
  ok(roomBytes(F, 2048).hostBytes > 0);
});
