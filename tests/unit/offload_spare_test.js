// What a device that offloads experts keeps free past its layers and its cache (room/plan.js offloadSpare), and the
// speculation switch for deals with offload (specWithOffload). The 122B room (Mac Studio + RTX 5070 PC, layers 15-47
// offloaded) ran its PC's VRAM to 11.5 of 12.2 GB on a 12k-token prompt and WDDM spilled 6 GB into its RAM; the deal
// now leaves room for the engine's buffers and the host's checkpoints, and keeps RAM for the process past the experts.
import { dealRoom, offloadPlan, offloadNeed, specWithOffload, dealOffloads, parseForce, offloadSpareOf,
  OFFLOAD_CKPT_SLOTS, OFFLOAD_CKPT_TOKENS, OFFLOAD_RAM_BASE, OFFLOAD_MIN_SLOTS } from "../../room/plan.js";
import { roomBytes, expertsOf, stateBytesPerLayer, offloadWork, SHAPE } from "../../room/models.js";
import { offloadFor, pledgeGB } from "../../room/pledge.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const GB = 2 ** 30, MiB = 2 ** 20;
const pc = (gb, ram) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram });
const tab = (gb) => ({ webgpu: true, contribGB: gb });
const deal = (model, metas, ctx = 32768) => {
  const rb = roomBytes(model, ctx), pledges = metas.map((m) => pledgeGB(m) * GB);
  return { rb, d: dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges, mode: "speed", off: offloadFor(metas, rb.experts) }) };
};

Deno.test("offload spare: the models' recurrent state and working buffers (engine/qwen35.js sizes)", () => {
  // DeltaNet S: nVH x 128 x 128 f32, conv window 3 x convDim f32, on 3 of every 4 layers
  const s35 = (32 * 128 * 128 * 4 + 3 * (2 * 128 * 16 + 4096) * 4) * 3 / 4, s122 = (64 * 128 * 128 * 4 + 3 * (2 * 128 * 16 + 8192) * 4) * 3 / 4;
  eq(roomBytes("qwen3.6-35b-moe", 32768).experts.state, s35);
  eq(roomBytes("qwen3.5-122b-moe", 32768).experts.state, s122);
  eq(stateBytesPerLayer(SHAPE["qwen3.5-122b-moe"].meta), s122);
  eq(roomBytes("qwen3.5-122b-moe", 32768).experts.work, offloadWork(3072));
  ok(offloadWork(3072) > 0.5 * GB && offloadWork(3072) < 0.6 * GB, "the 122B: 0.30 GiB measured on Dawn/Vulkan, ~70% more under D3D12");
  // a header gives the same profile as SHAPE (roomnode.js deals from the file's own header)
  const G = { meta: { ...SHAPE["qwen3.5-122b-moe"].meta, "qwen35.attention.head_count_kv": 2, "qwen35.attention.key_length": 256 }, tensors: {} };
  for (let i = 0; i < 2; i++) for (const p of ["gate", "up", "down"]) G.tensors[`blk.${i}.ffn_${p}_exps.weight`] = { shape: [256, p === "down" ? 3072 : 1024, p === "down" ? 1024 : 3072], ggmlType: 2, byteLength: 1 };
  const ex = expertsOf(G, 2);
  eq([ex.kvPos, ex.state, ex.work], [512, s122, offloadWork(3072)]);
});

Deno.test("offload spare: the cache leaves VRAM for the engine and the checkpoints, the RAM check keeps room past the experts", () => {
  const ex = roomBytes("qwen3.5-122b-moe", 32768).experts, n = 33;
  const sp = offloadSpareOf(ex, n), ckpt = ex.kvPos * OFFLOAD_CKPT_TOKENS;
  eq(sp.vram, ex.work + n * (ex.state * (1 + OFFLOAD_CKPT_SLOTS) + ckpt));
  eq(sp.ram, OFFLOAD_RAM_BASE + n * (ex.state + ckpt / 2));
  ok(sp.vram > 1.9 * GB && sp.vram < 2.1 * GB, `the PC's 33 layers keep ${(sp.vram / GB).toFixed(2)} GiB of VRAM`);
  ok(sp.ram > 1.8 * GB && sp.ram < 1.95 * GB, `and ${(sp.ram / GB).toFixed(2)} GiB of RAM`);
  // the estimate-only profile (a number, the tests' toy models) keeps nothing, as before
  eq(offloadSpareOf({ E: 1 }, n), { vram: 0, ram: 0 });
});

Deno.test("offload spare: the 122B room as it ran (Mac 25 GB, the PC 10 GB / 47 GB of RAM) leaves the PC room", () => {
  const { d, rb } = deal("qwen3.5-122b-moe", [tab(25), pc(10, 47)]);
  ok(d.fit.fits && d.fit.offload);
  eq(d.ranges, [[0, 15], [15, 48]], "the same layers as the room that ran");
  const o = d.offload[1];
  // main parked the same 41.77 GiB with a 7.15 GiB cache (36 slots); the 5.17 GiB left keeps 23 slots
  ok(Math.abs(o.ramBytes / GB - 41.77) < 0.01, `parked ${o.ramBytes / GB}`);
  ok(o.vramBytes < 5.2 * GB && o.slots >= OFFLOAD_MIN_SLOTS, JSON.stringify(o));
  // VRAM: the resident part + the cache + the spare is the pledge, never more (#291)
  const resident = 33 * rb.layerBytes - 33 * rb.experts.E;
  ok(resident + o.vramBytes + o.vramSpare <= 10 * GB + 1, "within the pledge");
  // RAM: parked + spare within what it lends
  ok(o.ramBytes + o.ramSpare <= 47 * GB, "within its RAM");
  ok(o.ramSpare > 1.5 * GB);
});

Deno.test("offload spare: a device whose RAM holds the experts but not the spare can't take them; the deal stays within both", () => {
  const ex = roomBytes("qwen3.6-35b-moe", 32768).experts, rb = roomBytes("qwen3.6-35b-moe", 32768);
  const n = 40, bytes = 10 * GB - rb.hostBytes;
  // the least RAM that lets it take its 40 layers: the experts it parks and the spare on top
  let lo = 0, hi = 64 * GB;
  while (hi - lo > MiB) { const mid = (lo + hi) / 2; if (offloadNeed(n, bytes, mid, rb.layerBytes, ex, OFFLOAD_MIN_SLOTS, 256, n) > 0) hi = mid; else lo = mid; }
  const m = offloadNeed(n, bytes, hi, rb.layerBytes, ex, OFFLOAD_MIN_SLOTS, 256, n), p = offloadPlan(n, bytes, hi, rb.layerBytes, ex, OFFLOAD_MIN_SLOTS, 256, n);
  ok(m > 0 && p.layers === m, JSON.stringify(p));
  ok(hi >= p.ramBytes + p.ramSpare && hi < p.ramBytes + p.ramSpare + 2 * 6 * 2 ** 20 + 64 * MiB, `${hi / GB} GB for ${(p.ramBytes + p.ramSpare) / GB}`);
  ok(p.ramSpare > OFFLOAD_RAM_BASE);
  for (const ram of [12, 16, 24, 48]) {
    const { d } = deal("qwen3.6-35b-moe", [tab(6), pc(8, ram)]);
    const o = d.offload?.[1];
    if (o) ok(o.ramBytes + o.ramSpare <= ram * GB, `${ram} GB: ${JSON.stringify(o)}`);
  }
});

Deno.test("offload spare: devices that don't offload deal exactly as before", () => {
  for (const metas of [[tab(12), tab(12)], [tab(12), pc(12, 48)], [pc(24, 48)]]) {
    const { d } = deal("qwen3.6-35b-moe", metas);
    ok(d.fit.fits && !d.fit.offload && !d.offload, JSON.stringify(d.fit));
  }
});

Deno.test("speculation with expert offload: off by default when the deal offloads, an override wins, unchanged otherwise", () => {
  const o = { lo: 15, hi: 48, vramBytes: 1, ramBytes: 1 };
  eq([dealOffloads(null), dealOffloads([]), dealOffloads([null, null]), dealOffloads({}), dealOffloads([null, o]), dealOffloads({ pc: o }), dealOffloads(true), dealOffloads(false)],
    [false, false, false, false, true, true, true, false]);
  eq(specWithOffload([null, null]), true, "no offload: speculation as before");
  eq(specWithOffload([null, o]), false, "offload: plain");
  eq(specWithOffload([null, o], true), true, "forced on");
  eq(specWithOffload([null, null], false), false, "forced off");
  eq(specWithOffload([null, o], null), false);
  eq(["1", "on", "TRUE", " yes ", "0", "off", "false", "no", "", undefined, null, "2"].map(parseForce), [true, true, true, true, false, false, false, false, null, null, null, null]);
  // the deal the host makes: the 122B room offloads -> plain; a 35B room the pledges hold -> speculation
  eq(specWithOffload(deal("qwen3.5-122b-moe", [tab(25), pc(10, 47)]).d.offload), false);
  eq(specWithOffload(deal("qwen3.6-35b-moe", [tab(12), pc(12, 48)]).d.offload), true);
});
