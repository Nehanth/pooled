const B = process.argv[2];
const { dealRoom } = await import(B + "/room/plan.js");
const { roomBytes, ngramOf } = await import(B + "/room/models.js");
const { offloadFor, ngramFor, pledgeGB } = await import(B + "/room/pledge.js");
const GB = 2 ** 30, F = "qwen3.8-flash";
const tab = (gb, ram = 3) => ({ webgpu: true, contribGB: gb, offload: true, diskOnly: true, ramGB: ram, disk: true, local: [F] });
const pc = (gb, ram, disk = true) => ({ webgpu: true, contribGB: gb, offload: true, ramGB: ram, ...(disk ? { disk: true, local: [F] } : {}) });
for (const [name, metas] of [["mac20+pc(10,43,disk)", [tab(20), pc(10, 43)]], ["mac24+pc(10,43,disk)", [tab(24), pc(10, 43)]], ["mac20+pc(10,32,disk)", [tab(20), pc(10, 32)]], ["mac20+pc(10,24,disk)", [tab(20), pc(10, 24)]]]) {
  const rb = roomBytes(F, 2048);
  const d = dealRoom({ L: rb.L, layerBytes: rb.layerBytes, hostBytes: rb.hostBytes, pledges: metas.map((m) => pledgeGB(m) * GB), mode: "memory", off: offloadFor(metas, rb.experts, F), ngram: ngramFor(metas, ngramOf(F)) });
  console.log(name, JSON.stringify({ ranges: d.ranges, off: d.offload?.map((o) => o && { lo: o.lo, hi: o.hi, diskLo: o.diskLo, layers: o.layers, slots: o.slots, ramGB: +((o.ramBytes || 0) / GB).toFixed(1), t1GB: +((o.diskRamBytes || 0) / GB).toFixed(2), diskOnly: o.diskOnly }) }));
}
