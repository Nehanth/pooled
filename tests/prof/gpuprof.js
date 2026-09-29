// GPU/CPU profiler for the Qwen 3.5/3.6 engine, for Deno and the browser (same module).
//
//   const P = installProf(device, eng, { mode: "submit" | "kernel" });
//   P.start(); ...engine work...; P.stop(); const r = await P.resolve();
//
// Counts every queue.submit, mapAsync and writeBuffer while recording, and timestamps on the GPU:
//   mode "submit": one empty timestamped compute pass at the start and at the end of every command
//     encoder, so each submitted command buffer gets its GPU span (start of its first pass to end of
//     its last). The engine's passes are untouched, so the span is the real GPU time of that submit.
//   mode "kernel": every dispatch runs in its own timestamped pass (as tests/prof_ts.js does), so
//     each kernel gets its GPU time, attributed to a category (router, expert gate/up, attention...)
//     from the engine op or layer it was encoded for. Per-pass overhead is included, so the sum is
//     larger than the submit span.
// Every submit/map/encoder carries the label of the engine method that was running when it was
// created (forwardToken, _verifyFused, _mtpRefill, _readback...), set by wrappers on the engine.
// Timestamps need the device to have "timestamp-query"; without it only CPU-side numbers exist.

import { pipeFamily } from "./families.js";

const LABELED = ["forwardToken", "embedRun", "runHidden", "embedRunBatch", "runHiddenBatch", "headFromHidden", "headBatch",
  "_verifyFused", "_draftChain", "_mtpRun", "_mtpRefill", "_preDraft0", "_restoreDN", "_adoptHidden", "_readback", "prefillTokens", "_mtpFillBatch"];
const DRAFT = new Set(["_draftChain", "_mtpRun", "_mtpRefill", "_preDraft0", "_mtpFillBatch"]);

export const now = () => performance.now();

// engine op object -> category, from the keys the engine stores them under
function opCategories(eng) {
  const m = new Map();
  const put = (op, cat) => { if (op && typeof op === "object" && (op.pipe || op.gemm) && !m.has(op)) m.set(op, cat); };
  const cat = (key, full) => {
    if (/^(mvRouter|mvRS|mvShRouter|router|rs|shRouter)$/.test(key)) return "MoE router GEMV";
    if (/^(gu|mvGate|mvUp|mvShDown|gateUp|shDown)$/.test(key)) return "MoE shared expert";
    if (/^(mvDown|down)$/.test(key)) return "dense FFN";
    return full ? "attention proj (q/kv/o)" : "DeltaNet proj (qkvz/ba/out)";
  };
  const walk = (obj, full) => { for (const [k, v] of Object.entries(obj || {})) {
    if (Array.isArray(v)) v.forEach((o) => put(o, cat(k, full))); else put(v, cat(k, full)); } };
  (eng.layers || []).forEach((L, i) => { walk(L, L.isFull); walk(eng.layerB?.[i], L.isFull); });
  put(eng.headOp, "LM head"); put(eng.headOpDraft, "MTP draft"); put(eng.headB, "LM head");
  if (eng.mtp) { put(eng.mtp.proj, "MTP draft"); put(eng.mtp.projB, "MTP draft"); }
  return m;
}

function pipeCategory(name, ctx) {
  if (/^moe_route/.test(name)) return "MoE router top-k";
  if (/^moe_gu/.test(name)) return "MoE expert gate/up";
  if (/^moe_dn/.test(name)) return "MoE expert down(+combine)";
  if (/^moe_combine/.test(name)) return "MoE combine";
  if (/^rmsnorm/.test(name)) return ctx.L ? "norms" : "LM head";
  if (/^silu_mul/.test(name)) return "MoE shared expert";
  if (/^add_res/.test(name)) return "residual add";
  if (/^dn_/.test(name)) return "DeltaNet core (conv/gates/delta)";
  if (/^(attn|kv|ks_|flash|fa_|sigmoid_mul|qsplit|head_norm|rope)/.test(name)) return "attention core";
  if (/^argmax/.test(name)) return "LM head";
  if (/^emb_gather/.test(name)) return "MTP draft";
  if (/^xpose/.test(name)) return "transpose (GEMM prefill)";
  return pipeFamily(name);   // every other pipeline by its family (tests/prof/families.js), never lost as "other"
}

export function installProf(device, eng, { mode = "submit", maxQ = 8 * 4096 } = {}) {
  const ts = mode !== "none" && device.features.has("timestamp-query");
  const P = { mode: ts ? mode : "none", on: false, label: "-", ctx: { L: null, op: null, draft: 0 }, subs: [], maps: [], writes: 0, writeBytes: 0,
    encs: 0, kern: [], nq: 0, overflow: false };
  // WebGPU caps a query set at 4096 queries: a list of sets, query i lives in set i >> 12 (pairs never straddle)
  const QS = 4096, sets = ts ? Array.from({ length: Math.ceil(maxQ / QS) }, () => device.createQuerySet({ type: "timestamp", count: QS })) : null;
  const qsOf = (i) => sets[i >> 12], qi = (i) => i & (QS - 1);
  const resBuf = ts ? device.createBuffer({ size: maxQ * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const rdBuf = ts ? device.createBuffer({ size: maxQ * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  const q2 = () => { if (P.nq + 2 > maxQ) { P.overflow = true; return -1; } const i = P.nq; P.nq += 2; return i; };
  const origCreate = device.createCommandEncoder.bind(device);
  const cbRec = new WeakMap();
  const opCat = opCategories(eng);
  // pipeline -> name, from the engine's own pipeline list (rebuilt if the engine has added one since)
  let pname = new Map(), pnameN = -1;
  const pipeName = (p) => {
    if (!pname.has(p) && Object.keys(eng.pipes).length !== pnameN) { const e = Object.entries(eng.pipes); pname = new Map(e.map(([k, v]) => [v, k])); pnameN = e.length; }
    return pname.get(p) || "?";
  };

  device.createCommandEncoder = (d) => {
    const enc = origCreate(d);
    if (!P.on) return enc;
    P.encs++;
    const rec = { label: P.label, tEnc: now(), q: -1, tSubmit: 0, gpu: 0, kern: [] };
    if (P.mode === "submit") {
      const i = q2();
      if (i >= 0) { rec.q = i; enc.beginComputePass({ timestampWrites: { querySet: qsOf(i), beginningOfPassWriteIndex: qi(i) } }).end(); }
    }
    // encoder structure (every mode but kernel, which re-splits the passes): compute passes, dispatches,
    // copyBufferToBuffer calls (a compute -> blit -> compute encoder switch on Metal) and bytes copied
    rec.passes = 0; rec.dispatches = 0; rec.copies = 0; rec.copyBytes = 0;
    const obp = enc.beginComputePass.bind(enc), ocp = enc.copyBufferToBuffer.bind(enc);
    if (P.mode !== "kernel") {
      enc.beginComputePass = (d) => {
        const p = obp(d); rec.passes++;
        const od = p.dispatchWorkgroups.bind(p), oi = p.dispatchWorkgroupsIndirect?.bind(p);
        p.dispatchWorkgroups = (...a) => { rec.dispatches++; return od(...a); };
        if (oi) p.dispatchWorkgroupsIndirect = (...a) => { rec.dispatches++; return oi(...a); };
        return p;
      };
    }
    enc.copyBufferToBuffer = (...a) => { rec.copies++; rec.copyBytes += a.length === 5 ? a[4] : a.length === 3 ? a[2] : 0; return ocp(...a); };
    if (P.mode === "kernel") {
      const ob = enc.beginComputePass.bind(enc);
      enc.beginComputePass = () => {
        let pipe = null; const bgs = {};
        return {
          setPipeline(p) { pipe = p; }, setBindGroup(i, b) { bgs[i] = b; },
          dispatchWorkgroups(x, y = 1, z = 1) {
            const name = pipeName(pipe);
            const c = P.ctx;
            let cat = c.op ? opCat.get(c.op) || "other GEMV" : pipeCategory(name, c);
            if (c.draft || (c.L && c.L === eng.mtpLayer)) cat = "MTP draft";
            const i = q2();
            if (i < 0) { const p = ob(); p.setPipeline(pipe); for (const k in bgs) p.setBindGroup(+k, bgs[k]); p.dispatchWorkgroups(x, y, z); p.end(); return; }
            rec.kern.push({ q: i, name, cat, label: rec.label });
            const p = ob({ timestampWrites: { querySet: qsOf(i), beginningOfPassWriteIndex: qi(i), endOfPassWriteIndex: qi(i) + 1 } });
            p.setPipeline(pipe); for (const k in bgs) p.setBindGroup(+k, bgs[k]); p.dispatchWorkgroups(x, y, z); p.end();
          },
          end() {},
        };
      };
    }
    const ofin = enc.finish.bind(enc);
    enc.finish = (dd) => {
      if (P.mode === "submit" && rec.q >= 0) obp({ timestampWrites: { querySet: qsOf(rec.q), endOfPassWriteIndex: qi(rec.q) + 1 } }).end();
      const cb = ofin(dd); cbRec.set(cb, rec); return cb;
    };
    return enc;
  };
  const queue = device.queue;
  const oSubmit = queue.submit.bind(queue), oWrite = queue.writeBuffer.bind(queue);
  queue.submit = (cbs) => {
    if (P.on) { const t = now(); for (const cb of cbs) { const r = cbRec.get(cb) || { label: P.label + "(unwrapped)", q: -1, kern: [] }; r.tSubmit = t; P.subs.push(r); } }
    return oSubmit(cbs);
  };
  queue.writeBuffer = (buf, off, data, ...rest) => {
    if (P.on) { P.writes++; P.writeBytes += rest.length > 1 ? rest[1] : (data.byteLength ?? 0) - (rest[0] || 0) * (data.BYTES_PER_ELEMENT || 1); }
    return oWrite(buf, off, data, ...rest);
  };
  // mapAsync on every buffer (prototype): time from the call to resolution
  const proto = Object.getPrototypeOf(rdBuf || device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  if (!proto.__profMap) {
    const oMap = proto.mapAsync;
    proto.__profMap = true;
    proto.mapAsync = function (...a) {
      const prof = globalThis.__gpuprofActive;
      if (!prof || !prof.on) return oMap.apply(this, a);
      const r = { label: prof.label, t0: now(), t1: 0, bytes: a[2] ?? this.size };
      prof.maps.push(r);
      return oMap.apply(this, a).then((v) => { r.t1 = now(); return v; });
    };
  }
  // engine wrappers: labels, and the context kernel mode attributes dispatches with
  const wrapSync = (name, before, after) => {
    const f = eng[name]; if (typeof f !== "function") return;
    eng[name] = function (...a) { const s = before(a); try { return f.apply(this, a); } finally { after(s); } };
  };
  for (const n of LABELED) wrapSync(n, () => { const s = [P.label, P.ctx.draft]; P.label = n; if (DRAFT.has(n)) P.ctx.draft++; return s; },
    ([l, d]) => { P.label = l; P.ctx.draft = d; });
  wrapSync("_encodeLayerR", (a) => { const s = P.ctx.L; P.ctx.L = a[1]; return s; }, (s) => { P.ctx.L = s; });
  wrapSync("_encodeLayerBatch", (a) => { const s = P.ctx.L; P.ctx.L = a[1] < eng.layers.length ? eng.layers[a[1]] : eng.mtpLayer; return s; }, (s) => { P.ctx.L = s; });
  wrapSync("_encodeDraftChain", () => { P.ctx.draft++; }, () => { P.ctx.draft--; });
  wrapSync("_dop", (a) => { const s = P.ctx.op; P.ctx.op = a[1]; return s; }, (s) => { P.ctx.op = s; });

  P.start = () => { globalThis.__gpuprofActive = P; P.subs = []; P.maps = []; P.writes = 0; P.writeBytes = 0; P.encs = 0; P.nq = 0; P.overflow = false; P.on = true; };
  P.stop = () => { P.on = false; };
  // read the timestamps back; fills rec.gpu (ms) per submit (mode submit) or rec.kern[].ms (mode kernel)
  P.resolve = async () => {
    if (!ts || !P.nq) return P;
    const was = P.on; P.on = false;
    const enc = origCreate();
    for (let s = 0; s * QS < P.nq; s++) enc.resolveQuerySet(sets[s], 0, Math.min(QS, P.nq - s * QS), resBuf, s * QS * 8);
    enc.copyBufferToBuffer(resBuf, 0, rdBuf, 0, P.nq * 8);
    oSubmit([enc.finish()]);
    await rdBuf.mapAsync(GPUMapMode.READ, 0, P.nq * 8);
    const t = new BigUint64Array(rdBuf.getMappedRange(0, P.nq * 8).slice(0));
    rdBuf.unmap();
    const d = (i) => (t[i + 1] > t[i] ? Number(t[i + 1] - t[i]) / 1e6 : 0);
    for (const s of P.subs) {
      if (s.q >= 0) { s.gpu = d(s.q); s.g0 = Number(t[s.q]) / 1e6; s.g1 = Number(t[s.q + 1]) / 1e6; }
      for (const k of s.kern) k.ms = d(k.q);
    }
    P.on = was;
    return P;
  };
  return P;
}

// ---- summaries ----
export function bySum(items, key, val) {
  const o = {};
  for (const it of items) { const k = key(it); (o[k] ||= { n: 0, ms: 0 }); o[k].n++; o[k].ms += val(it); }
  return o;
}
export const r2 = (x) => Math.round(x * 100) / 100;
export function table(rows, cols) {
  const w = cols.map((c) => Math.max(c.length, ...rows.map((r) => String(r[c] ?? "").length)));
  const line = (vals) => "| " + vals.map((v, i) => String(v ?? "").padEnd(w[i])).join(" | ") + " |";
  return [line(cols), "|" + w.map((x) => "-".repeat(x + 2)).join("|") + "|", ...rows.map((r) => line(cols.map((c) => r[c])))].join("\n");
}
