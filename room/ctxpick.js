// The context a room opens a model at, and its fit there: one computation for the picker (room.js roomFitFor: the
// rows, the need line, the pool) and Start (room.js aiStart), so what the picker says fits is what Start deals (the
// #386 lesson: the picker and Start measured Flash's layers two ways, and "fits" in the picker failed at Start).
// DOM-free (tests/unit/ctxpick_test.js runs the picker's inputs and Start's inputs through it and compares).
//
// Qwen3.8-Flash-Next (rooms long-context PR 5): the room's K / V format and the longest context its devices take
// (room/models.js flashRoomLimits), the ladder of proven lengths (CTX ladder), and the user's context choice in the
// picker (want: the longest it may open at; the room walks the ladder down from there when its memory is short).
import { MODELS, CTX, FLASH_CTX_OPTIONS, maxSeqFor, kvModeFor, pickCtx, roomBytes, flashRoomLimits, flashTabHostMargin, ctxK } from "./models.js";
import { roomFit, dealRoom, dealWithHostMargin, minFitPledge, gbUp } from "./plan.js";
import { offloadFor } from "./pledge.js";

// costs(ctx, kv) -> { L, layers (a number: one layer's bytes, or Flash's per-layer costs), hostBytes, experts }: the
// picker's from SHAPE (roomCosts), Start's from the model's header (the same numbers for Flash: models.js flashLayerCost)
export const roomCosts = (model) => (c, kv) => {
  const rb = roomBytes(model, c, kv);
  return rb && { L: rb.L, layers: rb.layerCost || rb.layerBytes, hostBytes: rb.hostBytes, experts: rb.experts || rb.expertBytes };
};
// A tab model host's load margin for Flash (models.js flashTabHostMargin): a tab has no load-time check, so its deal
// keeps the biggest layer's staging free; a room node host (meta.native) checks at load instead. src: the model key
// (SHAPE) or its header: the same numbers
export const hostMarginFor = (model, hostMeta, src = model) =>
  (MODELS[model]?.kind === "qwen4exp" && !hostMeta?.native ? [flashTabHostMargin(src), flashTabHostMargin(src, true)] : null);

// The fit at one context: roomFit, or for a tab host of Flash the deal with its load margin (dealWithHostMargin).
// deal: dealRoom's other options (mode, ms, phone, ngram: Start's split; the picker leaves them out)
// wrapOff: Start's test switches on the offload (room.js withSplit: ?flashsplit=0, ?hostlayers=)
export function fitAtCtx(model, { c, kv, pledges, metas, costs, margin = null, deal = {}, wrapOff = (o) => o }) {
  const x = costs(c, kv);
  if (!x) return null;
  const off = wrapOff(offloadFor(metas, x.experts, model));
  const per = typeof x.layers === "number" ? { layerBytes: x.layers } : { layerCost: x.layers };
  if (!margin && !Object.keys(deal).length) return { fit: roomFit(x.L, pledges, x.layers, x.hostBytes, off), costs: x, off };
  const dx = (m) => dealRoom({ L: x.L, ...per, hostBytes: x.hostBytes + m, pledges, off, ...deal });
  const d = margin ? dealWithHostMargin(dx, ...margin) : dx(0);
  return { fit: d.fit, deal: d, costs: x, off };
}

// The room's context for `model` and its fit there.
// metas / pledges: the devices that may be dealt layers, host first (pledges in bytes: room/pledge.js dealPledgeBytes);
// ask: an explicit context (?ctx=, --ctx: tried alone); choice: the picker's context choice (the longest the room opens
// at; the ladder below it when short); kvAsk: the room's ?kv= (Qwen 3.5 only); costs / margin / deal: as fitAtCtx.
// -> { ctx, want, fits, fellBack, kv, cap, fitAt(c, pledges?) }: kv the format the planner sized (Flash: the engine's
// K / V format, flashRoomLimits; else the room's KV mode), cap the longest context every device takes
// cap (optional): a longer limit of the caller's own (Start: a Qwen 3.5 room's GPU binding limit, ctxForBinding).
// fitAt(c, pledges?, deal?): fitAtCtx at c, with dealRoom's other options when given (Start's final deal: its split)
export function roomCtxPlan(model, { metas, pledges, ask = 0, choice = 0, kvAsk = null, costs = roomCosts(model), margin = null, cap = Infinity, wrapOff }) {
  const lim = flashRoomLimits(model, metas);
  const kv = lim.kv || kvModeFor(model, kvAsk), top = Math.min(lim.cap, cap);
  const want = ask > 0 ? maxSeqFor(model, ask) : choice > 0 ? Math.min(maxSeqFor(model), maxSeqFor(model, choice)) : maxSeqFor(model);
  const fitAt = (c, pl = pledges, deal = {}) => fitAtCtx(model, { c, kv, pledges: pl, metas, costs, margin, deal, wrapOff });
  const pick = pickCtx(model, { want, ask, cap: top, fitsAt: (c) => !!fitAt(c)?.fit.fits });
  return { ...pick, kv, cap: top, fitAt };
}

// The picker's context choices for a model (Qwen3.8-Flash-Next: FLASH_CTX_OPTIONS, proven lengths only), each with
// what it needs in this room: [{ ctx, label, fits, needGB, shortGB, why, capped }] (needGB: the GPU memory the planner
// needs from these devices at that context, minFitPledge, as the picker's need line; shortGB / why: how much more, when
// it doesn't fit; capped: a device here can't take that long a context). [] for a model without choices.
export function ctxOptions(model, { metas, pledges, costs = roomCosts(model), margin = null, kvAsk = null }) {
  if (!CTX[model]?.ladder) return [];
  const lim = flashRoomLimits(model, metas);
  const kv = lim.kv || kvModeFor(model, kvAsk);
  const has = pledges.reduce((a, b) => a + (b > 0 ? b : 0), 0) / 2 ** 30;
  return FLASH_CTX_OPTIONS.map((c) => {
    const capped = c > lim.cap;
    const fitAt = (pl) => fitAtCtx(model, { c, kv, pledges: pl, metas, costs, margin });
    const f = capped ? null : fitAt(pledges);
    const m = capped ? null : minFitPledge(pledges, (pl) => !!fitAt(pl)?.fit.fits);
    const needGB = m ? gbUp(m.bytes) : null;
    const fits = !!f?.fit.fits;
    const shortGB = fits || needGB == null ? 0 : Math.max(0.1, +(needGB - has).toFixed(1));
    const why = fits ? "" : capped ? "a device here runs an older Pooled" : needGB == null ? "this room can't hold it" : `needs ${shortGB} GB more`;
    return { ctx: c, label: ctxK(c), fits, needGB, shortGB, why, capped };
  });
}

// How long reading a prompt of `tokens` takes, from the rooms' measured prefill rates (2026-10-13, this repo's GB10;
// docs/long-context-and-sessions.md), by how the room reads a prompt (mode):
//   batch     every device runs batched passes (room/flash.js): two room nodes, f16 K / V (flash_gold.mjs pair):
//             189 tok/s at 8K, 200 at 32K and 64K, 186 at 128K
//   perToken  a device steps prompt frames one column at a time with its experts in GPU memory or RAM (an offloader
//             without the region, POOLED_FLASH_BATCH=0): 15.8 tok/s at 8K, about flat with length (QSA reads at most
//             2051 cells a token)
//   disk      a device reads experts from its disk (a tab's disk tier: those layers never batch): 1.8 tok/s
//             (flash_tab_gold.mjs tab, a 12 GB tab on the GB10; a Mac's own disk may differ)
// -> seconds, or null for a model without measurements
export const FLASH_PREFILL_TPS = { batch: { 8192: 189, 32768: 200, 65536: 200, 131072: 186 }, perToken: { 8192: 15.8, 131072: 15.8 }, disk: { 8192: 1.8, 131072: 1.8 } };
export function prefillSeconds(model, tokens, { mode = "batch" } = {}) {
  if (MODELS[model]?.kind !== "qwen4exp" || !(tokens > 0)) return null;
  const t = FLASH_PREFILL_TPS[mode] || FLASH_PREFILL_TPS.batch, ks = Object.keys(t).map(Number).sort((a, b) => a - b);
  // the rate measured at the nearest length at or above it (rates fall a little with length)
  const k = ks.find((x) => x >= tokens) ?? ks[ks.length - 1];
  return tokens / t[k];
}
// How a room reads prompts, from its fit (roomFitFor's): disk when some layers read experts from disk, perToken when a
// device offloads (its frames usually run per token: #381's region rule), else batch
export const prefillMode = (fit) => (fit?.disk ? "disk" : fit?.offload ? "perToken" : "batch");
// "about 2 min", "about 1.5 h": a duration in words for the picker
export function durationWords(s) {
  if (!(s > 0)) return "";
  if (s < 90) return `about ${Math.max(10, Math.round(s / 10) * 10)} s`;
  if (s < 90 * 60) return `about ${Math.round(s / 60)} min`;
  return `about ${+(s / 3600).toFixed(1)} h`;
}
