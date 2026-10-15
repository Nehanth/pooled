// How much memory a device may lend the room. DOM-free so it can be unit tested (#207).
//
// Phones are capped because the browser tab, not the GPU, is the limit. Measured on an iPhone 14
// Pro Max (6 GB, iOS 26.6): Safari's page process has a 1536 MB soft limit, WebGPU buffers count
// against it like JS memory does, and after a few back-to-back loads iOS killed the tab just over
// 1.5 GB. The tab must also fit the page itself (~0.15 GB), the engine (~0.1 GB) and the load's
// spikes (0.3-0.6 GB), so 1 GB of layers is the most a phone can safely hold. Android tabs have
// the same shape of limit and less data; navigator.deviceMemory (Chrome) sizes them.

import { MODELS, flashDealReserve } from "./models.js";

export const PHONE_DEFAULT_GB = 0.5;
export const IOS_MAX_GB = 1;
export const DESK_MAX_GB = 64;
// the most an explicit pledge (a room node's --gb N, --gb max, --here) may say: Qwen3.8-Flash-Next (~72 GB) on one big
// unified-memory machine (a GB10, a 128 GB Mac). The host counts it only for that model (maxPledgeGB).
export const PLEDGE_MAX_GB = 128;
// the most one device's pledge counts for `model` (a key): 128 GB for Qwen3.8-Flash-Next (kind qwen4exp), else 64,
// also when the model is not known
export const maxPledgeGB = (model) => (MODELS[model]?.kind === "qwen4exp" ? PLEDGE_MAX_GB : DESK_MAX_GB);

// kind: deviceKind() (room/preflight.js): "iPhone" | "iPad" | "Android" | "Android tablet" | "Mac" | "Device"
// deviceMemory: navigator.deviceMemory (GB, rounded down to a power of two, at most 8; Chrome only)
// -> { min, def, max, step, capped, why }: why is one sentence for the Lend screen when capped
export function pledgeRule(kind, deviceMemory) {
  if (kind === "iPhone" || kind === "iPad") {
    const dev = kind === "iPhone" ? "an iPhone" : "an iPad";
    return { min: 0.5, def: PHONE_DEFAULT_GB, max: IOS_MAX_GB, step: 0.5, capped: true,
      why: `Safari closes a tab that uses more than about 1.5 GB, so ${dev} lends at most ${IOS_MAX_GB} GB.` };
  }
  if (kind === "Android" || kind === "Android tablet") {
    const dm = Number(deviceMemory);
    const max = !(dm > 0) ? 1 : dm <= 2 ? 0.5 : dm < 8 ? 1 : 2;
    return { min: 0.5, def: Math.min(PHONE_DEFAULT_GB, max), max, step: 0.5, capped: true,
      why: `The browser closes a tab that uses too much of this ${kind === "Android" ? "phone" : "tablet"}'s memory, so it lends at most ${max} GB.` };
  }
  return { min: 1, def: null, max: DESK_MAX_GB, step: 1, capped: false, why: "" };
}

// A pledge as the model host counts it: what the device says it lends, held to its kind's cap and
// to any smaller cap it reported itself (meta.pledgeMax), and to a share the host lowered after the
// device's tab was killed while loading (hostCap, GB), and to the most a device lends for the room's model (model: its
// key; maxPledgeGB: 64 GB, 128 for Qwen3.8-Flash-Next; 64 when not given). Devices from before the cap still get it.
export function pledgeGB(meta, hostCap = Infinity, model = null) {
  const rule = pledgeRule(meta?.ua, NaN);
  const own = Number(meta?.pledgeMax) > 0 ? Number(meta.pledgeMax) : Infinity;
  const said = Number(meta?.contribGB);
  const gb = said > 0 ? said : (meta?.maxBufGB ? meta.maxBufGB * 0.5 : PHONE_DEFAULT_GB);
  return Math.min(gb, rule.capped ? rule.max : Infinity, own, hostCap > 0 ? hostCap : Infinity, maxPledgeGB(model));
}

// A pledge as a dealer deals it, in bytes: pledgeGB less what the room's model keeps back on every device
// (room/models.js flashDealReserve: Qwen3.8-Flash-Next's batched-pass scratch)
export function dealPledgeBytes(meta, hostCap = Infinity, model = null) {
  return Math.max(0, pledgeGB(meta, hostCap, model) * 2 ** 30 - flashDealReserve(model));
}

// A device's tab was killed while loading its layers (it came back with the "died" breadcrumb).
// Loading it again with the same share would likely kill it again, so the host re-deals: with a
// smaller share (half, in the device's steps, never below one layer), or without the device when
// it was already down to one layer or has been killed twice.
// layers: how many layers it was loading; layerGB: one layer's size; gb: its share when it died;
// deaths: how many times its tab was killed while loading, this one included.
// -> { drop: true } | { drop: false, gb }
export function afterLoadDeath({ layers, layerGB, gb, deaths }) {
  if (deaths >= 2 || !(layers > 1)) return { drop: true };
  const half = Math.floor(layers / 2) * layerGB;
  const next = Math.max(layerGB, Math.min(gb / 2, half));
  if (!(next < gb)) return { drop: true };
  return { drop: false, gb: Math.round(next * 100) / 100 };
}

// Expert offload (room/plan.js offloadNeed): the system RAM a device lets the room park a MoE model's routed experts
// in, in GB. Only a device that says it can (meta.offload: a room node on a discrete GPU; browsers never park)
// and as much as it says (meta.ramGB), held to the most one computer lends (DESK_MAX_GB). It is a promise like the
// pledge: the deal never parks more than this. 0 for any other device.
export function ramGB(meta) {
  const r = Number(meta?.ramGB);
  return meta?.offload && meta?.webgpu !== false && r > 0 ? Math.min(DESK_MAX_GB, r) : 0;
}
// Expert offload's disk tier in a browser tab (room/flashdisk.js tabDiskMeta; room.js): meta.tabDisk =
// { ramGB, models, ask? }, a field of its own. It is not meta.offload / meta.ramGB on purpose: a host that predates it
// (@pooled/cli 0.4.1's planner) reads those as RAM to park experts in, which a tab never does. ramGB: the RAM the
// tier's spare and RAM cache (T1) may take; models: the model keys it reads experts of from disk (Qwen3.8-Flash-Next);
// ask: its user has not said yes yet (the tab asks when a deal gives it disk layers, and refuses them on a no). A tab
// whose user said no sends no tabDisk at all. -> GB for this model (0: no disk tier for it), held to DESK_MAX_GB
export const TAB_DISK_MAX_GB = 16;
export function tabDiskGB(meta, model) {
  const t = meta?.tabDisk;
  if (!t || typeof t !== "object" || meta?.webgpu === false || !model) return 0;
  if (!Array.isArray(t.models) || !t.models.includes(model)) return 0;
  const r = Number(t.ramGB);
  return r > 0 ? Math.min(TAB_DISK_MAX_GB, r) : 0;
}
// Expert offload's disk tier (room/plan.js DISK_RAM_MIN): whether a device may leave some of this model's experts in
// its files and read them when used. Only a device that offloads (ramGB) and says it has a disk tier (meta.disk: a
// room node started with --disk) and has every file of this model on its own disk (meta.local: the model keys it
// holds whole, room-node env.js probeMeta).
export function diskFor(meta, model) {
  return !!model && ramGB(meta) > 0 && !!meta?.disk && Array.isArray(meta?.local) && meta.local.includes(model);
}
// The RAM a room node lets the room park this model's experts in (ramGB), or 0 for Qwen3.8-Flash-Next on a node whose
// meta does not say it offloads Flash (meta.offloadFlash): offloadFor's own rule, shared with storageOf
const nodeRam = (m, model) => (MODELS[model]?.kind === "qwen4exp" && !m?.offloadFlash ? 0 : ramGB(m));
// What a device brings beyond its GPU memory for `model`, as the pool's Storage bar shows it (room.js renderPool), by
// offloadFor's rules for one meta: "disk" (a tab with the disk tier for it, ask: its user has not said yes yet; a room
// node with --disk and the model's files, ramGB: the RAM it reads them into), "ram" (a room node that parks it in RAM,
// ramGB), or null (nothing).
// Read-only: the planner reads offloadFor, never this.
export function storageOf(meta, model) {
  if (tabDiskGB(meta, model) > 0) return { kind: "disk", ask: !!meta.tabDisk.ask, ramGB: 0 };
  const r = !meta?.tabDisk ? nodeRam(meta, model) : 0;
  if (r > 0) return diskFor(meta, model) ? { kind: "disk", ask: false, ramGB: r } : { kind: "ram", ask: false, ramGB: r };
  return { kind: null, ask: false, ramGB: 0 };
}
// room/plan.js's `off` for a room's devices (metas, host first) and the model's routed experts: a profile
// (room/models.js expertsOf, roomBytes experts: what ExpertStore parks, layer by layer) or one layer's bytes (an
// estimate), or null when the model has none or no device offloads. model: the model's key; with it, a device with a
// disk tier for it (diskFor) is in off.disk (the field is there only when one is)
// A browser tab with the OPFS disk tier for this model (meta.tabDisk: tabDiskGB) is in off.diskOnly: it never parks
// experts (its GPU shares the computer's memory) and offloads only in the disk pass, with tabDisk.ramGB as RAM for the
// spare and T1 (room/plan.js diskOnlyAt). For any other model it counts as a plain tab (no offload), field for field.
// A room node offloads Qwen3.8-Flash-Next's experts only when its meta says it can (meta.offloadFlash, set by a node
// whose shard.js has Flash offload): a released 0.4.1 node says offload for the Qwen 3.5 MoEs but refuses Flash's, so
// it is dealt Flash layers within its pledge (as a node without --ram).
export function offloadFor(metas, ex, model = null) {
  const only = metas.map((m) => tabDiskGB(m, model) > 0);
  const ram = metas.map((m, i) => (only[i] ? tabDiskGB(m, model) : m?.tabDisk ? 0 : nodeRam(m, model)) * 2 ** 30);
  const expertBytes = ex && typeof ex === "object" ? +ex.E || 0 : +ex || 0;
  if (!(expertBytes > 0) || !ram.some((r) => r > 0)) return null;
  const disk = metas.map((m, i) => only[i] || (!m?.tabDisk && nodeRam(m, model) > 0 && diskFor(m, model)));
  return { expertBytes, ...(ex && typeof ex === "object" ? { experts: ex } : {}), ram, ...(disk.some(Boolean) ? { disk } : {}), ...(only.some(Boolean) ? { diskOnly: only } : {}) };
}

// The disk a device lets the room keep a model's n-gram table on (Qwen3.8-Flash-Next: room/plan.js placeNgram), in
// bytes: meta.ngram true when it already has the table (a room node with the model file: Infinity), else meta.diskGB
// (what it has free for model files: a room node's filesystem, a tab's OPFS quota), else null (it did not say).
export function ngramDisk(meta) {
  if (meta?.ngram === true) return Infinity;
  const g = Number(meta?.diskGB);
  return meta?.diskGB != null && g >= 0 ? g * 2 ** 30 : null;
}
// room/plan.js's `ngram` for a room's devices (metas, host first) and the model's table (room/models.js ngramOf:
// { layer, bytes }), or null for a model without one
export function ngramFor(metas, ng) {
  return ng && ng.bytes > 0 ? { layer: ng.layer, bytes: ng.bytes, disk: metas.map(ngramDisk) } : null;
}
