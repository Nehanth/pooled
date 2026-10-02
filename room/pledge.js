// How much memory a device may lend the room. DOM-free so it can be unit tested (#207).
//
// Phones are capped because the browser tab, not the GPU, is the limit. Measured on an iPhone 14
// Pro Max (6 GB, iOS 26.6): Safari's page process has a 1536 MB soft limit, WebGPU buffers count
// against it like JS memory does, and after a few back-to-back loads iOS killed the tab just over
// 1.5 GB. The tab must also fit the page itself (~0.15 GB), the engine (~0.1 GB) and the load's
// spikes (0.3-0.6 GB), so 1 GB of layers is the most a phone can safely hold. Android tabs have
// the same shape of limit and less data; navigator.deviceMemory (Chrome) sizes them.

export const PHONE_DEFAULT_GB = 0.5;
export const IOS_MAX_GB = 1;
const DESK_MAX_GB = 64;

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
// device's tab was killed while loading (hostCap, GB). Devices from before the cap still get it.
export function pledgeGB(meta, hostCap = Infinity) {
  const rule = pledgeRule(meta?.ua, NaN);
  const own = Number(meta?.pledgeMax) > 0 ? Number(meta.pledgeMax) : Infinity;
  const said = Number(meta?.contribGB);
  const gb = said > 0 ? said : (meta?.maxBufGB ? meta.maxBufGB * 0.5 : PHONE_DEFAULT_GB);
  return Math.min(gb, rule.capped ? rule.max : Infinity, own, hostCap > 0 ? hostCap : Infinity);
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
// in, in GB. Only a device that says it can (meta.offload: a room node on a discrete GPU; browsers never offload)
// and as much as it says (meta.ramGB), held to the most one computer lends (DESK_MAX_GB). It is a promise like the
// pledge: the deal never parks more than this. 0 for any other device.
export function ramGB(meta) {
  const r = Number(meta?.ramGB);
  return meta?.offload && meta?.webgpu !== false && r > 0 ? Math.min(DESK_MAX_GB, r) : 0;
}
// room/plan.js's `off` for a room's devices (metas, host first) and the model's routed experts: a profile
// (room/models.js expertsOf, roomBytes experts: what ExpertStore parks, layer by layer) or one layer's bytes (an
// estimate), or null when the model has none or no device offloads
export function offloadFor(metas, ex) {
  const ram = metas.map((m) => ramGB(m) * 2 ** 30);
  const expertBytes = ex && typeof ex === "object" ? +ex.E || 0 : +ex || 0;
  if (!(expertBytes > 0) || !ram.some((r) => r > 0)) return null;
  return ex && typeof ex === "object" ? { expertBytes, experts: ex, ram } : { expertBytes, ram };
}
