// Pre-flight: can this browser hold layers, and if not, what exactly to do about it. The verdict
// is a pure function of what the browser reports, so it can be unit tested; probe() gathers it.

// iPadOS reports a Mac user agent; a touch screen gives it away
export function deviceKind({ ua = "", touchPoints = 0, mobile = false } = {}) {
  if (/iPhone|iPod/.test(ua)) return "iPhone";
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && touchPoints > 1)) return "iPad";
  if (/Android/.test(ua)) return mobile || /Mobile/.test(ua) ? "Android" : "Android tablet";
  if (/Mac/.test(ua)) return "Mac";
  return "Device";
}

// facts: { ua, secure, hasGpuApi, adapter: null | { vendor, architecture }, touchPoints, mobile }
// -> { ok, kind, line, detail }: one calm sentence (what this device can do), and the remedy
// (what to change for WebGPU) for a Details disclosure. Phones are checked first, in phone words.
const PHONE = "This phone's GPU isn't available to the browser. It can still join and chat.";
const DESK = "This browser can't use its graphics chip, so this device can chat but not help run the model. Chrome or Edge on a laptop works best.";
export function verdict(f) {
  const kind = deviceKind(f);
  const ua = f.ua || "";
  if (f.adapter) {
    const g = [f.adapter.vendor, f.adapter.architecture].filter(Boolean).join(" ");
    return { ok: true, kind, line: `WebGPU works here${g ? ` (${g})` : ""}: this ${kind === "Device" ? "device" : kind} can hold part of the model.`, detail: "" };
  }
  const no = (line, detail) => ({ ok: false, kind, line, detail });
  const phone = kind === "iPhone" || kind === "iPad" || kind === "Android" || kind === "Android tablet";
  if (phone) {
    if (!f.secure) return no(PHONE, "WebGPU needs a secure page: open this over https.");
    if (kind === "iPhone" || kind === "iPad") return no(PHONE, `WebGPU on an ${kind} needs Safari 26 (iOS 26) or later: update iOS.`);
    if (f.hasGpuApi) return no(PHONE, "The browser has WebGPU but offered no GPU on this phone (a blocklisted driver?). Chrome 121 or later may help.");
    return no(PHONE, "This Android browser has no WebGPU: use Chrome 121 or later.");
  }
  // when the fix is in this browser (a flag, an update), the line says so instead of "use Chrome",
  // and the page shows the remedy open, not behind a collapsed Details
  const CHAT = "so this device can chat but not help run the model yet.";
  if (!f.secure) return no(`This page isn't secure, ${CHAT}`, "WebGPU needs a secure page: open this over https (or localhost).");
  if (f.hasGpuApi) return no(`This browser has WebGPU but found no usable GPU, ${CHAT}`, "A blocklisted driver? Try chrome://flags/#enable-unsafe-webgpu, then reload, or another browser.");
  if (/Firefox\//.test(ua)) return no(`Firefox has no WebGPU here, ${CHAT}`, "Use Chrome or Edge 113+, or Safari 26+.");
  if (/Linux/.test(ua) && /Chrome\//.test(ua) && !/Android/.test(ua))
    return no(`WebGPU is off in Chrome on Linux, ${CHAT} Two flags turn it on:`, "Enable chrome://flags/#enable-unsafe-webgpu and chrome://flags/#enable-vulkan, then reload.");
  if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) return no(`This Safari has no WebGPU, ${CHAT}`, "Safari 26 has it: update macOS, or use Chrome.");
  return no(DESK, "Use Chrome or Edge 113+, or Safari 26+.");
}

export async function probe() {
  const f = {
    ua: navigator.userAgent, secure: window.isSecureContext, hasGpuApi: !!navigator.gpu,
    touchPoints: navigator.maxTouchPoints || 0, mobile: !!navigator.userAgentData?.mobile, adapter: null,
  };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) f.adapter = { vendor: a.info?.vendor || "", architecture: a.info?.architecture || "" };
    } catch {}
  }
  return verdict(f);
}
