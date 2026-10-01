// A compute pipeline whose compile failure names its kernel. Dawn's own error for a shader its
// backend compiler rejects (FXC on Windows without DXC, say) is the compiler's text plus the
// generated HLSL, with no kernel name in it; this one says which kernel failed in its first line and
// keeps the rest: err.shaderCompile = true, err.kernel (the entry point), err.raw (Dawn's message).
export async function compilePipeline(device, desc) {
  try {
    return await device.createComputePipelineAsync(desc);
  } catch (e) {
    const raw = String(e?.message || e);
    const kernel = desc?.compute?.entryPoint || "?";
    const first = raw.split("\n").map((l) => l.trim()).find(Boolean) || "compile error";
    throw Object.assign(new Error(`shader compile failed for ${kernel}: ${first.slice(0, 240)}`), { shaderCompile: true, kernel, raw, cause: e });
  }
}
