// Model catalogue for the room: URLs, layer counts, memory needs, context length.

export const NEED_GB = { "qwen3-0.6b": 0.8, "qwen3-1.7b": 4.0, "qwen3-4b": 4.6, "qwen3.8-27b": 17.0, "qwen3.6-35b-moe": 22.5, "smollm-135m": 0.6 };

// The whole weights file per picker model, in GB (the GGUF's size on Hugging Face). A room splits it:
// each device downloads about its share of the layers, so the picker can say what this device will fetch.
export const FILE_GB = { "qwen3-1.7b": 1.83, "qwen3.8-27b": 16.06, "qwen3.6-35b-moe": 20.84 };

// The models the room's picker offers. The others stay for tests and ?dev=1.
export const PICKER = ["qwen3-1.7b", "qwen3.8-27b", "qwen3.6-35b-moe"];

export const MODELS = {
  "qwen3-0.6b": { label: "Qwen3 0.6B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json" },
  "qwen3-1.7b": { label: "Qwen3 1.7B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json" },
  "qwen3-4b": { label: "Qwen3 4B · Q8", kind: "gguf",
    gguf: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    cfg: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json",
    tok: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json" },
  "qwen3.8-27b": { label: "Qwen3.8 27B \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_0.gguf" },
  // mixture of experts: 256 experts, 8 active per token (~3B of 35B), so decode reads far less than the 27B
  "qwen3.6-35b-moe": { label: "Qwen3.6 35B MoE \u00b7 Q4", kind: "qwen35",
    gguf: "https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf" },
  "smollm-135m": { label: "SmolLM 135M · bf16", kind: "safetensors",
    st: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
    cfg: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json",
    tok: "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json" },
};

// Context window per room, in tokens: prompt + answer. Each full-attention layer keeps K and V
// for this many positions (4 KB per position each for the 27B, so 16 MiB per attention layer at
// 2048); the kernels only use it as a stride. Generation stops before the cache would overflow.
export const MAX_SEQ = 2048;
// The 27B family keeps its KV cache in f16 with split-K flash attention (engine attnFlash), so its
// rooms get 8192 positions for the price of 4096 in f32: 32 MiB per attention layer, ~0.5 GB for
// the whole model, spread over the devices that hold the layers. The host reads the engine's
// maxSeq, so prompts, answer budgets and "context full" all follow it.
export const MAX_SEQ_LONG = 8192;
// Context per model, in tokens: the default a room opens with and the most ?ctx=N may ask for.
// These hybrids keep a KV cache only on their full-attention layers (1 in 4), and the DeltaNet
// layers hold a fixed-size state, so context is cheap in memory: f16 K+V is 64 KB per position
// for the 27B (16 attention layers, 4 KV heads x 256) and 20 KB for the 35B MoE (10 layers,
// 2 KV heads x 256). 32k on the MoE = 0.64 GB, 128k = 2.5 GB, spread over the devices holding
// those layers. One layer's K (or V) is one GPU buffer bound whole: 128 MiB for the MoE at 131072
// positions in f16, exactly the binding size every WebGPU device supports (so no device limits the
// MoE); the 27B needs 256 MiB at 131072 in f16 (128 MiB in int8), which the room only asks for
// when every device can bind it (ctxForBinding). The practical limit is prefill speed, not memory
// (docs/long-context-and-sessions.md: needle and speed at 32K..128K).
export const CTX = {
  "qwen3.8-27b": { def: 16384, max: 32768 },
  "qwen3.6-35b-moe": { def: 32768, max: 131072 },
  // the dense engine keeps an f32 KV cache (~224 KB per position on the 1.7B, 1.8 GB at 8k); 2k was
  // too small for Code mode, whose prompt alone is ~620 tokens (checked exact at 8k: tests pass)
  "qwen3-1.7b": { def: 8192, max: 16384 },
};
// WebGPU's default maxStorageBufferBindingSize: what a device binds when it reports nothing
export const WEBGPU_MIN_BIND = 128 * 2 ** 20;
// bytes of one attention layer's K (or V) cache buffer at ctx positions: kvDim x 1 (int8) or 2 (f16)
export const kvLayerBufBytes = (meta, ctx, kv = "f16") =>
  ctx * (meta["qwen35.attention.head_count_kv"] || 0) * (meta["qwen35.attention.key_length"] || 0) * (kv === "q8" ? 1 : 2);
// The longest context (<= ctx, a multiple of 256) whose per-layer K/V buffer fits bindBytes, the
// smallest binding limit among a room's devices (a device that reports none counts as WebGPU's 128 MiB).
export const ctxForBinding = (meta, ctx, kv = "f16", bindBytes = WEBGPU_MIN_BIND) => {
  const per = kvLayerBufBytes(meta, 1, kv);
  if (!per) return ctx;
  return Math.min(ctx, Math.floor(Math.max(bindBytes, WEBGPU_MIN_BIND) / per / 256) * 256);
};
export const maxSeqFor = (model, ask = 0) => {
  const c = CTX[model];
  if (!c) return MODELS[model]?.kind === "qwen35" ? MAX_SEQ_LONG : MAX_SEQ;
  return ask > 0 ? Math.min(c.max, Math.max(2048, Math.round(ask / 256) * 256)) : c.def;
};
// KV cache format per room: "f16" (the default) or "q8" (int8 values + one f32 scale per 32, engine
// kvQ8, ~56% of f16's memory; asked for with ?kv=q8). Only the qwen35 engine has the int8 kernels, so
// any other model, or any other ask, stays f16. The host decides and sends it with ai-load, so every
// device of a room keeps the same format and the layer deal counts the right bytes.
export const KV_MODES = ["f16", "q8"];
export const kvModeFor = (model, ask) => (ask === "q8" && MODELS[model]?.kind === "qwen35" ? "q8" : "f16");
// A device's format for an ai-load: the host's `kv` when it sent one (so a worker's own ?kv= cannot
// make it differ from the room), else this device's own ask (a host from before the field existed).
export const kvForLoad = (model, sent, ownAsk) => kvModeFor(model, sent ?? ownAsk);
// K+V bytes per position, averaged over a model's layers (for dealing layers by memory), f16 or int8
export const kvBytesPerLayerPos = (meta, kv = "f16") => {
  const kvDim = (meta["qwen35.attention.head_count_kv"] || 0) * (meta["qwen35.attention.key_length"] || 0);
  const perKV = kv === "q8" ? kvDim + kvDim / 32 * 4 : kvDim * 2;   // int8 values + f32 scales, or f16
  return perKV * 2 / (meta["qwen35.full_attention_interval"] || 1);
};
export const MAX_NEW = 400;    // longest answer, tokens
export const MAX_NEW_THINKING = 1200;   // with thinking on, the think block comes out of the same budget
export const MIN_ROOM = 32;    // a prompt must leave at least this many tokens for the answer
