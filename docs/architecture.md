# Architecture

Pooled has three parts: an inference **engine** that runs a model (or a slice of one) on a device's GPU through WebGPU, a **room runtime** that connects browsers over WebRTC and threads one generation through all of them, and **Code mode** (`harness/`), a coding agent that runs on the room's model.

```
                 ┌──────────────────────── host browser ────────────────────────┐
 text ──tokenize──► embed ──► layers 0..k ──┐                      ┌──► final norm ──► LM head ──► sample ──► text
                                             │ hidden state (f16)   │
                                             ▼                      │
                                       peer A: layers k+1..m ──► peer B: layers m+1..N
```

## One token, end to end

1. **Tokenize** (host CPU). Text → token ids using the tokenizer stored in the GGUF header.
2. **Embed** (host). The id selects a row of the embedding table: a `dim`-float hidden state (5,120 for Qwen 3.8).
3. **Layers** (every device, in order). Each device runs its contiguous range of transformer blocks on its GPU. Qwen 3.8 has 64: 48 Gated-DeltaNet blocks (a recurrent state matrix per head, constant memory in context length) and 16 full-attention blocks (with a KV cache). All ops of a device's range are recorded into one command buffer and submitted once.
4. **Hop.** The hidden state is packed to f16 (10 KB) and sent to the next device over a WebRTC data channel as a binary frame.
5. **Head** (host). Final RMSNorm, then the LM head matvec over the vocabulary (248,320 rows) → logits → readback → sampling on the CPU.
6. **Draft** (host). The model's built-in multi-token-prediction layer proposes the next token(s) from the last hidden state; see below.

Hosts that also hold layers (the usual case) run steps 2–3 for their own range before the first hop.

## Prefill

The prompt is known upfront, so it is processed in batches of up to 16 tokens per GPU pass and per network round. Column-batched kernels read each weight block once for all columns, and full 16-column passes go through the prefill GEMM (`engine/wgsl/gemm.js`, Q4_0 and Q8_0). Up to 6 prefill rounds are in flight around the chain. During prefill nothing runs the LM head; the pass only fills the KV caches and recurrent states. Causality inside a batch is preserved by strictly ordered column processing in the recurrent kernels and by appending K/V before attending.

## Speculative decoding

Qwen 3.5/3.8 ship a `nextn` draft block. Given the trunk's last hidden state and the just-sampled token, it predicts the next token; chaining it predicts several. The host then verifies `1 + K` tokens in **one** batched trunk pass (up to 16 columns, one network lap in a room). Drafts come from the draft block or, when the answer repeats text already in the context, from prompt lookup (up to 15 at once). Accepted drafts are those the trunk's own sampled token agrees with; the first mismatch ends acceptance, the trunk's token is used, and the recurrent state is rolled back by replaying the accepted columns from the pre-verify state. Because the trunk always decides, the output stream is identical to plain decoding for any sampler. Draft depth (3/5/7) is chosen per room by measured tokens per second.

## Memory and caching

Each device range-fetches only the byte spans of its tensors from the public model repository and streams them into GPU buffers, repacking Q4_0/Q8_0 blocks into separate nibble and f16-scale arrays for coalesced reads. Entries are stored in the browser Cache API with a size stamp, so rejoining a room reloads from disk in seconds.

## Where time goes (GB10, 27B)

Measured with `benchmarks/bench_breakdown.js` (skips kernel families and re-times):

| | per token |
|---|---|
| all matvecs (weights streamed at ~183 GB/s; roofline 184) | 82 ms |
| everything else (small kernels, encode, submit, readback) | ~30 ms |

Decode is at the memory roofline on this GPU; speculation is what raises tokens per second. Prefill now runs through the GEMM in `engine/wgsl/gemm.js` at 16 columns and is still the biggest gap to native (roadmap 02). The table above predates the GEMM, flash attention and the fused kernels.

## Files

- `engine/engine.js`: public entry point; re-exports the modules below.
- `engine/dense.js`: `DenseEngine` for dense models (Qwen3, SmolLM); `engine/qwen35.js`: `Qwen35Engine` for the hybrid family, batched paths and MTP speculation.
- `engine/wgsl/`: `base.js` (shared kernels), `coop.js` (the cooperative-GEMV generator), `qwen35.js` (DeltaNet and attention glue).
- `engine/gguf.js`: GGUF header/tensor parsing, tokenizer extraction, quantization/repacking, streaming upload. `engine/safetensors.js`: the SmolLM path.
- `engine/tokenizer.js`, `sampling.js`, `quant.js`, `autotune.js`, `selftest.js`: what their names say.
- `engine/wgsl/gemm.js` (prefill GEMM), `engine/wgsl/moe.js` (mixture-of-experts router and expert kernels).
- `room.js`: the room: signaling, links, layer assignment, download orchestration, the generation loop. `p2p.html` holds its markup and styles and is served at `/room`.
- `room/model-loader.js`: browser model headers, tensor prefetch/retry, converted-weight caching and engine construction, including the optional draft model. The room supplies its state, range transport and lazily imported engine constructors; GPU acquisition, progress/pacing, membership and recovery stay in `room.js`.
- `room/`: `transport.js` (the hidden-state wire, `PROTOCOL`), `wire.js` (frame packing), `conversation.js` (the conversation and the exact tokens every device holds), `models.js` (the model list and context per model), `plan.js` (layer split and model ladder), `compute.js` (the compute screen), `lookup.js` (prompt-lookup drafts), `preflight.js` (can this browser hold layers), `qr.js`, `card.js` (the room card), `markdown.js`, `sampling.js`, `visibility.js`, and `code.js` / `code-ui.js` (the Code pane).
- `index.html` + `site/`: the landing page.

## Code mode (`harness/`)

The host types a request; the room's model runs a small agent loop and edits a project in the browser.

- `agent.js`: the tool loop, with approval for edits. `tools.js` and `constrain.js`: Qwen's tool-call formats, a streaming parser, and a constraint that keeps tool names to the declared ones while sampling.
- `room-model.js`, `engine-model.js`, `model-common.js`: the agent's model, either the whole room or one local engine, with prefix reuse between steps.
- `codetools.js`, `workspace.js`, `projects.js`, `diff.js`: list, read, search, edit and write files in a scratch project (browser storage) or a folder on disk.
- `preview.js`, `preview-build.js`, `preview-frame.js`, `preview-relay.html`, `preview-sync.js`, `preview-tools.js`, `run-js.js`: serve the project on a virtual port (`:5173`) in a sandboxed frame, collect its console errors for the agent, and mirror it to peers.
- `sessions.js`, `statecache.js`, `prefix.js`: several conversations on one engine, parked in GPU slots or on disk (OPFS), resumed bit-exactly. See [long-context-and-sessions.md](long-context-and-sessions.md).

Design: [design/harness-app.md](design/harness-app.md).
