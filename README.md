<p align="center">
  <a href="https://pooled.run"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="site/logo/wordmark-dark.svg">
    <img src="site/logo/wordmark-light.svg" height="56" alt="pooled">
  </picture></a>
</p>
<p align="center"><b>Peer-to-peer LLM inference in the browser. Pool your devices to run big open models, for chat and coding agents.</b></p>
<p align="center">
  <a href="https://pooled.run">Site</a> ·
  <a href="https://pooled.run/room">Start a room</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/bench-log.md">Benchmarks</a> ·
  <a href="roadmap/">Roadmap</a> ·
  <a href="SECURITY.md">Threat model</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-2A45E0">
  <img alt="runtime" src="https://img.shields.io/badge/runs%20on-WebGPU%20%2B%20WebRTC-16171c">
</p>

https://github.com/user-attachments/assets/5142d877-e4d5-4f6b-9b5e-0ef48dcb73f6

<p align="center"><sub>Demo, recorded September 27, 2026 (2:29, with voice and subtitles): Qwen 3.6 35B MoE across a Mac and an iPhone in browser tabs. The phone chats with it, then Code mode builds a Tetris game, fixes its own bug and restyles it. <a href="https://github.com/Nehanth/pooled/releases/download/v1.0.0/pooled-demo-2026-09-27.mp4">Download</a> · <a href="https://github.com/Nehanth/pooled/releases/download/v0.2.0/swarmllm-demo-2026-09-07.mp4">the September 7 demo</a>.</sub></p>

Pooled runs one open model across the devices in a room. Your laptop, a friend's desktop and a phone each hold some of the model's layers, and together they run a model none of them could run alone. You can chat with it, or switch to Code mode and let it build and fix a small app right in the tab. There is nothing to install and no account, and no server does any of the thinking.

## How it works

Friends open one link. Each device (laptop, desktop, phone) holds some of the model's layers and runs them on its own GPU with our own WGSL kernels. For every token, a small hidden state (10 KB on the 27B, 4 KB on the MoE) passes from tab to tab over direct WebRTC connections, and the host turns the result into the next word.

```
host      embed the last token → hidden state
   ↓ a few KB over WebRTC
laptop    layers 0–21           ─┐
desktop   layers 22–42           ├─ each device runs its layers on its own GPU
phone     layers 43–63          ─┘
   ↓ back to the host
host      final norm → LM head → sample → next token (and draft the ones after it)
```

Models in the room today:

| Model | Size | Notes |
|---|---|---|
| Qwen 3.8 27B | Q4_0, needs ~17 GB across the room | hybrid Gated DeltaNet + attention, built-in draft layer for speculative decoding, 16K context (up to 32K) |
| Qwen 3.6 35B MoE | Q4_0, needs ~22.5 GB across the room | 256 experts, 8 active per token, so it decodes several times faster than the 27B; 32K context (up to 64K) |
| Qwen3 1.7B | Q8_0, needs ~4 GB across the room | small and quick, for rooms of phones and light laptops; 8K context |

Every device downloads only its own layers, from Hugging Face or from another device in the room that already has them, and keeps them cached for next time. If a device leaves, the room says so and the host deals its layers out again with one click. Speculative decoding, batched prefill and every kernel trick are checked by golden tests, so the speculative stream is the same as plain decoding.

Details: [docs/architecture.md](docs/architecture.md), [docs/kernels.md](docs/kernels.md), [docs/protocol.md](docs/protocol.md), [docs/models.md](docs/models.md).

## Code mode

Switch the room to Code and ask for something, like "build a tetris game". The room's model runs a small agent loop: it writes the files, serves them on a virtual localhost (`:5173`) in a sandboxed preview, reads its own console errors, and fixes them. Everyone in the room sees the files and can run the same preview on their own screen. The files live in the host's browser, or in a folder on disk the host picks, and edits to a real folder wait for the host's approval.

Design: [docs/design/harness-app.md](docs/design/harness-app.md). What the agent can and cannot touch: [SECURITY.md](SECURITY.md#code-mode).

## Use Pooled from your tools

A room can serve its model to anything that speaks the OpenAI Chat Completions, OpenAI Responses or Anthropic Messages API, tool calling included: coding agents (Codex CLI, Claude Code, opencode), Continue, Open WebUI, LiteLLM, the `openai` and `anthropic` SDKs, curl. In a room, the black **Serve API** button in the header opens the Serve API page with the command for that room; run it on your own computer (Node 22 or newer, no GPU needed there):

```bash
npx @pooled/cli serve "https://pooled.run/r/4TKG9P#k=…"   # the room's invite link, or its code
# pooled serve · room 4TKG9P · Qwen3.6 35B MoE · Q4 · 32768 tokens of context
#   OpenAI     http://127.0.0.1:8080/v1         (OPENAI_BASE_URL, any API key: chat/completions, responses)
#   Anthropic  http://127.0.0.1:8080            (ANTHROPIC_BASE_URL: messages)

curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "pooled", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

The bridge joins the room as an API client with no layers, so requests run on the room's GPUs, one at a time in the room's queue, and show in the chat under the room's visibility setting. It listens on 127.0.0.1 only; `POOLED_TOKEN` makes it require a key. Prompts go to the room's host and, unless the host limits who sees answers, to everyone in the room: see [who sees your prompts](cli/README.md#who-sees-your-prompts). Tool calls (streamed, parallel, every `tool_choice` form), tool results, JSON mode and JSON schemas, reasoning and Responses' `previous_response_id` work on all three APIs; every call and structured answer is grammar-constrained to its schema, so calls always parse, even from the 1.7B. The room's host must run a Pooled with tool calling (older hosts answer tools with a 400 asking to reload). Images become a note (the models read text only); the legacy `/v1/completions` is not served. Setup for each tool, including the context settings Codex, Claude Code and opencode need: [cli/README.md](cli/README.md). Design: [docs/design/serve.md](docs/design/serve.md).

## Run it locally

```bash
git clone https://github.com/Nehanth/pooled && cd pooled
npx -y serve -l 8080 .        # then open http://localhost:8080/room
```

`serve` reads `serve.json` for the `/room` and `/r/:code` rewrites. Production uses the same two rewrites in `vercel.json`, so keep the two files in step.

## Tests

```bash
npm test                                   # unit tests, no GPU (Deno)
npm run e2e:code                           # Code mode end to end, no GPU: two tabs, a real room, a scripted model
node tests/e2e/preview_browser.mjs         # the Code mode preview sandbox, no GPU
node tests/e2e/room_synth.mjs              # the real engine and room on a tiny synthetic model, no GPU
npm run test:gpu                           # golden tests on Qwen3 0.6B (needs a WebGPU GPU and Deno 2)
npm run test:q38                           # 27B suites, including speculative == plain
```

The no-GPU tests run headless Chromium with WebGPU on SwiftShader. They need `npm install` (Playwright and the `peer` server). GPU tests read model files from `models/`; see [docs/models.md](docs/models.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Architecture

```
engine/            the WebGPU engine (ES modules; engine.js re-exports the public API)
  dense.js         DenseEngine: dense models (Qwen3, SmolLM)
  qwen35.js        Qwen35Engine: hybrid Gated DeltaNet + attention, MoE, batched paths, speculation
  wgsl/            kernels: base, coop (GEMV family), qwen35 (DeltaNet), gemm (prefill), moe
  gguf.js · tokenizer.js · sampling.js · quant.js · autotune.js · selftest.js · safetensors.js
room.js            the room: signaling, links, layer split, downloads, the generation loop
room/              transport and wire, conversation, models, plan, lookup drafts, preflight, QR, room card, Code pane
harness/           Code mode: agent loop, tools, workspace, preview server and sandbox, sessions
p2p.html           the room's markup and styles (served at /room and /r/<code>)
index.html, site/  the landing page
tests/             unit/ (no GPU) · e2e/ (SwiftShader) · GPU golden tests · run.sh
benchmarks/        tok/s harnesses and the kernel-family profiler
docs/              architecture, kernels, protocol, models, bench log, design, research, archive
roadmap/           one file per planned item
```

More in [docs/architecture.md](docs/architecture.md).

## Performance

All numbers use greedy decoding, and the output matches plain one-device decoding. Full history, with commits and caveats, is in [docs/bench-log.md](docs/bench-log.md).

### One device

NVIDIA GB10 (DGX Spark), Chrome, 2026-09-26, coding prompts, speculative decoding with the model's own draft layer.

| Model | Decode plain | Decode speculative | llama.cpp CUDA (b10840), same GGUF, same machine |
|---|---|---|---|
| Qwen 3.6 35B MoE | 40–46 tok/s | 65–84 tok/s | 85.5 tok/s decode, 2520 tok/s prefill |
| Qwen 3.8 27B | 10.1–10.8 tok/s | 20–27 tok/s | 13.8 tok/s decode |

Native llama.cpp is still ahead on plain decode (about 2x on the MoE, 1.3x on the 27B). Speculative decoding closes most of that on code, where 70–85% of drafts are accepted. Prefill is the biggest gap: the first token of a 172-token prompt takes about 1.0 s on the MoE and 2.5 s on the 27B, against roughly 0.07 s and 0.5 s native. Faster prefill is the top item on the roadmap.

### Across devices, with emulated network latency

Same GB10, 2026-09-27. Each device is its own headless Chromium with real WebRTC over loopback, and every frame of model data a device sends is held back by the stated one-way delay (a timer in the page, not a real network). A 172-token itinerary prompt (`japan`), 128-token answers, mean of two. Decode tok/s, plain / speculative:

| Model | Devices | 0 ms | 5 ms | 20 ms | 50 ms | Time to first token |
|---|---|---|---|---|---|---|
| 35B MoE | 1 | 32.5 / 44.2 | | | | 1.04 s |
| 35B MoE | 2 | 27.9 / 32.7 | 21.1 / 27.9 | 12.9 / 20.6 | 7.2 / 13.5 | 1.10 s (1.13 s at 50 ms) |
| 35B MoE | 3 | 24.8 / 33.3 | 17.7 / 26.9 | 9.9 / 17.5 | 5.2 / 10.8 | 1.14 s |
| 27B | 1 | 9.2 / 14.6 | | | | 2.57 s |
| 27B | 2 | 8.6 / 11.9 | 7.8 / 10.9 | 6.3 / 9.3 | 4.6 / 7.6 | 2.52 s (2.53 s at 50 ms) |
| 27B | 3 | 8.4 / 11.0 | 7.3 / 10.2 | 5.5 / 8.6 | 3.7 / 6.6 | 2.53 s (2.55 s at 50 ms) |

- Plain decode pays one delay per device per token, because a token goes from the host through the workers and back. Three devices at 50 ms add about 150 ms to every token.
- Speculative decoding is what keeps a room usable on a real network. At 20–50 ms it is 1.6–2.1x plain on the MoE and 1.5–1.8x on the 27B.
- Time to first token barely moves with latency (at most +0.03 s at 50 ms), because the prompt goes out in pipelined 16-token frames.
- Splitting costs about 3.5–4 ms per extra device even at 0 ms, to read back, pack and upload the hidden state.
- The one-device rows are lower than the table above because the itinerary prompt drafts worse than code (47–59% accepted, against 70–85%) and a room answer includes the room's own per-token work (sampling, chat, telemetry).

Caveats: loopback has no bandwidth limit, loss or jitter, and only model-data frames are delayed. All devices share one GPU, so their compute never overlaps, which makes the 0 ms rows pessimistic compared with separate machines. Harness: [tests/e2e/room_latency.mjs](tests/e2e/room_latency.mjs).


## How it compares

Other projects split models across machines or run models in a browser tab. Each published speed below is for a different model on different hardware, so read it as what each project can do, not as a ranking. Of the projects that pool several devices, Pooled is the only one where every device just opens a URL, and none of the others ships a coding agent that runs in the browser.

| Project | Runs in | Pools several devices | Install per device | Published speed (source, hardware) | Coding agent built in |
|---|---|---|---|---|---|
| **Pooled** | browser tab (WebGPU) | yes, layers split across laptops, desktops and phones over WebRTC | none, open a URL | 35B MoE: 40–46 tok/s plain, 65–84 speculative on one GB10; 13 / 21 on two devices at 20 ms emulated latency ([Performance](#performance)) | yes, Code mode runs in the tab |
| [exo](https://github.com/exo-explore/exo) | native (MLX); macOS app, Linux CPU only for now | yes, with RDMA over Thunderbolt 5 on Macs | macOS app (DMG or `brew install --cask exo`) or build from source | 31.9 tok/s Qwen3 235B and 32.5 tok/s DeepSeek V3.1 671B, both on 4x M3 Ultra Mac Studio ([Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/)) | no |
| [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | native | yes, remote ggml devices | build with `-DGGML_RPC=ON` on every host; the docs call it fragile and insecure, never for open networks | Qwen3 235B: 20.4 tok/s on 1 node, 15.2 tok/s on 4 nodes, 4x M3 Ultra ([Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/)) | no |
| [distributed-llama](https://github.com/b4rtaz/distributed-llama) | native (CPU, experimental Vulkan) | yes, tensor parallel, power-of-2 node counts | C++ build plus Python launcher | Llama 3.2 3B Q40: 5.95 tok/s on 1 Pi 5, 13.68 tok/s on 4 Pi 5 ([#165](https://github.com/b4rtaz/distributed-llama/discussions/165)); Qwen3 30B A3B: 13.04 tok/s on 4 Pi 5 ([#255](https://github.com/b4rtaz/distributed-llama/discussions/255)) | no |
| [Petals](https://github.com/bigscience-workshop/petals) | native Python, public or private swarm | yes, layers served by volunteers | pip package, PyTorch | up to 6 tok/s for Llama 2 70B and up to 4 tok/s for Falcon 180B ([README](https://github.com/bigscience-workshop/petals)); last commit 2024-08-25 | no |
| [Mesh LLM](https://github.com/Mesh-LLM/mesh-llm) | native (Rust, llama.cpp) plus a web console | yes, layer splits for dense models, expert sharding for MoE, over QUIC | install script, Homebrew or distro package | none in the README | no own agent; launches outside agents against the mesh |
| [WebLLM](https://github.com/mlc-ai/web-llm) | browser tab (WebGPU) | no, whole model in one tab | none for users; npm or CDN for developers | 41.1 tok/s Llama 3.1 8B vs 57.7 native MLC-LLM on an M3 Max ([paper](https://arxiv.org/abs/2412.15803)) | no |
| [Transformers.js](https://github.com/huggingface/transformers.js) | browser or Node (ONNX Runtime, WASM or WebGPU) | no | none for users; npm or CDN for developers | WebGPU up to 100x faster than WASM ([v3 post](https://huggingface.co/blog/transformersjs-v3)) | no |

The engine underneath Pooled is our own WGSL, not WebLLM, MLC or llama.cpp. The model weights and tokenizer come from Qwen, hosting from Hugging Face, and signaling from PeerJS.

## Related projects

- [exo](https://github.com/exo-explore/exo): turns your Macs (and Linux, on CPU for now) into one cluster over MLX, with RDMA over Thunderbolt 5.
- [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc): exposes ggml devices on other hosts so llama.cpp can spread one model across them. Trusted networks only.
- [distributed-llama](https://github.com/b4rtaz/distributed-llama): tensor-parallel CPU and Vulkan inference across home devices, down to Raspberry Pis.
- [Petals](https://github.com/bigscience-workshop/petals): BitTorrent-style public swarm for running and fine-tuning large models in Python.
- [Mesh LLM](https://github.com/Mesh-LLM/mesh-llm): pools GPUs across machines behind one OpenAI-compatible API, built on llama.cpp and iroh.
- [WebLLM](https://github.com/mlc-ai/web-llm): single-tab WebGPU inference compiled with MLC and TVM.
- [Transformers.js](https://github.com/huggingface/transformers.js): Hugging Face models in the browser via ONNX Runtime, on WASM or WebGPU.
- [pi](https://github.com/earendil-works/pi): small, extensible terminal coding agent that works with Ollama, LM Studio, vLLM and other local servers.
- [little-coder](https://github.com/itayinbarr/little-coder): coding agent tuned for small local models, built on pi.
- [aider](https://github.com/Aider-AI/aider): terminal pair-programming agent that works with almost any LLM, including local ones.

## Browsers

Chrome on macOS is the tested host. Safari on an iPhone joins a room and holds a few layers. Safari on a Mac reloads the tab under memory pressure when it holds most of the 27B, so do not host from it. Firefox and Linux Chromium need WebGPU turned on and we have not tested them. Devices without WebGPU join as ask-only guests.

## Privacy

A room is a shared conversation: everyone in it sees the questions and answers, on purpose. No server sees them. The devices running layers work on hidden states, which are *not* private against a determined peer, so run rooms with people you would share a document link with. See [SECURITY.md](SECURITY.md).

## Roadmap

What's next, by area and priority (P0 now, P1 next, P2 later): [roadmap/README.md](roadmap/README.md). Each item will link its GitHub issue once it is filed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md). Benchmark reports from hardware we don't have are especially welcome (there's an issue template). Contributors are listed in [AUTHORS](AUTHORS).

## Citation

```bibtex
@software{pooled2026,
  author = {Narendrula, Nehanth},
  title  = {Pooled: peer-to-peer LLM inference in the browser},
  year   = {2026},
  url    = {https://github.com/Nehanth/pooled}
}
```

## Acknowledgements

Model weights and the GGUF format come from the [Qwen](https://huggingface.co/Qwen) team and [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), whose speculative-decoding graph for Qwen 3.5/3.8 was the reference for ours. Prior work that shaped this: [Petals](https://github.com/bigscience-workshop/petals), [exo](https://github.com/exo-explore/exo), [WebLLM](https://github.com/mlc-ai/web-llm), [LlamaWeb](https://arxiv.org/abs/2605.20706), and the Gated DeltaNet and PipeInfer papers.

## License

[MIT](LICENSE).
