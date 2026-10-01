---
title: Comparison
description: How Pooled differs from other projects that split models across machines or run models in a browser tab, and when to pick one of them instead.
eyebrow: Internals
sidebar:
  label: Comparison
  order: 6
---


How Pooled differs from other projects, and when to pick one of them instead. Other projects split models across machines, or run models in a browser tab. Pooled does both: it splits one model across devices, and every device runs its share in a browser tab.

:::caution[Not a ranking]
Each published speed below is for a different model on different hardware, taken from each project's own sources. Read it as what each project can do, not as a head-to-head result.
:::

## At a glance

| Project | Runs in | Pools several devices | Install per device | Coding agent built in |
|---|---|---|---|---|
| **Pooled** | Browser tab (WebGPU) | Yes: layers split across laptops, desktops and phones over WebRTC | None: open a URL | Yes: Code mode runs in the tab |
| [exo](https://github.com/exo-explore/exo) | Native (MLX); macOS app, Linux on CPU for now | Yes, with RDMA over Thunderbolt 5 on Macs | macOS app or build from source | No |
| [llama.cpp RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | Native | Yes: remote ggml devices | Build with `-DGGML_RPC=ON` on every host | No |
| [distributed-llama](https://github.com/b4rtaz/distributed-llama) | Native (CPU, experimental Vulkan) | Yes: tensor parallel, power-of-2 node counts | C++ build plus a Python launcher | No |
| [Petals](https://github.com/bigscience-workshop/petals) | Native Python, public or private swarm | Yes: layers served by volunteers | pip package, PyTorch | No |
| [Mesh LLM](https://github.com/Mesh-LLM/mesh-llm) | Native (Rust, llama.cpp) plus a web console | Yes: layer splits for dense models, expert sharding for MoE, over QUIC | Install script, Homebrew or distro package | No own agent; launches outside agents against the mesh |
| [WebLLM](https://github.com/mlc-ai/web-llm) | Browser tab (WebGPU) | No: the whole model in one tab | None for users; npm or CDN for developers | No |
| [Transformers.js](https://github.com/huggingface/transformers.js) | Browser or Node (ONNX Runtime, WASM or WebGPU) | No | None for users; npm or CDN for developers | No |

## Published speeds

<!-- TODO(fact-check): third-party speeds are copied from README.md "How it compares"; the cited sources were not re-fetched for this page. -->

| Project | Published speed | Hardware | Source |
|---|---|---|---|
| **Pooled** | 35B MoE: 40–46 tok/s plain, 65–84 speculative on one device; 12.9 / 20.6 on two devices at 20 ms emulated latency | One GB10 (DGX Spark) | [Benchmarks](/docs/internals/benchmarks) |
| exo | 31.9 tok/s Qwen3 235B; 32.5 tok/s DeepSeek V3.1 671B | 4× M3 Ultra Mac Studio | [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/) |
| llama.cpp RPC | Qwen3 235B: 20.4 tok/s on 1 node, 15.2 tok/s on 4 nodes | 4× M3 Ultra | [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/) |
| distributed-llama | Llama 3.2 3B Q40: 5.95 tok/s on 1 Pi 5, 13.68 on 4; Qwen3 30B A3B: 13.04 tok/s on 4 | Raspberry Pi 5 | [#165](https://github.com/b4rtaz/distributed-llama/discussions/165), [#255](https://github.com/b4rtaz/distributed-llama/discussions/255) |
| Petals | Up to 6 tok/s Llama 2 70B; up to 4 tok/s Falcon 180B | Public swarm | [README](https://github.com/bigscience-workshop/petals) (last commit 2024-08-25) |
| Mesh LLM | None published in the README | | |
| WebLLM | 41.1 tok/s Llama 3.1 8B, against 57.7 native MLC-LLM | M3 Max | [paper](https://arxiv.org/abs/2412.15803) |
| Transformers.js | WebGPU up to 100× faster than WASM | | [v3 post](https://huggingface.co/blog/transformersjs-v3) |

## What is different about Pooled

- **Nothing to install on any device.** Every device opens a URL. Of the projects that pool devices, the others all need a native install or a build on each machine.
- **Phones can join.** A phone can ask questions, and hold layers when the computers can't hold the model.
- **Direct links between browsers.** Model traffic goes device to device over WebRTC. No server does inference, and there is no account.
- **Our own engine.** Pooled's kernels are hand-written WGSL, not WebLLM, MLC or llama.cpp. That is what lets it split a model mid-way and run hybrid DeltaNet models. See [Engine and kernels](/docs/internals/engine).
- **A coding agent in the tab.** Code mode writes, runs and fixes a small web app on the room's model. The room can also serve its model to outside tools through the [Serve API](/docs/serve).

## When to pick something else

- **You need the fastest single machine.** Native llama.cpp is still about 2× faster on plain decode and much faster on prefill. See [Benchmarks](/docs/internals/benchmarks#one-device).
- **You have several Macs with Thunderbolt 5.** exo's RDMA links are far faster than Wi-Fi, and it runs much bigger models.
- **You want one model in one tab, embedded in your own app.** WebLLM and Transformers.js are libraries built for that.
- **You want to share a model with strangers.** Pooled rooms are for people you trust. See [Security model](/docs/internals/security).

## Related projects

Besides the projects in the table, these coding agents work well with local models: [pi](https://github.com/earendil-works/pi), [little-coder](https://github.com/itayinbarr/little-coder) (built on pi) and [aider](https://github.com/Aider-AI/aider).

Model weights and the tokenizer come from [Qwen](https://huggingface.co/Qwen), hosting from Hugging Face, and signaling from PeerJS. llama.cpp's speculative-decoding graph for Qwen 3.5 was the reference for Pooled's.
