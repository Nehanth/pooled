# Native Vulkan Dense backend

The runtime calls Vulkan 1.1 directly from C++ and compiles into both a Linux oracle runner and Android JNI library. It uses existing WGSL entry-point sources compiled ahead of time to SPIR-V. There is no WebGPU runtime in the native backend. Floating-point contraction is disabled for CPU embedding conversion; shader input layouts, quantization and reduction order are preserved.

`shaders.mjs` pins Naga 30.0.1, emits twelve SPIR-V entry points, an embedded C++ header and source/binary hashes. `vulkan.cpp` owns device selection, coherent buffers, descriptors, pipelines, barriers and fenced submissions. `gguf.cpp` bounds-checks GGUF v3 metadata/tensors; only Qwen3 Dense F32/Q8_0 is admitted. `dense.cpp` implements causal KV caches, QK norm, RoPE, attention, residual/MLP, tied/untied heads and memory/context limits. Large output matrices are split into descriptor ranges that fit the device; computation remains row-equivalent.

Oracle calibration uses actual execution of the original WebGPU sources, never a copy of the native algorithm as the expectation. `export-reference.mjs` exports the independently compiled Tint programs from pinned Dawn 0.6.2, assembles/validates core Vulkan 1.1 SPIR-V, and embeds them alongside the Naga production programs. Android executes both on the same GPU to avoid comparing unrelated drivers’ floating-point outputs. The exported reference is calibrated against ordinary WebGPU captures in CI; kernel equivalence is not a universal proof for all devices, models or browser-specific driver workarounds. All bit differences fail. Absolute/relative errors are diagnostic only. Use one physical GPU at a time; tests make no speed claims.

```sh
npm ci --prefix native/vulkan/oracle
cargo install naga-cli --version 30.0.1 --locked
export DUGA_WEBGPU_MODULE=native/vulkan/oracle/node_modules/webgpu/index.js
node native/vulkan/shaders.mjs
node native/vulkan/export-reference.mjs  # spirv-as and spirv-val required
# Requires system Vulkan headers and loader. Software Lavapipe is valid for correctness CI.
g++ -std=c++17 -O2 -ffp-contract=off -Inative/vulkan/build/shaders native/vulkan/vulkan.cpp native/vulkan/oracle.cpp -lvulkan -o native/vulkan/build/duga-vk-oracle
g++ -std=c++17 -O2 -ffp-contract=off -Inative/vulkan/build/shaders native/vulkan/vulkan.cpp native/vulkan/gguf.cpp native/vulkan/dense.cpp native/vulkan/model-run.cpp -lvulkan -o native/vulkan/build/duga-vk-model
export DUGA_WEBGPU_MODULE=native/vulkan/oracle/node_modules/webgpu/index.js
node native/vulkan/capture-oracle.mjs
native/vulkan/build/duga-vk-oracle native/vulkan/build/oracle.dvk
native/vulkan/build/duga-vk-oracle native/vulkan/build/oracle.dvk --reference
native/vulkan/build/duga-vk-oracle native/vulkan/build/oracle.dvk --on-device
node native/vulkan/check-model.mjs
```

Use `DUGA_ORACLE_ADAPTER` (Dawn adapter name) and `DUGA_VK_DEVICE` (native device-name substring) together to select the same physical GPU. Set `VK_ICD_FILENAMES` to a software ICD for deterministic CI. The default is the first adapter in both implementations; explicitly verify reported device names when several GPUs are installed.

Optional real-model check: set `DUGA_REAL_GGUF` and `DUGA_REAL_CONFIG` to locally obtained Qwen3 Dense Q8 files before running `check-model.mjs`. It compares all logits at three fixed input positions. This is not a conversational quality or long-context validation. Checksums, hardware/driver and test scope are recorded in the Android design note. Never treat a passed desktop test as proof for all mobile GPUs.

Oracle modes: default compares Naga against captured WebGPU bytes; `--reference` compares exported Tint against the same bytes; `--on-device` compares Naga against Tint executed on the current GPU. Android JNI uses the last mode. Exactly twelve unique known kernels, non-empty complete output ranges, finite reference values and changed reference outputs are required. Empty or partial fixtures cannot pass. No tolerance, dynamic shader download or WebGPU runtime is introduced.

Set `DUGA_CHAT_TOKENIZER` with a real model to compare the Duga system prompt, full prefill and greedy response (default prompt: `Reply with OK only.`). The original text/tokens are written to `build/chat-oracle.json`, while every logit is still checked exactly. See the Android README for the real native-UI check and test-only Android runners.
