# Models

Pooled ships no weights. Browsers fetch tensors by HTTP range request from public Hugging Face repositories; local development and tests read the same files from `models/` (git-ignored).

## Supported

| Model | File | Engine | Notes |
|---|---|---|---|
| Qwen 3.8 27B | GGUF Q4_0 (~15 GB), includes the `nextn` draft layer | `Qwen35Engine` | 64 layers: 48 Gated DeltaNet + 16 attention; MTP speculation; 16K context by default, up to 64K |
| Qwen 3.6 35B MoE (Qwen3.6-35B-A3B) | GGUF Q4_0 (bartowski) | `Qwen35Engine` (`qwen35moe`) | 256 experts, 8 active per token (~3B of 35B); router, experts and shared expert on the GPU; 32K context by default, up to 128K |
| Qwen3.5-122B-A10B MoE | GGUF Q4_0 (bartowski, two files, 72.5 GB) | `Qwen35Engine` (`qwen35moe`) | **Developer only, hidden from users**: in the room page's `?dev=1` select and, with `POOLED_DEV=1`, in `pooled list` / `host` / `pull` (pull / rm take its full key without it); `tests/test_moe.js MODEL=122b`. 256 experts, 8 active (~10B of 122B); 32K context by default, up to 128K |
| Qwen3.8 Flash Next (**experimental**) | GGUF Q4_0 (bartowski, three files, ~100.6 GB, of which a 28.8 GB n-gram table) | `Qwen4ExpEngine` (`engine/qwen4exp.js`, rooms via `room/flash.js`) | 125B, 512 experts, 10 active per token (~6B); most experts stay on disk and stream into a GPU cache (disk tier); context up to 64K, tested against llama.cpp; 128K in progress; from the command line, run the CLI from the repo ([cli/README.md](../cli/README.md#qwen38-flash-experimental-run-the-cli-from-the-repo)); in the room picker |
| Qwen3 4B / 1.7B / 0.6B | GGUF Q4_0 / Q8_0 | `DenseEngine` | dense; 0.6B is the golden-test model |
| SmolLM2 135M | safetensors f32 | `DenseEngine` | smallest demo; quantized to Q8 at load |

The room's model list lives in `room/models.js`: `MODELS` (URLs), `NEED_GB` (memory the whole model needs), `PICKER` (what the picker offers: Qwen3 1.7B, Qwen 3.8 27B, Qwen 3.6 35B MoE, plus the models in `DISK_COPY`, which keep a copy on each device's disk: Qwen3.8-Flash-Next; the others, the developer-only 122B included, stay for tests and `?dev=1`) and `CTX` (default and largest context per model). Architectures sharing Qwen 3.5/3.6/3.8's hybrid layout (e.g. Qwen 3.5 2B/9B) need only entries there.

## Local layout for tests and benchmarks

```
models/
  qwen/    model.gguf (Qwen3-0.6B Q8_0), model-q4.gguf, tokenizer.json, config.json
  qwen17/  model.gguf, tokenizer.json, config.json
  q38/     model.gguf (Qwen 3.8 27B Q4_0)
  model/   SmolLM2-135M: model.safetensors, tokenizer.json, config.json
```

## Adding a model

1. Confirm the GGUF tensor names and dims match one of the two engines (`gguf.js` has the name maps).
2. Add entries in `room/models.js` (`MODELS`, `NEED_GB`, and `CTX` if the default context should differ).
3. Generate a golden with the reference implementation in `tests/reference/` and add a test.
4. Record tok/s in `docs/bench-log.md`.
