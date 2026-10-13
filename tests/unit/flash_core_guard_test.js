// room/flash.js qwen4expCore checkpoints the dense f32 layout (f32 K / V rows, no QSA state); an engine built with
// kv: "f16" or QSA (engine/qwen4exp.js, the long-context options) must be refused until the room delegates to the
// engine's _stateParts / stateSignature, rather than checkpoint the wrong rows silently.
// No GPU.   deno test --allow-read tests/unit/flash_core_guard_test.js
import { qwen4expCore } from "../../room/flash.js";

const fake = (extra) => ({ device: {}, cfg: { hDim: 8, nKVHeads: 1, headDim: 2 }, dims: {}, lo: 0, hi: 1, hasEmbed: true, hasHead: false,
  maxSeq: 4, pos: 0, layers: [{ attnOps: [], ffnOps: [], st: {} }], ...extra });
const throws = (f, re) => { try { f(); } catch (err) { if (re.test(err.message)) return; throw err; } throw new Error("did not throw"); };

Deno.test("flash core: the dense f32 engine (the room loader's) is accepted", () => {
  qwen4expCore(fake({}));
  qwen4expCore(fake({ kv: "f32", qsa: false }));
});

Deno.test("flash core: an f16 KV or QSA engine is refused (its checkpoint layout differs)", () => {
  throws(() => qwen4expCore(fake({ kv: "f16", qsa: false })), /not kv f16/);
  throws(() => qwen4expCore(fake({ kv: "f32", qsa: true })), /with QSA/);
});
