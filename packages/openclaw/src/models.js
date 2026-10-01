// The models onboarding offers, where they live on disk, and how much memory this machine lends: the
// same math and the same files as `pooled host` / `pooled pull` (cli/lib hostui.js, lend.js, cache.js,
// room/*.js), so the plugin and the CLI agree on what fits, share ~/.pooled/models and never
// download a model twice.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import os from "node:os";
import { MODELS, FILES, NEED_GB, CTX, maxSeqFor, roomBytes, pickCtx, ctxShortNote, ctxK } from "../../../room/models.js";
import { roomFit, shortNote, shortBy, gbUp } from "../../../room/plan.js";
import { pledgeGB } from "../../../room/pledge.js";
import { LOCAL } from "../../room-node/source.js";
import { modelsDir, modelState, fmtBytes } from "../../../cli/lib/cache.js";
import { recommendModel, pledgeDefaults, modelNeedGB, modelFallback, roomFitNow } from "../../../cli/lib/hostui.js";
import { detectMemory, memoryRule } from "../../../cli/lib/lend.js";

export { MODELS, FILES, fmtBytes, modelsDir };
// what onboarding offers (room/models.js PICKER order)
export const MODEL_CHOICES = ["qwen3-1.7b", "qwen3.8-27b", "qwen3.6-35b-moe"];

// The model for OpenClaw: the 35B MoE when the room can hold it, else the 27B. Measured on the Spark
// (a Spark + a second Linux machine): with the MoE a 2nd turn took 4.6 s (17k tokens reused), a tool
// turn 10 s and a new session 21 s; with the 1.7B a 2nd turn took ~120 s with nothing reused, its
// background memory saves held the room for ~120 s, and a new session looped on tool search
// (85 tool_search/tool_describe calls in 580 s) until OpenClaw aborted the run.
export const OPENCLAW_MODEL = "qwen3.6-35b-moe";
export const OPENCLAW_FALLBACK = "qwen3.8-27b";
export const SMALL_MODELS = new Set(["qwen3-1.7b"]);
export const isSmall = (key) => SMALL_MODELS.has(key);
export const SMALL_WARNING = "Small models struggle with OpenClaw's long prompts and tools: expect slow turns and tool loops. Use the 35B MoE if your devices can hold it.";
// rows: modelChoices' rows -> the key to preselect: the MoE if this device's pledge holds it, else the
// 27B if it does; when neither fits on this device alone the room needs more devices anyway, and the
// MoE needs only ~3 GB more than the 27B, so the MoE
export function recommendForOpenClaw(rows) {
  const row = (k) => rows.find((r) => r.key === k);
  if (row(OPENCLAW_MODEL)?.fitsAlone) return OPENCLAW_MODEL;
  if (row(OPENCLAW_FALLBACK)?.fitsAlone) return OPENCLAW_FALLBACK;
  return row(OPENCLAW_MODEL) ? OPENCLAW_MODEL : recommendModel(rows);
}

// the context a node opens a room with (room-node roomnode.js nodeCtxFor): the qwen35 models' largest
// (OpenClaw's prompts alone are 8-12k tokens), the room's default otherwise
export const nodeCtxFor = (model, ask = 0) => (ask > 0 || MODELS[model]?.kind !== "qwen35" || !CTX[model] ? maxSeqFor(model, ask) : CTX[model].max);
export const lib = { MODELS, FILES, NEED_GB, roomBytes, roomFit, shortNote, shortBy, gbUp, pledgeGB, nodeCtxFor, pickCtx, ctxShortNote };

// the context the plugin opens a room with: what was asked (clamped by room/models.js), else the
// model's largest room context (CTX: 16k on the 1.7B, 64k on the 27B, 128k on the MoE; OpenClaw's
// prompts alone are 8-12k tokens)
export const pluginCtx = (key, ask = 0) => (ask > 0 ? maxSeqFor(key, ask) : CTX[key]?.max ?? maxSeqFor(key));
// what the plugin asks the room node for: the context asked for, else none for a model with a fallback
// context (the node opens the 1.7B at 16k, or at 8k when the room is short of memory for 16k:
// room/models.js pickCtx), else the model's largest (pluginCtx)
export const pluginAsk = (key, ask = 0) => (ask > 0 ? ask : CTX[key]?.fallback ? 0 : pluginCtx(key));

// OpenClaw's own instructions and tools are about 12k tokens and it keeps 4k for the answer: a room
// with a shorter context (`pooled host qwen3-1.7b` from @pooled/cli 0.3.0-0.3.1 opened at 8k) ends every
// turn in "Context overflow"
export const OPENCLAW_MIN_CTX = 16384;
// a joined room's context is too short for OpenClaw: what to tell the owner, else null. A room at the
// model's fallback context (the 1.7B at 8k) is one whose memory was short for 16k: more memory fixes it
export function shortCtxNote(model, ctx, host = "the host") {
  if (!(Number.isInteger(ctx) && ctx > 0 && ctx < OPENCLAW_MIN_CTX)) return null;
  const max = CTX[model]?.max;
  const how = max >= OPENCLAW_MIN_CTX ? `\`pooled host ${model} --ctx ${max}\`` : `a model with a longer context (at least ${OPENCLAW_MIN_CTX / 1024}k)`;
  const short = CTX[model]?.fallback === ctx && max >= OPENCLAW_MIN_CTX
    ? ` (the room opens ${String(MODELS[model]?.label || model).split("·")[0].trim()} at ${ctxK(ctx)} when its memory is short for ${ctxK(max)}: a device lending more, or one more device, gets ${ctxK(max)})` : "";
  return `${host} runs this room with a ${ctx}-token context, but OpenClaw needs ${OPENCLAW_MIN_CTX / 1024}k: its own instructions and tools take about 12k tokens, so every answer would end in "Context overflow". Ask ${host} to open the room again with ${how}${short}.`;
}
// This machine's own room opened the model at its fallback context (room/models.js pickCtx): what to
// tell the owner, else null. ctxNote: the room node's status().ctxNote; needGB: the model's need at 16k
export function ownShortCtxNote(model, ctx, ctxNote, needGB = null) {
  if (!ctxNote || !(ctx > 0 && ctx < OPENCLAW_MIN_CTX)) return null;
  return `${ctxNote}. OpenClaw needs ${OPENCLAW_MIN_CTX / 1024}k (its own instructions and tools take about 12k tokens), so answers may end in "Context overflow". ` +
    `Raise this machine's pledge (/pooled pledge <GB>${needGB ? `: about ${Math.ceil(needGB)} GB across the room` : ""}) or add a device with the room's invite link (/pooled link), then ask again.`;
}

// needGB: the whole deal at that context (room/models.js roomBytes, as the room page counts it)
// minNeed: { ctx, needGB } at the model's fallback context (the 1.7B: 8k), null without one
export function modelInfo(key, ctxAsk = 0) {
  const ctx = pluginCtx(key, ctxAsk);
  return { name: String(MODELS[key]?.label || key).split("·")[0].trim(), needGB: modelNeedGB(lib, key, ctx) ?? NEED_GB[key] ?? null, ctx, fileBytes: FILES[key]?.bytes || null,
    minNeed: modelFallback(lib, key, pluginAsk(key, ctxAsk)) };
}

// on disk in dir (pooled pull's layout, or the room node's older test layout)?
export const diskState = (dir, key) => modelState(dir, key, MODELS, FILES, LOCAL);
export const isPulled = (dir, key) => diskState(dir, key).pulled;

// This GPU, as `pooled host` reads it (no adapter needed): -> { mem, rule, def, max, label }
export function memoryDefaults({ detect = detectMemory } = {}) {
  let mem;
  try {
    mem = detect({
      run: (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", timeout: 5000, stdio: ["ignore", "pipe", "ignore"] }),
      read: (p) => readFileSync(p, "utf8"), totalmem: os.totalmem, freemem: os.freemem,
    });
  } catch (e) { mem = { kind: "unknown", why: e.message }; }
  const rule = memoryRule(mem, null, { maxBufGB: 0 });
  const maxGB = memoryRule(mem, { max: true }, { maxBufGB: 0 }).gb || rule.gb;
  const smallest = Math.min(...MODEL_CHOICES.map((k) => modelInfo(k).needGB || 99));
  const pd = pledgeDefaults(mem, { maxGB, ruleGB: rule.gb, smallestNeedGB: Math.min(smallest, 4) });
  const label = mem.kind === "discrete" ? `${mem.name} · ${Math.round(mem.totalGB)} GB`
    : mem.kind === "unified" ? `${mem.name} · ${Math.round(mem.totalGB)} GB unified memory` : "GPU memory unknown";
  return { mem, rule, def: pd.def, max: Math.max(pd.max, pd.def), label };
}

// onboarding's model list: smallest first, with what is on disk, the download, the need and the
// context -> { rows: [{ key, label, hint, pulled, fileBytes, needGB, fitsAlone }], recommended }
export function modelChoices(dir, pledge) {
  const rows = MODEL_CHOICES.map((key) => {
    const info = modelInfo(key);
    const alone = roomFitNow(lib, { model: key, devices: [{ name: "this machine", meta: { contribGB: pledge, webgpu: true } }], ctxAsk: pluginAsk(key) });
    return { key, name: info.name, ctx: info.ctx, fileBytes: info.fileBytes, needGB: info.needGB, minNeed: info.minNeed, pulled: isPulled(dir, key),
      fitsAlone: alone.fits, aloneCtx: alone.fits ? alone.ctx : null };
  }).sort((a, b) => (a.needGB ?? 99) - (b.needGB ?? 99));
  const recommended = recommendForOpenClaw(rows);
  for (const r of rows) {
    r.hint = [
      r.pulled ? "downloaded ✓" : `${fmtBytes(r.fileBytes)} download`,
      `needs about ${r.needGB} GB across the room${r.minNeed?.needGB ? ` (${r.minNeed.needGB} GB at ${Math.round(r.minNeed.ctx / 1024)}k)` : ""}`,
      `${Math.round(r.ctx / 1024)}k context`,
      r.fitsAlone && r.aloneCtx < r.ctx ? `fits on this machine alone at ${Math.round(r.aloneCtx / 1024)}k (OpenClaw needs ${OPENCLAW_MIN_CTX / 1024}k)`
        : r.fitsAlone ? "fits on this machine alone" : "needs more devices",
      r.key === recommended ? "recommended for OpenClaw" : null,
      isSmall(r.key) ? "small: slow turns and tool loops in OpenClaw" : null,
    ].filter(Boolean).join(" · ");
  }
  return { rows, recommended };
}
