import { buildIds, specials } from "../room/conversation.js";
import { pickSampler } from "../room/sampling.js";
import { deliveryRate } from "./bench_room_report.js";

export function checkRequest({ prompt, mode = "plain", maxNew = 400 } = {}) {
  if (!["plain", "defaults", "spec-greedy"].includes(mode)) throw new Error("unknown benchmark mode");
  if (typeof prompt !== "string" || !Number.isSafeInteger(maxNew) || maxNew < 1) throw new Error("invalid benchmark request");
  return { prompt, mode, maxNew };
}

// Called with the room's normal generation lock held. Timing never includes IPC
// from the automation process; these timestamps all belong to the model host.
export async function sampleRoom({ state: ai, snapshot, reset, generate, tok, maxSeq }, { prompt, mode, maxNew }) {
  const before = snapshot(), started = performance.now(), tokenTimes = [], observed = [], drafted = [];
  let inputIds = [], result = null, error = null;
  const sampling = mode === "defaults" ? ai.settings.sampling : "exact";
  try {
    reset();
    if (ai.fed.length || ai.pos) throw new Error("prefix reset failed");
    const S = specials(tok);
    inputIds = buildIds(tok, { turns: [{ role: "user", text: prompt }], thinking: false });
    if (inputIds.length + maxNew + 2 > maxSeq) throw new Error("benchmark exceeds context");
    result = await generate(inputIds, { maxNew, spec: mode !== "plain", sample: pickSampler(sampling),
      stop: new Set([S.imEnd, S.eot].filter(Number.isInteger)),
      onToken(id, draft) { observed.push(id); drafted.push(!!draft); tokenTimes.push(performance.now() - started); } });
  } catch (e) { error = String(e?.message || e); }
  const elapsedMs = performance.now() - started, after = snapshot();
  const invalid = result?.resumed ? "generation recovered during measurement"
    : result && result.reused !== 0 ? "prefix reuse was not cleared"
    : JSON.stringify(before.chain) !== JSON.stringify(after.chain) || JSON.stringify(before.layers) !== JSON.stringify(after.layers)
      ? "layer assignment changed" : null;
  const interrupted = result?.reason === "abort" || result?.reason === "ctx";
  return { status: error || interrupted ? "failed" : invalid ? "non-equivalent" : "pass", error, invalid, before, after,
    prompt, inputIds, mode, sampling, thinking: false, maxNew, result, observed, drafted, tokenTimes,
    ttftMs: tokenTimes[0] ?? null, elapsedMs, deliveryTps: deliveryRate(tokenTimes) };
}

export function checkEpoch(epoch) {
  if (!(typeof epoch === "string" && epoch) && !(Number.isSafeInteger(epoch) && epoch >= 0)) throw new Error("missing chain epoch");
}
