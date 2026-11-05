import { checkEpoch, checkRequest, sampleRoom } from "./bench_room_sample.js";
import { wireSnapshot } from "./bench_room_wire.js";
import { buildIds } from "../room/conversation.js";
import { pairClient } from "./bench_room_client.js";

// Reject unsupported values before changing either select. Browsers otherwise
// accept an unknown value as empty, which can silently select a fallback policy.
export function configureBrowser({ model, split }) {
  const settings = [["ai-model", model], ["ai-split", split]].filter(([, value]) => value != null);
  for (const [id, value] of settings) {
    const select = document.getElementById(id);
    if (!select || ![...select.options].some((option) => option.value === value)) throw new Error(`unsupported ${id}: ${value}`);
  }
  for (const [id, value] of settings) {
    const select = document.getElementById(id);
    select.value = value; select.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

// Evaluated in the dedicated browser by the CLI while normal model loading runs.
export function browserReady() {
  const errors = window.__benchGpuErrors;
  if (errors?.length) throw new Error("GPU failure during startup: " + errors.join("; "));
  const status = document.getElementById("ai-status")?.textContent || "";
  if (status.startsWith("failed:")) throw new Error(status);
  const state = window.__pooledBench?.snapshot();
  return !!state?.ready && !state.busy;
}

// Opt-in automation for dedicated benchmark rooms (?bench=1). Uses the normal
// room lock, prompt builder and resumable generator, with fresh prefix state.
export function browserBenchmark({ state: ai, pipeline, api, timing, options, connections = new Map() }) {
  let epoch = null;
  const wire = wireSnapshot(connections, api.nameOf);
  const snapshot = () => ({ model: ai.model, role: ai.role, name: api.name(),
    ready: api.ready(), busy: !!ai.busy, chain: ai.chain.map(api.nameOf),
    range: ai.range?.slice() ?? null, layers: { ...ai.layersByName },
    settings: { ...ai.settings }, context: api.maxSeq(), options,
    loaded: !!ai.engine, readyWorkers: ai.readyPeers.size, degraded: !!ai.degraded, wire: wire() });
  return {
    snapshot,
    prepareClient(request) {
      const { prompt, mode, maxNew } = checkRequest(request);
      if (mode === "plain" && (options.lookup || options.denseSpec || ai.engine?.mtp)) throw new Error("plain client runs require speculation disabled");
      if (mode !== "plain" && (!options.lookup || !options.denseSpec)) throw new Error("spec runs require default room flags");
      if (ai.role !== "host" || ai.readyPeers.size < ai.chain.length || !api.lock("benchmark")) throw new Error("host is not ready");
      try {
        pipeline.ckptClear(true); pipeline.resetState();
        if (ai.fed.length || ai.pos) throw new Error("prefix reset failed");
        const inputIds = buildIds(api.tok(), { turns: [{ role: "user", text: prompt }], thinking: false });
        if (inputIds.length + maxNew + 2 > api.maxSeq()) throw new Error("benchmark exceeds context");
        return { before: snapshot(), inputIds };
      } finally { api.unlock(); }
    },
    pairClient(client, capture, prepared) {
      const after = snapshot(), before = prepared.before;
      if (JSON.stringify([before.model, before.chain, before.layers]) !== JSON.stringify([after.model, after.chain, after.layers])) {
        throw new Error("host assignment changed during client request");
      }
      return pairClient(client, capture, prepared.inputIds, (ids) => api.tok().decode(ids));
    },
    begin(id, chainEpoch) {
      if (ai.busy || ai.waiters.size) throw new Error("room is busy");
      checkEpoch(chainEpoch);
      timing?.begin(id);
      epoch = chainEpoch;
    },
    async end() {
      await ai.q;
      if (ai.waiters.size) throw new Error("frames still outstanding");
      return timing ? { ...timing.end(), epoch } : null;
    },
    async run(request) {
      const checked = checkRequest(request), { mode } = checked;
      // The browser's generator uses room flags rather than request.spec for these paths.
      if (mode === "plain" && (options.lookup || options.denseSpec || ai.engine?.mtp)) {
        throw new Error("plain browser runs require ?lookup=0&densespec=0 and a model without MTP");
      }
      if (mode !== "plain" && (!options.lookup || !options.denseSpec)) throw new Error("spec runs require default room flags");
      if (ai.role !== "host" || ai.readyPeers.size < ai.chain.length || !api.lock("benchmark")) throw new Error("host is not ready");
      try {
        return await sampleRoom({ state: ai, snapshot, tok: api.tok(), maxSeq: api.maxSeq(), generate: api.generate,
          reset() { pipeline.ckptClear(true); pipeline.resetState(); } }, checked);
      } finally { api.unlock(); }
    },
  };
}
