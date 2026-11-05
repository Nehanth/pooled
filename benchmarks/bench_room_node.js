import { checkEpoch, checkRequest, sampleRoom } from "./bench_room_sample.js";
import { wireSnapshot } from "./bench_room_wire.js";
import { buildIds } from "../room/conversation.js";
import { pairClient } from "./bench_room_client.js";

// The node owns its CkptIndex and promise lock; the browser owns different helpers.
// Both adapters measure the same prompt/generation boundaries in sampleRoom.
export function nodeBenchmark(room, { timing = null } = {}) {
  const ai = room.ai;
  let epoch = null;
  let restoreGenerate = null;
  const nameOf = (id) => room.conns.get(id)?.name || id;
  const wire = wireSnapshot(room.conns, nameOf);
  const snapshot = () => ({ model: ai.model, role: ai.role, name: room.name,
    ready: room.whole(), busy: !!ai.busy, chain: ai.chain.map(nameOf), range: ai.range?.slice() ?? null,
    layers: Object.fromEntries([[room.name, ai.range?.slice() ?? null], ...[...ai.plan].map(([id, p]) => [nameOf(id), p.msg.range.slice()])]),
    settings: { ...ai.settings }, context: room.ctxMax(), options: { profile: "node", flags: room.flags },
    loaded: !!ai.engine, readyWorkers: ai.readyPeers.size, degraded: !!ai.degraded, wire: wire() });
  return {
    snapshot,
    prepareClient(request) {
      const { prompt, mode, maxNew } = checkRequest(request);
      if (!room.hosting() || !room.whole() || ai.busy || restoreGenerate) throw new Error("host is not ready");
      return room.locked(() => {
        room.ckptClear(true); room.resetState();
        if (ai.fed.length || ai.pos) throw new Error("prefix reset failed");
        const inputIds = buildIds(ai.tok, { turns: [{ role: "user", text: prompt }], thinking: false });
        if (inputIds.length + maxNew + 2 > room.ctxMax()) throw new Error("benchmark exceeds context");
        if (mode === "plain") {
          // The normal API does not transport a speculation option. Apply the
          // supported generate option on this dedicated instance for this attempt;
          // keep normal API validation, locking, recovery and sampling intact.
          const generate = room.generate, own = Object.getOwnPropertyDescriptor(room, "generate");
          room.generate = (ids, opts) => generate.call(room, ids, { ...opts, spec: false });
          restoreGenerate = () => { if (own) Object.defineProperty(room, "generate", own); else delete room.generate; };
        }
        return { before: snapshot(), inputIds };
      });
    },
    pairClient(client, capture, prepared) {
      const after = snapshot(), before = prepared.before;
      if (JSON.stringify([before.model, before.chain, before.layers]) !== JSON.stringify([after.model, after.chain, after.layers])) {
        throw new Error("host assignment changed during client request");
      }
      return pairClient(client, capture, prepared.inputIds, (ids) => ai.tok.decode(ids));
    },
    begin(id, chainEpoch) {
      if (ai.busy || ai.waiters.size) throw new Error("room is busy");
      checkEpoch(chainEpoch);
      timing?.begin(id); epoch = chainEpoch;
    },
    async end() {
      try {
        await ai.q;
        if (ai.waiters.size) throw new Error("frames still outstanding");
        return timing ? { ...timing.end(), epoch } : null;
      } finally { restoreGenerate?.(); restoreGenerate = null; }
    },
    run(request) {
      const checked = checkRequest(request);
      if (!room.hosting() || !room.whole() || ai.busy) throw new Error("host is not ready");
      return room.locked(() => sampleRoom({ state: ai, snapshot, tok: ai.tok, maxSeq: room.ctxMax(),
        reset() { room.ckptClear(true); room.resetState(); },
        generate: (ids, options) => room.generate(ids, options) }, checked));
    },
  };
}
