import { checkRequest } from "./bench_room_sample.js";
import { generationSample, deliveryRate } from "./bench_room_report.js";
import { pieceDecoder } from "../room/api.js";

// Reproduce the API's existing UTF-8 holding rule to associate exact emitted IDs
// with their text message. Several IDs can arrive in one chunk and share a time.
// If the message boundaries cannot be explained, leave delivery rate unavailable.
function clientDelivery(tokens, decode, events) {
  const pieces = pieceDecoder({ decode }), groups = [];
  let pending = 0;
  for (const id of tokens) {
    pending++;
    const text = pieces.push(id);
    if (text) { groups.push({ text, count: pending }); pending = 0; }
  }
  const tail = pieces.flush();
  if (tail) { groups.push({ text: tail, count: pending }); pending = 0; }
  const chunks = events.filter((e) => e.type === "ai-token" && !e.thinking && e.text);
  if (pending || groups.length !== chunks.length || groups.some((g, i) => g.text !== chunks[i].text)) {
    return { tokenTimes: null, tps: null, reason: "API text boundaries do not identify every token delivery" };
  }
  const times = groups.flatMap((g, i) => Array(g.count).fill(chunks[i].at));
  return { tokenTimes: times, tps: deliveryRate(times), reason: null };
}

// Pair only an isolated, freshly reset host attempt with this client's reply.
// The caller obtains expectedInput from the host tokenizer before sending, and
// decodes the observed IDs on that same host after measurement (outside TTFT).
export function pairClient(client, capture, expectedInput, decode) {
  if (client.status !== "pass") throw new Error("client request did not pass");
  const host = generationSample(capture), result = host.result;
  if (JSON.stringify(host.inputIds) !== JSON.stringify(expectedInput)) throw new Error("host prompt differs from prepared request");
  if (result.reused !== 0) throw new Error("host reused a cached prefix");
  if (result.reason !== client.done?.reason || result.tokens.length !== client.done?.usage?.out ||
      host.inputIds.length !== client.done?.usage?.in) throw new Error("client and host generation evidence disagree");
  const text = client.events.filter((e) => e.type === "ai-token" && !e.thinking).map((e) => e.text).join("");
  if (decode(result.tokens) !== text) throw new Error("client text differs from host tokens");
  return { ...host, clientTtftMs: client.ttftMs, clientElapsedMs: client.elapsedMs,
    clientDelivery: clientDelivery(result.tokens, decode, client.events), text };
}

// Uses cli/lib/room.js Bridge without owning its connection. This is the API
// client's clock, not automation IPC or the model host's generation clock.
export async function measureRequest(bridge, request, { rid, timeoutMs = 600000, capacity = 2048 } = {}) {
  const { prompt, mode, maxNew } = checkRequest(request);
  if (!/^[A-Za-z0-9_-]{1,32}$/.test(rid || "") || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 ||
      !Number.isSafeInteger(capacity) || capacity < 1 || maxNew > 65536) throw new Error("invalid client measurement options");
  if (bridge.reqs.has(rid)) throw new Error("request id is already in use");
  const body = { system: "", messages: [{ role: "user", text: prompt }],
    params: { maxTokens: maxNew, thinking: false, client: "benchmark", ...(mode === "defaults" ? {} : { temperature: 0 }) } };
  const events = [];
  let started = null, firstTextMs = null, lastTextMs = null, textChunks = 0, dropped = 0, starts = 0;
  return new Promise((resolve) => {
    let timer = null, settled = false;
    const finish = (status, error = null, done = null, cancel = false) => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      const elapsedMs = started === null ? null : performance.now() - started;
      if (cancel) { try { bridge.stop(rid); } catch {} }
      resolve({ status, error, request: { prompt, mode, maxNew }, body, events, dropped, done,
        boundary: "before Bridge.ask to first nonempty answer text on this client; includes queueing and transport",
        ttftMs: firstTextMs, elapsedMs, firstTextMs, lastTextMs, textChunks,
        tokenIds: null, tokenIdsReason: "the existing API transports text chunks, not token IDs" });
    };
    if (!bridge.connected || !bridge.ready) { finish("blocked", "requesting client is not connected to a ready host"); return; }
    timer = setTimeout(() => finish("failed", "client request timed out", null, true), timeoutMs);
    const receive = (m) => {
      if (settled) return;
      // Bridge already routes by rid. Guard this boundary too: unrelated room
      // chat must never supply this request's first text or completion.
      if (m.rid !== rid) { finish("incomplete", "response request id mismatch", null, true); return; }
      const at = performance.now() - started;
      const event = { type: m.t, at };
      if (m.t === "ai-token") {
        event.text = m.text; event.thinking = !!m.th; event.drafted = !!m.d;
        if (typeof m.text !== "string") { finish("incomplete", "invalid response text", null, true); return; }
        if (!starts) { finish("incomplete", "text arrived before generation started", null, true); return; }
        if (m.text && !m.th) { firstTextMs ??= at; lastTextMs = at; textChunks++; }
      } else if (m.t === "ai-genstart") {
        event.promptTokens = m.promptTokens; event.model = m.model;
        if (++starts > 1) { finish("non-equivalent", "generation restarted during measurement", null, true); return; }
      } else if (m.t === "ai-queued") event.position = m.pos;
      if (events.length < capacity) events.push(event); else dropped++;
      if (m.t === "ai-gendone") {
        const done = { reason: m.reason, usage: m.usage, reused: m.reused, stats: m.stats, failed: !!m.failed, error: m.err ?? null };
        if (m.failed || m.err || !["stop", "max"].includes(m.reason)) finish("failed", m.err || `generation ended: ${m.reason}`, done);
        else if (starts !== 1 || firstTextMs === null) finish("incomplete", "completion without an observed answer", done);
        else if (dropped) finish("incomplete", "client events were dropped", done);
        else finish("pass", null, done);
      } else if (m.t === "ai-busy" || m.t === "x-fail") finish("failed", m.why || m.code || m.kind || m.t);
      else if (!["ai-queued", "ai-genstart", "ai-token"].includes(m.t)) finish("incomplete", "unexpected API response: " + m.t, null, true);
    };
    started = performance.now();
    try { if (!bridge.ask(rid, body, receive)) finish("failed", "client could not send request", null, true); }
    catch (e) { finish("failed", String(e?.message || e), null, true); }
  });
}
