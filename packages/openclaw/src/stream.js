// OpenClaw's model call, answered by the Pooled room: a custom StreamFn (provider hook
// createStreamFn), so there is no HTTP hop. The conversation becomes the same API ask `pooled serve`
// sends (convert.js: finishRequest + askBody); it goes to the room's host (this process, or another
// device over WebRTC: pool.js transport), which renders it with the model's chat template, constrains
// tool calls with the call grammar and parses them (room/api.js apiRun2). The answer is checked the
// way `pooled serve` checks it (cli/lib/answer.js Ask) and streamed back as OpenClaw assistant events.
//
// sdk: OpenClaw's { createAssistantMessageEventStream, createEmptyTransportUsage, failTransportStream }
// (index.js passes the real ones from openclaw/plugin-sdk; the unit tests pass stand-ins).
import { Ask, Collector } from "../../../cli/lib/answer.js";
import { ids, outcome, ApiError } from "../../../cli/lib/common.js";
import { ensureRoom, roomSettings, transport, PooledError } from "./pool.js";
import { toAsk } from "./convert.js";
import { remember } from "./prewarm.js";

let seq = 0;
const newRid = () => "oc" + Date.now().toString(36) + (seq++).toString(36);

// a refusal from the room (ai-busy, or a failed link) -> a message for the chat
export function busyMessage(ev, r) {
  switch (ev.code) {
    case "ctx": return `the conversation is ${ev.n} tokens; Pooled room ${r.code}'s context holds ${ev.max}. Start a new session (/new) or compact it`;
    case "loading": return `Pooled room ${r.code} is still loading the model; ask again in a moment`;
    case "degraded": return `a device left Pooled room ${r.code} while it held layers of the model; re-open ${r.link} on it, or wait for the room to re-deal the layers`;
    case "off": return `the host of Pooled room ${r.code} does not allow API clients: turn on "Allow API clients" in the room's Serve API panel`;
    case "queue": return `Pooled room ${r.code}'s queue is full; try again after the current answer`;
    case "gone": return `lost the link to Pooled room ${r.code}'s host (${ev.err})`;
    default: return `Pooled room ${r.code}: ${ev.err || "the answer failed"}`;
  }
}

// the checked answer -> OpenClaw's assistant events on `message`
function openclawEncoder(stream, message) {
  let textIdx = -1, thinkIdx = -1;
  const calls = new Map();   // i -> { idx, tc }
  const closeText = () => { if (textIdx >= 0) { stream.push({ type: "text_end", contentIndex: textIdx, content: message.content[textIdx].text, partial: message }); textIdx = -1; } };
  const closeThink = () => { if (thinkIdx >= 0) { stream.push({ type: "thinking_end", contentIndex: thinkIdx, content: message.content[thinkIdx].thinking, partial: message }); thinkIdx = -1; } };
  return {
    calls,
    close() { closeThink(); closeText(); },
    start() {},
    think(t) {
      closeText();
      if (thinkIdx < 0) { thinkIdx = message.content.length; message.content.push({ type: "thinking", thinking: "" }); stream.push({ type: "thinking_start", contentIndex: thinkIdx, partial: message }); }
      message.content[thinkIdx].thinking += t;
      stream.push({ type: "thinking_delta", contentIndex: thinkIdx, delta: t, partial: message });
    },
    text(t) {
      closeThink();
      if (textIdx < 0) { textIdx = message.content.length; message.content.push({ type: "text", text: "" }); stream.push({ type: "text_start", contentIndex: textIdx, partial: message }); }
      message.content[textIdx].text += t;
      stream.push({ type: "text_delta", contentIndex: textIdx, delta: t, partial: message });
    },
    callStart(i, id, name) {
      closeThink(); closeText();
      const idx = message.content.length, tc = { type: "toolCall", id, name, arguments: {} };
      message.content.push(tc);
      calls.set(i, { idx, tc });
      stream.push({ type: "toolcall_start", contentIndex: idx, partial: message });
    },
    callArgs(i, a) { const c = calls.get(i); if (c) stream.push({ type: "toolcall_delta", contentIndex: c.idx, delta: a, partial: message }); },
    callEnd(i, args) {
      const c = calls.get(i); if (!c) return;
      try { c.tc.arguments = JSON.parse(args || "{}"); } catch { c.tc.arguments = {}; }
      c.ended = true;
      stream.push({ type: "toolcall_end", contentIndex: c.idx, toolCall: c.tc, partial: message });
    },
    done() {}, error() {}, keepAlive() {},
  };
}

export function createPooledStream({ getPluginConfig, log = () => {}, sdk }) {
  const { createAssistantMessageEventStream, createEmptyTransportUsage, failTransportStream } = sdk;
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    // (the released 2026.9.6 SDK has no buildAssistantMessage yet: the same shape by hand, as its apple-fm does)
    const message = { role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id, usage: createEmptyTransportUsage(), stopReason: "stop", timestamp: Date.now() };
    stream.push({ type: "start", partial: message });
    void (async () => {
      let r = null;
      const signal = options?.signal;
      try {
        const s = roomSettings(getPluginConfig());
        r = await ensureRoom(s, (m) => log(m));
        const t = await transport(r, { signal });
        r.asked = true;   // (a warm-up after this would only queue behind real work)
        const { req, v2, body } = toAsk(context, options, model, { hostMeta: t.hostMeta, log });
        if (r.s.mode === "host" && r.s.prewarm) remember(body, r.s.model);   // the next gateway start warms up with it
        const enc = openclawEncoder(stream, message);
        const ask = new Ask({ req, v2, encoders: [new Collector(), enc], idFor: () => ids.call(), log, label: `room ${r.code}` });
        let stats = "", promptTokens = 0;
        const answer = await new Promise((resolve, reject) => {
          const rid = newRid();
          let h = null;
          if (signal?.aborted) { reject(Object.assign(new Error("aborted"), { name: "AbortError" })); return; }   // an abort event already fired never fires again
          const onAbort = () => h?.stop();
          signal?.addEventListener?.("abort", onAbort);
          const end = (fn) => { signal?.removeEventListener?.("abort", onAbort); fn(); };
          h = t.ask(rid, body, (d) => {
            if (d.t === "ai-busy") { end(() => reject(new PooledError(d.code || "busy", busyMessage({ code: d.code, err: d.why, n: d.n, max: d.max }, r)))); return; }
            if (d.t === "x-fail") { end(() => reject(new PooledError("gone", busyMessage({ code: "gone", err: d.why }, r)))); return; }
            if (d.t === "ai-genstart") { promptTokens = d.promptTokens; log(`answering in room ${r.code}: ${d.promptTokens} prompt tokens`); }
            if (d.t === "ai-gendone") stats = d.stats || d.err || "";
            const x = ask.feed(d);
            if (x?.error) end(() => reject(x.error));
            else if (x?.answer) end(() => resolve(x.answer));
          });
        });
        enc.close();
        log(`done in room ${r.code}: ${answer.reason}, ${promptTokens} prompt tokens (${answer.reused} reused), ${answer.usage.out} out, calls ${answer.calls.map((c) => c.name).join(",") || "none"}; ${stats}`);
        // the complete calls are the truth (docs/protocol.md): fix up any call whose fragments differ
        answer.calls.forEach((c, i) => {
          const have = enc.calls.get(i);
          let args = {}; try { args = JSON.parse(c.args || "{}"); } catch {}
          if (have) { have.tc.arguments = args; if (!have.ended) stream.push({ type: "toolcall_end", contentIndex: have.idx, toolCall: have.tc, partial: message }); }
        });
        const u = answer.usage;
        message.usage.input = Math.max(0, u.in - answer.reused); message.usage.output = u.out;   // input without the cached prefix (cacheRead)
        message.usage.cacheRead = answer.reused;
        message.usage.totalTokens = u.in + u.out;
        const how = outcome(answer, req);
        message.stopReason = how === "tool" && !answer.open ? "toolUse" : how === "length" || how === "ctx" ? "length" : "stop";
        if (signal?.aborted) throw Object.assign(new Error("aborted"), { name: "AbortError" });
        stream.push({ type: "done", reason: message.stopReason, message });
        stream.end(message);
      } catch (error) {
        const e = error instanceof PooledError || error instanceof ApiError ? new Error(`Pooled: ${error.message}`) : error;
        log(`request failed: ${e.message}`);
        failTransportStream({ stream, output: message, error: e, signal });
      }
    })();
    return stream;
  };
}
