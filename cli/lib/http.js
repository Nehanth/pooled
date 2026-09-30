// The local HTTP server: 127.0.0.1 only, Host/Origin checks, the optional token, routing to the API
// adapters (openai.js, anthropic.js, responses.js), the local FIFO (one request in flight in the
// room at a time), and turning the room's messages for a request into the adapter's response or
// stream (answer.js). docs/design/serve.md sections 3, 4 and 11.
import http from "node:http";
import { timingSafeEqual, createHash } from "node:crypto";
import { ApiError, bad, clientFromUA, newRid, LIMITS, cleanText, finishRequest, askBody, needsV2, OLD_HOST_MSG, outcome } from "./common.js";
import { adapter as openai, openaiModels } from "./openai.js";
import { adapter as anthropic, anthropicModels } from "./anthropic.js";
import { adapter as responses } from "./responses.js";
import { SSEWriter } from "./sse.js";
import { Ask, Collector } from "./answer.js";

const KEEPALIVE_MS = 10000;
// How long a request the room has may go without a message from it before it fails with 504: the
// room took it and went quiet (a bug, or a host that dropped it). Generous: prefilling a long prompt
// in a room of phones can take minutes before the first token. While the host holds it in its own
// queue behind other answers (ai-queued), the wait can be long and legitimate: 30 minutes.
const IDLE_MS = 300000, HOST_QUEUED_MS = 1800000;
const count = (x) => Math.max(0, Math.floor(+x) || 0);

// Every API's adapter (the contract: docs/design/serve.md 11). Each lists its routes; a route with
// a handler is a plain one (GET / DELETE of stored objects), one without is an ask (POST a request,
// answered by the room).
export const ADAPTERS = [openai, anthropic, responses];
// extra plain routes, for anything else (register before createServer): handler(req, res, ctx)
const EXTRA = [];
export function register(method, path, handler) { EXTRA.push({ method, path, handler }); }
const matches = (route, path) => (typeof route.path === "string" ? route.path === path : route.path.test(path));

// Settings for the coding agents, from the room's context: neither knows the size of a model it has
// never heard of, so without these it never compacts before the room refuses a prompt that is too long
// Claude Code keeps its output limit (32000 by default) free in the window: at a 16 k or 24 k window
// it refuses its own ~15 k prompt ("Prompt is too long") before sending anything. A quarter of the
// room's context, at most its default, leaves the prompt room (checked with 2.1.285 at 8 k to 64 k).
export const claudeOutput = (ctx) => Math.max(1024, Math.min(32000, Math.floor(ctx / 4)));
// opencode (@ai-sdk/openai-compatible) takes the context and an output limit per model: a quarter of
// the context, at most 8192, as for Claude Code
export const opencodeOutput = (ctx) => Math.min(8192, claudeOutput(ctx));
// a whole opencode.json for the room: the provider, and the model picked (model: the /v1/models id)
export function opencodeConfig(ctx, port, model = "pooled", label = model) {
  return { provider: { pooled: { npm: "@ai-sdk/openai-compatible", name: "Pooled room", options: { baseURL: `http://127.0.0.1:${port}/v1`, apiKey: "x" },
    models: { [model]: { name: label, tool_call: true, limit: { context: ctx, output: opencodeOutput(ctx) } } } } }, model: `pooled/${model}` };
}
export function agentSettings(ctx, port, { model, label } = {}) {
  if (!ctx) return [];
  return [`For this room's ${ctx}-token context:`,
    `  Codex        model_context_window = ${ctx}, model_auto_compact_token_limit = ${Math.floor(ctx * 0.8)}  (~/.codex/config.toml)`,
    `  Claude Code  ANTHROPIC_BASE_URL=http://127.0.0.1:${port} CLAUDE_CODE_MAX_CONTEXT_TOKENS=${ctx} CLAUDE_CODE_MAX_OUTPUT_TOKENS=${claudeOutput(ctx)} CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 claude --model pooled`,
    `  opencode     OPENCODE_DISABLE_CLAUDE_CODE=1 opencode, with this opencode.json:`,
    `    ${JSON.stringify(opencodeConfig(ctx, port, model || "pooled", label || model || "pooled"))}`, ""];
}

export function createServer({ bridge, port, token = null, maxQueue = 8, log = () => {}, version = "", keepAliveMs = KEEPALIVE_MS, idleMs = IDLE_MS, hostQueuedMs = HOST_QUEUED_MS }) {
  const queue = [];          // jobs waiting here, oldest first
  let active = null;         // the job the room is working on
  const served = { n: 0 };
  let bound = port;          // the port actually listened on (port 0 in tests picks a free one)

  const modelId = () => (bridge.model ? `pooled/${bridge.model}` : null);
  // the room's context in tokens, when the host said it (a v2 host: hello meta / ai-ready-all)
  const ctxOf = () => { const c = +bridge.hostMeta?.ctx; return Number.isInteger(c) && c > 0 ? c : null; };
  const modelLabel = () => `${bridge.modelLabel || bridge.model || "model"} (Pooled room ${bridge.code})`;
  // why a request cannot be served now, as an ApiError, or null
  const unavailable = () => {
    if (bridge.kicked) return new ApiError("unavailable", `the room is closed to this client: ${bridge.kicked}`);
    if (!bridge.connected) return new ApiError("unavailable", bridge.gone || `not connected to room ${bridge.code}`, { retryAfter: 5 });
    if (!bridge.ready) return new ApiError("unavailable", "the room's model is not ready yet (loading, or a device left and the host has to re-deal the layers)", { retryAfter: 5 });
    return null;
  };

  function json(res, status, body, headers = {}) {
    if (res.headersSent) { res.end(); return; }
    const s = JSON.stringify(body);
    res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(s), ...headers });
    res.end(s);
  }
  function fail(res, adapter, e) {
    const { status, body } = adapter.error(e);
    json(res, status, body, e.retryAfter ? { "retry-after": String(e.retryAfter) } : {});
  }

  // --token: "Authorization: Bearer <t>" or "x-api-key: <t>", compared in constant time
  const digest = (s) => createHash("sha256").update(String(s)).digest();
  const tokenHash = token ? digest(token) : null;
  function keyOf(req) {
    const auth = String(req.headers.authorization || "");
    return auth.toLowerCase().startsWith("bearer ") ? auth.slice(7).trim() : String(req.headers["x-api-key"] || "");
  }
  const authorized = (req) => !token || timingSafeEqual(digest(keyOf(req)), tokenHash);

  function readBody(req) {
    return new Promise((resolve, reject) => {
      const len = +req.headers["content-length"];
      if (len > LIMITS.body) { reject(new ApiError("toolarge", `the request body is over ${LIMITS.body} bytes`)); req.resume(); return; }
      const parts = []; let n = 0, over = false;
      req.on("data", (c) => { n += c.length; if (n > LIMITS.body) over = true; else parts.push(c); });
      req.on("end", () => {
        if (over) return reject(new ApiError("toolarge", `the request body is over ${LIMITS.body} bytes`));
        const s = Buffer.concat(parts).toString("utf8");
        try { resolve(s ? JSON.parse(s) : null); } catch { reject(bad("the body is not valid JSON")); }
      });
      req.on("error", reject);
    });
  }

  // ---- the queue ----
  function ahead(job) {
    const i = queue.indexOf(job);
    return i >= 0 ? i + (active ? 1 : 0) + (active?.hostPos ? active.hostPos - 1 : 0) : Math.max(0, (job.hostPos || 1) - 1);
  }
  function finish(job) {
    if (job.state === "done") return;
    job.state = "done";
    clearInterval(job.keep);
    clearTimeout(job.idle);
    const i = queue.indexOf(job); if (i >= 0) queue.splice(i, 1);
    if (active === job) active = null;
    setImmediate(pump);
  }
  function jobError(job, e) {
    if (job.state === "done") return;
    log(`${job.label}: ${e.message}`);
    if (job.sse?.started) { job.enc.error(e); job.sse.end(); }
    else fail(job.res, job.adapter, e);
    finish(job);
  }
  function pump() {
    if (active || !queue.length) return;
    const job = queue.shift();
    if (job.state === "done") { setImmediate(pump); return; }
    const why = unavailable();
    if (why) { jobError(job, why); setImmediate(pump); return; }
    // negotiated per ask: v2 only to a host that said it answers v2 (docs/protocol.md "API clients")
    job.v2 = (bridge.hostMeta?.api ?? 1) >= 2;
    const r = job.req;
    if (!job.v2 && job.needsV2) { jobError(job, bad(OLD_HOST_MSG)); setImmediate(pump); return; }
    active = job;
    job.state = "asked";
    job.ask = new Ask({ req: r, meta: job.meta, v2: job.v2, encoders: job.encoders, idFor: job.idFor, log, label: job.label });
    watch(job, idleMs);
    const ok = bridge.ask(job.rid, askBody(r, job.v2), (d) => onRoom(job, d));
    if (!ok) jobError(job, new ApiError("unavailable", "not connected to the room", { retryAfter: 5 }));
  }
  // (re)arm the job's silence timer: on expiry the room is told to stop and the client gets a 504
  function watch(job, ms) {
    clearTimeout(job.idle);
    job.idle = setTimeout(() => {
      if (job.state === "done") return;
      bridge.stop(job.rid);
      jobError(job, new ApiError("timeout", `the room sent nothing for this request in ${+(ms / 1000).toFixed(1)} s`));
    }, ms);
  }
  // the answer is complete: the final events, or the whole response
  function complete(job, a) {
    served.n++;
    // the finish reason the client sees: a turn that ends in calls is tool_calls, not the room's "stop"
    const why = outcome(a, job.req) === "tool" ? "tool_calls" : a.reason;
    log(`${job.label}: ${a.usage.in} prompt + ${a.usage.out} tokens, ${why}${a.calls.length ? `, ${a.calls.length} call${a.calls.length > 1 ? "s" : ""}` : ""}${a.reused ? `, ${a.reused} reused` : ""}`);
    if (job.stream) { job.enc.done(a); job.sse.end(); }
    else json(job.res, 200, job.adapter.final(a, job.req));
    try { job.adapter.after?.(a, job.req); } catch (e) { log(`${job.label}: ${e.message}`); }
    finish(job);
  }
  function onRoom(job, d) {
    if (job.state === "done") return;
    watch(job, d.t === "ai-queued" ? hostQueuedMs : idleMs);
    switch (d.t) {
      case "ai-queued": job.hostPos = count(d.pos); return;
      case "ai-genstart": case "ai-token": case "ai-call": case "ai-gendone": {
        const r = job.ask.feed(d);
        if (d.t === "ai-genstart" && job.ask.state === "streaming") { job.state = "streaming"; }
        if (!r) return;
        if (r.error) { if (d.t !== "ai-gendone") bridge.stop(job.rid); jobError(job, r.error); return; }
        if (d.t !== "ai-gendone") bridge.stop(job.rid);   // cut short by the bridge (the room sent too much)
        complete(job, r.answer);
        return;
      }
      case "ai-busy": {
        const code = d.code, why = cleanText(d.why, 300);
        const e = code === "gone" ? new ApiError("unavailable", why ? `the room dropped the request: ${why}` : "the room dropped the request", { retryAfter: 5 })
          : code === "queue" ? new ApiError("busy", `the room's queue is full: ${why || "try again later"}`, { retryAfter: 5 })
          : code === "ctx" ? new ApiError("ctx", job.api === "anthropic" ? `prompt is too long: ${count(d.n)} tokens > ${count(d.max)} maximum`
            : `This model's maximum context length is ${count(d.max)} tokens. However, your messages resulted in ${count(d.n)} tokens.`, { param: "messages" })
          : code === "bad" ? bad(why || "the room refused the request")
          : new ApiError("unavailable", why || "the room cannot answer now", { retryAfter: 5 });
        jobError(job, e);
        return;
      }
      case "x-fail": jobError(job, new ApiError(d.kind || "unavailable", d.why)); return;
    }
  }

  async function ask(req, res, adapter) {
    const body = await readBody(req);
    let r = adapter.parse(body, req.headers, { log });
    // the label the room shows: the client program from User-Agent, never the request's user /
    // metadata.user_id (often an account or session id)
    r.client = clientFromUA(req.headers["user-agent"]) || "API";
    for (const w of r.extra?.warnings || []) log(`${adapter.label || adapter.api} (${r.client}): ${w}`);
    if (adapter.prepare) r = await adapter.prepare(r);
    const why = unavailable();
    if (why) throw why;
    r = finishRequest(r, { hostMeta: bridge.hostMeta, log });
    // --max-queue counts requests waiting behind the one the room is answering: 0 means answer only when idle
    if ((active || queue.length) && queue.length >= maxQueue) throw new ApiError("busy", `${queue.length} requests are already waiting here (--max-queue ${maxQueue})`, { retryAfter: 5 });
    const rid = newRid();
    const meta = { id: rid, created: Math.floor(Date.now() / 1000), model: modelId() };
    const job = { rid, api: adapter.api, adapter, req: r, res, stream: r.stream, meta, state: "waiting", hostPos: 0,
      needsV2: needsV2(r), idFor: adapter.idFor(r), label: `${adapter.label || adapter.api} ${rid} (${r.client})` };
    const collector = new Collector(meta);
    job.encoders = [collector];
    if (job.stream) {
      job.sse = new SSEWriter(res);
      job.enc = adapter.encoder(r, job.sse, meta);
      job.encoders.push(job.enc);
      // keep-alives every 10 s of silence (an SSE comment / an Anthropic ping), so proxies and
      // clients do not time out: while it waits (a request that runs at once keeps a proper HTTP
      // status for errors) and while the room thinks without sending anything
      job.keep = setInterval(() => {
        if (job.state === "done") return;
        if (job.state !== "streaming") job.enc.keepAlive(ahead(job));
        else if (Date.now() - job.sse.last >= keepAliveMs) job.enc.keepAlive(null);
      }, keepAliveMs);
    }
    res.on("close", () => {
      if (job.state === "done") return;
      log(`${job.label}: the client went away${job.state === "waiting" ? " while it waited" : ""}`);
      if (job.state !== "waiting") bridge.stop(rid);
      finish(job);
    });
    queue.push(job);
    pump();
  }

  function models(req, res, rest) {
    const anthropicShape = req.headers["anthropic-version"] != null || (req.headers["x-api-key"] != null && req.headers.authorization == null);
    const id = modelId();
    if (rest) {
      const want = decodeURIComponent(rest);
      if (!id || want !== id) throw new ApiError("notfound", `model ${want} not found (this room serves ${id || "no model yet"})`);
      if (anthropicShape) json(res, 200, anthropicModels(id, modelLabel(), (bridge.readySince || 0) * 1000).data[0]);
      else json(res, 200, openaiModels(id, bridge.readySince || 0, ctxOf()).data[0]);
      return;
    }
    if (anthropicShape) json(res, 200, anthropicModels(bridge.ready ? id : null, modelLabel(), (bridge.readySince || 0) * 1000));
    else json(res, 200, openaiModels(bridge.ready ? id : null, bridge.readySince || 0, ctxOf()));
  }

  function banner() {
    const ctx = ctxOf();
    return [`pooled serve ${version} · room ${bridge.code} · ${bridge.ready ? modelLabel() : "model not ready yet"}${ctx ? ` · ${ctx} tokens of context` : ""}`, "",
      `OpenAI     http://127.0.0.1:${bound}/v1        POST /v1/chat/completions, POST /v1/responses, GET /v1/models`,
      `Anthropic  http://127.0.0.1:${bound}           POST /v1/messages`,
      `Health     http://127.0.0.1:${bound}/health`, "", ...(bridge.ready ? agentSettings(ctx, bound, { model: modelId(), label: bridge.modelLabel || bridge.model }) : [])].join("\n");
  }

  // which adapter's error shape a path gets
  function adapterFor(req, path) {
    for (const a of ADAPTERS) if (a.routes.some((r) => matches(r, path))) return a;
    if (path.startsWith("/v1/messages")) return anthropic;
    if (path.startsWith("/v1/models") && (req.headers["anthropic-version"] != null || (req.headers["x-api-key"] != null && req.headers.authorization == null))) return anthropic;
    return openai;
  }
  // the methods a known path takes (for 405 with Allow), or null
  function allowed(path) {
    const fixed = { "/v1/models": "GET", "/health": "GET", "/": "GET" }[path] || (path.startsWith("/v1/models/") ? "GET" : null);
    if (fixed) return fixed;
    const m = new Set();
    for (const a of ADAPTERS) for (const r of a.routes) if (matches(r, path)) m.add(r.method);
    for (const r of EXTRA) if (matches(r, path)) m.add(r.method);
    return m.size ? [...m].join(", ") : null;
  }

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const adapter = adapterFor(req, path);
    try {
      // a web page must not drive this endpoint through the user's browser: foreign Host names (DNS
      // rebinding) and any Origin are refused, and no CORS headers are ever sent
      const host = String(req.headers.host || "").toLowerCase();
      if (host !== `127.0.0.1:${bound}` && host !== `localhost:${bound}`) throw new ApiError("forbidden", `Host ${host || "(none)"} is not allowed: use http://127.0.0.1:${bound}`);
      if (req.headers.origin != null) throw new ApiError("forbidden", "requests from web pages (with an Origin header) are not allowed");
      if (path === "/health" && req.method === "GET") {
        // with --token, a caller without it learns only that something is up: the room code alone
        // would let it join the room directly, around the token
        if (!authorized(req)) { json(res, 200, { ok: true }); return; }
        json(res, 200, { ok: true, room: bridge.code, connected: bridge.connected, ready: bridge.ready, model: modelId(), ctx: ctxOf(), queue: queue.length + (active ? 1 : 0), served: served.n, ...(bridge.kicked ? { closed: bridge.kicked } : {}) });
        return;
      }
      if (!authorized(req)) throw new ApiError("auth", keyOf(req) ? "invalid API key" : "missing API key: this pooled serve was started with a token");
      if (path === "/" && req.method === "GET") { res.writeHead(200, { "content-type": "text/plain; charset=utf-8" }); res.end(banner()); return; }
      if (path === "/v1/models" && req.method === "GET") { models(req, res, ""); return; }
      if (path.startsWith("/v1/models/") && req.method === "GET") { models(req, res, path.slice("/v1/models/".length)); return; }
      for (const a of ADAPTERS) {
        const route = a.routes.find((r) => r.method === req.method && matches(r, path));
        if (!route) continue;
        if (route.handler) { await route.handler(req, res, { json, bridge, path, url, readBody: () => readBody(req), log }); return; }
        await ask(req, res, a);
        return;
      }
      const extra = EXTRA.find((r) => r.method === req.method && matches(r, path));
      if (extra) { await extra.handler(req, res, { json, bridge, path, url, readBody: () => readBody(req), log }); return; }
      // a known path with the wrong method is 405 with Allow, as the real APIs answer
      const allow = allowed(path);
      if (allow) { res.setHeader("allow", allow); throw new ApiError("method", `${req.method} ${path} is not allowed: use ${allow}`); }
      if (/^\/v1\/completions$/.test(path)) throw new ApiError("notfound", `${req.method} ${path}: the legacy completions API is not served; use POST /v1/chat/completions`);
      throw new ApiError("notfound", /^\/v1\/(embeddings|messages\/count_tokens)/.test(path) ? `${req.method} ${path} is not available in pooled serve` : `unknown path ${req.method} ${path}`);
    } catch (e) {
      if (!(e instanceof ApiError)) { log(`error: ${e.stack || e}`); e = new ApiError("server", String(e.message || e)); }
      fail(res, adapter, e);
    }
  });
  // a request lasts as long as the room takes (the per-job silence timer above bounds it); only local
  // processes can connect at all, and at most 64 at once
  server.requestTimeout = 0; server.headersTimeout = 60000; server.keepAliveTimeout = 5000;
  server.maxConnections = 64;

  // Ctrl-C: open requests end with an error
  function closeAll(why) {
    for (const job of [...queue, active].filter(Boolean)) jobError(job, new ApiError("unavailable", why));
  }
  // the room went away: the jobs that were waiting here fail now too
  bridge.on("state", () => { if (bridge.kicked || !bridge.connected) for (const job of [...queue]) jobError(job, unavailable()); });

  return {
    server,
    closeAll,
    banner,
    listen: () => new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(port, "127.0.0.1", () => { server.off("error", reject); bound = server.address().port; resolve(bound); });
    }),
  };
}
