// The local HTTP server: 127.0.0.1 only, Host/Origin checks, the optional token, routing, the
// local FIFO (one request in flight in the room at a time), and turning the room's messages for a
// request into an OpenAI or Anthropic response or stream.
import http from "node:http";
import { timingSafeEqual, createHash } from "node:crypto";
import { ApiError, bad, clientFromUA, newRid, LIMITS } from "./common.js";
import { parseOpenAI, openaiError, openaiResponse, OpenAIStream, openaiModels } from "./openai.js";
import { parseAnthropic, anthropicError, anthropicResponse, AnthropicStream, anthropicModels } from "./anthropic.js";
import { SSEWriter } from "./sse.js";

const KEEPALIVE_MS = 10000;
// How long a request the room has may go without a message from it before it fails with 504: the
// room took it and went quiet (a bug, or a host that dropped it). Generous: prefilling a long prompt
// in a room of phones can take minutes before the first token. While the host holds it in its own
// queue behind other answers (ai-queued), the wait can be long and legitimate: 30 minutes.
const IDLE_MS = 300000, HOST_QUEUED_MS = 1800000;
const NOT_V1 = "is not available in pooled serve v1";

export function createServer({ bridge, port, token = null, maxQueue = 8, log = () => {}, version = "", keepAliveMs = KEEPALIVE_MS, idleMs = IDLE_MS, hostQueuedMs = HOST_QUEUED_MS }) {
  const queue = [];          // jobs waiting here, oldest first
  let active = null;         // the job the room is working on
  const served = { n: 0 };
  let bound = port;          // the port actually listened on (port 0 in tests picks a free one)

  const modelId = () => (bridge.model ? `pooled/${bridge.model}` : null);
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
  function fail(res, api, e) {
    const { status, body } = api === "anthropic" ? anthropicError(e) : openaiError(e);
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
    if (job.sse?.started) job.sse.end(job.fmt.error(e));
    else fail(job.res, job.api, e);
    finish(job);
  }
  function pump() {
    if (active || !queue.length) return;
    const job = queue.shift();
    if (job.state === "done") { setImmediate(pump); return; }
    const why = unavailable();
    if (why) { jobError(job, why); setImmediate(pump); return; }
    active = job;
    job.state = "asked";
    watch(job, idleMs);
    const r = job.req;
    const ok = bridge.ask(job.rid, {
      system: r.system, messages: r.messages,
      params: { maxTokens: r.maxTokens, temperature: r.temperature ?? undefined, topK: r.topK ?? undefined, stop: r.stop, thinking: r.thinking, client: r.client },
    }, (d) => onRoom(job, d));
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
  function onRoom(job, d) {
    if (job.state === "done") return;
    watch(job, d.t === "ai-queued" ? hostQueuedMs : idleMs);
    switch (d.t) {
      case "ai-queued": job.hostPos = d.pos; return;
      case "ai-genstart":
        job.state = "streaming"; job.promptTokens = d.promptTokens;
        clearInterval(job.keep);
        if (job.stream) job.sse.write(job.api === "anthropic" ? job.fmt.start(d.promptTokens) : job.fmt.start());
        return;
      case "ai-token":
        if (job.stream) job.sse.write(job.fmt.token(d.text, !!d.th));
        else if (d.th) job.think += d.text; else job.text += d.text;
        return;
      case "ai-gendone": {
        const usage = { in: d.usage?.in ?? job.promptTokens ?? 0, out: d.usage?.out ?? 0 };
        if (d.reason === "error" || (d.failed && !d.reason)) { jobError(job, new ApiError("server", `generation failed in the room: ${d.err || "unknown error"}`)); return; }
        // the host pressed Stop (the only abort a live client sees): a cut-off answer must not read as a
        // finished one, so it ends as an error (an error event, or 503 / 529), never with stop / end_turn
        if (d.reason === "abort") { jobError(job, new ApiError("unavailable", `the room's host stopped this answer after ${usage.out} tokens`)); return; }
        const out = { reason: d.reason, stopSeq: d.stopSeq, usage, reused: d.reused || 0 };
        served.n++;
        log(`${job.label}: ${usage.in} prompt + ${usage.out} tokens, ${d.reason}${d.reused ? `, ${d.reused} reused` : ""}`);
        if (job.stream) job.sse.end(job.fmt.done(out));
        else if (job.api === "anthropic") json(job.res, 200, anthropicResponse({ id: job.rid, model: modelId(), text: job.text, think: job.think, thinking: job.req.thinking, ...out }));
        else json(job.res, 200, openaiResponse({ id: job.rid, created: job.created, model: modelId(), text: job.text, think: job.think, ...out }));
        finish(job);
        return;
      }
      case "ai-busy": {
        const code = d.code;
        const e = code === "gone" ? new ApiError("unavailable", d.why ? `the room dropped the request: ${d.why}` : "the room dropped the request", { retryAfter: 5 })
          : code === "queue" ? new ApiError("busy", `the room's queue is full: ${d.why || "try again later"}`, { retryAfter: 5 })
          : code === "ctx" ? new ApiError("ctx", job.api === "anthropic" ? `prompt is too long: ${d.n} tokens > ${d.max} maximum`
            : `This model's maximum context length is ${d.max} tokens. However, your messages resulted in ${d.n} tokens.`, { param: "messages" })
          : code === "bad" ? bad(d.why || "the room refused the request")
          : new ApiError("unavailable", d.why || "the room cannot answer now", { retryAfter: 5 });
        jobError(job, e);
        return;
      }
      case "x-fail": jobError(job, new ApiError(d.kind || "unavailable", d.why)); return;
    }
  }

  async function chat(req, res, api) {
    const body = await readBody(req);
    const r = api === "anthropic" ? parseAnthropic(body) : parseOpenAI(body);
    // the label the room shows: the client program from User-Agent, never the request's user /
    // metadata.user_id (often an account or session id)
    r.client = clientFromUA(req.headers["user-agent"]) || "API";
    const why = unavailable();
    if (why) throw why;
    // --max-queue counts requests waiting behind the one the room is answering: 0 means answer only when idle
    if ((active || queue.length) && queue.length >= maxQueue) throw new ApiError("busy", `${queue.length} requests are already waiting here (--max-queue ${maxQueue})`, { retryAfter: 5 });
    const rid = newRid();
    const job = { rid, api, req: r, res, stream: r.stream, created: Math.floor(Date.now() / 1000), state: "waiting", text: "", think: "", hostPos: 0,
      label: `${api === "anthropic" ? "messages" : "chat"} ${rid} (${r.client})` };
    if (job.stream) {
      job.sse = new SSEWriter(res);
      job.fmt = api === "anthropic" ? new AnthropicStream({ id: rid, model: modelId(), thinking: r.thinking }) : new OpenAIStream({ id: rid, created: job.created, model: modelId(), includeUsage: r.includeUsage });
      // while it waits: keep-alives every 10 s (an SSE comment / an Anthropic ping), so proxies and
      // clients do not time out; a request that runs at once keeps a proper HTTP status for errors
      job.keep = setInterval(() => { if (job.state !== "streaming" && job.state !== "done") job.sse.write(job.fmt.keepAlive(ahead(job))); }, keepAliveMs);
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
      else json(res, 200, openaiModels(id, bridge.readySince || 0).data[0]);
      return;
    }
    if (anthropicShape) json(res, 200, anthropicModels(bridge.ready ? id : null, modelLabel(), (bridge.readySince || 0) * 1000));
    else json(res, 200, openaiModels(bridge.ready ? id : null, bridge.readySince || 0));
  }

  function banner() {
    return [`pooled serve ${version} · room ${bridge.code} · ${bridge.ready ? modelLabel() : "model not ready yet"}`, "",
      `OpenAI     http://127.0.0.1:${bound}/v1        POST /v1/chat/completions, GET /v1/models`,
      `Anthropic  http://127.0.0.1:${bound}           POST /v1/messages`,
      `Health     http://127.0.0.1:${bound}/health`, ""].join("\n");
  }

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const path = url.pathname.replace(/\/+$/, "") || "/";
    const api = path.startsWith("/v1/messages") ? "anthropic"
      : path.startsWith("/v1/models") && (req.headers["anthropic-version"] != null || (req.headers["x-api-key"] != null && req.headers.authorization == null)) ? "anthropic" : "openai";
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
        json(res, 200, { ok: true, room: bridge.code, connected: bridge.connected, ready: bridge.ready, model: modelId(), queue: queue.length + (active ? 1 : 0), served: served.n, ...(bridge.kicked ? { closed: bridge.kicked } : {}) });
        return;
      }
      if (!authorized(req)) throw new ApiError("auth", keyOf(req) ? "invalid API key" : "missing API key: this pooled serve was started with a token");
      if (path === "/" && req.method === "GET") { res.writeHead(200, { "content-type": "text/plain; charset=utf-8" }); res.end(banner()); return; }
      if (path === "/v1/models" && req.method === "GET") { models(req, res, ""); return; }
      if (path.startsWith("/v1/models/") && req.method === "GET") { models(req, res, path.slice("/v1/models/".length)); return; }
      if (path === "/v1/chat/completions" && req.method === "POST") { await chat(req, res, "openai"); return; }
      if (path === "/v1/messages" && req.method === "POST") { await chat(req, res, "anthropic"); return; }
      // a known path with the wrong method is 405 with Allow, as the real APIs answer
      const allow = { "/v1/chat/completions": "POST", "/v1/messages": "POST", "/v1/models": "GET", "/health": "GET", "/": "GET" }[path] || (path.startsWith("/v1/models/") ? "GET" : null);
      if (allow) { res.setHeader("allow", allow); throw new ApiError("method", `${req.method} ${path} is not allowed: use ${allow}`); }
      throw new ApiError("notfound", /^\/v1\/(embeddings|completions|responses|messages\/count_tokens)/.test(path) ? `${req.method} ${path} ${NOT_V1}` : `unknown path ${req.method} ${path}`);
    } catch (e) {
      if (!(e instanceof ApiError)) { log(`error: ${e.stack || e}`); e = new ApiError("server", String(e.message || e)); }
      fail(res, api, e);
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
    listen: () => new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(port, "127.0.0.1", () => { server.off("error", reject); bound = server.address().port; resolve(bound); });
    }),
  };
}
