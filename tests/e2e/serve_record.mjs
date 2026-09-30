// Record real model outputs for the API unit tests (tests/fixtures/api/<model>-<case>.json): each
// case's request is rendered to ids exactly as the room host renders a v2 ask (room/conversation.js
// renderApi, the model's own tokenizer and chat template), a llama.cpp server (CPU is fine) samples
// the answer greedily and unconstrained, and the fixture keeps the ids and each token's text.
// The unit tests replay them through the call parser, the grammar (every recorded token of a
// well-formed answer must be allowed) and the host's answer pipeline.
//
//   llama-server -m ~/bello/models/qwen17/model.gguf --port 18501 -c 16384 &
//   node tests/e2e/serve_record.mjs --llama http://127.0.0.1:18501 --model qwen3-1.7b \
//     --gguf ~/bello/models/qwen17/model.gguf [--tokjson ~/bello/models/qwen17/tokenizer.json] [--cases a,b]
//
// --model names the fixture (and the template in tests/fixtures/api/templates/, unless the GGUF has
// one); --tokjson: the tokenizer.json the room loads for this model (the 1.7B), else the GGUF's.
import fs from "node:fs";
import path from "node:path";
import { parseGGUFHeader, tokenizerFromGGUF } from "../../engine/gguf.js";
import { makeTokenizer } from "../../engine/tokenizer.js";
import { renderApi, templateProfile } from "../../room/conversation.js";
import { apiRun2, apiPrompt2, TurnCache, validateApiAsk } from "../../room/api.js";
import { makeSampler } from "../../room/sampling.js";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const OUT = path.join(ROOT, "tests/fixtures/api");
const LLAMA = arg("llama", "http://127.0.0.1:18501");
const MODEL = arg("model", "qwen3-1.7b");
const GGUF = arg("gguf", "");
const TOKJSON = arg("tokjson", "");
const ONLY = arg("cases", "");
// --grammar: the answers go through the host's whole v2 pipeline (room/api.js apiRun2: the tool-call
// grammar, the call parser, ai-call messages), sampling greedily from llama.cpp's top 64 candidates
// per step (the grammar's candidate fast path; a step with no allowed candidate falls back to the
// full mask over the rest). Fixtures: <model>-g-<case>.json.
const GRAMMAR = process.argv.includes("--grammar");

// the GGUF header (metadata + tensor infos): read growing prefixes until it parses
function ggufMeta(file) {
  const fd = fs.openSync(file, "r");
  for (let n = 16 << 20; ; n *= 2) {
    const buf = Buffer.alloc(n);
    const got = fs.readSync(fd, buf, 0, n, 0);
    try { const h = parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + got)); fs.closeSync(fd); return h.meta; }
    catch (e) { if (got < n) throw e; }
  }
}
const meta = GGUF ? ggufMeta(GGUF) : null;
const tok = makeTokenizer(TOKJSON ? JSON.parse(fs.readFileSync(TOKJSON, "utf8")) : tokenizerFromGGUF(meta));
const template = meta?.["tokenizer.chat_template"] || fs.readFileSync(path.join(OUT, "templates", MODEL + ".jinja"), "utf8");
const profile = templateProfile(template, tok);

export const TOOLS = [
  { name: "get_weather", description: "Get the current weather in a city.", parameters: { type: "object", properties: { city: { type: "string", description: "City name" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } }, required: ["city"] } },
  { name: "search", description: "Search the web.", parameters: { type: "object", properties: { query: { type: "string" }, limit: { type: "integer", description: "How many results" }, tags: { type: "array", items: { type: "string" } }, exact: { type: "boolean" }, filters: { type: "object", properties: { site: { type: "string" }, year: { type: "integer" } } } }, required: ["query"] } },
  { name: "write_file", description: "Write a text file.", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } },
];
const user = (text) => ({ role: "user", text });
const CASES = {
  plain: { req: { system: "", tools: null, messages: [user("What is 2+2? Answer in one short sentence.")] }, thinking: false, n: 60 },
  call: { req: { system: "", tools: TOOLS, messages: [user("What's the weather in Paris right now?")] }, thinking: false, n: 120 },
  parallel: { req: { system: "", tools: TOOLS, messages: [user("What's the weather in Paris and in Tokyo? Call the tool once per city, both at once.")] }, thinking: false, n: 200 },
  text_then_call: { req: { system: "Before any tool call, write one short sentence saying what you are about to do.", tools: TOOLS, messages: [user("Check the weather in Oslo for me.")] }, thinking: false, n: 160 },
  args: { req: { system: "", tools: TOOLS, messages: [user("Search the web for \"rust async runtimes\": at most 3 results, tags web and news, exact match, only site docs.rs from year 2024.")] }, thinking: false, n: 240 },
  write: { req: { system: "", tools: TOOLS, messages: [user("Create hello.py that prints \"hi\" and a <b>tag</b> string, using the write_file tool.")] }, thinking: false, n: 200 },
  think_call: { req: { system: "", tools: TOOLS, messages: [user("What's the weather in Rome?")] }, thinking: true, n: 400 },
  think_parallel: { req: { system: "", tools: TOOLS, messages: [user("What's the weather in Paris and in Tokyo?")] }, thinking: true, n: 500 },
  truncated: { req: { system: "", tools: TOOLS, messages: [user("Create notes.txt containing a 10-line poem about the sea, using the write_file tool.")] }, thinking: false, n: 40 },
  after_tool: { req: { system: "", tools: TOOLS, messages: [user("What's the weather in Paris right now?"),
    { role: "assistant", text: "", calls: [{ name: "get_weather", args: { city: "Paris" } }] }, { role: "tool", text: "{\"temp\": 18, \"sky\": \"cloudy\"}" }] }, thinking: false, n: 80 },
  json_schema: { req: { system: "Reply with only a JSON object: {\"city\": string, \"population_millions\": number}.", tools: null, messages: [user("Largest city in Japan?")] }, thinking: false, n: 60 },
};

async function complete(ids, n) {
  const r = await fetch(LLAMA + "/completion", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: ids, n_predict: n, temperature: 0, top_k: 1, return_tokens: true, cache_prompt: false, special: true }) });
  if (!r.ok) throw new Error(`llama.cpp: ${r.status} ${await r.text()}`);
  return r.json();
}

// v2 asks for --grammar: [request, params]
const G = {
  parallel: [{ system: "", tools: TOOLS, messages: [user("What's the weather in Paris and in Tokyo? Call the tool once per city, both at once.")] }, { thinking: false }, 200],
  think_parallel: [{ system: "", tools: TOOLS, messages: [user("What's the weather in Paris and in Tokyo?")] }, { thinking: true }, 600],
  args: [{ system: "", tools: TOOLS, messages: [user("Search the web for \"rust async runtimes\": at most 3 results, tags web and news, exact match, only site docs.rs from year 2024.")] }, { thinking: false }, 240],
  required: [{ system: "", tools: TOOLS, messages: [user("Hi! How are you?")] }, { thinking: false, toolChoice: "required" }, 120],
  named: [{ system: "", tools: TOOLS, messages: [user("What's the weather in Paris?")] }, { thinking: false, toolChoice: { name: "search" } }, 120],
  none: [{ system: "", tools: TOOLS, messages: [user("What's the weather in Paris right now?")] }, { thinking: false, toolChoice: "none" }, 80],
  single: [{ system: "", tools: TOOLS, messages: [user("What's the weather in Paris and in Tokyo? Call the tool once per city, both at once.")] }, { thinking: false, parallel: false }, 200],
  json_schema: [{ system: "", tools: null, messages: [user("Largest city in Japan, and its population in millions?")] },
    { thinking: false, format: { type: "schema", schema: { type: "object", properties: { city: { type: "string" }, population_millions: { type: "number" } }, required: ["city", "population_millions"], additionalProperties: false } } }, 80],
  think_required: [{ system: "", tools: TOOLS, messages: [user("Is it cold in Oslo today?")] }, { thinking: true, toolChoice: "required" }, 700],
};
async function top64(ids) {
  const r = await fetch(LLAMA + "/completion", { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: ids, n_predict: 1, n_probs: 64, temperature: 0, return_tokens: true, cache_prompt: true, special: true }) });
  if (!r.ok) throw new Error(`llama.cpp: ${r.status} ${await r.text()}`);
  return (await r.json()).completion_probabilities[0].top_logprobs;
}
if (GRAMMAR) {
  let vocab = 0;
  for (const v of Object.values(tok.vocab)) if (v >= vocab) vocab = v + 1;
  for (const [name, [r, p, n]] of Object.entries(G)) {
    if (ONLY && !ONLY.split(",").includes(name)) continue;
    const v = validateApiAsk({ api: 2, rid: "rec", system: r.system, messages: r.messages, tools: r.tools, params: { maxTokens: n, temperature: 0, ...p } }, { profile });
    if (v.err) throw new Error(v.err);
    const req = v.req;
    const prompt = apiPrompt2(tok, req, 65536, { profile, cache: new TurnCache(), model: MODEL });
    const sent = [], seq = [];
    let steps = 0, fell = 0;
    const generate = async (ids, { stop, maxNew, sample, signal, onToken }) => {
      for (let k = 0; k < maxNew; k++) {
        if (signal.aborted) return { reason: "abort", reused: 0 };
        const cands = await top64([...ids, ...seq.slice(ids.length - prompt.ids.length)]);
        const lg = new Float32Array(vocab).fill(-1e9);
        for (const c of cands) lg[c.id] = c.logprob;
        const t = sample(lg);
        steps++;
        if (!cands.some((c) => c.id === t)) fell++;
        if (stop.has(t)) return { reason: "stop", reused: 0 };
        seq.push(t);
        onToken(t, 0);
      }
      return { reason: "max", reused: 0 };
    };
    const res = await apiRun2({ tok, req, prompt, generate, send: (m) => sent.push(m), cache: new TurnCache(), fallback: makeSampler({ temp: 0 }), ctxMax: 65536, signal: null, log: (m) => console.log("  log:", m) });
    const fixture = {
      note: "Recorded by tests/e2e/serve_record.mjs --grammar: the host's v2 pipeline (grammar, parser, ai-call) over greedy llama.cpp top-64 candidates.",
      model: MODEL, case: name, grammar: true, profile, req: { system: r.system, tools: r.tools, messages: r.messages }, params: p, maxTokens: n,
      promptTokens: prompt.ids.length, ids: seq, texts: seq.map((t) => tok.decode([t])), text: tok.decode(seq), fellBack: fell,
      sent, result: { reason: res.reason, calls: res.calls, open: res.open, text: res.text, think: res.think, usage: res.usage, err: res.err },
    };
    fs.writeFileSync(path.join(OUT, `${MODEL}-g-${name}.json`), JSON.stringify(fixture, null, 1) + "\n");
    console.log(name, seq.length, "tokens,", res.reason, fell ? `(${fell} steps past the top 64)` : "", JSON.stringify(res.calls), JSON.stringify(res.text.slice(0, 100)));
  }
  process.exit(0);
}

for (const [name, c] of Object.entries(CASES)) {
  if (ONLY && !ONLY.split(",").includes(name)) continue;
  const req = { ...c.req, params: { thinking: c.thinking } };
  const { ids } = renderApi(tok, req, profile, { thinking: c.thinking });
  const res = await complete(ids, c.n);
  const out = res.tokens || [];
  const fixture = {
    note: "Recorded by tests/e2e/serve_record.mjs: greedy, unconstrained llama.cpp output for this request as the room renders it.",
    model: MODEL, case: name, profile, thinking: c.thinking, maxTokens: c.n, req: c.req,
    stop: res.stop_type || null, promptTokens: ids.length,
    ids: out, texts: out.map((t) => tok.decode([t])), text: tok.decode(out),
  };
  fs.writeFileSync(path.join(OUT, `${MODEL}-${name}.json`), JSON.stringify(fixture, null, 1) + "\n");
  console.log(name, out.length, "tokens,", res.stop_type, JSON.stringify(fixture.text.slice(0, 160)));
}
