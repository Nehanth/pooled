// No GPU: OpenClaw's real requests (recorded from a gateway by a mock OpenAI server, one JSON line per
// request: { path, body }) -> the plugin's ask -> the host's checks and prompt, with a real tokenizer.
// Prints the prompt size per request against the room contexts the plugin offers.
//   node packages/openclaw/test/offline.mjs <requests.jsonl> <tokenizer.json> [template.jinja]
import fs from "node:fs";
import { toAsk } from "../src/convert.js";
import { modelInfo } from "../src/pool.js";
import { validateApiAsk, apiPrompt2, TurnCache, EncodeCache } from "../../../room/api.js";
import { templateProfile } from "../../../room/conversation.js";
import { makeTokenizer } from "../../../engine/tokenizer.js";

const [REQ, TOK, TPL] = process.argv.slice(2);
if (!REQ || !TOK) { console.error("usage: offline.mjs <requests.jsonl> <tokenizer.json> [template.jinja]"); process.exit(2); }
const tok = makeTokenizer(JSON.parse(fs.readFileSync(TOK, "utf8")));
const profile = templateProfile(TPL ? fs.readFileSync(TPL, "utf8") : "", tok);
const text = (c) => typeof c === "string" ? c : (c || []).map((p) => p.text || "").join("");
let n = 0, bad = 0;
for (const line of fs.readFileSync(REQ, "utf8").trim().split("\n")) {
  const rec = JSON.parse(line), b = rec.body;
  if (!b?.messages || !String(rec.path || "").includes("chat")) continue;
  // the OpenAI body OpenClaw sent -> the context OpenClaw's agent loop hands a provider's StreamFn
  const ids = new Map();
  const messages = [];
  for (const m of b.messages) {
    if (m.role === "system") continue;
    if (m.role === "user") messages.push({ role: "user", content: text(m.content) });
    else if (m.role === "assistant") messages.push({ role: "assistant", content: [...(text(m.content) ? [{ type: "text", text: text(m.content) }] : []),
      ...(m.tool_calls || []).map((c) => { ids.set(c.id, c.function.name); return { type: "toolCall", id: c.id, name: c.function.name, arguments: JSON.parse(c.function.arguments || "{}") }; })] });
    else if (m.role === "tool") messages.push({ role: "toolResult", toolCallId: m.tool_call_id, toolName: ids.get(m.tool_call_id), content: [{ type: "text", text: text(m.content) }] });
  }
  const context = { systemPrompt: b.messages.filter((m) => m.role === "system").map((m) => text(m.content)).join("\n"), messages,
    tools: (b.tools || []).map((t) => ({ name: t.function.name, description: t.function.description, parameters: t.function.parameters })) };
  n++;
  try {
    const { body, v2 } = toAsk(context, { maxTokens: b.max_tokens || b.max_completion_tokens || 4096 }, { maxTokens: 4096 }, { hostMeta: { api: 2, ctx: 65536 }, log: (m) => console.log("  log:", m) });
    const v = validateApiAsk({ rid: "oc" + n, ...body }, { profile });
    if (v.err) { bad++; console.log(JSON.stringify({ n, err: v.err })); continue; }
    const sizes = {};
    for (const m of ["qwen3-1.7b", "qwen3.6-35b-moe"]) {
      const p = apiPrompt2(tok, v.req, modelInfo(m).ctx, { profile, cache: new TurnCache(), encoder: new EncodeCache(), model: m });
      sizes[m] = p.err ? `${p.code}: ${p.n ?? ""}` : p.ids.length;
    }
    console.log(JSON.stringify({ n, v2, messages: body.messages.length, tools: body.tools?.length || 0, bytes: JSON.stringify(body).length, promptTokens: sizes, style: profile.style }));
  } catch (e) { bad++; console.log(JSON.stringify({ n, err: e.message })); }
}
console.log(bad ? `FAIL: ${bad} of ${n} requests did not convert` : `PASS: ${n} requests converted`);
process.exit(bad || !n ? 1 : 0);
