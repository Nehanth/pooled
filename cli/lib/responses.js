// The OpenAI Responses API (docs/design/serve.md section 7). Not built yet: this stub holds the
// routes so the server knows them; the Responses workstream replaces this file (and adds store.js).
import { ApiError } from "./common.js";
import { openaiError } from "./openai.js";

const notYet = () => { throw new ApiError("notfound", "the Responses API (/v1/responses) is not built yet in pooled serve: use /v1/chat/completions or /v1/messages"); };
export const adapter = {
  api: "responses",
  label: "responses",
  routes: [
    { method: "POST", path: "/v1/responses", handler: notYet },
    { method: "GET", path: /^\/v1\/responses\/[^/]+$/, handler: notYet },
    { method: "DELETE", path: /^\/v1\/responses\/[^/]+$/, handler: notYet },
    { method: "GET", path: /^\/v1\/responses\/[^/]+\/input_items$/, handler: notYet },
  ],
  parse: notYet,
  idFor: () => (i) => `call_${i}`,
  encoder: notYet,
  final: notYet,
  error: openaiError,
  streamError: (e) => `data: ${JSON.stringify(openaiError(e).body)}\n\n`,
};
