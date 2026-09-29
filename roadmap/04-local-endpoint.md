# 04 · `npx pooled serve`: an OpenAI-compatible local endpoint

**Phase:** later · **Status:** built on branch `feat/serve` (not published): `npx @pooled/cli serve <CODE>` on port 8080, OpenAI and Anthropic APIs. The npm name `pooled` is taken; the `@pooled` scope is free. The design that replaced the sketch below: [docs/design/serve.md](../docs/design/serve.md)

## Why
Every local-AI tool (Continue.dev, Open WebUI, LangChain, LiteLLM) speaks the OpenAI chat API to a `localhost` URL. One base-URL swap makes all of them use the room's model. A browser tab cannot accept inbound HTTP, so a small bridge is required.

## Design
- A ~5 MB CLI (Deno-compiled) that opens `http://localhost:11435/v1/chat/completions` (SSE streaming, JSON mode) and forwards requests to the host tab over the same WebRTC channel peers use.
- Bound to 127.0.0.1 with a generated bearer token printed as `OPENAI_API_KEY`; requests attributed in the room UI ("a Continue.dev session asked a question").
- Docs on day one: Continue.dev (chat/edit roles; autocomplete needs a local small model), Open WebUI, LiteLLM.

## Done when
- `curl localhost:11435/v1/chat/completions` streams an answer from a room; Continue.dev chat works with only a base URL change.
