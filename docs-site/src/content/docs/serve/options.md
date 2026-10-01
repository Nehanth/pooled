---
title: Tokens and security
eyebrow: Serve API reference
description: How to require a token on pooled serve, why it only listens locally, and who can see what your tools send.
sidebar:
  label: Tokens and security
  order: 2
---
<!-- Sources (origin/main): cli/bin/pooled.js (token precedence --token-file > POOLED_TOKEN > --token, ps warning),
     cli/lib/http.js (listen on 127.0.0.1, Host/Origin 403, /health without token -> {ok:true}), cli/lib/room.js
     (KNOCK_MS 3000, HOST_WAIT_MS 60000), cli/README.md (Options, Who sees your prompts), SECURITY.md (activations are
     not encryption). Every flag: reference/cli.mdx. -->

Every flag of `pooled serve` is on the [CLI reference](/docs/reference/cli#options).

## Require a token

By default the bridge accepts any API key, because some tools refuse to run without one. To require a real one:

```bash
openssl rand -hex 24 > ~/.pooled-token
npx @pooled/cli serve 4TKG9P --token-file ~/.pooled-token
```

Clients send it as their API key, which the SDKs put in `Authorization: Bearer <token>` or `x-api-key: <token>`. A missing or wrong key is `401`.

You can also set `POOLED_TOKEN` or pass `--token`. If more than one is set, `--token-file` wins, then `POOLED_TOKEN`, then `--token`. Prefer the first two: `--token` shows in the process list.

With a token, every path needs it except `GET /health`, which then answers only `{"ok": true}`.

## Local only

- The bridge listens on `127.0.0.1` only. Other machines on your network can't reach it.
- A request whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>` gets `403`. This blocks DNS rebinding.
- Any request with an `Origin` header gets `403`, so a web page in your browser can't use the room through the bridge.

From inside a Docker container the bridge is not at `127.0.0.1`. Run the tool on the host, or its container with `--network=host`.

## Who sees your prompts

Everything a tool sends goes to the room's host, a browser tab on someone's device, and the other devices compute on it.

- The host's screen shows every API request: the last user message (up to 2,000 characters) and the answer.
- Everyone else in the room sees the same, unless the host sets **Who sees the chat** to **Only me** or **Whoever asked**.
- Devices that run layers receive the model's hidden states. Those are not encryption: published attacks recover much of the text from them. Assume anyone in the room can read your prompts. See the [security model](/docs/internals/security).

:::danger[Only use a room you trust]
Coding tools send file contents, and sometimes secrets.
:::

### When the host's page closes

The bridge knocks on the room's code every 3 s for a minute, in case the page reloads. The room code is the only name involved, so a page that opened a room with that code within the minute would get the next requests. Stop the bridge with Ctrl-C when the room ends.

## What the host controls

On the Serve API page the host can disconnect one client, or turn off **Allow API clients** for the whole room. The bridge then stops taking work and answers `503` (OpenAI) or `529` (Anthropic). It does not reconnect by itself, and `GET /health` shows the reason under `closed`. After a Disconnect, restarting `pooled serve` gets back in.
