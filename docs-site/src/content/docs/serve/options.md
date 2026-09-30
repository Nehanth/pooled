---
title: CLI options and security
eyebrow: Serve API
description: Every flag of pooled serve, how to require a token, and who can see what your tools send.
sidebar:
  label: CLI options and security
  order: 2
---
<!-- Sources (origin/main 40bb0fa): cli/bin/pooled.js (HELP, parseArgs defaults, token precedence --token-file >
     POOLED_TOKEN > --token, room code 4 to 6 letters and digits, --name default), cli/lib/http.js (listen on 127.0.0.1,
     Host/Origin 403, /health without token -> {ok:true}, 405 with Allow, max-queue 429), cli/lib/room.js (KNOCK_MS 3000,
     HOST_WAIT_MS 60000), cli/README.md (Options, Who sees your prompts), cli/lib/common.js (LIMITS.client 40), SECURITY.md (activations are not encryption). -->

## Usage

```text
pooled serve <ROOM CODE | room link> [options]
```

The room code is 4 to 6 letters and digits. A full room link works too.

## Options

| Option | Default | What it does |
|---|---|---|
| `--port <n>` | `8080` | HTTP port. |
| `--token-file <f>` | none | Require the token in this file on every request. |
| `--token <t>` | none | The same, given on the command line. Other users on the machine can read it with `ps`. |
| `--name <s>` | `pooled serve` and 4 random letters | How the room shows this client. Everyone in the room sees it. |
| `--signal <host:port>` | PeerJS cloud | PeerJS signaling server. Use the same value as the room page's `?signal=`. |
| `--max-queue <n>` | `8` | Requests that may wait in the bridge behind the running one. Past that: `429` (OpenAI) or `529` (Anthropic). `0` accepts a request only when nothing is running. |
| `--quiet` | off | Print only errors. |
| `--json-log` | off | One JSON object per log line. |
| `-v`, `--version` | | Print the version. |
| `-h`, `--help` | | Print help. |

Environment: `POOLED_TOKEN` sets the token, like `--token-file`.

## Require a token

By default the bridge accepts any API key, because some tools refuse to run without one. To require a real one:

```bash
echo "$(openssl rand -hex 24)" > ~/.pooled-token
npx @pooled/cli serve ABCD --token-file ~/.pooled-token
```

Clients then send it as `Authorization: Bearer <token>` or `x-api-key: <token>`. Both OpenAI and Anthropic SDKs do this with their `api_key` setting.

If more than one is set, `--token-file` wins, then `POOLED_TOKEN`, then `--token`. Prefer the first two: they keep the token out of the process list.

With a token set, every path needs it except `GET /health`, which then answers only `{"ok": true}` to callers without it. A missing or wrong key is `401`.

## Local only

- The endpoint listens on `127.0.0.1` only. Other machines on your network cannot reach it.
- A request whose `Host` header is not `127.0.0.1:<port>` or `localhost:<port>` is refused with `403`. This blocks DNS rebinding.
- Any request with an `Origin` header is refused with `403`. So a web page open in your browser cannot use the room through the bridge.

:::note[Docker]
From inside a container, the bridge on the host is not at `127.0.0.1`, and the bridge does not listen on other addresses. Run the tool on the host, or run its container with `--network=host`.
:::

## Who sees your prompts

Everything a tool sends goes to the room's host, a browser tab on someone's device. The room's other devices compute on it.

- The host's screen shows every API request: the last user message (up to 2000 characters; `(tool results)` for an agent step) and the answer.
- Other people in the room see the same, unless the host sets the room's answers to **Only me** or **Whoever asked**.
- Devices that run layers receive the model's activations. Those are not encryption: published attacks recover much of the text from them. Assume anyone in the room can read your prompts ([threat model](https://github.com/Nehanth/pooled/blob/main/SECURITY.md)).

:::danger[Only use a room you trust]
Coding tools send file contents, and sometimes secrets. Point them only at a room whose host and devices you trust.
:::

### When the host's page closes

The bridge knocks on the room's code every 3 s for a minute, in case the page reloads. The room code is the only name involved. A page that opens a room with that code within that minute would receive the next requests.

Stop the bridge with Ctrl-C when the room ends.

## What the host controls

On the room's Serve API page the host sees each connected client and can:

- disconnect one client, or
- turn API clients off for the room.

Either way the bridge stops accepting work and answers `503` (OpenAI) or `529` (Anthropic). It does not reconnect, and `GET /health` shows the reason under `closed`.
