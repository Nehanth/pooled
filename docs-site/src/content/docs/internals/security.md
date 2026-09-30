---
title: Security model
description: What a Pooled room protects and what it does not, how Code mode's preview is sandboxed, where weights come from, and how to report a vulnerability.
eyebrow: How it works
sidebar:
  label: Security model
  order: 7
---


This page is the honest version of what Pooled does and does not protect. Read it before you run a room with anyone you would not hand a shared document link to.

:::caution[The short version]
Everyone in a room can read your prompts. Run rooms with people you trust, not strangers.
:::

## How a room works, for security purposes

A room is a set of browsers that split one model's layers. They pass the model's intermediate activations (the "hidden state") to each other over direct WebRTC connections.

- The **host** holds the tokenizer, the embedding table, the LM head and the sampler. Only the host turns hidden states into text.
- The other devices run their layers on the hidden state they receive and pass the result on. They never make sampling decisions.
- The **signaling server** only introduces browsers to each other. It carries no model traffic.

## What the design gives you

- **No counterparty.** There is no account, no API key, and no server that sees your conversation.
- **Transport encryption.** WebRTC data channels are encrypted with DTLS on every hop.
- **Each device loads only its own layers**, from the public model repository or from another device in the room.
- **You choose who sees the chat.** The room is a shared conversation by design, and everyone sees questions and answers on their own screen. The host can limit that:

  | Visibility | Who sees the text |
  |---|---|
  | everyone in the room (default) | Every screen in the room |
  | only me | Only the host's screen |
  | only whoever asked | The host and the device that asked |

  Every device still computes the answer. This only decides which screens get the text.

## What it cannot promise

- **Hidden states are not encryption.** The hidden state that crosses each hop is a lossy transformation of your text. Published attacks reconstruct a large fraction of tokens from mid-model activations (for example [arXiv 2503.09022](https://arxiv.org/abs/2503.09022)). So the visibility setting hides text from screens, not from a determined peer that holds layers. **Assume anyone in your room can read your prompts.**
- **Remote compute is not verified.** A peer could return wrong or manipulated activations. Nothing in the current design detects this. Spot-check auditing is on the roadmap.
- **Weights shared inside the room are not verified either.** A device missing a byte range can take it from another device that has it cached, and only the length is checked. A malicious peer could hand out altered weights. That is the same trust a room already places in its peers' compute. `?peerweights=0` turns sharing off and fetches everything from the model repository.
- **Peers learn metadata** beyond the transcript: device names, memory pledges, layer assignments and timing, through the room roster.
- **Peers see your IP address**, as with any direct WebRTC connection. `?relay=1` with a [TURN relay](/docs/self-host/turn) sends all of a device's traffic through the relay so other devices never see its address.

Noise or permutation "privacy" tricks are not used on purpose. They are known to be breakable and would give a false sense of safety. For the same reason Pooled does not open rooms to strangers by default, and does not claim to be private, end-to-end encrypted or verified.

## The Serve API

`pooled serve` runs on your own computer and joins the room as an ask-only guest.

- It listens on `127.0.0.1` only. Set `POOLED_TOKEN` to require a key as well.
- Prompts sent through it go to the room's host and, unless the host limits visibility, to everyone in the room. The same rules as the chat apply.
- The host can switch API clients off, or disconnect one.

See [CLI options and security](/docs/serve/options).

## Code mode

In Code mode the room's model writes a small web app and runs it. That code is written by a model, so treat it as untrusted.

- **Files live on the host.** The project is in the host's browser storage, or in a folder on disk the host picked. Other devices get a read-only copy of the files and can choose to run the preview themselves.
- **The preview is a sandboxed frame.** It gets an opaque origin and `allow-scripts` only: no access to the room's storage, cookies, cached weights or page; no popups; no top-level navigation; no camera or microphone. A content security policy blocks network requests except to two public script CDNs. Messages from the frame are shown as plain text.
- **Peers check what they receive.** Preview files from the host are checked against their SHA-256 hashes and against the limits (400 files, 2 MB each, 8 MB in all) before a peer uses them. Timeline text is typed, capped and shown as plain text.
- **Edits to a real folder need approval.** Every edit shows a diff and waits for the host to approve it, unless the host allows edits for the current task. Scratch projects in browser storage apply edits without asking, and still show the diff.

### The preview origin

Without a separate preview origin, the preview runs in the same process as the room tab. A preview that loops forever can freeze the room.

A deployment can set `<meta name="preview-origin">` to a second site on another registrable domain. Previews then run there, in a separate process, and only then may the agent run code snippets (`run_js`). A preview origin on the page's own site (a subdomain), or plain http from an https page, is ignored, because it could not isolate the preview.

:::note
The preview origin is not turned on for pooled.run yet ([roadmap 29](https://github.com/Nehanth/pooled/tree/main/roadmap)). Until it is, `run_js` is off there.
:::

## Model weights and supply chain

- Each browser downloads weights directly from public Hugging Face repositories over HTTPS, and caches them in the browser's Cache API. Pooled ships no weights.
- The exact file a room runs is identified by its URL and size stamp. Checking a content hash against the upstream repository is planned (roadmap 19).
- pooled.run is static files. No code runs on a server.

## Reporting a vulnerability

Email **nehanthnarendrula@gmail.com** with "Pooled security" in the subject. Include steps to reproduce and the browser and OS. You will get an acknowledgement within 72 hours. Please don't open a public issue until a fix is available.

Activation inversion and unverified peers are known limits of the design, described above, not vulnerabilities. Discussion of them is welcome in [public issues](https://github.com/Nehanth/pooled/issues).
