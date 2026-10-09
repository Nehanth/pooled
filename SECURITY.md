# Security

This page is the honest version of what Pooled does and does not protect. Read it before running a room with anyone you would not hand a shared document link to.

## Threat model

A Pooled room is a set of browsers that split one model's layers and pass the model's intermediate activations (the "hidden state") between them over direct WebRTC connections.

**What the design gives you**

- **No counterparty.** There is no account, no API key, and no server that sees your conversation. The signaling broker only introduces browsers to each other and carries no model traffic.
- **The room is a shared conversation.** Everyone in a room sees the questions asked and the answers streamed, on their own screen; that is the product, not a leak. The host alone holds the tokenizer, embedding table, LM head and sampler, so the devices running layers receive activations and never make sampling decisions. Nothing leaves the room.
- **Each device loads only its own layers.** It fetches its layer range from the public model repository, or takes byte ranges another device in the room already has cached (see below).
- **Transport encryption.** WebRTC data channels are encrypted with DTLS between the two browsers on each hop.
- **The host decides who joins.** A room code (six characters, about 729 million of them) only finds the room. A device that types it waits until the host presses Allow; the host's invite link and QR code carry a random 128-bit key in the link's fragment (never sent to a server) that lets a device in without asking. A device the host let in gets a pass for coming back after a reload or a dropped link. Until it is in, a device gets nothing from the room: no layers, no chat, no roster. The host can turn asking off in Room settings (then the code is enough), and anyone holding the invite link can get in, so share it like a document link. Details: [docs/protocol.md](docs/protocol.md#joining-a-room).

- **Links prove who is on them.** The signaling server (the public PeerJS cloud unless you run your own) carries the certificates each WebRTC link is checked against, so a malicious one could try to sit in the middle of a link. Pooled never sends the invite key or a pass over a link: the joining device proves it holds one, bound to that exact link's certificates, and the host proves it back; links between devices prove a room key the host hands out. A device let in by Allow is trust on first use: both screens show a six-digit code that matches only when nobody is in the middle. A new device never sends the raw key; for one release a new host still accepts it from older devices, with a warning (`?legacyauth=0` / `POOLED_LEGACY_AUTH=0` on the host refuses it). Details: docs/protocol.md, "Proving who is on a link".

**What it cannot promise**

- **Activations are not encryption.** The hidden state that crosses each hop is a lossy transformation of your text. Published attacks reconstruct a large fraction of tokens from mid-model activations (see e.g. [arXiv 2503.09022](https://arxiv.org/abs/2503.09022)). **Assume anyone in your room can read your prompts.** The trust model is "people you would share a document link with", not "strangers".
- **No verification of remote compute.** A peer could return wrong or manipulated activations. Nothing in the current design detects this. Spot-check auditing is on the roadmap and will be documented here when it ships.
- **Weights shared inside the room are not verified either.** A device that is missing a byte range of the model can take it from another device in the room that has it cached (only the length is checked), so a malicious peer could hand out altered weights. That is the same trust a room already places in its peers' compute; `?peerweights=0` turns sharing off. Model URLs are pinned to immutable revisions; cryptographic verification of shared ranges remains roadmap 19.
- **Noise or permutation "privacy" tricks are not used, deliberately.** They are known to be breakable and would give a false sense of safety.
- **Saved prompt state stays on disk.** A room node (`pooled host` / `pooled join`, the OpenClaw plugin) keeps the KV cache of a pinned system prompt + tools in `~/.pooled/cache/ckpt/` (folder 0700, files 0600, at most 4 GB) so a restart does not read the prompt again; each device holds its own layers' part. That state is derived from the prompt, which can be roughly recovered from it: treat the folder like your prompts. Delete it to clear it, or set `POOLED_CKPT_DISK=0` to keep nothing on disk. The browser room keeps its checkpoints in the site's storage (OPFS) the same way.
- **Peers learn metadata** beyond the shared transcript: device names, memory pledges, layer assignments, and timing, via the room roster.

So Pooled does not, and will not, open rooms to strangers by default, and does not claim to be "private", "encrypted end-to-end", or "verified".

## Code mode

In Code mode the room's model writes a small web app and runs it. That code is written by a model, so treat it as untrusted.

- **It runs in the host's browser.** The agent's files live in the host's browser (a scratch project in browser storage, or a folder the host picked). Other devices in the room get a read-only copy of the files and can choose to run the preview themselves.
- **The preview is a sandboxed frame.** It gets an opaque origin and `allow-scripts` only: no access to the room's storage, cookies, cached weights or page, no popups, no top-level navigation, no camera or microphone. A content security policy blocks network requests except two public script CDNs. Messages from the frame are shown as plain text.
- **Without a `preview-origin`, the preview shares the room tab's process.** A preview that loops forever can freeze the room tab. Pages that set `<meta name="preview-origin">` to a second site (a deployment on another registrable domain) run previews there, in a separate process, and only then allow the agent to run code snippets. A `preview-origin` on the page's own site (a subdomain), or plain http from an https page, is ignored, since it could not isolate the preview. The relay page there is served with the preview's content security policy and can be framed only by pooled.run, staging and local development. Roadmap 29 tracks turning this on for pooled.run.
- **Edits to a folder on disk need approval.** When the host opens a real folder (File System Access API), every edit shows a diff and waits for the host to approve it, unless the host allows edits for the current task. Scratch projects in browser storage apply edits without asking, and still show the diff.

## Model weights and supply chain

Weights are downloaded by each browser directly from public Hugging Face repositories over HTTPS and cached in the browser's Cache API. Pooled ships no weights. The catalogue pins weights, configuration and tokenizers to immutable Hugging Face revisions. Browser GGUF HTTP weight ranges must have the requested offsets, the catalogue file total (when known), and the declared streamed length. Cache keys include the pinned URL, so old mutable-revision entries are not reused. These checks detect wrong or incomplete downloads, but do not authenticate the bytes: cryptographic verification of cached and peer-shared ranges is still planned. `pooled pull` checks a whole-file SHA-256 on a fresh download.

## Reporting a vulnerability

Please report security issues privately to **nehanthnarendrula@gmail.com** with "Pooled security" in the subject. Include steps to reproduce and the browser/OS involved. You will get an acknowledgement within 72 hours. Please do not open a public issue for security reports until a fix is available.

Issues in the threat-model sense above (activation inversion, unverified peers) are known limitations rather than vulnerabilities; discussion of them is welcome in public issues.
