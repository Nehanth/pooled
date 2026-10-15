# Duga: account-free first phase

Status: proposed product direction. The account-free invitation mechanism described below already exists in Pooled; this document changes no runtime protocol or deployed domain configuration.

## Decision

Build toward a website-first assistant at duga.app without introducing accounts. Use the existing browser room, invite links and room codes for device participation. Registration, login, account-bound pairing and guest-to-account migration are deferred. Preserve Pooled's browser-link flow, CLI compatibility and output correctness.

## Joining and trust

- An invite link carries a random 128-bit bearer secret in its fragment. Anyone holding it can join without a separate host approval; share it only with intended devices.
- A six-character room code locates a room. By default the host must approve the joining device. Preserve that default and the existing link verification flow.
- A returning device can use its room pass. This is room-scoped access, not an account or verified ownership identity.
- Expiring invitations, rotating credentials, kicking/revoking devices and durable recovery are future work, not current guarantees. A disconnect alone is not credential revocation.

See [the current security model](../../SECURITY.md) and [joining protocol](../protocol.md#joining-a-room). Every admitted inference device must be trusted with room data. Neither a code nor an invite proves that the device belongs to the user.

## Product boundaries

**Implemented building blocks:** browser rooms, distributed inference, chat, Code mode, CLI/API bridge and OpenClaw integration. Model availability, browser support and loading still constrain the first response.

**Proposed:** Duga's assistant experience, an owned-device Private Net, user-controlled memory, voice conversation, connectors and a restricted Global Net pilot. A room with invited friends must not be presented as an owned-device Private Net. The current runtime does not enforce device ownership.

**Open decisions:** dependable first-response inference and its cost/privacy disclosure; session storage and approved cross-device transfer; invitation lifecycle and recovery; device revocation; public-worker admission and abuse controls. Do not add a hosted inference fallback or send tasks to public devices silently.

Global Net remains separate from private rooms. Both contributing a device and submitting each public task require explicit consent. Public workers may learn prompts, intermediate activations and outputs. Activations are not encryption. Keep memory, private files, service credentials and tool execution within a trusted boundary. Account-free invitations do not resolve public-worker trust, correctness or Sybil resistance.

## Delivery

First preserve and present the existing invitation flow honestly. Then evaluate the first-response route and design session/device lifecycle before implementing persistent features. Do not advertise public compute or credits as available. Public-network changes remain subject to [governance](../../GOVERNANCE.md) and a separate security review; this direction does not amend Pooled's governance commitments.

## Implemented preview on `feat/duga-assistant`

The root page now provides a responsive assistant workspace, streaming Local chat, cancellation, visible loading/failure states, explicitly saved browser-only notes (edit, delete, export and per-conversation inclusion), ephemeral session activity, data deletion, and device invite/code entry. Code and multi-device controls reuse the existing room in an explicitly opened workspace, with a disconnect action. Local chat and notes are not transferred into it. No account service is introduced.

Local loads the existing Qwen3 0.6B model through `room/model-loader.js` and `harness/engine-model.js` on this device, with a 2,048-token context and 384-token answer limit. Model downloads contact Hugging Face; prompts are used only by the local engine. First use requires about 640 MB of downloads and a capable WebGPU device. The small model has limited quality. This is an explicit evaluation bootstrap, not a claim that every visitor is guaranteed an immediate useful response. Cost, quality, target hardware and a dependable production bootstrap remain open decisions.

No room connection or signaling library is loaded by Local. Opening a room releases the Local GPU allocation. A room iframe must be disconnected before another Local task starts; hidden frames can otherwise keep using compute. Room data and cache lifecycle remain the existing runtime's responsibility. No wire protocol, admission, peer visibility, model weights or engine math are changed by the new root UI.

Private Net and Global Net are explained as unavailable. An existing trusted room is not relabeled Private Net. Connections and Voice have truthful capability/status views; no OAuth grants, microphone recording, public workers, credits or write-capable everyday tools are simulated. Actual coding approvals and sandboxed previews remain in the room runtime. The UI adds no analytics and records no invitation secrets in its session activity or persistent notes. Chat remains in tab memory; notes are explicitly persisted under `duga.memories.v1` in localStorage, unencrypted and accessible to same-origin code. Session context already influenced by a note must be discarded with New conversation to remove that influence.

Validation commands: `npm run test:duga`, `npm run test:e2e:duga`, `npm run test:runtime`, and existing package tests. Browser tests inject a scripted local adapter through request interception, not a production mock switch, and exercise the real UI. They do not establish model quality or real-device GPU performance. The loader's network/cache contracts are tested independently. Real hardware/quality evaluation remains required before a production availability claim.

## Validation record (2026-10-15)

- Added seven Node tests: unavailable placement rejection, invitation parsing/secret handling, explicit note persistence/edit/delete, blocked storage, validated model ranges, caching and cancellation.
- Added a desktop/mobile browser suite: streaming, stopping, model failure without fallback, note opt-in, safe text rendering, export/deletion confirmation, invite-secret handling, room/Local compute exclusion and unsupported WebGPU behavior. Passed at 390, 768 and 1440 px widths.
- Full Deno unit suite: 1,415 passed, zero failed, five ignored. Existing headless runtime checks passed. Package tests on Node 22: 212 passed, seven skipped. Existing preview suite: 60/60. Join security: 24 passed; optional CLI and native room-node scenarios skipped because their package-local dependencies are absent.
- Existing Code acceptance suite: 54/55. The failing assertion is “peer has no Code tab before the host starts”. The same failure reproduced in a detached baseline at `7bbf15a`, before the UI implementation; the remainder, including coding edits, approvals, sandbox and peer preview, passed. No change was made to that existing behavior.
- Syntax checks, documentation build (60 pages) and static site assembly passed. Tests used Node 22.23.3 and Deno 2.9.6; browser UI tests used Playwright Chromium 131. No speed/quality claim is made.
- Added `npm run test:e2e:duga-local` to exercise the actual loader and dense engine using synthetic weights. This runner has no WebGPU adapter, so that check was skipped explicitly. Set `E2E_REQUIRE_GPU=1` to make an unavailable adapter a failure on a GPU-equipped validation host. Real-model end-to-end inference and production bootstrap quality are not verified here.
