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
