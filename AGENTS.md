# Duga: Product Direction and Agent Notes

## Purpose and status

Duga is a proposed website-first personal AI assistant at **duga.app**. A visitor should be able to open the site and start using an assistant without installing software or setting up a room. Over time, they can connect their own devices, contribute idle compute, and use agents for coding, everyday tasks, memory, voice, and data integrations.

**This is a product plan, not a description of shipped Duga functionality.** The current repository is Pooled: a peer-to-peer inference engine, room runtime, coding-agent harness, CLI/API bridge, and OpenClaw integration. Reuse those components where appropriate; do not represent accounts, the Global Net, credits, long-term personal memory, general connectors, or end-to-end voice conversation as implemented today.

The existing Pooled project expressly commits to *no accounts* and does not plan to open rooms to strangers by default. Duga's initial product also has **no accounts**: use invite links and room codes, preserving the current browser-room experience. Accounts are deferred, not a prerequisite for continuity or device participation. A public compute network still requires a distinct trust boundary and an explicit governance decision; it must not silently change the security model. See [GOVERNANCE.md](GOVERNANCE.md), [roadmap/README.md](roadmap/README.md), and [SECURITY.md](SECURITY.md).

## Target experience

1. Visit **duga.app** and get a useful answer immediately. Guest access should not require an account, installing a worker, pairing a device, or donating compute.
2. Use invite links or room codes to connect another device without registration or login. Preserve the existing distinction: invite links grant access to their holders; code-based requests require host approval by default. Cross-device continuity, durable memory and persistent device management remain proposed and must not imply an account service exists.
3. Let users connect their computers and other capable devices through invite links or host-approved room codes. Show availability, capacity, privacy mode, current jobs, and a one-click disconnect/revoke action.
4. Let users ask for coding help, everyday planning and actions, and voice interaction from one assistant interface. Show what an agent plans to do, what it did, and what requires approval.
5. Provide clear, per-task compute choices: **Local**, **Private Net**, or **Global Net**. Do not make network placement an invisible privacy decision.
6. Let users inspect, correct, export, and delete their memories and connected-service permissions.

An immediate first answer needs a dependable inference route. An active Global Net alone cannot guarantee availability. Plan a clearly disclosed bootstrap route (such as reserved/hosted capacity or a suitable on-device model); measure cost, privacy, and quality before selecting a default. Never silently send a private task to Global Net because a user's devices are offline.

## Two networks, plus local execution

| Mode | Compute membership | Intended use | Data rule |
| --- | --- | --- | --- |
| **Local** | The device currently in use | Tasks that fit locally; offline-capable work where supported | No other compute node receives task data. |
| **Private Net** | The user's own devices explicitly admitted through invite links or host-approved room codes | Personal conversations, memories, coding work, and connected-service data | Only the owner's approved devices may act as inference workers. |
| **Global Net** | Opt-in idle devices contributed by participants; shared pool for participants | Tasks whose inputs the user explicitly agrees to expose to untrusted workers | Treat assigned workers as untrusted. No default access to personal memories, connector credentials, or private files. |

Private Net membership is **the user's own devices only** in the initial product; invite-based teams are out of scope for now. A link or code does not prove ownership: the user must check which devices they admit. An existing Pooled room can include other people and must not automatically be labeled an owned-device Private Net. The current runtime does not enforce device ownership. Global Net is a participant-shared pool, not a free source of secretly enrolled devices. Contributing compute is optional and separate from using the website. The user can choose a mode per task or conversation; the system must visibly explain when a choice changes where data is processed.

A Global Net worker may see task text, intermediate activations, outputs, model weights, and operational metadata, depending on the job. **Activations are not encryption**: they may reveal prompt content. Do not claim that public nodes cannot learn users' prompts, or that transport encryption makes a public worker trustworthy. If a task includes potentially sensitive content, recommend Local or Private Net. Require explicit, informed consent before sending it to Global Net. Provenance-based blocking of memory/connector data is defense in depth, not a promise that arbitrary pasted text is automatically classified correctly. See [SECURITY.md](SECURITY.md).

## Execution architecture

```text
Browser at duga.app
  - account-free sessions, invite links/codes, chat, voice, agent activity, approvals
  - per-task compute-mode choice, memories and device controls
              |
Control plane (new product services, not the Pooled room broker)
  - room-scoped admission and proposed device revocation (no account identity)
  - presence, capability discovery, job scheduling and quotas
  - contribution/usage ledger and abuse controls
  - optional encrypted sync and connector authorization
              |
Trusted agent orchestrator (user device or explicitly disclosed service)
  - plans and routes tasks, enforces tool permissions and budgets
  - owns memory retrieval and service credentials
  - submits only permitted inference payloads to the selected compute mode
              |
Compute/data plane
  - Local: one device
  - Private Net: owned and paired devices
  - Global Net: opt-in, untrusted volunteer workers
```

Keep the control plane and inference data plane separate. An admission or scheduling service should not receive conversation plaintext merely because it handles membership or scheduling; whether it does receive plaintext through a specific fallback or hosted orchestrator must be stated truthfully. Never give Global Net workers connector tokens or unrestricted tool execution. Prefer running agent orchestration and sensitive tools in a trusted environment and using remote nodes only for explicitly approved inference jobs.

The existing Pooled runtime splits **one model's layers** across GPUs over WebRTC; this is different from running several independent agents. Support both patterns deliberately: model-level distribution for a large model, and agent-level delegation only when tasks can be partitioned, measured, and safely combined. Cross-device latency, the slowest worker, and worker churn can make distribution slower than a single device. Implement scheduling and fallbacks based on measured performance rather than assuming more nodes are faster. See [docs/architecture.md](docs/architecture.md) and [README.md](README.md).

## Agent capabilities and permissions

- **Coding:** reuse the existing coding harness and isolated preview where it fits. Present diffs and require approval before touching real projects or running consequential code. Keep generated code sandboxed.
- **Everyday tasks:** start with read-only lookup and drafts, then add calendar/email/file changes behind review and explicit confirmation. Maintain an audit trail, cancellation, retry limits, and clear failure states.
- **Memory:** separate transient conversation context from durable, user-visible facts. Request or clearly present proposed saves; support correction, deletion, and scoped retrieval. Never assume a fact inferred by the model is true.
- **Voice:** add microphone permission, speech-to-text, spoken responses, interruption, and visible recording state. Begin with push-to-talk rather than always-on listening. Browser speech synthesis exists in the room, but voice conversation and audio API support are not already complete.
- **Data integrations:** add connectors one at a time with narrowly scoped OAuth or equivalent grants. Store credentials in a trusted boundary, never in prompts sent to volunteer nodes. Show what each connector may read or change, and allow revocation.
- **Multiple agents:** begin with one orchestrator and bounded tool loops. Add specialist/parallel agents only where evaluation shows a benefit, and keep a single authority for permissions, budgets, and final actions.

Tool results and model output are untrusted. Validate arguments and schemas, isolate execution, bound steps/time/cost, and distinguish a proposed action from an action already performed. Do not automatically execute a volunteer worker's code or recommendations on a user's machine.

## Contributed-node experience and credits

A device owner must explicitly opt in to Global Net contribution and retain control over resource limits, allowed hours, idle/battery/thermal behavior, downloads, and immediate stop. A browser tab may contribute while active and supported; a closed or suspended tab is **not** an always-on worker. A future desktop or background node is an optional upgrade, not a prerequisite for using duga.app.

Use **non-cash contribution credits** initially: earn credits by providing verified work and spend them on Global Net jobs; show earnings, charges, balances, quotas, and job history. Provide a bounded way for guests and newcomers to try the service without first donating compute. Define exact accounting rates after benchmarking and abuse analysis. Do not mint credits merely for claiming idle capacity or self-reported work; account for verifiable completed work, failed jobs, duplicate identities, manipulated results, and disputes. Correct execution by an arbitrary public worker is not guaranteed by the existing Pooled system; redundancy/spot checks and reputation can reduce risk, not prove universal correctness. Do not promise monetary redemption or a trustless economy.

Public contribution also entails risks for providers: untrusted inputs, model and dependency downloads, disk use, power, heat, bandwidth, and potentially abusive workloads. Constrain jobs and isolate any code execution; do not turn a volunteer inference worker into a general remote shell.

## Delivery sequence

1. **Decision record and evaluation baseline:** define account-free session and invitation boundaries, guest inference bootstrap, explicit Global Net consent, data-retention policy, model licenses, target devices, task examples, cost and latency targets. Resolve Duga-versus-Pooled governance before changing core product commitments.
2. **Website-first assistant:** usable guest chat at duga.app with an honest, dependable inference route; responsive UI, streaming, visible privacy mode, and basic session handling.
3. **Private Net:** invite-link/code admission, key and device revocation, an explicit owned-device trust policy, offline handling, local/private routing, and quality/performance comparison against one device.
4. **Useful agents:** coding flow, limited everyday tools, user-controlled memory, push-to-talk voice, and at least one read-only connector. Add approvals and observable execution before write-capable integrations.
5. **Global Net pilot:** explicitly enrolled nodes and opted-in jobs, limited model/job types, admission and scheduling, rate limits, disconnect recovery, privacy warnings, abuse controls, and measured correctness/latency. Start with a restricted cohort, not an open public network.
6. **Credits and scale:** transparent ledger, verified-work policy, allocation fairness, audits, and multi-agent or background-task expansion once the pilot's reliability and security are demonstrated.

This ordering is a planning recommendation, not a statement that later capabilities already exist. Parallel research is fine, but do not expose high-risk public workloads before the trust model and safeguards exist.

## Working on this repository

- Existing building blocks include `engine/` (WebGPU inference), `room.js` and `room/` (P2P room and protocol), `harness/` (Code mode), `cli/` (local compatible API), and `packages/openclaw/` (agent integration). Read [docs/architecture.md](docs/architecture.md), [docs/design/harness-app.md](docs/design/harness-app.md), and [docs/agents.md](docs/agents.md) before modifying them.
- Preserve the existing browser-link room flow and current output-correctness guarantees. Core protocol, peer visibility, dependencies, and model changes require a design note and maintainer review under [GOVERNANCE.md](GOVERNANCE.md).
- Start with tests that match the layer being changed. The existing repository provides `npm test` (Deno unit tests), package tests, and browser/room end-to-end suites; see [package.json](package.json) and [docs/agents.md](docs/agents.md). Record hardware, commit, workload, and baseline for performance claims.
- Keep **implemented**, **proposed**, and **open decision** distinct in code comments, documentation, and the UI. In particular, a Pooled invite link grants bearer access and does not prove device ownership or a persistent identity, and the existing room authentication does not make public volunteer nodes safe for private prompts.

## Invitation and session rules

- **Implemented:** a room code locates a room; host approval is required by default. An invite link contains a random secret in its fragment and admits its holder without approval. Returning devices can use a room pass. See [SECURITY.md](SECURITY.md) and [docs/protocol.md](docs/protocol.md).
- **Required direction:** no registration, login, account database, or account-bound device enrollment in the initial Duga product. Keep invite secrets out of logs, analytics and server-visible URLs. Do not treat short room codes as authentication secrets or silently disable host approval.
- **Proposed:** invitation expiry/rotation, per-device removal and durable cross-device continuity need explicit lifecycle and recovery designs before being advertised. Do not claim current links expire or that disconnecting a device revokes its credentials.
- **Privacy:** every admitted inference device is trusted with room data under the existing threat model. Invitations do not make public workers safe for private tasks. Keep Global Net enrollment and task consent separate from room admission.

## Open decisions requiring explicit agreement

- What guarantees the guest's first response: on-device inference, reserved hosted capacity, or another clearly disclosed fallback? Who pays for it?
- Which data is synced across devices, where is it stored, how is it encrypted and deleted, and how can users transfer sessions through explicitly approved devices without accounts?
- What work is admissible on Global Net, what exactly does a worker learn, and what verification is sufficient before credits are awarded?
- How do account-free invitations, device ownership, revocation, recovery, rate limits, and Sybil resistance work without degrading the existing account-free Pooled rooms?
- Which first connector and voice stack provide useful UX without leaking credentials or promising unsupported browser background behavior?
