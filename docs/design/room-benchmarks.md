# Room benchmark measurements (#21)

Measure a room's normal inference path, preserving token output and frame delivery.
The common workload is Qwen3 1.7B Q8, the bench-log's `hello`, `meaning` and `japan`
prompts, a 400-token cap, plain greedy followed by a separately labelled room-default
column. Every result names the source commit, model revision or SHA-256, tokenizer,
devices, backend, actual layer ranges, settings and cache state.

## Local timing hook

`createPipeline` accepts an optional synchronous `hooks.onTiming(record)` observer.
`RoomNode` passes its optional `onTiming` constructor argument to the shared pipeline.
With no hook, it creates no timing records or additional GPU waits. Observer exceptions
are isolated from inference. Callbacks must be short; collect in memory and serialize
after the request. `room/timing.js` provides a bounded collector with an explicit
dropped-record count and attempt boundaries. Start every device's collector before
dispatch and stop only after its work drains; do not mix attempts in a buffer.

Frame records contain `phase`, `kind` (token/batch), `pos`, `n`, `spec`, `payloadBytes`
and local monotonic `at` in milliseconds. Payload bytes include model token IDs when
present, exclude transport framing overhead and are not wire counters.
Dispatch/return records describe the host's chain boundary.
Worker records include:

- `queueMs`: receipt to the start of the queued job (null for a direct worker call).
- `serviceMs`: job start, including checkpoint controls, through the forwarding call.
- `engineMs`: the awaited engine operation or batch loop, including its existing
  submission/readback and inter-column work. This is not pure GPU kernel time.
- `status`: forwarded or incomplete. Missing durations are null, never zero.

Worker observer notification follows forwarding. No timestamps or IDs are added to
the inference protocol. Normal and speculative dispatch share `sendChain`, including
the generator's verification frames. Solo host compute and head/sampling remain
separate from traversal; a solo room has no network hops.

Completed host operations use `phase: "operation"`, `operation`, `pos`, `n`,
local end time `at` and `durationMs`. The pipeline exposes a nullable `noteOperation`
to its shared generator so both consumers report through the same observer:

| Operation | Existing boundary |
| --- | --- |
| `host` | Embedding/layer operation or host batch loop, excluding traversal |
| `prefill` | Solo prefill, retaining only the synchronization already present |
| `head` | Head/candidate readback call outside a fused or speculative operation |
| `head-host` | Fused head and next-token host layers; cannot be separated without changing execution |
| `sample` | Synchronous CPU sampler call, preserving its GPU descriptor |
| `verify` | Inclusive speculative step, including its internal draft/head work, sampling and any traversal |

These are elapsed API operations, not GPU timestamp-query measurements. Async GPU
submission can make a later operation include earlier queued work. Verification
contains nested host/sampler/traversal intervals; do not add their totals. An engine's
internal speculative head work remains within `verify`, not a fabricated standalone
head interval. A failed operation may have no completed record; the failed generation
and partial trace remain available and cannot support a successful timing result.

Snapshot deltas reuse link `sent`/`recv` (frames), `dups` (duplicate slices), `skipped`
(frames), and `kaSent` (keepalive messages). These exclude PeerJS fallback and protocol
traffic outside the hidden-state channel. Replaced links or counter resets invalidate
the delta. No available counter measures SCTP/DTLS/IP bytes, so wire bytes are null
with a reason. Activation payload size is a separate frame measurement.

## Correlation and reporting

Each capture carries a run/attempt ID and a fixed chain epoch with ordered device
aliases. Correlate ordered occurrences of `(kind, pos, n, spec)` across the host and
workers. Positions may repeat after rollback. Reject missing, duplicate, unexpected,
ambiguous, dropped or incomplete traces rather than pairing by position alone. Match
returns to outstanding dispatches on the host's own clock. Never subtract clocks on
different devices or call half an RTT a measured one-way delay.

Retain failed requests and token IDs. Report client TTFT, engine prefill/reuse counts,
upstream decode rate, delivery rate `(N-1)/(last-first)` for N >= 2, total duration and
stop reason. Five measured repeats follow one warm-up; first-load/first-request runs
are separate. Aggregate median/min/max with sample and failure counts, not p95 from
five observations. Verify identical inputs and cleared prefix reuse between independent
requests; clearing the chat display alone is insufficient.

Measure timing off/on overhead with paired ordering on solo and split hardware.
If overhead exceeds run noise, publish detailed timing separately from headline rates.
Synthetic latency/TTFB controls validate the collector; they are not real-device results.

The contribution excludes kernel tuning, CPU adapters and compatibility patches.
Stock GTX failures and any downstream workaround results must remain distinct.

## Browser collection

For dedicated automation pages, `?bench=1` exposes `window.__pooledBench`; add
`timing=1` to collect frame records. Normal room pages expose neither the controller
nor an observer. The controller's `begin(attempt, epoch)` and asynchronous `end()`
bound each capture. Its `run({prompt, mode, maxNew})` takes the normal room lock,
clears checkpoint/live-prefix reuse, uses the normal prompt builder and generator,
and returns input/output IDs, token arrival times and generation statistics.

For the 1.7B plain column, load the browser host with `lookup=0&densespec=0` and use
`mode: "plain"`. Browser generation does not honor Node's per-request `spec: false`
for those paths; the adapter rejects plain runs without the room flags. Reload the
host with default flags for `mode: "defaults"` (current sampling) or `"spec-greedy"`
(the correctness comparison). MTP models are not supported by this plain browser
adapter. Recovery, changed assignments or unexpected prefix reuse make a run
non-equivalent; abort/context exhaustion is failed. These outcomes remain in raw
evidence and cannot silently enter successful throughput aggregates.

Controller TTFT starts after acquiring the normal host lock, before prefix reset and tokenization,
and ends at the first token callback. It excludes CDP/USB control latency and is not
a measurement from a separate asking device. A remote-client TTFT needs timestamps
in that client's normal request path, explicitly labelled in the result.

## Node collection and records

`nodeBenchmark(room, {timing})` has the same capture/run methods. It uses the Node
room's checkpoint index and promise lock. The shared sample collector preserves
partial token callbacks and their times if generation throws. It never substitutes
partial output for a successful answer.

For an isolated Node process, `node benchmarks/bench_room_device.js device.json` reads
local options from that file, creates or joins a room, and accepts JSON-line commands
on stdin. Config contains `options` for `createRoom`/`joinRoom`, optional `join` code,
and `timing: true` to enable collection. Commands carry an `id` and `action`:
`snapshot`, `start`, `begin` (with `attempt`, `epoch`), `run` (with `request`), `end`,
or `close`. Replies start with `@pooled-bench `; other stdout is engine diagnostics.
This is local process control, not a new room protocol or network endpoint. Keep
connection configuration and invite codes out of public evidence.

Node ingress retains its checkpoint disk queue; it stamps receipt before enqueuing
the shared worker call. Already-validated tail returns use the shared return observer.
Tests must enter through `RoomNode.aiOnData`, as a direct factory call alone cannot
verify that wiring.

`makeRecord` creates version 1 records with source commit and content fingerprint,
harness fingerprint, GGUF/config/tokenizer revisions and hashes, device inventory,
cache classification, request identity, raw sample and captures. Prompt hashes use
UTF-8 text. Input/output hashes use consecutive unsigned 32-bit little-endian token
IDs with no header; the raw IDs remain available for first-difference comparisons.
A failed request hashes its observed partial output and retains its failed status.

Inference and timing validity are separate. An otherwise successful answer with a
missing worker capture or null queue duration retains the raw trace but has timing
status `incomplete` and no derived per-hop breakdown. The orchestration layer must
also preserve control/transport failures that prevent a generation sample, collect
resource/wire observations, and write each attempt to a new file without overwriting
earlier evidence. Completed host operation summaries and wire counter deltas are
included. An optional API client measures delivery and pairs it with exact host
evidence in the repeated series. External resource sampling is optional and
recorded separately from generation success.

`measureRequest` in `benchmarks/bench_room_client.js` measures an ordinary v1 API ask
through the existing `Bridge` in `cli/lib/room.js`. The caller owns the bridge and
must connect it to the dedicated room first. Pass a unique `rid`, the standard
`{ prompt, mode, maxNew }` request and an optional timeout. The clock starts
immediately before `Bridge.ask`; TTFT ends at the first nonempty answer text on
that same client. Queueing is included; thinking-only and empty chunks are not the
answer. It retains bounded, timestamped response events and partial failures,
cancels a timed-out request, and never retries. Plain/spec-greedy send temperature
zero; defaults omit the override. The host's supported speculation settings must
still be selected and verified separately.

The API carries text chunks, sometimes combining several tokens. This collector
therefore reports chunk counts, not a token delivery rate or reconstructed token
IDs. Its result alone is not a complete benchmark record. `prepareClient` on each
host adapter clears checkpoints and prefix state under the normal lock and
returns the template-inclusive input IDs. After the request, `pairClient` requires
one complete observed host attempt, matching input/output IDs, client usage and
stop reason, identical decoded text, no prefix reuse and unchanged assignments.
The room must remain dedicated between preparation and request; the lock is
released so the normal API can acquire it. Admission stays with the existing room.
Client runtime, machine and network placement belong in the run manifest. Series
tables separate client TTFT/delivery from host measurements and never combine
direct-host and API-client samples into one row. Client failures retain their raw
partial events, end all begun collectors and stop the schedule.

Pairing reuses the API's `pieceDecoder` to match emitted token IDs to received text
chunks, including a Unicode character spanning multiple IDs. IDs held for one
text chunk share that chunk's arrival time. Client delivery rate is
`1000 * (N - 1) / (lastArrivalMs - firstArrivalMs)` for N >= 2; it is display delivery,
not per-token GPU time. If those text boundaries cannot account for every ID,
delivery rate stays null with a reason; no retokenization or chunk-count proxy is
used. Client TTFT remains the first nonempty answer text.

With timing enabled, the shared generator also emits `phase: "generation"`
records: `start` with copied input IDs, `token` with ID/draft flag, and `end` with
the result and copied output IDs, or `error`. These use the same bounded collector
and host clock as the frame records. They never cross the inference wire. Recovery
produces another start and is rejected for equivalent-output measurements. With
timing disabled the caller receives the original generator, with no token wrapper
or generation record allocation. Enabled overhead must be measured again after
these additions. Captures contain prompt/output IDs: collect only the agreed
public prompts in dedicated benchmark rooms.

Node's ordinary API does not transport a speculation option. For a prepared plain
client request, the benchmark adapter temporarily wraps that dedicated instance's
`generate` method to supply the existing supported `spec: false` option. `end`
restores the original own-property descriptor or inherited method, including on
queue failure. Defaults and spec-greedy leave it untouched. API validation,
locking, sampling, recovery and the inference loop remain the normal ones; no
wire field or engine-method disabling is needed. The controlled room must not
accept unrelated work during an attempt.

## Series runner

For an already prepared dedicated browser room, or Node devices configured to create
or join one, run `node benchmarks/bench_room.js run config.json`. The runner does
not create a browser profile, grant phone certificate trust, or choose a topology.
Those preparation steps stay with the device launcher. Example local configuration:

```json
{
  "run": "R-plain", "host": "R", "epoch": "solo-1",
  "manifest": "manifest.json", "prompts": "docs/bench-log.md",
  "output": "results/R-plain", "mode": "plain", "maxNew": 400,
  "repeats": 5, "timing": true, "timeoutMs": 600000,
  "start": { "model": "qwen3-1.7b" },
  "devices": {
    "R": { "kind": "browser", "dedicated": true,
      "cdp": "http://127.0.0.1:9222",
      "url": "http://localhost:8080/p2p.html?bench=1&timing=1&lookup=0&densespec=0" }
  }
}
```

Paths resolve from the command's working directory. The browser URL must match
exactly one tab. A Node entry instead uses `kind: "node"`, `command`, `args`, and
optional `cwd`; typically its command launches `bench_room_device.js` with local config.
The manifest follows the identity requirements above. Its compute-device aliases
must exactly match the configured devices (the separate API client is not in this
set). Before measuring, the runner checks the host and loaded workers' model names
against the manifest and compares Node readiness inventory with the declared
device identity; feature enumeration order is immaterial. Contradictions stop the
series and retain setup/attempt evidence. These checks do not verify model hashes
or browser hardware identity: those still require launcher provenance.
Browser launchers install
GPU error/loss monitoring before navigation; absent monitoring is reported as
unavailable. Node's device process installs its own monitor. Device clocks, power,
visibility and resource conditions still need external qualification.

To measure the requesting client, add `client` to the runner config with
`{ "kind": "node", "command": "node", "args": ["benchmarks/bench_room_request.js", "client.json"] }`.
The optional `cwd` follows the same rules as a Node device. `client.json` contains
`alias`, `network` (a public description of placement), `timeoutMs` and
`connection`, the existing Bridge options (`code`, `key` when required, `signal`,
`name`, `client`, `relay`). Keep connection details private. Run the client on the
machine that defines the desired delivery measurement; the automation controller
can be elsewhere. Set its request timeout below the runner's command timeout so
it can preserve a timeout result and cancel the ask before control expires.

The manifest must include `client: { alias, os, runtime, transport, network }`;
`os` is Node's platform plus kernel release, `runtime` is its version, and transport
is `Bridge API v1`. The runner checks this against client readiness. This client
is an ask-only guest, not a compute device in the GPU count. The runner starts it
after host loading and closes it on completion/failure. Pairing requires
`timing: true` for exact host evidence. Host-only off/on overhead runs remain a
separate request path.

The schedule is one first request, then one warm-up and five measured requests for
each pinned prompt. Every attempt is saved before the next starts. Request/control
failures retain a record and stop the series; setup failures get `runner-error.json`.
No retries or automatic recovery enter the measured series. Child devices are
closed; browser CDP connections are disconnected without closing caller-owned tabs.
Existing run files are never overwritten.

## Resource observations

The optional runner `resources` map selects a reader per device alias, for example
`"resources": { "R": { "kind": "nvidia", "gpu": 0 } }`. `resourceIntervalMs`
defaults to 1000 and `resourceCapacity` to 4096 entries per attempt. Readers run
outside inference callbacks and do not overlap polls. Each attempt has a sample
before work, periodic samples and a final sample after work. Short requests may
have only the surrounding samples. Command start/end timestamps use the controller
clock and bracket observation latency; they cannot be combined with another
device's clock to infer a temperature at a particular token.

The NVIDIA reader uses `nvidia-smi` for GPU temperature (C), utilization (%), used
and total memory (MiB), and power (W). These are whole-GPU readings, including other
applications. `command` and `args` may select an already-authorised remote command;
the reader appends the fixed query and numeric adapter index. It never runs a shell
or installs a tool. Select the correct adapter in the local configuration.

An Android entry uses `kind: "android"`, `command` for ADB and optional `args` for
the existing server port and serial selection. It reads battery service,
thermal service, low-power and brightness settings. Only allowlisted current
fields and the `Current temperatures from HAL` sensor section are retained; raw vendor history and identifiers
are discarded. Battery temperature is explicitly labelled and is not a GPU
temperature. Brightness stays a raw setting; thermal type/status codes stay as
reported. This reader never changes power, lock, charging or brightness settings.

Commands have a default 5-second timeout and a 256 KiB output bound. Failures,
missing fields and collector overflow are explicit in the resource capture and
reports; they do not erase valid inference evidence or become zero-valued readings.
Local command arguments are omitted from saved run metadata. Public resource data
use the same device aliases as the manifest. Reports separate rows with different
sampling intervals, buffer capacities, provider kinds or command timeouts. GPU
feature enumeration order does not separate otherwise identical device manifests.
Resource collection adds external work: measure its
overhead with the intended configuration before treating results as headline
performance. Collection alone does not implement the endurance controller's
thermal-stop/cooldown policy; sustained-phone runs still require that gate.

Regenerate tables with `node benchmarks/bench_room.js report results/R-plain
results/R-plain-regenerated`. JSON and Markdown outputs group by source, model,
cache, devices, assignments, settings and observer mode. First/warm-up requests stay
in raw evidence. Rows retain failure counts, output hashes, median/min/max and actual
sampling. Speculation activity comes from generation counters and, when available,
speculative dispatch records: enabling a setting is not evidence that verification
ran. Missing counters remain unknown. Defaults with inactive speculation are not a
speculative speedup comparison.

Transport/report integration is covered by
`node --test packages/room-node/test/benchmark_cli_test.mjs`; its child device is a
fixture and establishes no GPU performance result.
