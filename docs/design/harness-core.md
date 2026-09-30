# Code mode on the serve v2 core ("hcore")

Status: built behind `?hcore=1` (default off until the Code mode eval says otherwise, section 7).
Scope: `harness/core-model.js` and `harness/engine-gen.js` (new), `harness/agent.js`,
`harness/engine-model.js`, `harness/model-common.js`, `room/api.js` (additive), `room/code.js`,
`room.js` (three `roomApi` fields), the eval page and runner. No wire change, no `PROTOCOL` bump.

## 1. What changes

Code mode used its own tool prompt (`toolsSystemPrompt`), its own text parser (`ToolCallParser`
and the bare-call fallbacks) and a lenient call automaton (`ToolCallConstraint`). `pooled serve` v2
(serve.md section 11) has a second, stricter stack for the same job: the template's own tool block
and rendering (`templateProfile`, `renderApi`), the strict grammar (`GrammarConstraint`, names and
schema-typed arguments, the tags as special ids), the streaming parser (`CallStream`) and the think
split. With `?hcore=1` every Code mode model call goes through that stack, in process:

```
Agent ──ask({system, tools, turns, signal, on})──► coreModel (harness/core-model.js)
          ◄─{text, calls, open, reason, ids}──      toMessages(turns) -> v2 messages
                                                    apiPrompt2 (renderApi, the turns' exact ids)
                                                    apiRun2 (grammar sampler, CallStream, ThinkSplit)
                                                      └─ host.generate: roomApi.generate (the room)
                                                                        engineHost(...).generate (one engine)
```

No HTTP, no `ai-ask`, no `cli/` import. The request is the v2 ask `validateApiAsk` would normalize
(`coreRequest`; a unit test keeps them equal): every tool allowed, `toolChoice: "auto"`, parallel
calls, thinking off, the room's sampling preset ("exact" for the JSON style, "focused" for XML).

## 2. History

Assistant turns are `{ text, sampled: [{ name, args }] }`, args the JSON text as sampled; a
tool-results turn keeps `text` (the `<tool_response>` blocks, for sizes and repeat checks) and adds
`results` (the array). `toMessages` maps turns to v2 messages; results pair with the calls before
them, or go as the user's text when that answer has no call. Turns with raw call markup (sessions
saved by the legacy path) are parsed once with `CallStream` unconstrained, in either call style.
The legacy path renders core turns back as markup (`legacyText`), so a session survives switching
paths both ways.

Exact ids: `apiRun2` returns the answer's ids (header tail included) whenever the API cache would
store them; the Agent keeps them per turn object (`model.setIds`) only when the turn is exactly what
was sampled, and `apiPrompt2`'s `turnIds` override replays them. Per-turn storage survives
compaction (the turn objects persist), cannot collide, and is saved in sessions under the
tokenizer's tag.

## 3. The Agent on this path

- Calls come from `res.calls`; `fixArgs` still runs for execution, history keeps what was sampled.
- An open call (length cap, or the answer ended mid-call): `write_file` keeps its complete lines
  and runs (`salvageArgs`), anything else goes into history with the arguments it had and its error
  as its own result.
- The bare-call fallback runs on the content when there is no call (a bare JSON object, bare tags,
  a `<function=` block with its opener garbled), takes the call out of the content and appends the
  format hint; the `bare` event counts it (eval column). A garbled opener left before a real call
  (`<toolly_call>`, Qwen3.6) is stripped from history.
- Garbage (`garbage: "mass"`): only an engine fault (NaN / +Infinity logits, or a run of positions
  with almost no mass on anything allowed). Nothing runs; the errors go back as the user's text.
- A generation failure throws (the run shows an error); a room-side Stop (reason abort) is a stop.
  Calls that never ran are dropped from history.
- Unchanged: repeat and stuck guards, page-error guard, idle-done, EMPTY nudge, cards, result caps,
  approval, compaction tiers 1-4 (tier 1 stubs the `results` array and rewrites `text` from it).
- `_size` counts each call as the template renders it (`model.callText`: an XML value unescaped).

## 4. Shared core changes (additive)

`apiRun2` also returns `garbage`, `forced`, `forcedFree`, `openArgs`, `raw`, `gen` and `ids` /
`thinkEnd`; `open` stays `{i, name}` and every wire field is unchanged. It takes `garbage` (the
rule). `apiPrompt2` takes `turnIds`. `strictSampler` counts `forcedFree` (forced positions among the
model's own choices) and has the `"mass"` rule. `roomApi` gains `profile()`, `tokenTexts()` (the API
path's, so one mask cache) and `modelKey()`. `engineGenerate` (harness/engine-gen.js) is the room's
generate contract over one engine; `engineModel` is rebuilt on it with the same behaviour.

## 5. Streaming

The call being typed goes to the UI as append-only JSON-shaped text built from CallStream's
fragments: `{"name": "write_file", "arguments": {"path": "a.js", "content": "…`. `parseLive`'s JSON
branch reads it for both call styles and on older peers.

## 6. Review notes (the adversarial critique)

Accepted: generation errors throw (A1); per-turn ids instead of a content-keyed cache (A2); the bare
fallback kept as a post-pass on the content (A3, option a); `forcedFree` for the "forced" note and
the mass-only garbage rule, the note no longer blames the engine (A4); the legacy path renders core
turns (A5); models without a tool format stay on the legacy path, and `coreModel` refuses them (A6);
the room's profile and token texts (B1); an open call stays in history paired with its error, one
results helper (B3); call sizes as rendered (B4); decode time in the eval (C1, the column); a
16-token margin (C2); the larger of the tokenizer's and the engine's vocabulary (C3); the tools
compiled once per set, and a test that all Code tools compile under both styles (C4); the pin rule
stated both ways (C5); cumulative `tps` / `generated` (C6); the garbled opener stripped (C7).

Changed: B2. `engineGenerate` never emits a stop id, but `fed` stays exactly what the engine wrote:
a speculative step that accepted a stop id as a draft wrote tokens after it, and cutting `fed` back
would be wrong on a hybrid model (DeltaNet state cannot roll back). The next prompt resumes from the
pinned slot in that case; the common case (the stop is the step's last, unwritten sample) reuses
everything.

Not done here: the free-state fast path in `GrammarConstraint.mask` (C1): the eval's decode time
decides whether it is needed. A unit test with Qwen3.6's real token texts for the garbled opener
(C7) needs the tokenizer in the fixtures.

## 7. The switch and the eval

`room/code.js`: `?hcore=1` / `?hcore=0`, read once; `HCORE_DEFAULT` is `false`. The eval runs
both arms on the same commit: `tests/eval/run.mjs --model engine --codemode --hcore 0|1` (with
`--dense <dir>` for Qwen3 1.7B or `--weights` for the MoE, `--label`, `--repeat 2`). Default on
when: 1.7B hcore 1 >= hcore 0; MoE hcore 1 >= hcore 0 - 2 (a second run of each arm when within 3);
0 GPU errors, no more garbage-marked calls, wall time within +20%.

After hcore has been the default for a while, a follow-up deletes the legacy path:
`toolsSystemPrompt`, `ToolCallParser` in the Agent, the lenient `ToolCallConstraint` and the legacy
branch of `constrainedSampler`, `room-model.js`, `OwnIds` / `encodeTurn` / `buildIds` use for Code,
`detectStyle`, the XML branch of `parseLive`, `CARDS.format`, the `idsFor` / `adopt` plumbing.
`parseCallBody`, `parseLooseJSON`, the bare-call finders (the fallback), `fixArgs` and every loop
guard stay.
