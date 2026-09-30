# Code-mode evals

Twelve small tasks that measure the Code-mode harness end to end: the model gets the task's request
in a fresh in-memory project, the agent runs with every tool auto-approved, and an automatic check
decides pass or fail. Design: `docs/design/harness-light.md` §6.

| id | kind | the check (runs in the sandboxed page) |
|---|---|---|
| tetris | build, canvas | canvas drawn after 500 ms; it changes by itself within 2 s (gravity); ArrowLeft / ArrowUp / Space raise no errors; Space changes the lower half at once (a drop) |
| snake | build, canvas | canvas drawn; it moves within 600 ms; after ArrowUp the change is a column, after ArrowLeft a row (steering; growth is not checked) |
| todo | build, DOM | Enter in `#new` adds two `#list li`; the checkbox visibly marks one; Delete removes it |
| calculator | build, DOM | `12+7=` shows 19; C clears; `8/0=` does not throw; `9*3-4=` shows 23 |
| stopwatch | build, timers | `#time` starts at 00:00, 00:01-00:02 after 1.6 s, frozen after Stop, 00:00 after Reset |
| landing | build, static | one h1, ≥ 3 feature cards, a CTA, viewport meta, no horizontal scroll at 400 px |
| fix-bug | fix | `total()` in the seeded `cart.js` no longer skips the last item |
| fix-crash | fix | the seeded animation (a TDZ error at load) runs with no errors and draws |
| add-feature | extend | the seeded todo app gets a working "Clear completed" button, old features intact |
| css | style | header sticky, every button 8px radius, the page's text unchanged |
| refactor | multi-file | helpers moved to `utils.js` and imported by `app.js`; the page renders the same |
| logic | pure JS | `wordFreq()` in `lib.js` passes 4 cases |

A check fails when it throws, when anything logs a console error, or when the page itself errors.
It runs through the run_js probe (`harness/run-js.js`): the project is built exactly as the
preview builds it, the check becomes a module `__run.js` that can import project files, and it
runs in a hidden sandboxed frame on another site (its own process, cut after 8 s).

## Running

```
# no GPU: headless Chromium, the scripted golden run of each task (CI-safe; exit 1 on any failure)
NODE_PATH=<dir with playwright> node tests/eval/run.mjs --model mock

# one local WebGPU engine (only when nothing else uses the GPU); a model that fits one device
E2E_GPU=real node tests/eval/run.mjs --model engine --weights <file.gguf> --headed [--ctx 16384]

# a real room (host + other devices), e.g. the 35B MoE split over a laptop and a phone:
#   (localhost only: tests/eval/ is not deployed) open p2p.html?eval=all (or ?eval=tetris,todo), start the model, open Code, send "/eval"
#   ("/eval calculator,css" runs a subset). One line per task appears in the timeline and a
#   .jsonl of records and trajectories downloads at the end. Stop ends the run.
```

Options for `run.mjs`: `--tasks a,b`, `--repeat N` (sampling noise), `--no-selftest`, `--verbose`
(page errors, the apps' expected ones included), `--port`.

In mock mode `run.mjs` also runs a **self-test** first: each task's `bad` files (a deliberately
wrong solution, or the unfixed seed; a list for several) must fail its check, so no check passes vacuously.

## Output

`tests/eval/results/<stamp>-<model>.jsonl` (gitignored), one record per task run:

```
{ id, model, ok, reason: done|limit|stuck|context|stopped|error, check (the probe's report),
  steps, calls, cards: {id: n}, forced, prompt (tokens prefilled), reused (tokens the caches
  already held), generated, compactions, ctx (the agent's estimate of the conversation at the
  end), ms, firstMs, run }
```

and `tests/eval/results/<stamp>-<model>/<id>.json`, the task's full trajectory (system prompt,
every turn, the final files) for reading failures. The summary line reads
`success 10/12 (83 %) · steps 7.4 · prefilled 41k · reused 310k · generated 18k · ctx 5.2k/task · 612 s`.
Each task's line shows its own prefilled and reused counts too.
`forced` is the sum over steps of kept tokens the call grammar forced (a healthy model: ~0);
with the mock model the token columns are 0 and only `ctx` (estimated) moves.

## Adding a task

`tasks/<id>.js` exports `{ id, kind, prompt, files (seed), maxSteps, check, checkFiles?, page?,
viewport?, mock, bad }` and is listed in `tasks/index.js`. `check` is module source; `suite.js`
prepends helpers: `sleep ok $ $$ text button key press shot drawn`. `checkFiles(read)` checks the
workspace itself and returns an error string or null. Keep the prompt explicit about the ids and
labels the check relies on, so it measures the harness, not guessing.
