// Prefix reuse across Code mode agent steps (issue #73): harness/engine-model.js checkpoints the
// system prompt + tools, and harness/agent.js compacts old turns into a stable form, so a long
// session prefills only what is new, and after a compaction only what follows the system prompt.
//
// The fake engine's whole state is the list of tokens it has run (a stand-in for the KV rows and
// DeltaNet state, which are a function of exactly that list); saveSlot / loadSlot copy it. Its
// logits are a function of that state, so "same state" means "same answer".
import { engineModel } from "../../harness/engine-model.js";
import { Agent, stubResults } from "../../harness/agent.js";
import { buildIds } from "../../room/conversation.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// word-level tokenizer; any other character gets an id of its own the first time it is seen
const WORDS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<function=", "<parameter=", "</parameter>", "</function>", "hello", "done"];
function makeTok() {
  const V = [...WORDS], ID = Object.fromEntries(V.map((w, i) => [w, i]));
  const id = (w) => (ID[w] ??= (V.push(w), V.length - 1));
  return {
    V, ID, vocab: ID,
    encode(s) {
      const out = [];
      for (let i = 0; i < s.length;) {
        const w = WORDS.find((x) => s.startsWith(x, i)) || s[i];
        out.push(id(w)); i += w.length;
      }
      return out;
    },
    decode: (ids) => ids.map((i) => V[i]).join(""),
  };
}
const VOCAB = 4096;

// answer: (state) -> the next token id, or a queue of scripted ids (consumed in order)
function stateEngine(tok, { slots = true, answer } = {}) {
  const e = {
    maxSeq: 1 << 20, pos: 0, mtp: null, dims: { vocab: VOCAB }, st: [], log: [], saved: new Map(),
    reset() { this.st = []; this.pos = 0; this.log.push("reset"); },
    async prefillTokens(ids) { this.st.push(...ids); this.pos = this.st.length; this.log.push("pre" + ids.length); },
    async forwardToken(t) {
      this.st.push(t); this.pos = this.st.length;
      const lg = new Float32Array(VOCAB);
      lg[answer(this.st)] = 10;
      return lg;
    },
  };
  if (slots) {
    e.saveSlot = function (k) { this.saved.set(k, this.st.slice()); this.log.push("sv"); };
    e.loadSlot = function (k) {
      const s = this.saved.get(k);
      if (!s) throw new Error("no saved slot " + k);
      this.st = s.slice(); this.pos = s.length; this.log.push("ld" + s.length);
    };
    e.dropSlot = function (k) { this.saved.delete(k); };
  }
  return e;
}
// a 3-word answer that depends on every token in the state (then the end token)
function hashAnswer(tok) {
  const words = ["alpha", "beta", "gamma", "delta"].map((w) => tok.encode(w)[0] ?? 0);
  const head = tok.encode("assistant\n").length;
  return (st) => {
    let n = 0;
    for (let i = st.length - 1; i >= 0 && st[i] !== tok.ID["<|im_start|>"]; i--) n++;
    if (n >= head + 3) return tok.ID["<|im_end|>"];
    let h = 7;
    for (const t of st) h = (h * 31 + t) >>> 0;
    return words[h % words.length];
  };
}
const collect = async (it) => { let s = ""; for await (const d of it) s += d; return s; };
const SYS = "You are a coder. Tools: read_file(path).";

Deno.test("engine model: a changed middle resumes after the system prompt, with the same answer and state as a fresh prefill", async () => {
  const tok = makeTok();
  const E = stateEngine(tok, { answer: hashAnswer(tok) });
  const m = engineModel(E, tok, { spec: false, maxNew: 8 });
  const sysLen = buildIds(tok, { system: SYS, turns: [] }).length;

  const a1 = await collect(m.generate({ system: SYS, turns: [{ role: "user", text: "hi" }] }));
  eq(m.stats.pins, 1, "the system prompt was saved on the way");
  ok(E.log.includes("pre" + sysLen), "the prefill paused at the end of the system prompt");
  eq(E.saved.get("engine-model:pin"), buildIds(tok, { system: SYS, turns: [] }), "the slot holds exactly the system prompt");

  const t2 = [{ role: "user", text: "hi" }, { role: "assistant", text: a1 }, { role: "user", text: "<tool_response>\n" + "x".repeat(300) + "\n</tool_response>" }];
  await collect(m.generate({ system: SYS, turns: t2 }));
  eq(m.stats.pins, 1, "a follow-up extends the caches: no new save");
  ok(m.stats.last.reused > sysLen, "follow-up reused the whole previous conversation");

  // compaction rewrote the old tool output: the prompt changes right after the first answer
  const t3 = [t2[0], t2[1], { role: "user", text: stubResults(t2[2].text, ["read_file a"]) }, { role: "assistant", text: "ok" }, { role: "user", text: "next" }];
  E.log = [];
  const a3 = await collect(m.generate({ system: SYS, turns: t3 }));
  ok(E.log[0] === "ld" + sysLen && !E.log.includes("reset"), "loaded the system prompt's checkpoint: " + E.log.join(","));
  eq(m.stats.last.reused, sysLen);
  const ids3 = buildIds(tok, { system: SYS, turns: t3.map((t) => (t.role === "assistant" ? { role: "assistant", ids: tok.encode(t.text) } : t)) });
  eq(m.stats.last.prefilled, ids3.length - sysLen, "only what follows the system prompt was prefilled");

  // the same request on a fresh engine without checkpoints: the same answer, the same state
  const F = stateEngine(tok, { slots: false, answer: hashAnswer(tok) });
  const f = engineModel(F, tok, { spec: false, maxNew: 8 });
  await collect(f.generate({ system: SYS, turns: [{ role: "user", text: "hi" }] }));   // same sampled ids for turn 1
  const b3 = await collect(f.generate({ system: SYS, turns: t3 }));
  eq(a3, b3, "same answer");
  eq(E.st, F.st, "same engine state");
});

Deno.test("engine model: pin: false, a lost slot, or another system prompt fall back to a full prefill", async () => {
  const tok = makeTok();
  const run = async (m, sys, text) => collect(m.generate({ system: sys, turns: [{ role: "user", text }] }));
  // pin: false never touches slots
  const E0 = stateEngine(tok, { answer: hashAnswer(tok) });
  const m0 = engineModel(E0, tok, { spec: false, maxNew: 8, pin: false });
  await run(m0, SYS, "a"); await run(m0, SYS, "b");
  eq(E0.saved.size, 0); eq(m0.stats.last.reused, 0);
  // the slot was dropped by someone else: prefill again and save it again
  const E = stateEngine(tok, { answer: hashAnswer(tok) });
  const m = engineModel(E, tok, { spec: false, maxNew: 8 });
  await run(m, SYS, "a");
  E.saved.clear();
  await run(m, SYS, "b");
  eq(m.stats.last.reused, 0); eq(m.stats.pins, 2);
  eq(E.st.slice(0, 5), buildIds(tok, { system: SYS, turns: [{ role: "user", text: "b" }] }).slice(0, 5));
  // another system prompt: nothing reused, its own prefix saved instead
  await run(m, "other", "c");
  eq(m.stats.last.reused, 0); eq(m.stats.pins, 3);
  eq(E.saved.get("engine-model:pin"), buildIds(tok, { system: "other", turns: [] }));
  // and back: the pinned prefix is "other" now, so SYS is prefilled in full again
  await run(m, SYS, "d");
  eq(m.stats.last.reused, 0);
  // no system prompt: nothing to pin
  const E2 = stateEngine(tok, { answer: hashAnswer(tok) });
  const m2 = engineModel(E2, tok, { spec: false, maxNew: 8 });
  await run(m2, "", "a");
  eq(m2.stats.pins, 0);
});

// ---- a whole Code mode session: 24 steps, compactions on the way

Deno.test("agent session: 20+ steps never prefill more than what is new; after a compaction only what follows the system prompt", async () => {
  const tok = makeTok();
  const N = 22;
  const call = (i) => `<tool_call>\n<function=read_file>\n<parameter=path>\nf${i}.txt\n</parameter>\n</function>\n</tool_call>`;
  const replies = [...Array.from({ length: N }, (_, i) => call(i)), "done"];
  // the engine answers the scripted replies in order, whatever its state
  let queue = [], r = 0;
  const E = stateEngine(tok, {
    answer: () => {
      if (!queue.length) queue = [...tok.encode(replies[r++] ?? "done"), tok.ID["<|im_end|>"]];
      return queue.shift();
    },
  });
  const m = engineModel(E, tok, { spec: false, maxNew: 256 });
  const tools = [{ name: "read_file", description: "read a file", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    run: ({ path }) => `${path}:\n` + (path + " line\n").repeat(40) }];
  const log = [];   // per step: the prompt ids, what was reused, what the engine held before
  const gen = m.generate;
  const generate = async function* (req) {
    const before = E.st.slice();
    yield* gen(req);
    eq(E.st, m.fed, "the engine holds exactly what the adapter thinks it holds");
    log.push({ ...m.stats.last, before, compactedBefore: compacted });
    compacted = false;
  };
  let compacted = false, compactions = 0;
  const A = new Agent({
    generate, tools, system: SYS, maxSteps: 30, count: (t) => tok.encode(t).length,
    budget: () => A.count(A.system) + 6000,   // the system prompt + ~10 steps of full tool outputs
    onEvent: (e) => { if (e.type === "compacted" && e.tier < 4) { compacted = true; compactions++; } },
  });
  const res = await A.run("read every file");
  eq(res.reason, "done", "the session ran to its answer");
  ok(log.length >= 20, `${log.length} steps`);
  ok(compactions >= 2, `the budget forced compactions (${compactions})`);
  const sysLen = buildIds(tok, { system: A.system, turns: [] }).length;
  for (const [i, s] of log.entries()) {
    if (i === 0) { eq(s.reused, 0); continue; }
    ok(s.reused >= sysLen, `step ${i + 1}: the system prompt + tools is never prefilled again (reused ${s.reused})`);
    if (!s.compactedBefore) {
      // nothing changed in the conversation: everything the engine held is reused
      eq(s.reused, s.before.length, `step ${i + 1}: the whole previous state is reused`);
    } else eq(s.reused, sysLen, `step ${i + 1}: after a compaction, resumed at the system prompt`);
  }
  eq(m.stats.pins, 1, "the system prompt was prefilled and saved once for the whole session");
});

// ---- compaction's stable form

Deno.test("stubResults: stable, idempotent, keeps short results", () => {
  const long = "y".repeat(250);
  const text = `<tool_response>\nshort\n</tool_response>\n<tool_response>\n${long}\n</tool_response>`;
  const once = stubResults(text, ["read_file a", "read_file b"]);
  eq(once, "<tool_response>\nshort\n</tool_response>\n<tool_response>\n(output of read_file b dropped; run it again if needed)\n</tool_response>");
  eq(stubResults(once, ["read_file a", "read_file b"]), once, "compacting a compacted turn changes nothing");
  eq(stubResults(text, ["read_file a", "read_file b"]), once, "the same input gives the same stub");
  eq(stubResults(`<tool_response>\n${long}\n</tool_response>`, undefined), "<tool_response>\n(output of this call dropped; run it again if needed)\n</tool_response>");
  eq(stubResults("plain user text", []), "plain user text");
});

Deno.test("compaction: turns before the first changed one stay byte-identical, and a compacted turn never changes again", async () => {
  const results = (k) => `<tool_response>\n${"z".repeat(400 + k)}\n</tool_response>`;
  const A = new Agent({ generate: async function* () {}, tools: [], budget: 1000, count: (t) => t.length });
  const events = [];
  A.onEvent = (e) => events.push(e);
  // one finished request of 2 steps, then a running one
  A.turns = [
    { role: "user", text: "first", req: 1 },
    { role: "assistant", text: "call a", req: 1 }, { role: "user", text: results(1), req: 1, calls: ["a"] },
    { role: "assistant", text: "call b", req: 1 }, { role: "user", text: results(2), req: 1, calls: ["b"] },
    { role: "assistant", text: "answer 1", req: 1 },
    { role: "user", text: "second", req: 2 },
    { role: "assistant", text: "call c", req: 2 }, { role: "user", text: results(3), req: 2, calls: ["c"] },
    { role: "assistant", text: "call d", req: 2 }, { role: "user", text: results(4), req: 2, calls: ["d"] },
  ];
  A.reqs = { 1: { calls: [{ name: "read_file", arguments: { path: "a" } }], done: true, answer: "answer 1" }, 2: { calls: [], done: false, answer: "" } };
  A.req = 2;
  A.system = "";
  const before = A.turns.map((t) => t.text);
  A._compact(2);
  const ev = events.find((e) => e.type === "compacted");
  ok(ev && ev.tier >= 1, "compacted");
  ok(Number.isInteger(ev.at), "reports where the prompt first changed");
  const after = A.turns.map((t) => t.text);
  eq(after.slice(0, ev.at), before.slice(0, ev.at), "everything before `at` is unchanged");
  ok(after[ev.at] !== before[ev.at], "the turn at `at` did change");
  // the conversation grows under the budget: the compacted turns are left exactly as they are
  const snap = after.slice();
  A.turns.push({ role: "assistant", text: "call e", req: 2 }, { role: "user", text: "<tool_response>\nok\n</tool_response>", req: 2, calls: ["e"] });
  events.length = 0;
  A._compact(2);
  eq(events.filter((e) => e.type === "compacted").length, 0, "under the budget: no compaction");
  eq(A.turns.map((t) => t.text).slice(0, snap.length), snap, "the compacted prefix renders the same");
});
