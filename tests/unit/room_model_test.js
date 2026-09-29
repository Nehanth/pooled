// harness/room-model.js over a fake roomApi: the fake stands in for room.js roomGenerate (it keeps
// `fed` like the room's caches and asserts every call's ids extend it when they should), with the
// word tokenizer from engine_model_test.js.
import { roomModel, ContextFull } from "../../harness/room-model.js";
import { Agent } from "../../harness/agent.js";
import { reusablePrefix } from "../../room/conversation.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const BASE = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "<function=", "rm_rf>", "read_file>", "<parameter=", "path>", "</parameter>", "</function>", "</tool_call>", "hello", " world", "\n"];
const CHARS = [..."abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?'\"-_/<>=(){}[]#\n"];
function makeTok(extra = []) {
  const words = [...BASE, ...extra];
  const V = [...words, ...CHARS.filter((c) => !words.includes(c))];
  const ID = Object.fromEntries(V.map((w, i) => [w, i]));
  return {
    V, ID, vocab: ID,
    encode(s) {
      const out = [];
      for (let i = 0; i < s.length;) {
        let best = null;
        for (const w of V) if (s.startsWith(w, i) && (!best || w.length > best.length)) best = w;
        if (!best) throw new Error("untokenizable: " + s[i]);
        out.push(ID[best]); i += best.length;
      }
      return out;
    },
    decode: (ids) => ids.map((i) => V[i]).join(""),
  };
}
const argmax = (lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; };

// The fake room: every reply is a list of words; the model prefers script word k (logit 10) and
// read_file> second (5). Written tokens join `fed` like the room's caches (the stop token does not).
function fakeRoom(tok, replies, { maxSeq = 4096, onStep } = {}) {
  let call = 0;
  const room = {
    fed: [], calls: [], maxSeqV: maxSeq,
    tok: () => tok, maxSeq: () => room.maxSeqV,
    async generate(ids, { onToken, stop, maxNew, sample, signal, pin }) {
      const script = replies[call++] || ["<|im_end|>"];
      const reused = reusablePrefix(room.fed, ids);
      room.calls.push({ ids: ids.slice(), reused, fedBefore: room.fed.slice(), maxNew, pin });
      room.fed = ids.slice();
      const tokens = [];
      let reason = "stop";
      for (let k = 0; ; k++) {
        if (signal?.aborted) { reason = "abort"; break; }
        if (tokens.length >= maxNew) { reason = "max"; break; }
        const lg = new Float32Array(tok.V.length);
        lg[tok.ID["read_file>"]] = 5;
        lg[tok.ID[script[Math.min(k, script.length - 1)]]] = 10;
        const t = sample(lg);
        if (stop.has(t)) break;
        tokens.push(t); room.fed.push(t);
        onToken(t, 0);
        await onStep?.(k);
        await null;   // let the consumer run between tokens, like a lap
      }
      return { tokens, reason, reused, prefilled: ids.length - reused, count: tokens.length, tps: 10, stats: `${tokens.length} tok` };
    },
  };
  return room;
}
const collect = async (it, out = []) => { for await (const d of it) out.push(d); return out; };

Deno.test("room model: deltas stream in order and the stop token ends the answer", async () => {
  const tok = makeTok();
  const room = fakeRoom(tok, [["hello", " world", "!", "<|im_end|>", "x"]]);
  const m = roomModel(room, { sample: argmax });
  const parts = await collect(m.generate({ turns: [{ role: "user", text: "hi" }] }));
  eq(parts.join(""), "hello world!");
  ok(parts.length >= 2, "streamed in pieces: " + JSON.stringify(parts));
  eq(m.stats.last.reason, "stop");
  eq(room.calls[0].maxNew, Math.min(4096, 4096 - room.calls[0].ids.length - 16));
});

Deno.test("room model: the tool-name constraint goes through the sampler the room is given", async () => {
  const tok = makeTok();
  const CALL = ["<tool_call>", "\n", "<function=", "rm_rf>", "\n", "</function>", "\n", "</tool_call>", "<|im_end|>"];
  const free = roomModel(fakeRoom(tok, [CALL]), { sample: argmax });
  eq((await collect(free.generate({ turns: [{ role: "user", text: "hi" }] }))).join(""), "<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>");
  const tools = [{ name: "read_file", parameters: { properties: { path: {} } } }];
  const m = roomModel(fakeRoom(tok, [CALL]), { sample: argmax, tools });
  eq((await collect(m.generate({ turns: [{ role: "user", text: "hi" }] }))).join(""), "<tool_call>\n<function=read_file>\n</function>\n</tool_call>");
});

Deno.test("room model: an invented <tool_response> ends the answer (split over tokens: as text)", async () => {
  const tok = makeTok();
  const script = ["hello", ...[..."<tool_response>"], ..."fake result".split(""), "<|im_end|>"];
  const room = fakeRoom(tok, [script]);
  const m = roomModel(room, { sample: argmax });
  const parts = await collect(m.generate({ turns: [{ role: "user", text: "hi" }] }));
  eq(parts.join(""), "hello");
  ok(!parts.some((p) => p.includes("<")), "no part of the tag was streamed: " + JSON.stringify(parts));
  eq(m.stats.last.reason, "tool_response");
  eq(m.idsFor("hello"), [tok.ID.hello], "the kept ids decode to exactly the cut text");
});

Deno.test("room model: a single-token <tool_response> is a stop token, so the next step extends the caches", async () => {
  const tok = makeTok(["<tool_response>"]);
  const room = fakeRoom(tok, [["hello", "<tool_response>", "x"], ["ok", "<|im_end|>"]]);
  const m = roomModel(room, { sample: argmax });
  const a = (await collect(m.generate({ turns: [{ role: "user", text: "hi" }] }))).join("");
  eq(a, "hello");
  await collect(m.generate({ turns: [{ role: "user", text: "hi" }, { role: "assistant", text: a }, { role: "user", text: "more" }] }));
  eq(room.calls[1].reused, room.calls[1].fedBefore.length, "the whole previous state was reused");
});

Deno.test("room model: assistant turns replay their sampled ids, so each step extends the room's caches", async () => {
  const tok = makeTok();
  // "hello world" re-tokenized is ["hello", " world"]; sampled here as single characters
  const room = fakeRoom(tok, [[..."hello world", "<|im_end|>"], ["ok", "<|im_end|>"]]);
  const m = roomModel(room, { sample: argmax });
  const a = (await collect(m.generate({ system: "s", turns: [{ role: "user", text: "hi" }] }))).join("");
  eq(a, "hello world");
  eq(m.idsFor(a).length, 11, "the ids it was sampled as, one per character");
  await collect(m.generate({ system: "s", turns: [{ role: "user", text: "hi" }, { role: "assistant", text: a }, { role: "user", text: "more" }] }));
  const [c0, c1] = room.calls;
  const expected = [...c0.ids, ...[..."hello world"].map((ch) => tok.ID[ch])];
  eq(c1.ids.slice(0, expected.length), expected, "prompt = previous prompt + the answer's own ids");
  eq(c1.reused, c0.ids.length + 11, "everything the caches held was reused");
});

Deno.test("room model: the system prompt's length goes to the room as pin (its own checkpoint, issue #73)", async () => {
  const tok = makeTok();
  const room = fakeRoom(tok, [["ok", "<|im_end|>"], ["ok", "<|im_end|>"], ["ok", "<|im_end|>"]]);
  const m = roomModel(room, { sample: argmax });
  await collect(m.generate({ system: "the tools", turns: [{ role: "user", text: "hi" }] }));
  await collect(m.generate({ system: "the tools", turns: [{ role: "user", text: "other" }] }));
  await collect(m.generate({ turns: [{ role: "user", text: "hi" }] }));
  const sys = [tok.ID["<|im_start|>"], ...tok.encode("system\nthe tools"), tok.ID["<|im_end|>"], ...tok.encode("\n")];
  eq(room.calls[0].pin, sys.length);
  eq(room.calls[0].ids.slice(0, sys.length), sys, "the prompt starts with exactly those tokens");
  eq(room.calls[1].pin, sys.length);
  eq(room.calls[2].pin, 0, "no system prompt: nothing to pin");
});

Deno.test("room model: turns no longer in the conversation are pruned from the id map", async () => {
  const tok = makeTok();
  const room = fakeRoom(tok, [["hello", "<|im_end|>"], [" world", "<|im_end|>"]]);
  const m = roomModel(room, { sample: argmax });
  const a = (await collect(m.generate({ turns: [{ role: "user", text: "q1" }] }))).join("");
  ok(m.idsFor(a));
  await collect(m.generate({ turns: [{ role: "user", text: "new task" }] }));
  eq(m.idsFor(a), undefined);
  ok(m.idsFor(" world"));
});

Deno.test("room model: abort mid-stream keeps what was said and reports it", async () => {
  const tok = makeTok();
  const ac = new AbortController();
  const room = fakeRoom(tok, [["h", "e", "l", "l", "o", "<|im_end|>"]], { onStep: (k) => { if (k === 1) ac.abort(); } });
  const m = roomModel(room, { sample: argmax });
  const s = (await collect(m.generate({ turns: [{ role: "user", text: "hi" }], signal: ac.signal }))).join("");
  eq(s, "he");
  eq(m.stats.last.reason, "abort");
});

Deno.test("room model: a consumer that leaves early stops the room's step", async () => {
  const tok = makeTok();
  let seen = null;
  const room = fakeRoom(tok, [Array(50).fill("h")]);
  const gen = room.generate.bind(room);
  room.generate = (ids, o) => { seen = o.signal; return gen(ids, o); };
  const m = roomModel(room, { sample: argmax });
  for await (const d of m.generate({ turns: [{ role: "user", text: "hi" }] })) { void d; break; }
  ok(seen.aborted, "the signal handed to the room was aborted");
});

Deno.test("room model: ContextFull before the room is asked; budget and count", async () => {
  const tok = makeTok();
  const room = fakeRoom(tok, []);
  room.maxSeqV = 40;
  const m = roomModel(room, { sample: argmax });
  let err = null;
  try { await collect(m.generate({ turns: [{ role: "user", text: "x".repeat(30) }] })); } catch (e) { err = e; }
  ok(err instanceof ContextFull, "threw " + err);
  eq(room.calls.length, 0);
  room.maxSeqV = 16384;
  eq(m.budget(), 16384 - (4096 + 64));
  room.maxSeqV = 2048;
  eq(m.budget(), 2048 - (512 + 64));
  eq(m.count("hello world"), 2);
  eq(m.count("hello world"), 2, "cached");
});

Deno.test("room model: room failures reach the caller", async () => {
  const tok = makeTok();
  const m = roomModel({ tok: () => tok, maxSeq: () => 4096, generate: async () => { throw new Error("a device left"); } }, { sample: argmax });
  let err = null;
  try { await collect(m.generate({ turns: [{ role: "user", text: "hi" }] })); } catch (e) { err = e; }
  eq(err?.message, "a device left");
});

Deno.test("room model: drives the Agent loop through a tool call", async () => {
  const tok = makeTok(["<tool_response>"]);
  const CALL = ["<tool_call>", "\n", "<function=", "read_file>", "\n", "<parameter=", "path>", "\n", "a", "\n", "</parameter>", "\n", "</function>", "\n", "</tool_call>", "<|im_end|>"];
  const room = fakeRoom(tok, [CALL, ["d", "o", "n", "e", "<|im_end|>"]]);
  const tools = [{ name: "read_file", description: "read", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }, run: async ({ path }) => `contents of ${path}` }];
  const m = roomModel(room, { sample: argmax, tools });
  const agent = new Agent({ generate: m.generate, tools, style: "xml", budget: m.budget(), count: m.count });
  const r = await agent.run("read a");
  eq(r.text, "done");
  eq(r.calls, 1);
  eq(room.calls[1].reused, room.calls[1].fedBefore.length, "step 2 extended the caches of step 1");
  ok(agent.turns[2].text.includes("contents of a"));
});

Deno.test("model-common: streaming decode holds back a split UTF-8 character", async () => {
  const { deltaDecoder, asyncQueue } = await import("../../harness/model-common.js");
  const bytes = { decode: (ids) => new TextDecoder().decode(new Uint8Array(ids)) };
  const d = deltaDecoder(bytes);
  const enc = [...new TextEncoder().encode("aé€")];
  const out = enc.map((b) => d.push(b));
  eq(out, ["a", "", "é", "", "", "€"]);
  eq(d.text, "aé€");
  const q = asyncQueue();
  q.push(1); q.push(2); setTimeout(() => { q.push(3); q.end(); }, 1);
  const got = []; for await (const x of q) got.push(x);
  eq(got, [1, 2, 3]);
});

Deno.test("room model: a call whose tokens the grammar keeps forcing is ended as garbage; the agent runs none of it", async () => {
  const tok = makeTok();
  // a broken engine: after a good start, its top token is the end of turn (masked inside a call) every time
  const room = fakeRoom(tok, [["<tool_call>", "\n", "<function=", "read_file>", "\n", "<parameter=", "path>", "\n", "<|im_end|>"], ["hello"]]);
  const tools = [{ name: "read_file", description: "r", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }, mutates: false, run: () => { throw new Error("ran"); } }];
  const m = roomModel(room, { tools, sample: argmax });
  const A = new Agent({ generate: m.generate, tools, usage: () => m.stats.last });
  await A.run("go");
  const u = room.calls.length && A.turns[1].text;
  ok(u && u.length < 400, "stopped early, not at the cap: " + JSON.stringify(u));
  ok(/not run: \d+ tokens were forced/.test(A.turns[2].text), A.turns[2].text);
});
