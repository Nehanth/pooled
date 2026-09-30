// pooled chat <CODE | link>: talk to a room's model from the terminal (cli/lib/chat.js has the parts
// that are unit tested). It joins as an API client over the same Bridge as pooled serve (room.js:
// the gate, v2 asks), streams each answer, and keeps the conversation for the next turn.
import readline from "node:readline";
import { Bridge } from "./room.js";
import { Ask, Collector } from "./answer.js";
import { askBody, newRid, cleanText } from "./common.js";
import { parseChatArgs, parseLine, History, Renderer, statusLine, tokPerSec, explainChatError, fmtCode, UsageError, HELP_CHAT, DIM, RESET } from "./chat.js";

const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function busyText(d) {
  const why = cleanText(d.why, 300);
  if (d.code === "off") return "the host does not allow API clients in this room";
  if (d.code === "loading") return "the room's model is not started yet (no model, or still loading)";
  if (d.code === "degraded") return why || "a device left the room; the host has to re-deal the layers first";
  if (d.code === "ctx") return `the conversation no longer fits the room's context (${d.n} tokens of ${d.max}): /clear starts over`;
  if (d.code === "queue") return `the room's queue is full${why ? `: ${why}` : ""}`;
  return why || "the room cannot answer now";
}

// embedded: run inside pooled host (its room screen comes back after /exit): leaving resolves instead
// of ending the process, and Peer is the host's own WebRTC stack (one per process)
export async function chatMain(argv, { version = "", embedded = false, Peer = null } = {}) {
  let opts;
  try { opts = parseChatArgs(argv); }
  catch (e) {
    if (!(e instanceof UsageError)) throw e;
    process.stderr.write(`pooled chat: ${e.message}\n\n${HELP_CHAT}`);
    return 2;
  }
  if (opts.help) { process.stdout.write(HELP_CHAT); return 0; }

  const interactive = opts.prompt == null && !!process.stdin.isTTY;
  const errTTY = !!process.stderr.isTTY;
  const color = opts.color && !!process.stdout.isTTY;
  const ecolor = opts.color && errTTY;
  const edim = (s) => (ecolor ? DIM + s + RESET : s);
  const say = (s) => process.stderr.write(s + "\n");
  const fail = (err) => {
    const x = explainChatError(err, { code: opts.code });
    say(`pooled chat: ${x.message}`);
    if (x.hint) say(`  ${x.hint}`);
    return 1;
  };

  // one question from a script: the prompt argument, else all of stdin
  let oneShot = opts.prompt;
  if (!interactive && oneShot == null) {
    const chunks = [];
    for await (const c of process.stdin) chunks.push(c);
    oneShot = Buffer.concat(chunks).toString("utf8").trim();
    if (!oneShot) { say("pooled chat: nothing to ask (stdin was empty)"); return 2; }
  }

  // the bridge's own lines (waiting in the lobby, the host's link lost and back), dimmed on stderr
  let spinner = null;
  const log = (m) => { clearSpin(); say(edim(`· ${cleanText(m, 400).replace("start pooled serve with", "start pooled chat with")}`)); };
  const tag = Math.random().toString(36).slice(2, 6).padEnd(4, "0");
  const bridge = new Bridge({ code: opts.code, key: opts.key, signal: opts.signal, name: opts.name || `pooled chat ${tag}`,
    client: `pooled-chat/${version}`, log, Peer });

  // a spinner on stderr while nothing is on screen yet (joining, waiting for the model, the prompt being read)
  let spinLabel = "", spinAt = 0;
  function spin(label) {
    spinLabel = label;
    if (!errTTY || spinner) return;
    spinner = setInterval(() => { process.stderr.write(`\r\x1b[K${edim(`${SPIN[spinAt++ % SPIN.length]} ${spinLabel}`)}`); }, 100);
  }
  function clearSpin() {
    if (!spinner) return;
    clearInterval(spinner); spinner = null;
    process.stderr.write("\r\x1b[K");
  }

  let leaving = false, current = null, rl = null, left = null;
  const leftP = new Promise((r) => { left = r; });
  const leave = async (code = 0) => {
    if (leaving) { if (embedded) return; process.exit(code); }
    leaving = true;
    clearSpin();
    try { rl?.close(); } catch {}
    await bridge.leave().catch(() => {});
    await new Promise((r) => process.stdout.write("", r));
    if (embedded) { process.off("SIGINT", onInt); left(code); return; }
    process.exit(code);
  };
  if (!embedded) process.on("SIGTERM", () => leave(0));
  // Ctrl-C: stops the answer being written; otherwise leaves (readline sends its own below)
  const onInt = () => { if (current) current.stop(); else leave(interactive ? 0 : 130); };
  process.on("SIGINT", onInt);

  spin(`joining room ${fmtCode(opts.code)}`);
  try { await bridge.connect(); }
  catch (e) { clearSpin(); return fail(e); }
  // the host's ai-ready-all follows its hello
  if (!bridge.ready) await new Promise((r) => { const t = setTimeout(r, 1500); bridge.on("state", () => { if (bridge.ready) { clearTimeout(t); r(); } }); });
  clearSpin();

  // the room closed to us, or its host is gone for good: say so and leave
  bridge.on("state", () => {
    if (leaving) return;
    if (bridge.kicked) { clearSpin(); const x = explainChatError(new Error(bridge.kicked), { code: opts.code }); say(`pooled chat: ${x.message.startsWith("The host") ? x.message : `the host said: ${bridge.kicked}`}`); leave(1); }
    else if (!bridge.connected && /did not come back/.test(bridge.gone || "")) { clearSpin(); say(`pooled chat: the host of room ${fmtCode(opts.code)} left and did not come back`); leave(1); }
  });

  const model = () => bridge.modelLabel || bridge.model || null;
  // the model is up: at once, or once the host starts it (a script waits --wait seconds)
  async function ready() {
    if (bridge.ready) return;
    const noteFirst = interactive ? "no model started in this room yet: waiting for the host to start one (Ctrl-C to leave)" : null;
    if (noteFirst) say(edim(`· ${noteFirst}`));
    spin("waiting for the room's model");
    await new Promise((resolve, reject) => {
      const t = interactive ? null : setTimeout(() => { bridge.off("state", on); reject(new Error("no model started yet")); }, opts.waitMs);
      const on = () => { if (bridge.ready) { clearTimeout(t); bridge.off("state", on); resolve(); } };
      bridge.on("state", on);
    }).finally(clearSpin);
  }

  const history = new History(opts.system);
  let thinking = opts.thinking;

  // one answer: the whole conversation goes to the room; tokens stream to stdout as they come
  async function turn(text) {
    history.add("user", text);
    try { await ready(); }
    catch (e) { history.settle(""); throw e; }
    let req;
    try { req = history.request({ maxTokens: opts.maxTokens, thinking, temperature: opts.temperature, hostMeta: bridge.hostMeta }); }
    catch (e) { history.settle(""); throw e; }
    const v2 = bridge.hostApi >= 2;
    const rid = newRid();
    const collector = new Collector({ id: rid });
    let started = false;
    const R = new Renderer({ color, showThinking: interactive,
      write: (s) => { if (!started) { started = true; clearSpin(); } process.stdout.write(s); } });
    const ask = new Ask({ req, meta: { id: rid }, v2, encoders: [collector, R], log: () => {}, label: "chat" });
    spin("waiting for the room");
    const t0 = Date.now();
    let stats = "";
    const res = await new Promise((resolve) => {
      let done = false;
      const finish = (r) => { if (done) return; done = true; current = null; clearSpin(); resolve(r); };
      current = { stop: () => { bridge.stop(rid); finish({ stopped: true }); } };
      const ok = bridge.ask(rid, askBody(req, v2), (d) => {
        switch (d.t) {
          case "ai-queued": if (!started) spinLabel = `queued in the room${Number.isInteger(d.pos) && d.pos > 0 ? ` (${d.pos} ahead)` : ""}`; return;
          case "ai-genstart": case "ai-token": case "ai-call": case "ai-gendone": {
            if (d.t === "ai-genstart" && !started) spinLabel = "reading the conversation";
            if (d.t === "ai-gendone") stats = d.stats || "";
            const r = ask.feed(d);
            if (!r) return;
            if (d.t !== "ai-gendone") bridge.stop(rid);
            finish(r.error ? { error: r.error } : { answer: r.answer });
            return;
          }
          case "ai-busy": finish({ error: new Error(busyText(d)) }); return;
          case "x-fail": finish({ error: new Error(cleanText(d.why, 300) || "the room dropped the request") }); return;
        }
      });
      if (!ok) finish({ error: new Error(`not connected to room ${fmtCode(opts.code)}`) });
    });
    R.end();
    const a = res.answer || collector.answer;
    if (res.error) { history.settle(R.answer ? collector.answer.text : ""); throw res.error; }
    // the text the model wrote (not the terminal's cleaned copy): the next turn's prompt starts with it
    history.settle(collector.answer.text || R.answer, { stopped: !!res.stopped });
    const tps = tokPerSec(stats, { tokens: R.tokens, tFirst: R.tFirst, tEnd: Date.now() });
    const usage = res.stopped ? { ...a.usage, out: R.tokens, in: R.promptTokens } : a.usage;
    const line = statusLine({ code: opts.code, model: model(), answer: { ...a, usage }, tps, stopped: !!res.stopped });
    if (interactive) say(edim(line));
    else if (errTTY) say(edim(line));
    void t0;
    return res;
  }

  if (!interactive) {
    try { const r = await turn(oneShot); await bridge.leave(); return r.stopped ? 130 : 0; }
    catch (e) { clearSpin(); await bridge.leave().catch(() => {}); return fail(e); }
  }

  const hostBit = bridge.hostName ? ` · host ${bridge.hostName}` : "";
  say(`pooled chat · room ${fmtCode(opts.code)} · ${model() || "no model started yet"}${hostBit}`);
  say(edim("  type a message · /help for commands · Ctrl-C stops an answer · /exit or Ctrl-D leaves"));
  rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: color ? "\x1b[1m>>>\x1b[0m " : ">>> ", terminal: true, historySize: 200 });
  // readline takes Ctrl-C in raw mode: the same rule (stop the answer, else leave)
  rl.on("SIGINT", () => {
    if (current) { current.stop(); return; }
    process.stdout.write("\n");
    leave(0);
  });
  rl.on("close", () => { if (!leaving) { process.stdout.write("\n"); leave(0); } });
  const queue = [];
  let busy = false;
  const handle = async (line) => {
    const p = parseLine(line);
    if (!p) return;
    if (p.cmd === "exit") { leave(0); return; }
    if (p.cmd === "clear") { history.clear(); say(edim("· conversation cleared")); return; }
    if (p.cmd === "help") { say(HELP_CHAT.split("\nIn the chat\n")[1].split("\nOptions\n")[0].trimEnd()); return; }
    if (p.cmd === "think") {
      thinking = p.arg === "on" ? true : p.arg === "off" ? false : !thinking;
      say(edim(`· thinking ${thinking ? "on: the model reasons first (dimmed), then answers" : "off"}`));
      return;
    }
    if (p.cmd === "unknown") { say(edim(`· unknown command /${p.arg}; /help lists them`)); return; }
    try { await turn(p.text); }
    catch (e) { const x = explainChatError(e, { code: opts.code }); say(`pooled chat: ${x.message}`); if (x.hint) say(`  ${x.hint}`); }
  };
  rl.on("line", async (line) => {
    queue.push(line);
    if (busy) return;
    busy = true;
    while (queue.length && !leaving) { await handle(queue.shift()); }
    busy = false;
    if (!leaving) rl.prompt();
  });
  rl.prompt();
  return embedded ? leftP : new Promise(() => {});   // until leave()
}
