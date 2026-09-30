// OpenAI Codex CLI (0.104) against `pooled serve`'s Responses API: Codex reads a file with its
// exec_command tool and answers from what it read. Two modes:
//
//   node tests/e2e/agents/codex.mjs --mock        a scripted stand-in room behind the real HTTP server
//                                                 (cli/lib/http.js + responses.js): no GPU, no PeerJS;
//                                                 checks Codex parses our event stream and resends
//                                                 our items (function_call, reasoning) as input
//   node tests/e2e/agents/codex.mjs --base http://127.0.0.1:PORT
//                                                 a live `pooled serve` (a real room and model; used by
//                                                 tests/e2e/serve_responses.mjs)
//
// Needs `codex` on PATH (or --codex PATH). Prints one JSON line of checks; exit 0 only when all passed.
// runCodex() is also imported by serve_responses.mjs.
import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { EventEmitter } from "events";

const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../../..");

// one `codex exec` run in a fresh CODEX_HOME whose only provider is pooled serve at base
// -> { code, stdout, stderr, events: [parsed --json lines], dir }
export async function runCodex({ base, prompt, files = {}, codex = "codex", timeoutMs = 600000, contextWindow = 32768, extraArgs = [] }) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-codex-home-"));
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pooled-codex-work-"));
  for (const [f, text] of Object.entries(files)) fs.writeFileSync(path.join(dir, f), text);
  // Codex needs the context window of a provider it does not know, and a compaction limit under it
  fs.writeFileSync(path.join(home, "config.toml"), [
    `model = "pooled"`, `model_provider = "pooled"`,
    `model_context_window = ${contextWindow}`, `model_auto_compact_token_limit = ${Math.floor(contextWindow * 0.8)}`,
    `[model_providers.pooled]`, `name = "Pooled"`, `base_url = "${base}/v1"`, `wire_api = "responses"`, `stream_max_retries = 0`, `request_max_retries = 0`, "",
  ].join("\n"));
  const args = ["exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only", "-C", dir, ...extraArgs, prompt];
  return await new Promise((resolve) => {
    const p = spawn(codex, args, { env: { ...process.env, CODEX_HOME: home }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    p.stdout.on("data", (c) => { stdout += c; });
    p.stderr.on("data", (c) => { stderr += c; });
    const timer = setTimeout(() => p.kill("SIGKILL"), timeoutMs);
    p.on("close", (code) => {
      clearTimeout(timer);
      const events = stdout.split("\n").filter((l) => l.startsWith("{")).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
      fs.rmSync(home, { recursive: true, force: true });
      fs.rmSync(dir, { recursive: true, force: true });
      resolve({ code, stdout, stderr, events });
    });
  });
}

// what a finished run did: the commands it ran and the agent's last message
export function summarize(run) {
  const items = run.events.filter((e) => e.type === "item.completed").map((e) => e.item);
  return {
    commands: items.filter((i) => i.type === "command_execution").map((i) => ({ command: i.command, output: i.aggregated_output, exit: i.exit_code })),
    answer: items.filter((i) => i.type === "agent_message").map((i) => i.text).at(-1) ?? null,
    failed: run.events.filter((e) => e.type === "turn.failed" || e.type === "error").map((e) => e.error?.message || e.message),
  };
}

async function mock(codex) {
  const { createServer } = await import(path.join(ROOT, "cli/lib/http.js"));
  const asks = [];
  class Room extends EventEmitter {
    constructor() { super(); this.code = "MOCK"; this.connected = true; this.ready = true; this.model = "qwen3-1.7b"; this.hostMeta = { api: 2, ctx: 32768 }; }
    stop() {}
    // step 1: reasoning, a line of text, then exec_command; step 2 (a tool result came back): the answer
    ask(rid, body, h) {
      asks.push(body);
      setImmediate(() => {
        h({ t: "ai-genstart", rid, promptTokens: 1000, api: 2 });
        const last = body.messages.at(-1);
        if (last.role !== "tool") {
          for (const t of ["The user wants", " to see a.txt."]) h({ t: "ai-token", rid, text: t, th: 1 });
          h({ t: "ai-token", rid, text: "Reading it." });
          h({ t: "ai-call", rid, i: 0, name: "exec_command" });
          for (const a of ['{"cmd": ', '"cat a.txt"}']) h({ t: "ai-call", rid, i: 0, a });
          h({ t: "ai-call", rid, i: 0, end: 1 });
          h({ t: "ai-gendone", rid, api: 2, reason: "stop", usage: { in: 1000, out: 20, think: 2 }, reused: 0, calls: [{ name: "exec_command", args: '{"cmd": "cat a.txt"}' }] });
        } else {
          const said = /hello from a\.txt/.test(last.text) ? "a.txt says: hello from a.txt" : "I could not read it.";
          for (const t of said.match(/.{1,6}/g)) h({ t: "ai-token", rid, text: t });
          h({ t: "ai-gendone", rid, api: 2, reason: "stop", usage: { in: 1100, out: 8, think: 0 }, reused: 1000, calls: [] });
        }
      });
      return true;
    }
  }
  const logs = [];
  const api = createServer({ bridge: new Room(), port: 0, log: (m) => logs.push(m) });
  const port = await api.listen();
  try {
    const run = await runCodex({ base: `http://127.0.0.1:${port}`, prompt: "show a.txt", files: { "a.txt": "hello from a.txt\n" }, codex, timeoutMs: 120000 });
    const s = summarize(run);
    const second = asks[1];
    const checks = [
      ["codex exited 0", run.code === 0, run.stderr.slice(-400)],
      ["codex ran our call: cat a.txt", s.commands.some((c) => /cat a\.txt/.test(c.command) && /hello from a\.txt/.test(c.output)), JSON.stringify(s.commands)],
      ["codex printed the answer streamed as output_text", s.answer === "a.txt says: hello from a.txt", JSON.stringify(s)],
      ["two asks: the call, then its result", asks.length === 2, asks.length],
      ["Codex's hosted web_search tool was skipped, its function tools declared", !!asks[0]?.tools?.some((t) => t.name === "exec_command") && !asks[0].tools.some((t) => /web_search/.test(t.name)), JSON.stringify(asks[0]?.tools?.map((t) => t.name))],
      ["parallel_tool_calls false reached the room", asks[0]?.params?.parallel === false, JSON.stringify(asks[0]?.params)],
      ["Codex resent our function_call (name, args) and the reasoning with it", !!second && second.messages.some((m) => m.role === "assistant" && m.calls?.[0]?.name === "exec_command" && m.calls[0].args.cmd === "cat a.txt" && m.reasoning === "The user wants to see a.txt."), JSON.stringify(second?.messages?.slice(-3))],
      ["the tool result reached the room", second?.messages.at(-1)?.role === "tool" && /hello from a\.txt/.test(second.messages.at(-1).text), JSON.stringify(second?.messages?.at(-1))],
    ];
    return { checks, logs };
  } finally {
    api.closeAll("done"); api.server.closeAllConnections?.(); api.server.close();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
  const codex = arg("codex", "codex");
  let checks;
  if (process.argv.includes("--mock")) ({ checks } = await mock(codex));
  else {
    const base = arg("base");
    if (!base) { console.error("usage: codex.mjs --mock | --base http://127.0.0.1:PORT"); process.exit(2); }
    const run = await runCodex({ base, prompt: "Run `cat a.txt` and tell me what it says.", files: { "a.txt": "hello from a.txt\n" }, codex });
    const s = summarize(run);
    checks = [["codex exited 0", run.code === 0, run.stderr.slice(-400)], ["codex ran cat a.txt", s.commands.some((c) => /a\.txt/.test(c.command)), JSON.stringify(s.commands)],
      ["the answer quotes the file", /hello from a\.txt/i.test(s.answer || ""), JSON.stringify(s)]];
  }
  const out = checks.map(([name, ok, detail]) => ({ name, ok: !!ok, ...(ok ? {} : { detail: String(detail).slice(0, 400) }) }));
  for (const c of out) console.error(c.ok ? "ok  " : "FAIL", c.name, c.ok ? "" : c.detail);
  console.log(JSON.stringify({ passed: out.filter((c) => c.ok).length, failed: out.filter((c) => !c.ok).map((c) => c.name), checks: out }));
  process.exit(out.every((c) => c.ok) ? 0 : 1);
}
