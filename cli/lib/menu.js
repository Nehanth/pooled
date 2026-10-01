// pooled with no command, in a terminal: a short menu of what pooled does. Choosing one runs that
// command's own interactive flow (host and join ask for what they need; chat and serve ask for the
// room here first). run(cmd, argv) is bin/pooled.js runCommand; streams and env are injected so the
// menu is tested without a terminal (test/menu_test.mjs).
import { choose, askLine, termCaps } from "./tui.js";
import { style, header, I } from "./style.js";

export const MENU = [
  { key: "host", label: "Host a room", hint: "pick a model and invite other computers" },
  { key: "join", label: "Join a room", hint: "lend this computer to someone's room" },
  { key: "chat", label: "Chat with a room", hint: "talk to a room's model from here" },
  { key: "serve", label: "Serve a room", hint: "a local OpenAI and Anthropic API" },
  { key: "models", label: "Models", hint: "download, list or delete models" },
];

// a room code or invite link from the terminal -> the text as typed (a link keeps its #k= key) | null
export async function askRoom({ input, output, roomCodeFrom, terminal }) {
  for (;;) {
    const a = await askLine("Room code or invite link: ", { input, output, terminal });
    if (a == null) return null;
    const t = a.trim().replace(/^["']|["']$/g, "");
    if (!t) continue;
    if (roomCodeFrom(t)) return t;
    output.write(`  "${t.slice(0, 60)}" is not a room code (like 4TK-G9P) or an invite link\n`);
  }
}

export async function menuMain({ version = "", run, input = process.stdin, output = process.stderr, env = process.env,
  roomCodeFrom = null, models = null } = {}) {
  const caps = termCaps({ input, output, env });
  const S = style({ stream: output, env, depth: caps.color ? null : "none" });
  const cols = output.columns || 80;
  const top = ["", ...header(S, cols, [S.bold("pooled") + " " + S.ink3(version), "Run one AI model across several computers.", S.ink3(S.link("https://pooled.run", "pooled.run"))]), ""];
  const i = await choose({ title: "What do you want to do?", items: MENU, input, output, env, top, extra: { h: "help" },
    keys: [[S.g.up, "choose"], ["enter", "go"], ["h", "all commands"], ["q", "quit"]],
    tail: [I + S.ink3("Next time, skip this: ") + S.ink2("pooled host") + S.ink3(" or ") + S.ink2("pooled join <code>")] });
  if (i === "help") { output.write((await import("./cli.js")).overview(version)); return 0; }
  if (i == null) return 0;
  const key = MENU[i].key;
  if (key === "host" || key === "join") return run(key, []);
  if (key === "chat" || key === "serve") {
    roomCodeFrom ||= (await import("./room.js")).roomCodeFrom;
    const room = await askRoom({ input, output, roomCodeFrom, terminal: termCaps({ input, output, env }).ansi });
    if (room == null) return 130;
    output.write(`${I}${S.ink3("next time: ")}${S.ink2(`pooled ${key} "${room}"`)}\n`);
    return run(key, [room]);
  }
  // models: what is here, then what to do with them
  await run("list", []);
  models ||= await modelsHere();
  const j = await choose({ title: "Models", items: [
    { label: "Download a model", hint: "pooled pull <model>" },
    { label: "Delete a downloaded model", hint: "pooled rm <model>" },
    { label: "Done", hint: "" },
  ], def: models.pulled.length ? 2 : 0, input, output, env });
  if (j == null || j === 2) return 0;
  const pool = j === 0 ? models.notPulled : models.pulled;
  if (!pool.length) { output.write(j === 0 ? "Every model is downloaded already.\n" : "No models are downloaded yet.\n"); return 0; }
  const k = await choose({ title: j === 0 ? "Download which model?" : "Delete which model?", items: pool.map((m) => ({ label: m.key, hint: m.hint })), input, output, env });
  if (k == null) return 0;
  return run(j === 0 ? "pull" : "rm", [pool[k].key]);
}

// the models on this computer and the ones not downloaded, for the models menu
async function modelsHere() {
  const { loadRoomNode } = await import("./lendrun.js");
  const { listModels, modelsDir, fmtBytes } = await import("./cache.js");
  const rn = await loadRoomNode();
  const keys = Object.keys(rn.MODELS).filter((k) => rn.MODELS[k].gguf && (rn.MODELS[k].kind === "gguf" || rn.MODELS[k].kind === "qwen35"));
  const rows = listModels(modelsDir(), rn.MODELS, rn.FILES, keys, rn.LOCAL).sort((a, b) => (a.fileBytes || 0) - (b.fileBytes || 0));
  const row = (x, bytes) => ({ key: x.key, hint: `${x.label} · ${fmtBytes(bytes)}` });
  return { pulled: rows.filter((x) => x.pulled).map((x) => row(x, x.bytes)), notPulled: rows.filter((x) => !x.pulled).map((x) => row(x, x.fileBytes)) };
}
