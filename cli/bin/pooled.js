#!/usr/bin/env node
// pooled: pool computers' GPUs into one room and run a model across them.
//   pooled               a menu in a terminal (lib/menu.js), a short usage otherwise
//   pooled host / join   lend this computer's GPU (lib/lendrun.js)
//   pooled chat          talk to a room (lib/chatrun.js)
//   pooled serve         a room as a local OpenAI and Anthropic API (lib/serverun.js)
//   pooled pull / list / rm   the models on this computer (lib/pullrun.js)
// Which command a word names, the help texts and "did you mean": lib/cli.js.
import { readFileSync } from "node:fs";

const major = +process.versions.node.split(".")[0];
if (major < 22) { console.error("pooled needs Node 22 or newer"); process.exit(1); }

const VERSION = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
const { route, overview, shortUsage, unknownCommand } = await import("../lib/cli.js");

// the exit status, once stdout is all out (a piped answer or list)
const finish = async (status) => {
  await new Promise((r) => process.stdout.write("", r));
  process.exit(status ?? 0);
};

export async function runCommand(cmd, argv, alias = cmd) {
  if (cmd === "chat") {
    const { chatMain } = await import("../lib/chatrun.js");
    return chatMain(argv, { version: VERSION });
  }
  if (cmd === "pull" || cmd === "list" || cmd === "rm") {
    const { pullMain } = await import("../lib/pullrun.js");
    const { loadRoomNode } = await import("../lib/lendrun.js");
    return pullMain(alias, argv, { loadRoomNode });
  }
  if (cmd === "join" || cmd === "host") {
    const { lendMain } = await import("../lib/lendrun.js");
    return lendMain(cmd, argv, { version: VERSION });
  }
  if (cmd === "serve") {
    const { serveMain } = await import("../lib/serverun.js");
    return serveMain(argv, { version: VERSION });
  }
  throw new Error(`no command ${cmd}`);
}

const r = route(process.argv.slice(2), { tty: !!process.stdin.isTTY && !!process.stdout.isTTY });
if (r.do === "version") { console.log(VERSION); await finish(0); }
if (r.do === "usage") { process.stdout.write(shortUsage(VERSION)); await finish(0); }
if (r.do === "error") { process.stderr.write(r.lines.join("\n") + "\n"); await finish(r.code); }
if (r.do === "help") {
  if (r.topic == null) { process.stdout.write(overview(VERSION)); await finish(0); }
  const t = route([r.topic]);
  if (t.do === "run" && t.cmd === "help") { process.stdout.write(overview(VERSION)); await finish(0); }
  if (t.do !== "run") { process.stderr.write(unknownCommand(r.topic).join("\n") + "\n"); await finish(2); }
  await finish(await runCommand(t.cmd, ["--help"], t.alias));
}
if (r.do === "menu") {
  const { menuMain } = await import("../lib/menu.js");
  await finish(await menuMain({ version: VERSION, run: runCommand }));
}
if (r.do === "run") {
  if (r.cmd === "help") { process.stdout.write(overview(VERSION)); await finish(0); }
  await finish(await runCommand(r.cmd, r.argv, r.alias));
}
