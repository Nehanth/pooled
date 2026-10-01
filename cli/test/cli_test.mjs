// The front door (cli/lib/cli.js and bin/pooled.js): a short usage without a terminal, a short
// --help, one line for a mistyped command with the nearest real one, and short errors for a
// missing room or model (no help dump anywhere).
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { route, suggestCommand, editDistance, overview, shortUsage, unknownCommand, unknownOption, argsError } from "../lib/cli.js";

const BIN = new URL("../bin/pooled.js", import.meta.url).pathname;
const models = mkdtempSync(path.join(os.tmpdir(), "pooled-cli-test-"));
process.on("exit", () => rmSync(models, { recursive: true, force: true }));
// no terminal here: stdin is /dev/null, output is piped
const pooled = (...args) => {
  const r = spawnSync(process.execPath, [BIN, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, POOLED_MODELS: models, NO_COLOR: "1" } });
  return { code: r.status, out: r.stdout, err: r.stderr, lines: (r.stdout + r.stderr).trimEnd().split("\n") };
};

test("edit distance counts a swapped pair as one edit", () => {
  assert.equal(editDistance("hsot", "host"), 1);
  assert.equal(editDistance("jion", "join"), 1);
  assert.equal(editDistance("hel", "help"), 1);
  assert.equal(editDistance("serve", "serve"), 0);
  assert.equal(editDistance("", "abc"), 3);
});

test("did you mean: typos, near words and synonyms", () => {
  const want = { hel: "help", hsot: "host", jion: "join", pul: "pull", pulll: "pull", chta: "chat", srve: "serve", sevre: "serve",
    models: "list", model: "list", lst: "list", start: "host", connect: "join", api: "serve", talk: "chat", remvoe: "remove" };
  for (const [w, s] of Object.entries(want)) assert.equal(suggestCommand(w), s, w);
  assert.equal(suggestCommand("xyzzy"), null);
  assert.equal(suggestCommand("frobnicate"), null);
});

test("route: no command is the menu in a terminal and the short usage otherwise; aliases still run", () => {
  assert.deepEqual(route([], { tty: true }), { do: "menu" });
  assert.deepEqual(route([], { tty: false }), { do: "usage" });
  assert.deepEqual(route(["-h"]), { do: "help", topic: null });
  assert.deepEqual(route(["help", "host"]), { do: "help", topic: "host" });
  assert.deepEqual(route(["--version"]), { do: "version" });
  assert.deepEqual(route(["download", "35b"]), { do: "run", cmd: "pull", argv: ["35b"], alias: "download" });
  assert.deepEqual(route(["ls"]), { do: "run", cmd: "list", argv: [], alias: "ls" });
  assert.deepEqual(route(["serve", "4TKG9P", "--port", "9"]), { do: "run", cmd: "serve", argv: ["4TKG9P", "--port", "9"], alias: "serve" });
  assert.equal(route(["hel"]).do, "error");
  assert.equal(route(["--frob"]).do, "error");
});

test("the texts: usage <= 12 lines, --help <= 20 lines, all within 80 columns", () => {
  const u = shortUsage("0.3.1").trimEnd().split("\n"), h = overview("0.3.1").trimEnd().split("\n");
  assert.ok(u.length <= 12, `usage has ${u.length} lines`);
  assert.ok(h.length <= 20, `--help has ${h.length} lines`);
  for (const l of [...u, ...h]) assert.ok(l.length <= 80, `too wide: ${l}`);
  // every command in the overview, each with an example
  for (const c of ["host", "join", "chat", "serve", "pull", "list", "rm"]) assert.match(overview(), new RegExp(`pooled ${c}\\b`), c);
  assert.deepEqual(unknownCommand("hel"), ['pooled: unknown command "hel". Did you mean "help"?', "Run pooled --help for all commands."]);
  assert.deepEqual(unknownCommand("zzz"), ['pooled: unknown command "zzz".', "Run pooled --help for all commands."]);
  assert.deepEqual(unknownOption("serve", "--prot", ["port", "token"]), ['pooled serve: unknown option "--prot". Did you mean "--port"?', "Run pooled serve --help for all options."]);
  assert.deepEqual(argsError("serve", Object.assign(new Error("Option '--port <value>' argument missing"), { code: "ERR_PARSE_ARGS_INVALID_OPTION_VALUE" }), []),
    ["pooled serve: --port needs a value.", "Run pooled serve --help for all options."]);
});

test("pooled without a terminal: the short usage, exit 0", () => {
  const r = pooled();
  assert.equal(r.code, 0);
  assert.ok(r.lines.length <= 12, r.lines.join("\n"));
  assert.match(r.out, /Usage: pooled <command>/);
  assert.equal(r.err, "");
});

test("pooled -h / --help / help: the overview, <= 20 lines; pooled help <cmd> is that command's help", () => {
  for (const a of [["-h"], ["--help"], ["help"]]) {
    const r = pooled(...a);
    assert.equal(r.code, 0, a[0]);
    assert.ok(r.lines.length <= 20, `${a[0]}: ${r.lines.length} lines`);
    assert.match(r.out, /pooled host qwen3-1\.7b/);
  }
  const s = pooled("help", "serve");
  assert.equal(s.code, 0); assert.match(s.out, /--token-file/);
  assert.match(pooled("serve", "--help").out, /--max-queue/);
  assert.match(pooled("host", "--help").out, /--gb <n\|max>/);
});

test("an unknown command: one line with the nearest command, then where to look; nothing else", () => {
  for (const [w, s] of [["hel", "help"], ["hsot", "host"], ["jion", "join"], ["pul", "pull"], ["models", "list"]]) {
    const r = pooled(w);
    assert.equal(r.code, 2, w);
    assert.equal(r.out, "", w);
    assert.deepEqual(r.lines, [`pooled: unknown command "${w}". Did you mean "${s}"?`, "Run pooled --help for all commands."], w);
  }
  // download and ls were aliases before: they still run pull and list
  assert.match(pooled("download", "--help").out, /pooled pull <model>/);
  assert.match(pooled("ls", "--help").out, /pooled list/);
});

test("unknown options: the same two lines, on the command's side", () => {
  assert.deepEqual(pooled("host", "--gbb", "4").lines, ['pooled host: unknown option "--gbb". Did you mean "--gb"?', "Run pooled host --help for all options."]);
  assert.deepEqual(pooled("serve", "4TKG9P", "--prot", "9").lines, ['pooled serve: unknown option "--prot". Did you mean "--port"?', "Run pooled serve --help for all options."]);
  assert.deepEqual(pooled("chat", "4TKG9P", "--thinkk").lines, ['pooled chat: unknown option "--thinkk". Did you mean "--think"?', "Run pooled chat --help for all options."]);
  assert.deepEqual(pooled("pull", "--model", "x").lines, ['pooled pull: unknown option "--model". Did you mean "--models"?', "Run pooled pull --help for all options."]);
  assert.deepEqual(pooled("--frob").lines, ['pooled: unknown option "--frob".', "Run pooled --help for all options."]);
});

test("missing arguments: short errors that say what to type next", () => {
  const one = (args, line) => { const r = pooled(...args); assert.equal(r.code, 2, args.join(" ")); assert.deepEqual(r.lines, [line], args.join(" ")); };
  one(["serve"], "pooled serve needs a room: pooled serve 4TK-G9P, or paste the invite link in quotes.");
  one(["chat"], "pooled chat needs a room: pooled chat 4TK-G9P, or paste the invite link in quotes.");
  one(["join"], "pooled join needs a room: pooled join 4TK-G9P, or paste the invite link in quotes.");
  one(["serve", "XYZ"], 'pooled serve: "XYZ" is not a room code (like 4TK-G9P) or an invite link.');
  one(["chat", "nope!"], 'pooled chat: "nope!" is not a room code (like 4TK-G9P) or an invite link.');
  one(["pull"], "pooled pull needs a model: pooled pull qwen3-1.7b (pooled list shows them all).");
  one(["rm"], "pooled rm needs a model: pooled rm qwen3-1.7b (pooled list shows them all).");
  for (const a of [["serve"], ["join"], ["chat"], ["pull"], ["host", "--gbb"]]) assert.ok(pooled(...a).lines.length <= 2, a.join(" "));
});
