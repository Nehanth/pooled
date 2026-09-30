// pooled pull / pooled list / pooled rm: the models this computer keeps (cli/lib/cache.js).
import { parseArgs } from "node:util";
import os from "node:os";
import { modelsDir, pullModel, listModels, removeModel, resolveModel, modelState, fmtBytes, rateMeter, progressLine, progressPlain } from "./cache.js";
import { colors } from "./hostui.js";
import { colorOn } from "./tui.js";

export const HELP_PULL = `Usage
  pooled pull <model>      download a model, so pooled host and pooled join start from disk
  pooled list              the models on this computer, and the ones not downloaded yet
  pooled rm <model>        delete a downloaded model

  The files go to ~/.pooled/models/<model> (--models <dir> or POOLED_MODELS picks another place).
  A download resumes where it stopped: Ctrl-C, then pooled pull again. Each file is checked
  against its size and SHA-256 before it is used. <model> may be part of a name ("35b", "1.7b").

Options
  --models <dir>   where the models are kept
  -h, --help       this help
`;

const home = (p) => (p.startsWith(os.homedir()) ? "~" + p.slice(os.homedir().length) : p);
const ALIASES = { download: "pull", ls: "list", remove: "rm", delete: "rm" };

// Pull one model with a progress bar on a terminal (plain lines every few seconds otherwise).
// -> { ok, bytes } | { ok: false, aborted | error }. rn: the room node module (MODELS, FILES)
export async function pullWithProgress(rn, key, dir, { stream = process.stderr, signal, quiet = false, onProgress = null } = {}) {
  const tty = !!stream.isTTY;
  const c = colors(colorOn(stream));
  const rate = rateMeter();
  let lastPlain = 0, lastDraw = 0, last = { done: 0, total: 0 };
  const draw = (p) => {
    last = p;
    const bps = rate(p.done);
    onProgress?.({ ...p, bps });
    if (quiet) return;
    // at most ~10 redraws a second (a chunk arrives every few KB)
    const now = Date.now();
    if (tty && now - lastDraw < 100 && p.done !== p.total) return;
    lastDraw = now;
    if (tty) stream.write("\r\x1b[K" + progressLine({ done: p.done, total: p.total, bps, width: (stream.columns || 80) - 1 }));
    else if (Date.now() - lastPlain > 5000 || p.done === p.total) { lastPlain = Date.now(); stream.write(`${new Date().toTimeString().slice(0, 8)} ${key}: ${progressPlain({ done: p.done, total: p.total, bps })}\n`); }
  };
  let verifying = false;
  try {
    const r = await pullModel(key, { dir, MODELS: rn.MODELS, FILES: rn.FILES, signal, onProgress: draw,
      onVerify: (d, t) => {
        if (quiet) return;
        if (!verifying) { verifying = true; if (tty) stream.write("\r\x1b[K"); else stream.write(`checking the SHA-256 of ${key}\n`); }
        if (tty) stream.write(`\r\x1b[K${c.dim(`checking SHA-256 ${Math.floor((d / t) * 100)}%`)}`);
      } });
    if (tty && !quiet) stream.write("\r\x1b[K");
    return { ok: true, bytes: r.bytes, skipped: r.skipped };
  } catch (e) {
    if (tty && !quiet) stream.write("\r\x1b[K");
    if (e.type === "aborted") return { ok: false, aborted: true, done: last.done, total: last.total };
    return { ok: false, error: e };
  }
}

export async function pullMain(cmd0, argv, { loadRoomNode }) {
  const cmd = ALIASES[cmd0] || cmd0;
  let r;
  try { r = parseArgs({ args: argv, options: { models: { type: "string" }, help: { type: "boolean", short: "h" } }, allowPositionals: true, strict: true }); }
  catch (e) { process.stderr.write(`pooled ${cmd0}: ${e.message.replace(/^.*?: /, "")}\n\n${HELP_PULL}`); return 2; }
  if (r.values.help) { process.stdout.write(HELP_PULL); return 0; }
  const rn = await loadRoomNode();
  const dir = modelsDir({ flag: r.values.models });
  const keys = Object.keys(rn.MODELS).filter((k) => rn.MODELS[k].gguf && (rn.MODELS[k].kind === "gguf" || rn.MODELS[k].kind === "qwen35"));
  const c = colors(colorOn(process.stdout));
  const say = (s) => process.stdout.write(s + "\n");
  const err = (s) => process.stderr.write(s + "\n");
  const pick = () => {
    if (r.positionals.length !== 1) { err(`pooled ${cmd0}: give one model: ${keys.join(", ")}`); return null; }
    const m = resolveModel(r.positionals[0], keys);
    if (m.error) { err(`pooled ${cmd0}: ${m.error}; one of: ${m.choices.join(", ")}`); return null; }
    return m.key;
  };

  if (cmd === "list") {
    if (r.positionals.length) { err(`pooled ${cmd0} takes no model`); return 2; }
    const rows = listModels(dir, rn.MODELS, rn.FILES, keys, rn.LOCAL).sort((a, b) => (a.fileBytes || 0) - (b.fileBytes || 0));
    const have = rows.filter((x) => x.pulled), not = rows.filter((x) => !x.pulled);
    if (have.length) {
      say(c.dim(`${"MODEL".padEnd(18)} ${"NAME".padEnd(24)} SIZE`));
      for (const x of have) say(`${x.key.padEnd(18)} ${x.label.padEnd(24)} ${fmtBytes(x.bytes)}`);
    } else say(`No models downloaded yet in ${home(dir)}.`);
    if (not.length) {
      say("");
      say(c.dim("Not downloaded (pooled pull <model>):"));
      for (const x of not) say(`${c.dim("  ")}${x.key.padEnd(16)} ${x.label.padEnd(24)} ${c.dim(fmtBytes(x.fileBytes))}${x.partBytes ? c.yellow(`  ${fmtBytes(x.partBytes)} so far: pooled pull resumes it`) : ""}`);
    }
    say(c.dim(`\n${home(dir)}`));
    return 0;
  }
  if (cmd === "rm") {
    const key = pick(); if (!key) return 2;
    const freed = removeModel(dir, key);
    if (freed == null) { err(`pooled rm: ${key} is not downloaded (in ${home(dir)})`); return 1; }
    say(`removed ${key} (${fmtBytes(freed)})`);
    return 0;
  }
  // pull
  const key = pick(); if (!key) return 2;
  const st = modelState(dir, key, rn.MODELS, rn.FILES, rn.LOCAL);
  const total = rn.FILES?.[key]?.bytes;
  if (st.pulled) { say(`${key} is already downloaded (${fmtBytes(st.bytes)} in ${home(dir)}/${key}): pooled host ${key}`); return 0; }
  say(`pulling ${c.bold(key)} (${rn.MODELS[key].label}): ${fmtBytes(total)}${st.partBytes ? `, ${fmtBytes(st.partBytes)} already here` : ""} into ${home(dir)}/${key}`);
  const ac = new AbortController();
  const onInt = () => ac.abort();
  process.on("SIGINT", onInt);
  const res = await pullWithProgress(rn, key, dir, { signal: ac.signal, stream: process.stderr });
  process.off("SIGINT", onInt);
  if (res.aborted) {
    err(`paused${res.total ? ` at ${Math.floor((res.done / res.total) * 100)}% (${fmtBytes(res.done)} of ${fmtBytes(res.total)})` : ""}: pooled pull ${key} picks it up from there`);
    return 130;
  }
  if (!res.ok) {
    err(`pooled pull: ${res.error.message}`);
    if (res.error.type === "network" || res.error.type === "short") err(`  pooled pull ${key} again resumes it`);
    return 1;
  }
  say(`${c.green("✓")} ${key} is ready (${fmtBytes(res.bytes)}): ${c.bold(`pooled host ${key}`)}`);
  return 0;
}
