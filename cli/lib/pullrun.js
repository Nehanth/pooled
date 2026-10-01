// pooled pull / pooled list / pooled rm: the models this computer keeps (cli/lib/cache.js).
import { parseArgs } from "node:util";
import os from "node:os";
import { modelsDir, ensureModelsDir, pullModel, listModels, removeModel, resolveModel, modelState, fmtBytes, rateMeter, progressLine, progressPlain } from "./cache.js";
import { colors } from "./hostui.js";
import { style, label, progressRow, padStart, clip, gb, I } from "./style.js";
import { liveRegion } from "./tui.js";
import { colorOn } from "./tui.js";
import { argsError } from "./cli.js";

export const HELP_PULL = `Usage
  pooled pull <model>      download a model, so pooled host and pooled join start from disk
  pooled list              the models on this computer, and the ones not downloaded yet
  pooled rm <model>        delete a downloaded model

  Models live in ~/.pooled/models (made the first time pooled needs it); pooled host and pooled
  join use the same folder, and pooled join downloads the room's model there by itself.
  A download resumes where it stopped: Ctrl-C, then pooled pull again. Each file is checked
  against its size and SHA-256 before it is used. <model> may be part of a name ("35b", "1.7b").

Options
  -h, --help       this help

Advanced
  --models <dir>   keep models somewhere else (also POOLED_MODELS=<dir>); give the same to
                   pooled host and pooled join (they take --models and POOLED_MODELS too)
`;

const home = (p) => (p.startsWith(os.homedir()) ? "~" + p.slice(os.homedir().length) : p);
const ALIASES = { download: "pull", ls: "list", remove: "rm", delete: "rm" };

// Pull one model with the progress rows on a terminal (plain lines every few seconds otherwise):
//     Qwen3.8 27B · Q4 · 15.0 GB
//
//     download  ━━━━━━━━━━━━━━━━━━━━   54%   8.1 GB of 15.0 GB  69 MB/s  2 min left
//     check     waits for the download
//
//     ctrl-c stop (resumes next time)
// -> { ok, bytes, ms } | { ok: false, aborted | error }. rn: the room node module (MODELS, FILES)
export async function pullWithProgress(rn, key, dir, { stream = process.stderr, signal, quiet = false, onProgress = null, title = true } = {}) {
  const tty = !!stream.isTTY && process.env.TERM !== "dumb";
  const S = style({ stream });
  const rate = rateMeter();
  const t0 = Date.now();
  let lastPlain = 0, lastDraw = 0, last = { done: 0, total: rn.FILES?.[key]?.bytes || 0 }, check = null, bps = null;
  const region = tty && !quiet ? liveRegion(stream) : null;
  const [name, ...rest] = String(rn.MODELS[key]?.label || key).split("·").map((x) => x.trim());
  const lines = () => {
    const cols = stream.columns || 80;
    const L = [];
    if (title) L.push("", I + S.bold(name) + S.ink3(` · ${[...rest, gb(last.total)].filter(Boolean).join(" · ")}`), "");
    // rate and time left only after 2 s (the first chunks say little)
    L.push(progressRow(S, "download", { done: last.done, total: last.total, bps: Date.now() - t0 > 2000 ? bps : null }));
    L.push(check == null ? label(S, "check") + S.ink3("waits for the download") : label(S, "check") + S.bar(check, 20) + "  " + padStart(`${Math.floor(check * 100)}%`, 4) + "  " + S.ink3("SHA-256"));
    L.push("", I + S.keys([["ctrl-c", "stop (resumes next time)"]]));
    return L.map((l) => clip(l, cols - 1));
  };
  const draw = (p) => {
    last = p;
    bps = rate(p.done);
    onProgress?.({ ...p, bps });
    if (quiet) return;
    // at most ~10 redraws a second (a chunk arrives every few KB)
    const now = Date.now();
    if (tty && now - lastDraw < 100 && p.done !== p.total) return;
    lastDraw = now;
    if (region) region.render(lines());
    else if (Date.now() - lastPlain > 5000 || p.done === p.total) { lastPlain = Date.now(); stream.write(`${new Date().toTimeString().slice(0, 8)} ${key}: ${progressPlain({ done: p.done, total: p.total, bps })}\n`); }
  };
  if (region) region.render(lines());
  try {
    const r = await pullModel(key, { dir, MODELS: rn.MODELS, FILES: rn.FILES, signal, onProgress: draw,
      onVerify: (d, t) => {
        if (quiet) return;
        if (check == null && !region) stream.write(`checking the SHA-256 of ${key}\n`);
        check = t ? d / t : 0;
        if (region) region.render(lines());
      } });
    region?.clear();
    return { ok: true, bytes: r.bytes, skipped: r.skipped, ms: Date.now() - t0, sha: check != null };
  } catch (e) {
    region?.clear();
    if (e.type === "aborted") return { ok: false, aborted: true, done: last.done, total: last.total };
    return { ok: false, error: e };
  }
}

const took = (ms) => { const s = Math.round(ms / 1000); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`; };

export async function pullMain(cmd0, argv, { loadRoomNode }) {
  const cmd = ALIASES[cmd0] || cmd0;
  let r;
  try { r = parseArgs({ args: argv, options: { models: { type: "string" }, help: { type: "boolean", short: "h" } }, allowPositionals: true, strict: true }); }
  catch (e) { process.stderr.write(argsError(cmd0, e, ["models", "help"]).join("\n") + "\n"); return 2; }
  if (r.values.help) { process.stdout.write(HELP_PULL); return 0; }
  const rn = await loadRoomNode();
  const dir = modelsDir({ flag: r.values.models });
  try { ensureModelsDir(dir); } catch (e) { process.stderr.write(`pooled ${cmd0}: can't make ${home(dir)}: ${e.message}\n`); return 1; }
  const keys = Object.keys(rn.MODELS).filter((k) => rn.MODELS[k].gguf && (rn.MODELS[k].kind === "gguf" || rn.MODELS[k].kind === "qwen35"));
  const c = colors(colorOn(process.stdout));
  const say = (s) => process.stdout.write(s + "\n");
  const err = (s) => process.stderr.write(s + "\n");
  const pick = () => {
    if (!r.positionals.length) { err(`pooled ${cmd0} needs a model: pooled ${cmd0} ${keys.includes("qwen3-1.7b") ? "qwen3-1.7b" : keys[0]} (pooled list shows them all).`); return null; }
    if (r.positionals.length > 1) { err(`pooled ${cmd0} takes one model, not ${r.positionals.length}: ${r.positionals.slice(0, 3).join(" ")}`); return null; }
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
      for (const x of not) say(`${c.dim("  ")}${x.key.padEnd(16)} ${x.label.padEnd(24)} ${c.dim(fmtBytes(x.fileBytes))}${x.partBytes ? c.dim(`  ${fmtBytes(x.partBytes)} so far: pooled pull resumes it`) : ""}`);
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
  const tty = !!process.stderr.isTTY && process.env.TERM !== "dumb";
  if (!tty) say(`pulling ${key} (${rn.MODELS[key].label}): ${fmtBytes(total)}${st.partBytes ? `, ${fmtBytes(st.partBytes)} already here` : ""} into ${home(dir)}/${key}`);
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
  if (tty) {
    const S = style({ stream: process.stderr });
    const [name, ...rest] = String(rn.MODELS[key].label).split("·").map((x) => x.trim());
    const size = gb(total || res.bytes), cols = (process.stderr.columns || 80) - 1;
    process.stderr.write(["", I + S.bold(name) + S.ink3(` · ${[...rest, size].join(" · ")}`), "",
      label(S, "download") + S.bar(1, 20) + "  " + padStart("100%", 4) + "  " + S.ink3(`${size} in ${took(res.ms)}`),
      label(S, "check") + S.bar(1, 20) + "  " + padStart("100%", 4) + "  " + S.ink3("SHA-256 matches"),
      "", I + S.acc(S.g.live) + " Downloaded and checked" + S.ink3(` in ${took(res.ms)} · ${home(dir)}`),
      label(S, "next") + S.bold(`pooled host ${key}`), ""].map((l) => clip(l, cols)).join("\n") + "\n");
  } else say(`${key} is ready (${fmtBytes(res.bytes)}): pooled host ${key}`);
  return 0;
}
