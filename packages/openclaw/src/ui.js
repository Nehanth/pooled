// How the plugin looks: the Pooled CLI's design (cli/lib/style.js, the CLI design spec "minimal") inside
// OpenClaw. One blue accent, dim labels, the 3x3 dot mark, the room code as a pill, thin progress bars,
// rows on one grid, plain status sentences.
//
// Two places draw it, with different rules:
//   onboarding: OpenClaw's own prompts (clack). In a terminal the blocks between the prompts are
//     written with the prompter's plain() behind clack's guide bar, in color (NO_COLOR: bold / dim /
//     reverse only). A hosted wizard (the Control UI, the macOS app) gets the same lines as plain text
//     in a note: no escapes, because its client prints them as they are.
//   the chat (/pooled, the room's notices): markdown only. OpenClaw's TUI strips escape codes and
//     renders markdown, the Control UI renders markdown, and channels convert it. The aligned rows go
//     in a code block, the one thing every surface draws in a monospace font.
import { mkStyle, style, header, label, table, I, padEnd, padStart, width, wrap, gb as fmtGB, gbNum, minsLeft, stripAnsi } from "../../../cli/lib/style.js";
import { gpuLabel } from "../../../cli/lib/hostui.js";

export { gpuLabel, stripAnsi };
// no escapes at all: chat text and hosted wizards
export const PLAIN = mkStyle({ depth: "none", plain: true, ascii: false });
export const TAGLINE = "Peer-to-peer inference engine for your claw";
export const fmtCode = (c) => (String(c || "").length === 6 ? `${c.slice(0, 3)}-${c.slice(3)}` : String(c || ""));

// ---------------- onboarding

// the prompter's look: { S, terminal, cols }. terminal: OpenClaw's clack prompter at a terminal (it has
// plain() and none of the hosted wizard's openUrl / deviceCode)
export function promptUI(p, { stream = process.stdout, env = process.env } = {}) {
  const terminal = typeof p?.plain === "function" && typeof p.openUrl !== "function" && typeof p.deviceCode !== "function" && !!stream.isTTY;
  if (!terminal) return { S: PLAIN, terminal: false, cols: 1000 };   // (no wrapping: the client wraps)
  return { S: style({ stream, env }), terminal: true, cols: Math.max(60, stream.columns || 80) };
}

// lines -> the screen. Terminal: behind clack's grey guide bar, so the block sits in the flow of the
// prompts; hosted: a note titled `title`
export async function block(p, ui, lines, title) {
  if (ui.terminal) {
    const bar = ui.S.none ? "│" : "\x1b[90m│\x1b[39m";
    await p.plain(["", ...lines].map((l) => (l ? bar + l : bar)).join("\n"));
    return;
  }
  // plain text: the grid's 2-column indent is dropped (a note has its own margin)
  const text = lines.map((l) => stripAnsi(l).replace(/^ {2}/, "").replace(/\s+$/, ""));
  if (text[0] === title) text.shift();   // the header's first line is the note's title
  await p.note(text.join("\n").replace(/^\n+|\n+$/g, ""), title);
}

// a label row whose value wraps under itself at column 12
export function para(S, name, text, cols = 80, style = (x) => x) {
  const ws = wrap(text, Math.max(30, cols - 16));
  return ws.map((w, i) => (i ? I + " ".repeat(10) : label(S, name)) + style(w));
}
// a sentence across the block, wrapped, dim
export const note = (S, text, cols = 80) => wrap(text, Math.max(30, cols - 8)).map((w) => I + S.ink3(w));

// the header beside the mark: what onboarding shows first
export function introLines(S, cols, mem) {
  return [
    ...header(S, cols, [S.bold("Pooled"), TAGLINE, S.ink3("pooled.run")]),
    "",
    label(S, "gpu") + (mem?.label || "GPU memory unknown").replace(/ · /g, S.sep),
  ];
}

// option labels in a column: the longest + 2, so the hints OpenClaw puts after them line up
export function padLabels(options) {
  const w = Math.max(...options.map((o) => width(o.label)));
  return options.map((o) => ({ ...o, label: padEnd(o.label, w + 1) }));
}

// the model list as rows: name, what it needs, the download -> select options
//   Qwen3 1.7B       needs  4 GB   downloaded
//   Qwen3.6 35B MoE  needs 23 GB   19.4 GB download
export function modelOptions(S, rows, { recommended } = {}) {
  const nameW = Math.max(...rows.map((r) => width(r.name)));
  return rows.map((r) => ({
    value: r.key,
    label: padEnd(r.name, nameW + 2) + S.ink3("needs ") + padStart(`${Math.ceil(r.needGB ?? 0)} GB`, 5) + "   " + (r.pulled ? S.ink2("downloaded") : S.ink3(`${fmtGB(r.fileBytes)} download`)),
    hint: [r.key === recommended ? "recommended" : null, r.small ? "small: slow in OpenClaw" : null,
      r.fitsAlone ? "fits on this device" : "needs another device"].filter(Boolean).join(" · "),
  }));
}

// a download in one line, for OpenClaw's spinner (the CLI's progress row without the label):
//   Qwen3 1.7B  ━━━━━━━━━━━━━━━━   41%  0.7 GB of 1.8 GB  48 MB/s  1 min left
export function progressText(S, name, st, { cols = 80 } = {}) {
  if (!st) return name;
  if (st.state === "waiting") return `${name}${S.sep}${S.ink3("waiting for another download of it to finish")}`;
  if (st.state === "checking") return `${name}${S.sep}checking the download${S.ink3(` · SHA-256 ${st.total ? Math.floor((st.done / st.total) * 100) : 0}%`)}`;
  if (st.state === "done") return `${name} downloaded${S.ink3(` · ${fmtGB(st.total)} · checked`)}`;
  if (st.state === "error") return `${name}: download failed${S.ink3(` · ${st.error}`)}`;
  const frac = st.total ? st.done / st.total : 0;
  const bar = cols >= 90 ? 20 : 12;
  const rate = st.bps ? `${st.bps >= 1e6 ? Math.round(st.bps / 1e6) : (st.bps / 1e6).toFixed(1)} MB/s` : "";
  const left = st.bps && st.total ? `${minsLeft((st.total - st.done) / st.bps)} left` : "";
  // OpenClaw's spinner paints its message in its accent and re-opens that color after every reset
  // (chalk), so each part names its own color: values in ink2, units in ink3, the bar in the accent
  const ink = (x) => (S.none ? x : S.ink2(x));
  return ink(`${name}  `) + S.bar(frac, bar) + ink(`  ${padStart(`${Math.floor(frac * 100)}%`, 4)}  ${fmtGB(st.done)}`) + S.ink3(` of ${st.total ? fmtGB(st.total) : "?"}`) +
    (rate ? `  ${S.ink3(rate)}` : "") + (left ? `  ${S.ink3(left)}` : "");
}

// ---------------- chat (markdown)

const range = (rg) => (rg ? `${rg[0]}–${rg[1] - 1}` : "—");

// the device rows in a code block, the CLI's table (DEVICE GPU LENDS HOLDS, then a state)
export function deviceBlock(rows, { waiting = [], extra = [] } = {}) {
  const S = PLAIN;
  const cut = (x, n) => (width(x) > n ? [...x].slice(0, n - 1).join("") + "…" : x);
  const nameW = Math.min(22, Math.max(8, ...rows.map((d) => width(d.name)), ...waiting.map((q) => width(q.name || ""))));
  const cols = [{ h: "DEVICE", w: nameW }, { h: "GPU", w: 14 }, { h: "LENDS", w: 5, align: "r" }, { h: "HOLDS", w: 6 }, { h: "", w: 20 }];
  const body = rows.map((d) => [cut(d.name, nameW), cut(d.gpu || "", 14), d.gb ? gbNum(d.gb) : "—", range(d.range), d.self ? "this device" : ""]);
  for (const q of waiting) body.push([cut(q.name || "a device", nameW), cut(q.gpu || "", 14), q.gb ? gbNum(q.gb) : "", "", "wants to join"]);
  const lines = table(S, cols, body).map((l) => l.replace(/^ {2}/, ""));
  if (extra.length) lines.push("", ...extra.map((l) => l.replace(/^ {2}/, "")));
  return ["```text", ...lines, "```"];   // "text": the Control UI highlights a block without a language
}
export const memoryLine = (lent, need) => label(PLAIN, "memory").replace(/^ {2}/, "") + PLAIN.bar(need ? Math.min(1, lent / need) : 0, 20) + `  ${gbNum(lent)} lent · ${gbNum(need)} needed`;
export function downloadLine(st) {
  if (!st || st.state === "done") return null;
  if (st.state !== "running") return label(PLAIN, "download").replace(/^ {2}/, "") + stripAnsi(progressText(PLAIN, "", st)).trim();
  const frac = st.total ? st.done / st.total : 0;
  const rate = st.bps ? `  ${st.bps >= 1e6 ? Math.round(st.bps / 1e6) : (st.bps / 1e6).toFixed(1)} MB/s` : "";
  const left = st.bps && st.total ? `  ${minsLeft((st.total - st.done) / st.bps)} left` : "";
  return label(PLAIN, "download").replace(/^ {2}/, "") + PLAIN.bar(frac, 20) + `  ${Math.floor(frac * 100)}%  ${fmtGB(st.done)} of ${fmtGB(st.total)}${rate}${left}`;
}

// the first line of a chat reply: **Pooled** · `4TK-G9P` · what follows
export const headLine = (code, ...parts) => [`**Pooled** · \`${fmtCode(code)}\``, ...parts.filter(Boolean)].join(" · ");

// The room's notices in the chat (stream.js): the room talking, not the model. NOTICE starts each one,
// so convert.js can drop them when the conversation is replayed (LEGACY_NOTICE: 0.2.x's)
export const NOTICE = "**Pooled** · ";
export const LEGACY_NOTICE = "⚠️ Pooled: ";
export const isNoticeText = (t) => typeof t === "string" && (t.startsWith(NOTICE) || t.startsWith(LEGACY_NOTICE));
const TITLES = {
  setup: "not set up", noroom: "can't reach the room", lobby: "waiting to be let in", denied: "turned away",
  waiting: "waiting for devices", memory: "not enough memory", degraded: "a device left", downloading: "downloading the model",
  install: "missing a piece", start: "couldn't open the room", off: "API requests are off", queue: "the room is busy",
  loading: "loading the model", older: "the host runs an older Pooled",
};
// code: the PooledError's; text: its message -> the notice's markdown
export function noticeText(code, text, room = null) {
  const body = String(text || "").replace(/^Pooled: /, "");
  const first = body.charAt(0).toUpperCase() + body.slice(1);
  return `${room ? headLine(room, TITLES[code] || "notice") : `${NOTICE}${TITLES[code] || "notice"}`}\n\n${/[.!?)]$/.test(first) ? first : `${first}.`}`;
}
