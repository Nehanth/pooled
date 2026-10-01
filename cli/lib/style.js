// How pooled's terminal screens look (docs: the CLI design spec, "minimal"): one accent (the site's
// blue), greys for labels and hints, the terminal's own foreground for values, a white-on-blue pill
// for the room code, thin bars, and the 3x3 dot mark. Everything that draws takes an S from style().
//
// depth: "truecolor" (COLORTERM=truecolor|24bit), "256" (any other color terminal), "none" (NO_COLOR,
// TERM=dumb, not a TTY: bold / dim / reverse only). theme: "dark" | "light" (POOLED_THEME, else the
// terminal's background from OSC 11 or COLORFGBG, else dark). ascii: no UTF-8 locale, or TERM=linux.
// Plain ANSI only (SGR, OSC 8 links): nothing here reads terminfo.

export const PALETTE = {
  //            light                  dark
  ink2:   { light: ["#4B4E58", 239], dark: ["#A3A6B0", 248] },
  ink3:   { light: ["#6B6E78", 243], dark: ["#7C7F8A", 244] },
  track:  { light: ["#E3E5EA", 254], dark: ["#2A2E3B", 236] },
  accent: { light: ["#2A45E0", 26], dark: ["#7F93FF", 105] },
  err:    { light: ["#C4302B", 160], dark: ["#F0605A", 203] },
  pillBg: { light: ["#2A45E0", 26], dark: ["#2A45E0", 26] },
  pillFg: { light: ["#FFFFFF", 231], dark: ["#FFFFFF", 231] },
};

// escape sequences: CSI (colors, cursor) and OSC (links, clipboard)
export const ANSI_RE = /\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g;
const ANSI_AT = /^(?:\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\))/;
export const stripAnsi = (s) => String(s).replace(ANSI_RE, "");
// the columns a string takes (per code point; escapes take none)
export const visible = (s) => stripAnsi(s);
export const width = (s) => [...stripAnsi(s)].length;
export const padEnd = (s, n) => s + " ".repeat(Math.max(0, n - width(s)));
export const padStart = (s, n) => " ".repeat(Math.max(0, n - width(s))) + s;

// cut to `cols` columns with "…", keeping escapes balanced: attributes and colors closed, an open
// link closed
export function clip(s, cols) {
  s = String(s);
  if (width(s) <= cols) return s;
  let out = "", n = 0, link = false, esc = false;
  const cps = s;
  for (let i = 0; i < cps.length;) {
    const m = ANSI_AT.exec(cps.slice(i));
    if (m) {
      out += m[0]; i += m[0].length; esc = true;
      if (m[0].startsWith("\x1b]8;")) link = !/^\x1b\]8;[^;]*;(?:\x07|\x1b\\)$/.test(m[0]);
      continue;
    }
    if (n >= cols - 1) { out += "…"; break; }
    const cp = String.fromCodePoint(cps.codePointAt(i));
    out += cp; n++; i += cp.length;
  }
  return out + (esc ? "\x1b[22;27;39;49m" : "") + (link ? "\x1b]8;;\x1b\\" : "");
}

// words to lines of at most w columns
export function wrap(text, w) {
  const out = [];
  let line = "";
  for (const word of String(text).split(/\s+/).filter(Boolean)) {
    if (line && width(line) + 1 + width(word) > w) { out.push(line); line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => { const A = hex(a), B = hex(b); return A.map((x, i) => Math.round(x + (B[i] - x) * t)); };

// is color on for this stream?
export function colorOn(stream = process.stdout, env = process.env) {
  if (env.NO_COLOR != null && env.NO_COLOR !== "") return false;
  if (env.TERM === "dumb" || env.FORCE_COLOR === "0" || env.CLICOLOR === "0") return false;
  if ((env.FORCE_COLOR && env.FORCE_COLOR !== "0") || (env.CLICOLOR_FORCE && env.CLICOLOR_FORCE !== "0")) return true;
  return !!stream.isTTY;
}
export const colorDepth = (env = process.env) => (/^(truecolor|24bit)$/i.test(env.COLORTERM || "") ? "truecolor" : "256");
export const asciiOnly = (env = process.env) => {
  if (env.TERM === "linux") return true;
  const loc = env.LC_ALL || env.LC_CTYPE || env.LANG || "";
  return !!loc && !/utf-?8/i.test(loc) && process.platform !== "win32" && process.platform !== "darwin";
};

// light or dark background, decided once: POOLED_THEME, the answer to OSC 11 (queried by
// detectTheme() before any raw-mode screen), COLORFGBG, else dark
let themeCache = null;
export function themeOf(env = process.env) {
  if (env.POOLED_THEME === "light" || env.POOLED_THEME === "dark") return env.POOLED_THEME;
  if (themeCache) return themeCache;
  const bg = String(env.COLORFGBG || "").split(";").pop();
  if (bg === "7" || bg === "15") return "light";
  return "dark";
}
// OSC 11 background query (at most `ms`); only on a TTY, before keys are read
export async function detectTheme({ input = process.stdin, output = process.stderr, env = process.env, ms = 100 } = {}) {
  if (themeCache || env.POOLED_THEME || !input.isTTY || !output.isTTY || env.TERM === "dumb" || typeof input.setRawMode !== "function") return themeOf(env);
  const got = await new Promise((resolve) => {
    let buf = "";
    const wasRaw = !!input.isRaw;
    const done = (v) => { clearTimeout(t); input.off("data", on); try { input.setRawMode(wasRaw); } catch {} input.pause(); resolve(v); };
    const on = (b) => {
      buf += b.toString("latin1");
      const m = /\x1b\]11;rgb:([0-9a-f]+)\/([0-9a-f]+)\/([0-9a-f]+)/i.exec(buf);
      if (m) {
        const c = [m[1], m[2], m[3]].map((x) => parseInt(x, 16) / (16 ** x.length - 1));
        const lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        done(lum > 0.5 ? "light" : "dark");
      }
    };
    const t = setTimeout(() => done(null), ms);
    try { input.setRawMode(true); } catch {}
    input.on("data", on); input.resume();
    output.write("\x1b]11;?\x1b\\");
  });
  themeCache = got || themeOf(env);
  return themeCache;
}

// S: the drawing roles for one stream
export function style({ stream = process.stderr, env = process.env, depth = null, theme = null, ascii = null } = {}) {
  depth ||= colorOn(stream, env) ? colorDepth(env) : "none";
  theme ||= themeOf(env);
  ascii = ascii ?? asciiOnly(env);
  // no escapes at all (not a TTY, TERM=dumb): the plain layout
  const plain = depth === "none" && (!stream.isTTY || env.TERM === "dumb");
  return mkStyle({ theme, depth, ascii, plain });
}

export function mkStyle({ theme = "dark", depth = "truecolor", ascii = false, plain = false } = {}) {
  const none = depth === "none";
  const sgrFg = (role) => { const [h, n] = PALETTE[role][theme]; return depth === "truecolor" ? `\x1b[38;2;${hex(h).join(";")}m` : `\x1b[38;5;${n}m`; };
  const sgrBg = (role) => { const [h, n] = PALETTE[role][theme]; return depth === "truecolor" ? `\x1b[48;2;${hex(h).join(";")}m` : `\x1b[48;5;${n}m`; };
  const wrapFg = (role) => (s) => (none ? String(s) : `${sgrFg(role)}${s}\x1b[39m`);
  const attr = (on, off) => (s) => (plain ? String(s) : `\x1b[${on}m${s}\x1b[${off}m`);
  const g = ascii
    ? { sel: ">", live: "*", bar: "=", track: "-", sep: " - ", none: "-", dash: "-", up: "up/down", lr: "left/right", spin: ["-", "\\", "|", "/"], dots: [".", "o", "O"] }
    : { sel: "›", live: "●", bar: "━", track: "━", sep: " · ", none: "—", dash: "–", up: "↑↓", lr: "←→", spin: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"], dots: ["·", "•", "●"] };
  const S = {
    theme, depth, none, plain, ascii, g,
    ink: (s) => String(s),
    ink2: wrapFg("ink2"),
    ink3: none ? attr(2, 22) : wrapFg("ink3"),
    acc: none ? attr(1, 22) : wrapFg("accent"),
    err: none ? attr(1, 22) : wrapFg("err"),
    bold: attr(1, 22),
    rev: attr(7, 27),
    // the room code: white bold on the brand blue, one space each side; reverse video without color
    pill: (code) => (plain ? `[${code}]` : none ? `\x1b[7m\x1b[1m ${code} \x1b[22m\x1b[27m` : `${sgrBg("pillBg")}${sgrFg("pillFg")}\x1b[1m ${code} \x1b[22m\x1b[39m\x1b[49m`),
    link: (url, text = url) => (plain ? text : `\x1b]8;;${url}\x1b\\${text}\x1b]8;;\x1b\\`),
    sep: null,
  };
  S.sep = S.ink3(g.sep);
  // a thin two-tone bar, `cols` wide; the edge cell blends accent into track (truecolor)
  S.bar = (frac, cols) => {
    frac = Math.max(0, Math.min(1, Number(frac) || 0));
    const exact = frac * cols, full = Math.floor(exact), rem = exact - full;
    if (none) return g.bar.repeat(full) + S.ink3((ascii ? "-" : "─").repeat(cols - full));
    let edge = "", rest = cols - full;
    if (rem > 0.2 && full < cols && depth === "truecolor") {
      const [h1] = PALETTE.accent[theme], [h2] = PALETTE.track[theme];
      edge = `\x1b[38;2;${mix(h2, h1, rem).join(";")}m${g.bar}\x1b[39m`; rest -= 1;
    }
    return `${sgrFg("accent")}${g.bar.repeat(full)}\x1b[39m${edge}${sgrFg("track")}${g.track.repeat(rest)}\x1b[39m`;
  };
  // the mark: 3x3 dots growing toward the bottom right, the last one in the accent
  S.mark = () => {
    const [a, b, c] = g.dots;
    return [S.ink3(`${a} ${a} ${b}`), S.ink3(`${a} ${b} ${c}`), S.ink3(`${b} ${c} `) + S.acc(c)];
  };
  // key hints: [key, verb, state]; "primary": the one action to take now; "off": not available yet
  // width: the columns they may take. A hint never breaks inside a word: hints that don't fit are
  // dropped (they are listed most important first), never the primary one, q last
  S.keys = (pairs, { width: max = Infinity } = {}) => {
    const one = ([k, v, st]) => (st === "primary" ? `${S.bold(S.acc(k))} ${S.ink(v)}`
      : st === "off" ? S.ink3(`${k} ${v}`) : `${S.ink2(k)} ${S.ink3(v)}`);
    let ps = pairs.slice();
    const text = () => ps.map(one).join(S.sep);
    // the first to go: the last that is neither the primary one nor q (how to get out)
    while (ps.length > 1 && width(text()) > max) {
      let i = ps.findLastIndex((p) => p[2] !== "primary" && p[0] !== "q");
      if (i < 0) i = ps.findLastIndex((p) => p[2] !== "primary");
      if (i < 0) break;
      ps.splice(i, 1);
    }
    return text();
  };
  S.spin = (i) => S.acc(g.spin[((i % g.spin.length) + g.spin.length) % g.spin.length]);
  return S;
}

// ---------------- layout helpers (the spec's grid: indent 2, labels 10 wide, values at column 12)
export const I = "  ";
export const label = (S, text, w = 10) => I + padEnd(S.ink3(text), w);
// the 3-row header beside the mark (no mark below 60 columns or without escapes)
export function header(S, cols, lines) {
  if (cols < 60 || S.plain) return lines.map((l) => I + l);
  const m = S.mark();
  return lines.map((l, i) => `${I}${m[i]}     ${l}`);
}
// a table: cols [{ h, w, align }], rows of cells; the last cell is free width
export function table(S, cols, rows, { head = true } = {}) {
  const line = (cells) => I + cells.map((c, i) => {
    const col = cols[i];
    if (i === cells.length - 1) return c;
    return col.align === "r" ? padStart(c, col.w) + "  " : padEnd(c, col.w + 2);
  }).join("").replace(/\s+$/, "");
  const out = [];
  if (head) out.push(S.ink3(line(cols.map((c) => c.h))));
  for (const r of rows) out.push(line(r));
  return out;
}
// a left part and a right part flush with column cols-2 (the right part dropped when it doesn't fit)
export const leftRight = (l, r, cols) => (width(l) + width(r) + 4 <= cols ? l + " ".repeat(cols - 2 - width(l) - width(r)) + r : l);

// file sizes: "15.0 GB", "1.7 GB", "610 MB"; memory (gbNum): "12 GB" from 10 up, "4.5 GB" below
export function gb(bytes) {
  const G = 2 ** 30;
  if (!(bytes > 0)) return "0 GB";
  if (bytes < G) return `${Math.round(bytes / 2 ** 20)} MB`;
  return `${(bytes / G).toFixed(1)} GB`;
}
export const gbNum = (x) => (x >= 10 ? `${Math.round(x)} GB` : `${Math.round(x * 10) / 10} GB`);
export const minsLeft = (s) => (s == null ? "…" : s < 60 ? `${Math.max(1, Math.round(s))} s` : `${Math.round(s / 60)} min`);
export const clock = (ms) => { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
export const upFor = (ms) => { const m = Math.floor(ms / 60000); return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`; };

// the progress row shared by pull, the host's download and join: fixed-width columns so nothing moves
//   download  ━━━━━━━━━━━━━━━━━━━━   41%   6.2 GB of 15.0 GB  48 MB/s  3 min left
export function progressRow(S, name, { done = 0, total = 0, bps = null, left = null, cols = 20 } = {}) {
  const frac = total ? done / total : 0;
  const pct = `${Math.floor(frac * 100)}%`;
  const rate = bps ? `${bps >= 1e6 ? Math.round(bps / 1e6) : (bps / 1e6).toFixed(1)} MB/s` : "…";
  if (left == null && bps && total) left = (total - done) / bps;
  return label(S, name) + S.bar(frac, cols) + "  " + padStart(pct, 4) + "  " + padStart(gb(done), 7) +
    S.ink3(" of " + padEnd(total ? gb(total) : "?", 7)) + " " + S.ink3(padStart(rate, 8) + padStart(bps ? `${minsLeft(left)} left` : "", 12));
}
