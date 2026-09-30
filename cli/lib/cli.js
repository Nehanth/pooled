// The pooled command line's front door, with no I/O so it is unit tested: which command a word
// names, the short usage and the overview (pooled --help), "did you mean" for a mistyped command or
// option, and the one-line errors for usage mistakes. bin/pooled.js routes with it.

export const COMMANDS = ["host", "join", "chat", "serve", "pull", "list", "rm", "help"];
// words that already run a command (kept from earlier versions)
export const ALIASES = { download: "pull", ls: "list", remove: "rm", delete: "rm" };
// words people try that are not commands: what they probably meant
const MEANT = {
  models: "list", model: "list", show: "list", get: "pull", fetch: "pull", install: "pull",
  start: "host", open: "host", create: "host", new: "host", run: "host", server: "serve", api: "serve",
  connect: "join", lend: "join", enter: "join", talk: "chat", ask: "chat", prompt: "chat",
  uninstall: "rm", del: "rm", version: "--version",
};

// Damerau-Levenshtein (optimal string alignment): a swapped pair ("hsot") counts as one edit
export function editDistance(a, b) {
  a = String(a); b = String(b);
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  }
  return d[a.length][b.length];
}

// the closest of `choices` to `word`, or null when none is close (at most 1 edit for short words, 2 for longer)
export function closest(word, choices, { meant = {} } = {}) {
  const w = String(word || "").toLowerCase();
  if (!w) return null;
  if (meant[w]) return meant[w];
  let best = null, bestD = Infinity;
  for (const c of choices) {
    const d = editDistance(w, c.toLowerCase());
    if (d < bestD) { best = c; bestD = d; }
  }
  const room = Math.max(w.length, best?.length || 0) <= 4 ? 1 : 2;
  if (bestD <= room) return best;
  // a prefix of a command ("ser", "pu"): the one it starts
  const pre = choices.filter((c) => c.startsWith(w) && w.length >= 2);
  return pre.length === 1 ? pre[0] : null;
}

export const suggestCommand = (word) => closest(word, [...COMMANDS, ...Object.keys(ALIASES)], { meant: MEANT });

// `pooled hel` -> the two lines to print
export function unknownCommand(word) {
  const s = suggestCommand(word);
  const shown = String(word).slice(0, 40);
  return [`pooled: unknown command "${shown}".${s ? ` Did you mean "${s}"?` : ""}`, "Run pooled --help for all commands."];
}

// an unknown option of a command -> the two lines. known: its long option names without "--"
export function unknownOption(cmd, flag, known) {
  const bare = String(flag).replace(/^-+/, "").split("=")[0];
  const s = bare.length > 1 ? closest(bare, known) : null;
  const shown = String(flag).split("=")[0].slice(0, 40);
  const who = cmd ? `pooled ${cmd}` : "pooled";
  return [`${who}: unknown option "${shown}".${s ? ` Did you mean "--${s}"?` : ""}`, `Run ${who} --help for all options.`];
}

// node:util parseArgs error -> the lines to print (unknown option: with a suggestion; else its reason)
export function argsError(cmd, err, known) {
  const m = /Unknown option '([^']+)'/.exec(String(err?.message || ""));
  if (m) return unknownOption(cmd, m[1], known);
  const missing = /Option '(-\w, )?(--[\w-]+)[^']*' argument missing/.exec(String(err?.message || ""));
  const why = missing ? `${missing[2]} needs a value` : String(err?.message || err).replace(/^.*?: /, "").split(". ")[0];
  const who = cmd ? `pooled ${cmd}` : "pooled";
  return [`${who}: ${why}.`.replace(/\.\.$/, "."), `Run ${who} --help for all options.`];
}

// a room (code or link) that is missing: "pooled serve needs a room: ..." (one line)
export const needsRoom = (cmd) => `pooled ${cmd} needs a room: pooled ${cmd} 4TK-G9P, or paste the invite link in quotes.`;
export const notARoom = (cmd, given) => `pooled ${cmd}: "${String(given).slice(0, 60)}" is not a room code (like 4TK-G9P) or an invite link.`;

export function shortUsage(version = "") {
  return `pooled${version ? ` ${version}` : ""}: pool computers' GPUs into one room and run a model across them

Usage: pooled <command>
  host [model]   open a room on this computer
  join <room>    lend this computer's GPU to a room
  chat <room>    talk to a room's model
  serve <room>   use a room from your tools (a local OpenAI / Anthropic API)
  pull <model>   download a model (list and rm manage them)

Run pooled --help for examples, pooled <command> --help for a command's options.
`;
}

export function overview(version = "") {
  return `pooled${version ? ` ${version}` : ""}: pool computers' GPUs into one room and run a model across them

Commands                                    Example
  host [model]   open a room on this computer  pooled host qwen3-1.7b
  join <room>    lend this GPU to a room       pooled join 4TK-G9P
  chat <room>    talk to the room's model      pooled chat 4TK-G9P
  serve <room>   a local OpenAI/Anthropic API  pooled serve 4TK-G9P --port 8080
  pull <model>   download a model              pooled pull qwen3-1.7b
  list           models on this computer       pooled list
  rm <model>     delete a downloaded model     pooled rm qwen3-1.7b

<room> is a room code like 4TK-G9P, or the room's invite link in quotes.
pooled alone opens a menu. pooled <command> --help shows a command's options.
-v, --version prints the version.
`;
}

// argv (after "pooled") + whether this is a terminal -> what to do:
//   { do: "menu" } | { do: "usage" } | { do: "help", topic } | { do: "version" }
//   | { do: "run", cmd, argv, alias } | { do: "error", lines, code }
export function route(argv, { tty = false } = {}) {
  const [w, ...rest] = argv;
  if (w == null) return tty ? { do: "menu" } : { do: "usage" };
  if (w === "-h" || w === "--help") return { do: "help", topic: null };
  if (w === "-v" || w === "--version" || w === "version") return { do: "version" };
  if (w === "help") return { do: "help", topic: rest[0] ?? null };
  if (w.startsWith("-")) {
    const lines = unknownOption("", w, ["help", "version"]);
    return { do: "error", lines, code: 2 };
  }
  const cmd = ALIASES[w] || w;
  if (COMMANDS.includes(cmd)) return { do: "run", cmd, argv: rest, alias: w };
  return { do: "error", lines: unknownCommand(w), code: 2 };
}
