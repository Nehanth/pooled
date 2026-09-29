// Load named top-level functions out of room.js without a DOM, for unit tests.
//
// room.js is one DOM-bound module (it reads location.search and touches the page at import), so it
// cannot be imported under Deno. The checkpoint and pending-control logic in it is plain code over
// the `ai` state object, though, so the tests cut those functions out of the source text and run
// them against a stub `ai` and whatever globals they name. The tests therefore exercise the exact
// code the room ships, not a copy of it.
const SRC = await Deno.readTextFile(new URL("../../room.js", import.meta.url));

// The source text of `function name(...) { ... }` (or `async function`) at the start of a line.
// Brace matching skips strings, template literals and comments; the functions pulled out here have
// no regex literals with braces in them.
export function fnSource(name) {
  const re = new RegExp(`^(async )?function ${name}\\(`, "m");
  const m = re.exec(SRC);
  if (!m) throw new Error(`room.js has no top-level function ${name}`);
  // skip the parameter list's default values: the body's brace is the first one after the ")" that
  // closes the parameters
  let depth = 0, j = m.index + m[0].length - 1;
  for (; j < SRC.length; j++) {
    const c = SRC[j];
    if (c === "(") depth++;
    else if (c === ")" && --depth === 0) break;
  }
  const i = SRC.indexOf("{", j);
  depth = 0;
  for (let k = i; k < SRC.length; k++) {
    const c = SRC[k];
    if (c === '"' || c === "'" || c === "`") { k = skipString(k, c); continue; }
    if (c === "/" && SRC[k + 1] === "/") { k = SRC.indexOf("\n", k); continue; }
    if (c === "/" && SRC[k + 1] === "*") { k = SRC.indexOf("*/", k) + 1; continue; }
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return SRC.slice(m.index, k + 1);
  }
  throw new Error(`unbalanced braces in room.js function ${name}`);
}

function skipString(k, q) {
  for (let i = k + 1; i < SRC.length; i++) {
    if (SRC[i] === "\\") { i++; continue; }
    if (q === "`" && SRC[i] === "$" && SRC[i + 1] === "{") {
      // a template substitution: skip to its closing brace (nested strings inside are skipped too)
      let depth = 0;
      for (let j = i + 1; j < SRC.length; j++) {
        const c = SRC[j];
        if (c === '"' || c === "'" || c === "`") { j = skipString(j, c); continue; }
        if (c === "{") depth++;
        else if (c === "}" && --depth === 0) { i = j; break; }
      }
      continue;
    }
    if (SRC[i] === q) return i;
  }
  throw new Error("unterminated string in room.js");
}

// Build the named functions over `globals` (an object of name -> value: the `ai` object, constants,
// stubs). Returns { name: fn }. The functions see each other, so resetState can call a stubbed or
// a real ckptClear alike.
export function roomFns(names, globals) {
  const keys = Object.keys(globals);
  const body = names.map(fnSource).join("\n") + `\nreturn { ${names.join(", ")} };`;
  return new Function(...keys, body)(...keys.map((k) => globals[k]));
}
