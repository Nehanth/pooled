// Barrier lint for the engine's WGSL (engine/wgsl/*.js): finds workgroup barriers that some threads of
// a workgroup can skip, the code Windows' FXC compiler (Dawn's D3D12 backend without DXC) rejects with
// X3663 "thread sync operation found in varying flow control" while Metal, Vulkan and DXC take it.
//
// It reads the WGSL straight out of the JS sources (every template literal that holds a `fn`, with
// each `${...}` replaced by the text of the templates inside it, so every variant's code is seen) and
// flags, per function:
//   - a barrier under an `if` / `switch` whose condition depends on the thread (or on data it read)
//   - a loop that holds a barrier (directly or through a call) and whose trip count depends on the
//     thread: its condition, or a break / return taken under a thread-dependent condition
//   - a `continue` under a thread-dependent condition ahead of a barrier in the same loop
//   - a `return` under a thread-dependent condition ahead of a barrier in the function
//   - a loop with a run-time, thread-dependent trip count nested in a loop that calls a helper holding
//     a barrier: FXC rejected topk_b for this (for e = t; e < m inside the round loop around tk_reduce)
//     even with its `continue` gone, though it takes the same shape with the barriers written inline
//     (moe_router, attn_flash). Give the inner loop a trip count every thread shares and guard its body.
// "Depends on the thread": local_invocation_id/index, global_invocation_id, anything read from a
// storage or workgroup variable (except through workgroupUniformLoad), a helper's parameters, and every
// value computed from those or assigned under such a condition. workgroup_id, uniforms, constants and
// literals are uniform. It errs on the side of flagging.
//
// node scripts/wgsl-barrier-lint.mjs [files...]   (default engine/wgsl/*.js) -> one line per finding, exit 1 if any

const BARRIERS = new Set(["workgroupBarrier", "storageBarrier", "textureBarrier", "workgroupUniformLoad"]);
const VARYING_BUILTINS = new Set(["local_invocation_id", "local_invocation_index", "global_invocation_id", "subgroup_invocation_id"]);

// ---- JS source -> WGSL texts ----
// Each top-level template literal, with `${expr}` replaced by the templates inside expr (joined) or by
// the placeholder identifier _X_ when expr holds none (a constant such as ${WG}).
export function templatesOf(js) {
  const out = [];
  let i = 0;
  const n = js.length;
  function skipString(q) { i++; while (i < n && js[i] !== q) { if (js[i] === "\\") i++; i++; } i++; }
  function readTemplate() {   // at the opening backtick; returns the inlined text
    i++;
    let s = "";
    while (i < n && js[i] !== "`") {
      if (js[i] === "\\") { s += js[i + 1]; i += 2; continue; }
      if (js[i] === "$" && js[i + 1] === "{") { i += 2; const inner = readExpr(); s += inner.length ? " " + inner.join(" ") + " " : "_X_"; continue; }   // no spaces: ${P}_x stays one name
      s += js[i++];
    }
    i++;
    return s;
  }
  function readExpr() {   // after `${`, up to the matching `}`; returns the templates inside it
    const found = [];
    let depth = 0;
    while (i < n) {
      const c = js[i];
      if (c === "`") { found.push(readTemplate()); continue; }
      if (c === "'" || c === '"') { skipString(c); continue; }
      if (c === "/" && js[i + 1] === "/") { while (i < n && js[i] !== "\n") i++; continue; }
      if (c === "/" && js[i + 1] === "*") { i = js.indexOf("*/", i + 2) + 2; continue; }
      if (c === "{") depth++;
      if (c === "}") { if (depth === 0) { i++; return found; } depth--; }
      i++;
    }
    return found;
  }
  while (i < n) {
    const c = js[i];
    if (c === "`") { out.push(readTemplate()); continue; }
    if (c === "'" || c === '"') { skipString(c); continue; }
    if (c === "/" && js[i + 1] === "/") { while (i < n && js[i] !== "\n") i++; continue; }
    if (c === "/" && js[i + 1] === "*") { i = js.indexOf("*/", i + 2) + 2; continue; }
    i++;
  }
  return out.filter((t) => /\bfn\s+\w+\s*\(/.test(t));
}

// ---- WGSL tokens ----
function tokenize(src) {
  const toks = [];
  const re = /\/\/[^\n]*|\/\*[\s\S]*?\*\/|(\n)|([A-Za-z_][A-Za-z0-9_]*)|(0[xX][0-9a-fA-F]+[iuf]?|\d+\.?\d*(?:[eE][+-]?\d+)?[iufh]?)|(>>=|<<=|&&|\|\||==|!=|<=|>=|\+=|-=|\*=|\/=|%=|&=|\|=|\^=|\+\+|--|->|[{}()\[\];,.:<>=+\-*\/%&|^!~@])/g;
  let m, line = 1;
  while ((m = re.exec(src))) {
    if (m[0].startsWith("//") || m[0].startsWith("/*")) { line += (m[0].match(/\n/g) || []).length; continue; }
    if (m[1]) { line++; continue; }
    toks.push({ v: m[0], id: !!m[2], line });
  }
  return toks;
}

const KEYWORDS = new Set(["fn", "let", "var", "const", "if", "else", "for", "while", "loop", "return", "break", "continue", "continuing",
  "switch", "case", "default", "true", "false", "struct", "override", "discard", "enable", "requires", "diagnostic", "alias", "bitcast"]);

// ---- parse: module declarations + function bodies as statement trees ----
function parseModule(toks, mod) {
  let p = 0;
  const peek = (k = 0) => toks[p + k]?.v;
  const fns = [];
  while (p < toks.length) {
    const v = peek();
    if (v === "var" || v === "const" || v === "override") {
      let kind = v === "var" ? "private" : "uniform";
      if (v === "var" && peek(1) === "<") kind = toks[p + 2].v;   // storage | workgroup | uniform | private
      let q = p + 1;
      if (v === "var" && peek(1) === "<") { while (toks[q].v !== ">") q++; q++; }
      const name = toks[q]?.v;
      // read-only storage read at a uniform index is uniform (WGSL's own rule); read_write storage,
      // workgroup and private variables are not
      const ro = kind === "storage" && toks.slice(p, q).every((t) => t.v !== "read_write");
      if (name && name !== "_X_") mod.vars.set(name, ro ? "ro" : kind === "storage" || kind === "workgroup" || kind === "private" ? "varying" : "uniform");
      while (p < toks.length && peek() !== ";") p++;
      p++;
      continue;
    }
    if (v === "fn" && toks[p + 1]?.id) {
      const line = toks[p].line;
      let name = "";
      p++;
      while (p < toks.length && toks[p].v !== "(") name += toks[p++].v;   // fn foo${X}_b(...) -> "foo_X__b"
      // params: ( [@builtin(x)] name : type, ... )
      const params = [];
      let depth = 0;
      let pendingBuiltin = null;
      do {
        const t = peek();
        if (t === "(") depth++;
        else if (t === ")") depth--;
        else if (t === "@" && peek(1) === "builtin") { pendingBuiltin = toks[p + 3].v; p += 5; continue; }
        else if (depth === 1 && toks[p].id && peek(1) === ":") { params.push({ name: t, builtin: pendingBuiltin }); pendingBuiltin = null; }
        p++;
      } while (depth > 0 && p < toks.length);
      while (p < toks.length && peek() !== "{") p++;
      const r = parseBlock(toks, p);
      p = r.end;
      fns.push({ name, line, params, body: r.block, entry: false });
      continue;
    }
    p++;
  }
  return fns;
}

// statement nodes: { k: "block"|"if"|"loop"|"simple"|"jump", ... }
function parseBlock(toks, p) {   // at "{"
  const stmts = [];
  p++;
  while (p < toks.length && toks[p].v !== "}") {
    const r = parseStmt(toks, p);
    if (r.stmt) stmts.push(r.stmt);
    p = r.end;
  }
  return { block: { k: "block", stmts }, end: p + 1 };
}
function untilBrace(toks, p) { const s = p; let d = 0; while (p < toks.length && !(toks[p].v === "{" && d === 0)) { if (toks[p].v === "(" || toks[p].v === "[") d++; if (toks[p].v === ")" || toks[p].v === "]") d--; p++; } return { toks: toks.slice(s, p), end: p }; }
function untilSemi(toks, p) { const s = p; let d = 0; while (p < toks.length && !(toks[p].v === ";" && d === 0) && !(toks[p].v === "}" && d === 0)) { if ("([{".includes(toks[p].v)) d++; if (")]}".includes(toks[p].v)) d--; p++; } return { toks: toks.slice(s, p), end: toks[p]?.v === ";" ? p + 1 : p }; }
function parseStmt(toks, p) {
  const v = toks[p].v, line = toks[p].line;
  if (v === ";") return { stmt: null, end: p + 1 };
  if (v === "{") { const r = parseBlock(toks, p); return { stmt: r.block, end: r.end }; }
  if (v === "@") { return { stmt: null, end: p + 1 + (toks[p + 2]?.v === "(" ? 0 : 0) + 1 }; }   // attribute on a statement: skip name
  if (v === "if") {
    const c = untilBrace(toks, p + 1);
    const t = parseBlock(toks, c.end);
    let els = null, end = t.end;
    if (toks[end]?.v === "else") {
      if (toks[end + 1]?.v === "if") { const r = parseStmt(toks, end + 1); els = { k: "block", stmts: [r.stmt] }; end = r.end; }
      else { const r = parseBlock(toks, end + 1); els = r.block; end = r.end; }
    }
    return { stmt: { k: "if", cond: c.toks, then: t.block, els, line }, end };
  }
  if (v === "switch") {
    const c = untilBrace(toks, p + 1);
    let q = c.end + 1;
    const arms = [];
    while (q < toks.length && toks[q].v !== "}") {
      while (q < toks.length && toks[q].v !== "{" && toks[q].v !== "}") q++;
      if (toks[q]?.v === "{") { const r = parseBlock(toks, q); arms.push(r.block); q = r.end; }
    }
    return { stmt: { k: "switch", cond: c.toks, arms, line }, end: q + 1 };
  }
  if (v === "for") {
    // for ( init ; cond ; update ) { body }
    let q = p + 2, d = 0;
    const parts = [[]];
    while (q < toks.length && !(toks[q].v === ")" && d === 0)) {
      const t = toks[q];
      if (t.v === "(") d++; if (t.v === ")") d--;
      if (t.v === ";" && d === 0) parts.push([]); else parts[parts.length - 1].push(t);
      q++;
    }
    const b = parseBlock(toks, q + 1);
    return { stmt: { k: "loop", init: parts[0] || [], cond: parts[1] || [], update: parts[2] || [], body: b.block, line }, end: b.end };
  }
  if (v === "while") {
    const c = untilBrace(toks, p + 1);
    const b = parseBlock(toks, c.end);
    return { stmt: { k: "loop", init: [], cond: c.toks, update: [], body: b.block, line }, end: b.end };
  }
  if (v === "loop") {
    const b = parseBlock(toks, p + 1);
    return { stmt: { k: "loop", init: [], cond: [], update: [], body: b.block, line }, end: b.end };
  }
  if (v === "continuing") { const b = parseBlock(toks, p + 1); return { stmt: b.block, end: b.end }; }
  if (v === "break" && toks[p + 1]?.v === "if") { const s = untilSemi(toks, p + 2); return { stmt: { k: "jump", j: "break", cond: s.toks, line }, end: s.end }; }
  if (v === "break" || v === "continue" || v === "return" || v === "discard") {
    const s = untilSemi(toks, p + 1);
    return { stmt: { k: "jump", j: v === "discard" ? "return" : v, expr: s.toks, line }, end: s.end };
  }
  const s = untilSemi(toks, p);
  return { stmt: { k: "simple", toks: s.toks, line }, end: s.end };
}

// ---- analysis ----
const join = (ts) => (ts || []).map((t) => t.v).join(" ");
const srcOf = (s) => s.k === "loop" ? `for/while (${join(s.cond)})` : s.k === "jump" ? `${s.j} ${join(s.cond || s.expr)}`.trim() : join(s.toks || s.cond).slice(0, 80);
function walk(node, f) {
  f(node);
  if (node.k === "block") node.stmts.forEach((s) => walk(s, f));
  else if (node.k === "if") { walk(node.then, f); if (node.els) walk(node.els, f); }
  else if (node.k === "switch") node.arms.forEach((a) => walk(a, f));
  else if (node.k === "loop") walk(node.body, f);
}
const callsIn = (toks) => toks.filter((t, i) => t.id && toks[i + 1]?.v === "(").map((t) => t.v);

export function lintWGSL(text, { file = "", mod = { vars: new Map() }, fnsOut = null } = {}) {
  const toks = tokenize(text);
  const fns = parseModule(toks, mod);
  if (fnsOut) fnsOut.push(...fns);
  return { fns, toks };
}

// fns: every function of the module(s) (helpers may live in another template)
export function analyze(fns, mod, file = "") {
  const byName = new Map(fns.map((f) => [f.name, f]));
  // which functions hold a barrier (transitively)
  const hasBar = new Map();
  const barrierFn = (name, seen = new Set()) => {
    if (BARRIERS.has(name)) return true;
    const f = byName.get(name);
    if (!f || seen.has(name)) return false;
    if (hasBar.has(name)) return hasBar.get(name);
    seen.add(name);
    let b = false;
    walk(f.body, (s) => { for (const ts of [s.toks, s.cond, s.init, s.update, s.expr]) if (ts) for (const c of callsIn(ts)) if (barrierFn(c, seen)) b = true; });
    hasBar.set(name, b);
    return b;
  };
  for (const f of fns) barrierFn(f.name);
  // functions whose result can differ by thread even for uniform arguments (they read storage/workgroup)
  const readsVarying = new Map();
  for (const f of fns) {
    let r = false;
    walk(f.body, (s) => { for (const ts of [s.toks, s.cond, s.init, s.update, s.expr]) if (ts) for (const t of ts) if (t.id && mod.vars.get(t.v) === "varying") r = true; });
    readsVarying.set(f.name, r);
  }
  const findings = [];
  for (const f of fns) {
    scopeNames(f);
    const vary = new Set();
    for (const p of f.params) if (!p.builtin || VARYING_BUILTINS.has(p.builtin)) vary.add(p.name);
    // names declared in the function shadow module variables of the same name
    const local = new Set(f.params.map((p) => p.name));
    walk(f.body, (x) => { for (const ts of [x.toks, x.init]) if (ts && (ts[0]?.v === "let" || ts[0]?.v === "var" || ts[0]?.v === "const")) local.add(ts[1]?.v === "<" ? ts[ts.findIndex((t) => t.v === ">") + 1]?.v : ts[1]?.v); });
    const isVarying = (ts) => {
      if (!ts) return false;
      for (let i = 0; i < ts.length; i++) {
        const t = ts[i];
        if (!t.id || ts[i - 1]?.v === "." || t.v === "_X_") continue;
        if (t.v === "workgroupUniformLoad") { // its argument is read uniformly: skip to the matching ")"
          let d = 0, j = i + 1;
          for (; j < ts.length; j++) { if (ts[j].v === "(") d++; if (ts[j].v === ")" && --d === 0) break; }
          i = j; continue;
        }
        if (vary.has(t.v)) return true;
        const mk = local.has(t.v) ? null : mod.vars.get(t.v);
        if (mk === "varying") return true;
        if (mk === "ro") {   // name[index]: varying only when the index is
          if (ts[i + 1]?.v !== "[") return true;
          let d = 0, j = i + 1;
          for (; j < ts.length; j++) { if (ts[j].v === "[") d++; if (ts[j].v === "]" && --d === 0) break; }
          if (isVarying(ts.slice(i + 2, j))) return true;
          i = j; continue;
        }
        if (ts[i + 1]?.v === "(" && readsVarying.get(t.v)) return true;
      }
      return false;
    };
    const target = (ts) => {   // assigned name of a simple statement, or null
      if (!ts.length) return null;
      if (ts[0].v === "let" || ts[0].v === "var" || ts[0].v === "const") {
        let q = 1; if (ts[q]?.v === "<") { while (ts[q] && ts[q].v !== ">") q++; q++; }
        return { name: ts[q]?.v, rhs: ts.slice(ts.findIndex((t) => t.v === "=") + 1 || ts.length) };
      }
      const eq = ts.findIndex((t) => ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ">>=", "<<="].includes(t.v));
      if (eq > 0 && ts[0].id) return { name: ts[0].v, rhs: ts.slice(eq + 1), lhs: ts.slice(0, eq) };
      if ((ts[1]?.v === "++" || ts[1]?.v === "--") && ts[0].id) return { name: ts[0].v, rhs: [] };
      return null;
    };
    // taint to a fixpoint (loops carry values around); control dependence included
    const taint = (node, ctl) => {
      let changed = false;
      const mark = (n) => { if (n && n !== "_X_" && !vary.has(n)) { vary.add(n); changed = true; } };
      const go = (s, c) => {
        if (s.k === "block") s.stmts.forEach((x) => go(x, c));
        else if (s.k === "simple") { const a = target(s.toks); if (a && (c || isVarying(a.rhs) || (a.lhs && isVarying(a.lhs.slice(1))))) mark(a.name); }
        else if (s.k === "if") { const c2 = c || isVarying(s.cond); go(s.then, c2); if (s.els) go(s.els, c2); }
        else if (s.k === "switch") { const c2 = c || isVarying(s.cond); s.arms.forEach((a) => go(a, c2)); }
        else if (s.k === "loop") {
          const a = target(s.init); if (a && (c || isVarying(a.rhs))) mark(a.name);
          const u = target(s.update); if (u && (c || isVarying(u.rhs))) mark(u.name);
          go(s.body, c || isVarying(s.cond) || loopExitVaries(s));
        }
      };
      go(node, ctl);
      return changed;
    };
    // a loop's trip count varies: its condition, or a break/return under a varying condition (not in a nested loop)
    const loopExitVaries = (L) => {
      if (isVarying(L.cond)) return true;
      let v = false;
      const go = (s, c) => {
        if (s.k === "block") s.stmts.forEach((x) => go(x, c));
        else if (s.k === "if") { const c2 = c || isVarying(s.cond); go(s.then, c2); if (s.els) go(s.els, c2); }
        else if (s.k === "switch") { const c2 = c || isVarying(s.cond); s.arms.forEach((a) => goSwitch(a, c2)); }
        else if (s.k === "loop") goInner(s.body, c);   // a nested loop's break is its own; its return is ours
        else if (s.k === "jump" && (s.j === "break" || s.j === "return") && (c || (s.cond && isVarying(s.cond)))) v = true;
      };
      const goSwitch = (s, c) => { walkJumps(s, (j, cc) => { if (j.j === "return" && (c || cc)) v = true; }); };
      const goInner = (s, c) => { walkJumps(s, (j, cc) => { if (j.j === "return" && (c || cc)) v = true; }); };
      go(L.body, false);
      return v;
    };
    const walkJumps = (s, f, c = false) => {
      if (s.k === "block") s.stmts.forEach((x) => walkJumps(x, f, c));
      else if (s.k === "if") { const c2 = c || isVarying(s.cond); walkJumps(s.then, f, c2); if (s.els) walkJumps(s.els, f, c2); }
      else if (s.k === "switch") { const c2 = c || isVarying(s.cond); s.arms.forEach((a) => walkJumps(a, f, c2)); }
      else if (s.k === "loop") walkJumps(s.body, f, c || isVarying(s.cond));
      else if (s.k === "jump") f(s, c || (s.cond ? isVarying(s.cond) : false));
    };
    for (let it = 0; it < 20 && taint(f.body, false); it++);
    const stmtHasBarrier = (s) => {
      let b = false;
      walk(s, (x) => { for (const ts of [x.toks, x.cond, x.init, x.update, x.expr]) if (ts) for (const c of callsIn(ts)) if (barrierFn(c)) b = true; });
      return b;
    };
    // a barrier reached through a helper (tk_reduce), not written in the loop itself
    const barViaCall = (b) => {
      let r = false;
      walk(b, (x) => { for (const ts of [x.toks, x.cond, x.init, x.update, x.expr]) if (ts) for (const c of callsIn(ts)) if (!BARRIERS.has(c) && barrierFn(c)) r = true; });
      return r;
    };
    const say = (node, msg) => findings.push({ file, fn: f.name, what: srcOf(node), msg });
    // pass over the tree with the control context
    const check = (s, ctl, loops) => {
      if (s.k === "block") {
        s.stmts.forEach((x, i) => {
          // a varying continue/return here, with a barrier later in this block's loop iteration / function
          check(x, ctl, loops);
          if (x.k !== "loop" && x.k !== "simple") {
            const later = s.stmts.slice(i + 1).some(stmtHasBarrier);
            if (later) walkJumps(x, (j, c) => {
              if (!(c || ctl)) return;
              if (j.j === "continue" && loops.length && !insideNestedLoop(x, j)) say(j, "continue under a thread-dependent condition skips a later barrier in its loop");
              if (j.j === "return") say(j, "return under a thread-dependent condition skips a later barrier");
            });
          }
        });
      } else if (s.k === "simple" || s.k === "jump") {
        const ts = s.toks || s.expr || [];
        if (ctl && callsIn(ts).some((c) => barrierFn(c))) say(s, "barrier under a thread-dependent condition");
      } else if (s.k === "if") { const c2 = ctl || isVarying(s.cond); check(s.then, c2, loops); if (s.els) check(s.els, c2, loops); }
      else if (s.k === "switch") { const c2 = ctl || isVarying(s.cond); s.arms.forEach((a) => check(a, c2, loops)); }
      else if (s.k === "loop") {
        const bar = stmtHasBarrier(s.body);
        const varies = loopExitVaries(s);
        if (bar && (ctl || varies)) say(s, "loop with a barrier runs a thread-dependent number of times");
        if (barViaCall(s.body)) walk(s.body, (x) => { if (x !== s && x.k === "loop" && !stmtHasBarrier(x.body) && !constBound(x) && (isVarying(x.cond) || loopExitVaries(x))) say(x, "thread-dependent loop inside a loop that calls a barrier helper (FXC X3663, topk_b): give it a shared trip count and guard its body"); });
        check(s.body, ctl || varies, [...loops, s]);
      }
    };
    // `i < 512u` / `i < ${N}u`: a compile-time trip count FXC can unroll (moe_gusg's staging loop compiles)
    const constBound = (L) => L.cond.length >= 3 && L.cond[1].v === "<" && L.cond.slice(2).every((t) => !t.id || t.v.startsWith("_X_")) && !loopExitVaries({ ...L, cond: [] });
    const insideNestedLoop = (root, j) => { let inner = false; walk(root, (x) => { if (x.k === "loop") walk(x.body, (y) => { if (y === j) inner = true; }); }); return inner; };
    check(f.body, false, []);
  }
  return findings;
}

// Give every local declaration a name of its own (x, x#1, ...), so a `let b` in one block and a
// `let b` in another are told apart: block scoping, as WGSL has it.
function scopeNames(f) {
  if (f.scoped) return;
  f.scoped = true;
  const count = new Map();
  const stack = [new Map(f.params.map((p) => [p.name, p.name]))];
  const resolve = (n) => { for (let i = stack.length - 1; i >= 0; i--) if (stack[i].has(n)) return stack[i].get(n); return n; };
  const rw = (ts) => ts && ts.map((t, i) => (t.id && ts[i - 1]?.v !== "." && !KEYWORDS.has(t.v) ? { ...t, v: resolve(t.v) } : t));
  const decl = (ts) => {   // let/var/const name ... = rhs: the rhs sees the outer names, then the name is declared
    if (!ts?.length || !(ts[0].v === "let" || ts[0].v === "var" || ts[0].v === "const")) return rw(ts);
    let q = 1; if (ts[q]?.v === "<") { while (ts[q] && ts[q].v !== ">") q++; q++; }
    const name = ts[q]?.v;
    const out = rw(ts);
    const k = (count.get(name) || 0) + 1; count.set(name, k);
    const u = k === 1 ? name : `${name}#${k}`;
    stack[stack.length - 1].set(name, u);
    out[0] = ts[0]; out[q] = { ...ts[q], v: u };
    return out;
  };
  const go = (s) => {
    if (s.k === "block") { stack.push(new Map()); s.stmts.forEach(go); stack.pop(); }
    else if (s.k === "simple") s.toks = decl(s.toks);
    else if (s.k === "jump") { s.expr = rw(s.expr); s.cond = rw(s.cond); }
    else if (s.k === "if") { s.cond = rw(s.cond); go(s.then); if (s.els) go(s.els); }
    else if (s.k === "switch") { s.cond = rw(s.cond); s.arms.forEach(go); }
    else if (s.k === "loop") { stack.push(new Map()); s.init = decl(s.init); s.cond = rw(s.cond); s.update = rw(s.update); go(s.body); stack.pop(); }
  };
  go(f.body);
}

// lint JS sources: [{ file, js }] -> findings (functions are resolved within their file)
export function lintSources(sources) {
  const mod = { vars: new Map() };
  const perFile = sources.map(({ file, js }) => {
    const fns = [];
    for (const t of templatesOf(js)) lintWGSL(t, { file, mod, fnsOut: fns });
    return { file, fns };
  });
  const seen = new Set(), out = [];
  for (const { file, fns } of perFile) for (const x of analyze(fns, mod, file)) {
    const key = `${x.fn}:${x.what}:${x.msg}`;
    if (!seen.has(key)) { seen.add(key); out.push(x); }
  }
  return out;
}

if (typeof process !== "undefined" && process.argv?.[1] && import.meta.url === new URL(process.argv[1], "file://").href) {
  const fs = await import("node:fs");
  const path = await import("node:path");
  const dir = new URL("../engine/wgsl/", import.meta.url).pathname;
  const files = process.argv.slice(2).length ? process.argv.slice(2) : fs.readdirSync(dir).filter((f) => f.endsWith(".js")).map((f) => path.join(dir, f));
  const found = lintSources(files.map((file) => ({ file: path.basename(file), js: fs.readFileSync(file, "utf8") })));
  for (const x of found) console.log(`${x.file} fn ${x.fn}: \`${x.what}\`: ${x.msg}`);
  console.log(`${found.length} finding(s)`);
  process.exit(found.length ? 1 : 0);
}
