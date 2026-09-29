// The eval suite, page side (docs/design/harness-light.md C): runs a task's request through a fresh
// Agent over a fresh in-memory project, then checks the result in the sandbox with the run_js
// probe (harness/run-js.js). No Node APIs: eval.html loads it for the mock and a single engine,
// Code mode loads it (room/code.js, ?eval=) for a real room.
//
//   runTask(task, { makeModel, root, onEvent, model, signal }) -> { rec, trajectory }
//   runSuite(tasks, opts & { repeat, onResult }) -> [rec]
//   selfTest(task, { root }) -> { ok, detail }   the task's `bad` files must fail its check
//   summary(recs) -> one line
// makeModel({ task, tools }) -> { generate, stats?, budget?, count? } (a new model per task, so its
// token counters are the task's; a room model is built over the task's tools for the constraint).
//
// rec: { id, model, ok, reason, check, steps, calls, cards, forced, prompt, reused, generated,
//        compactions, ctx, ms, firstMs }. prompt (tokens prefilled) / reused (tokens already in
// the caches) / generated come from the model's `usage`; compactions counts the agent's; ctx is the
// agent's own estimate of the conversation at the end (system + turns, in tokens).
import { Agent } from "../../harness/agent.js";
import { CODE_SYSTEM } from "../../harness/code-prompt.js";
import { codingTools } from "../../harness/codetools.js";
import { previewTools } from "../../harness/preview-tools.js";
import { PreviewServer } from "../../harness/preview.js";
import { mountPreview } from "../../harness/preview-frame.js";
import { MemoryWorkspace, watch } from "../../harness/workspace.js";
import { probeSnapshot, runProbe, formatProbe, runJsTool, browserRunner } from "../../harness/run-js.js";

export const CHECK_MS = 8000;

// prepended to every check (one line, so check line numbers are off by one only)
export const HELPERS = [
  "const sleep=(ms)=>new Promise((r)=>setTimeout(r,ms));",
  "const ok=(c,m)=>{if(!c)throw new Error('check failed: '+m)};",
  "const $=(s)=>document.querySelector(s),$$=(s)=>[...document.querySelectorAll(s)];",
  "const text=(e)=>(e?(e.tagName==='INPUT'||e.tagName==='TEXTAREA'?e.value:e.textContent):'').trim();",
  "const button=(l)=>$$('button,input[type=button],input[type=submit],[role=button]').find((b)=>text(b)===l||b.value===l);",
  "const KC={ArrowLeft:37,ArrowUp:38,ArrowRight:39,ArrowDown:40,' ':32,Enter:13,Escape:27};",
  "const key=(k,el=document.activeElement||document.body,type='keydown')=>{const e=new KeyboardEvent(type,{key:k,code:k===' '?'Space':k,bubbles:true,cancelable:true});for(const p of ['keyCode','which'])Object.defineProperty(e,p,{get:()=>KC[k]||0});el.dispatchEvent(e);if(type==='keydown')key(k,el,'keyup');return e};",
  "const press=(k,el)=>{key(k,el);key(k,el,'keypress')};",
  "const shot=(c)=>c.toDataURL();",
  "const drawn=(c)=>{const g=c.getContext('2d');if(!g){const b=document.createElement('canvas');b.width=c.width;b.height=c.height;return c.toDataURL()!==b.toDataURL()}const d=g.getImageData(0,0,c.width,c.height).data,s=new Set();for(let i=0;i<d.length;i+=16){s.add((d[i]<<24)|(d[i+1]<<16)|(d[i+2]<<8)|d[i+3]);if(s.size>1)return true}return false};",
].join("");

const now = () => performance.now();
const readOr = (ws) => async (p) => ((await ws.exists(p)) ? ws.read(p) : null);

// the task's check against the workspace: -> { ok, detail }
export async function checkTask(task, ws, { root } = {}) {
  if (task.checkFiles) {
    const bad = await task.checkFiles(readOr(ws));
    if (bad) return { ok: false, detail: bad };
  }
  const snap = await probeSnapshot(ws);
  const page = task.page || (snap.files.has("index.html") ? "index.html" : null);
  const [width, height] = task.viewport || [800, 600];
  const r = await runProbe(snap, { code: HELPERS + "\n" + task.check, page, timeout: CHECK_MS, width, height, doc: root?.ownerDocument });
  return { ok: r.status === "ok", detail: formatProbe(r) };
}

export async function runTask(task, { makeModel, root = document.body, onEvent = () => {}, model: label = "?", signal } = {}) {
  const ws = watch(new MemoryWorkspace({ ...(task.files || {}) }));
  const server = new PreviewServer(ws);
  // a preview frame per served port, as Code mode shows one, so serve reports the page's errors
  const box = root.ownerDocument.createElement("div");
  box.style.cssText = "position:fixed;left:0;top:0;width:800px;height:600px;opacity:0;pointer-events:none;z-index:-1";
  root.appendChild(box);
  const mounts = new Map();
  const offUpdate = server.onUpdate((u) => {
    if (u.stopped || mounts.has(u.port)) return;
    const el = root.ownerDocument.createElement("div");
    el.style.cssText = "width:800px;height:600px";
    box.appendChild(el);
    mounts.set(u.port, mountPreview(el, server, u.port, {}));
  });
  const tools = [...codingTools(ws, { server }), ...previewTools(server), runJsTool(server, { runner: browserRunner(root.ownerDocument) })];   // as Code mode (room/code.js)
  const model = await makeModel({ task, tools });
  const rec = { id: task.id, model: label, ok: false, reason: "", check: "", steps: 0, calls: 0, cards: {}, forced: 0, prompt: 0, reused: 0, generated: 0, compactions: 0, ctx: 0, ms: 0, firstMs: 0 };
  const t0 = now();
  const agent = new Agent({
    generate: model.generate, tools, style: model.style || "xml", system: CODE_SYSTEM, maxSteps: task.maxSteps || 20, approve: async () => true,
    budget: model.budget || Infinity, count: model.count || null, usage: model.stats ? () => model.stats.last : null,
    onEvent: (e) => {
      if (e.type === "delta" && !rec.firstMs) rec.firstMs = Math.round(now() - t0);
      if (e.type === "usage") { rec.prompt += (e.prompt || 0) - (e.reused || 0); rec.reused += e.reused || 0; rec.generated += e.generated || 0; rec.forced += e.forced || 0; }
      if (e.type === "card") rec.cards[e.id] = (rec.cards[e.id] || 0) + 1;
      if (e.type === "compacted" && e.tier < 4) rec.compactions++;
      onEvent({ task: task.id, ...e });
    },
  });
  let r;
  try { r = await agent.run(task.prompt, { signal }); }
  catch (err) { r = { steps: 0, calls: 0, reason: "error", text: String(err?.message || err) }; }
  rec.ms = Math.round(now() - t0);
  Object.assign(rec, { reason: r.reason, steps: r.steps, calls: r.calls });
  try { rec.ctx = agent._size(); } catch {}
  for (const m of mounts.values()) m.destroy();
  offUpdate(); server.close(); box.remove();
  try {
    const c = await checkTask(task, ws, { root });
    rec.ok = c.ok; rec.check = c.detail;
  } catch (err) { rec.check = "check crashed: " + (err?.message || err); }
  if (r.reason === "error") rec.check = `run failed: ${r.text}\n${rec.check}`;
  const files = {};
  for (const p of await ws.walk()) files[p] = await ws.read(p).catch(() => "(binary)");
  return { rec, trajectory: { id: task.id, model: label, prompt: task.prompt, system: agent.system, result: r, ...agent.toJSON(), files } };
}

// `bad` is one set of files or a list of them (each must fail the check)
export async function selfTest(task, { root = document.body } = {}) {
  const out = [];
  for (const bad of [task.bad || {}].flat()) {
    const c = await checkTask(task, new MemoryWorkspace({ ...(task.files || {}), ...bad }), { root });
    if (c.ok) return { ok: false, detail: `bad variant ${out.length + 1} passes: ${c.detail}` };
    out.push(c.detail);
  }
  return { ok: true, detail: out.join("\n---\n") };
}

export async function runSuite(tasks, { repeat = 1, onResult = () => {}, ...opts } = {}) {
  const out = [];
  for (let k = 0; k < repeat; k++) {
    for (const task of tasks) {
      if (opts.signal?.aborted) return out;
      const res = await runTask(task, opts);
      out.push(res.rec);
      await onResult(res);
    }
  }
  return out;
}

const kt = (n) => (n >= 10000 ? Math.round(n / 1000) + "k" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n));
export function summary(recs) {
  const n = recs.length, ok = recs.filter((r) => r.ok).length, sum = (k) => recs.reduce((s, r) => s + (r[k] || 0), 0);
  const forced = sum("forced");
  return `success ${ok}/${n} (${n ? Math.round((100 * ok) / n) : 0} %) · steps ${(sum("steps") / (n || 1)).toFixed(1)}`
    + ` · prefilled ${kt(sum("prompt"))} · reused ${kt(sum("reused"))} · generated ${kt(sum("generated"))} · ctx ${kt(Math.round(sum("ctx") / (n || 1)))}/task`
    + (forced ? ` · forced ${forced}` : "") + ` · ${Math.round(sum("ms") / 1000)} s`;
}
