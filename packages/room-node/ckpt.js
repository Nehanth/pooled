// The room node's checkpoints (DOM-free, no GPU): which prefix states the host keeps on every device
// of the chain, and where a prompt's prefill pauses to save one. The same design as the browser
// room's (room.js ckptSave / ckptResume, harness/prefix.js, #251 / #260), with more room for an
// agent's traffic:
//
//   - pinned checkpoints: a prompt's fixed start (the system prompt + tools, and an agent's own
//     cache boundary inside its system prompt, see cacheBoundary). Kept apart from the answer
//     checkpoints and never evicted by them; up to `pins` of them, least recently used out first,
//     so two agents (or two tool sets) sharing a room do not keep replacing each other's.
//   - answer checkpoints: the state after each answer, up to `answers`. Which one goes is by
//     GreedyDual (Young / Cao-Irani): each has a value of the clock when it was last saved or used plus
//     what it would cost to rebuild (its tokens past the longest pinned prefix of it); the lowest goes,
//     and the clock moves up to it. So a side request (a session title, a compaction) that is cheap to
//     redo goes before the main conversation's thousands of tokens, and anything unused ages out.
//   - one index for the whole host, so a second session with the same system prompt (a new OpenClaw
//     chat, another agent with the same tools) resumes from the pinned one.
//
// A hybrid model's DeltaNet state cannot be cut back, so a resume works only at checkpoints: the
// longest checkpoint that is a strict prefix of the new prompt (PrefixIndex.best).
import { PrefixIndex, isPrefix } from "../../harness/prefix.js";
import { renderApi } from "../../room/conversation.js";

export const CKPT_DEFAULTS = {
  answers: 3,       // answer checkpoints kept (the browser room's ?ckpt default is 2)
  pins: 4,          // pinned prefixes kept (two per agent: its cache boundary and its system prompt + tools)
  minPin: 1024,     // a fixed start shorter than this is not worth a checkpoint of its own
};

// Where an agent marks the end of the part of its system prompt that stays the same across
// sessions (its prompt-cache boundary). OpenClaw wraps it in <!-- openclaw:attempt:STABLE -->
// ... <!-- /openclaw:attempt:STABLE -->, and puts the date, the model's name and other per-run
// text after it (src/agents/embedded-agent-runner/run/attempt-system-prompt.ts).
export const CACHE_MARKS = ["<!-- /openclaw:attempt:STABLE -->"];
// -> the character offset just after the last cache mark in the system text, or 0
export function cacheBoundary(system) {
  const s = String(system || "");
  let at = 0;
  for (const m of CACHE_MARKS) { const i = s.lastIndexOf(m); if (i >= 0) at = Math.max(at, i + m.length); }
  return at < s.length ? at : 0;   // a mark at the very end is the whole system prompt (systemLen covers it)
}
// the length of the common prefix of two id lists
export function commonPrefix(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}
// The token position of the cache boundary in a rendered v2 prompt: the prompt rendered with the
// system text cut there (and no messages) shares exactly the tokens up to it with the real one.
// Tokens around the cut that encode differently are left out by the common prefix, so the pin is
// always a prefix of the real prompt's ids. With the xml tool style (Qwen3.5+) the tools come
// first, so this pin covers them too. -> 0 when there is no boundary or it is too short.
export function boundaryPin(tok, req, prompt, { encode, minPin = CKPT_DEFAULTS.minPin } = {}) {
  const cut = cacheBoundary(req.system);
  if (!cut || !prompt?.ids?.length) return 0;
  const { ids } = renderApi(tok, { ...req, system: String(req.system).slice(0, cut), messages: [] }, prompt.profile, { encode, thinking: prompt.thinking });
  const n = commonPrefix(ids, prompt.ids);
  return n >= minPin && n < prompt.ids.length ? n : 0;
}
// The pinned points of a v2 prompt: its cache boundary and the end of its system prompt + tools,
// each when it is long enough. -> sorted, unique, each < ids.length
export function pinPoints(prompt, { boundary = 0, minPin = CKPT_DEFAULTS.minPin } = {}) {
  const n = prompt?.ids?.length || 0;
  return [...new Set([boundary, prompt?.systemLen || 0])].filter((p) => Number.isInteger(p) && p >= minPin && p < n).sort((a, b) => a - b);
}
// Where a prefill pauses to save pinned checkpoints: the pins past what the caches already hold,
// with at least one prompt token after each (the last one has to go through the head).
export function cutPoints(reused, pins, total) {
  return [...new Set(pins || [])].filter((p) => Number.isInteger(p) && p > reused && p < total).sort((a, b) => a - b);
}

// The host's index: PrefixIndex plus the pinned / answer split and the eviction rules. Pure: the
// caller does the engine and chain side (saveSlot, the frame's sv / dp) with what these return.
export class CkptIndex {
  constructor({ answers = CKPT_DEFAULTS.answers, pins = CKPT_DEFAULTS.pins } = {}) {
    this.answers = answers; this.pins = pins;
    this.ix = new PrefixIndex(1 << 30);
    this.L = 0;       // GreedyDual's clock for the answer checkpoints
    this.n = 0;       // last slot number handed out (1..65534, they ride the frame header as u16)
    this.hits = { pin: 0, answer: 0, miss: 0 };
  }
  get items() { return this.ix.items; }
  get size() { return this.ix.items.length; }
  find(key) { return this.ix.items.find((x) => x.key === key) || null; }
  same(ids) { return this.ix.items.find((x) => x.ids.length === ids.length && isPrefix(x.ids, ids)) || null; }
  nextKey() { return (this.n = this.n % 65534 + 1); }
  remove(key) { this.ix.remove(key); }
  // tokens to prefill again if this state were gone: past the longest pinned prefix of it
  cost(ids) {
    let base = 0;
    for (const x of this.ix.items) if (x.pin && x.ids.length > base && x.ids.length <= ids.length && isPrefix(x.ids, ids)) base = x.ids.length;
    return Math.max(1, ids.length - base);
  }
  touch(x) { x.t = ++this.ix.clock; if (!x.pin) x.h = this.L + this.cost(x.ids); }
  clear() { const keys = this.ix.items.map((x) => x.key); this.ix.items = []; return keys; }
  // Plan a save of the state holding `ids`.
  // -> { skip: true } (an equal checkpoint is kept instead, and touched) or
  //    { key, drop: [keys to evict first] }; commit(key, ids, pin, extra) records it after the save
  plan(ids, { pin = false } = {}) {
    const same = this.same(ids);
    if (same && (same.pin || !pin)) { this.touch(same); return { skip: true, key: same.key }; }
    const drop = [];
    if (same) drop.push(same.key);   // an answer checkpoint with exactly these tokens becomes the pinned one
    // pins by last use; answers by GreedyDual value (the lowest goes, the clock moves up to it)
    const kind = this.ix.items.filter((x) => !!x.pin === !!pin && x.key !== same?.key)
      .sort(pin ? (a, b) => a.t - b.t : (a, b) => (a.h - b.h) || (a.t - b.t));
    const cap = Math.max(1, pin ? this.pins : this.answers);
    for (const x of kind.slice(0, Math.max(0, kind.length - (cap - 1)))) { drop.push(x.key); if (!pin) this.L = Math.max(this.L, x.h); }
    return { key: this.nextKey(), drop };
  }
  commit(key, ids, pin = false, extra = {}) {
    this.ix.add(ids, key, { pin });
    const x = Object.assign(this.find(key), extra);
    this.touch(x);
  }
  // the longest checkpoint that is a strict prefix of ids and longer than `reused`, touched -> item or null
  best(ids, reused = 0) {
    const b = this.ix.best(ids);
    const x = b && b.n > reused ? this.find(b.key) : null;
    if (x) { this.touch(x); this.hits[x.pin ? "pin" : "answer"]++; } else this.hits.miss++;
    return x;
  }
}
