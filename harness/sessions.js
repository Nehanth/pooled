// Several agent sessions on one engine: the engine holds one conversation state at a time; the
// others wait in GPU slots (instant to switch back to) or, past `gpuSlots`, on disk in the OPFS
// state cache (a read plus an upload). Switching is exact: a session resumes bit for bit where it
// stopped (tests/e2e/sessions_synth.mjs).
//
// Planned for rooms (not wired yet; only the e2e tests use this today): every device runs the
// same manager with the same calls in the same order (the host drives it), so the devices' slots
// and files stay in step; see docs/long-context-and-sessions.md.
import { StateCache } from "./statecache.js";

export class Sessions {
  // engine: Qwen35Engine (or anything with saveSlot / loadSlot / dropSlot / exportSlot /
  // importState / reset / pos). gpuSlots: sessions kept on the GPU besides the active one.
  constructor(engine, { gpuSlots = 2, cache = null, prefix = "tabby" } = {}) {
    this.e = engine; this.gpuSlots = gpuSlots; this.prefix = prefix;
    this.cache = cache || new StateCache({ dirName: prefix + "-sessions" });
    this.active = null;
    this.where = new Map();   // id -> "gpu" | "disk"
    this.used = new Map();    // id -> last use (for LRU)
    this.clock = 0;
    this.stats = { gpuHits: 0, diskHits: 0, fresh: 0, spills: 0 };
  }
  _slot(id) { return `${this.prefix}:${id}`; }
  _key(id) { return `${this.prefix}-${String(id).replace(/[^A-Za-z0-9_-]/g, "_")}`; }
  // Make `id` the engine's current conversation. A new id starts empty (engine.reset()).
  // Returns where it came from: "active" | "gpu" | "disk" | "new".
  async switchTo(id) {
    this.used.set(id, ++this.clock);
    if (this.active === id) return "active";
    if (this.active != null) {   // park the current session on the GPU
      this.e.saveSlot(this._slot(this.active));
      this.where.set(this.active, "gpu");
    }
    let from;
    const w = this.where.get(id);
    if (w === "gpu") { this.e.loadSlot(this._slot(id)); this.e.dropSlot(this._slot(id)); from = "gpu"; this.stats.gpuHits++; }
    else if (w === "disk" && (await this._loadDisk(id))) { from = "disk"; this.stats.diskHits++; }
    else { this.e.reset(); from = "new"; this.stats.fresh++; }
    this.where.delete(id);
    this.active = id;
    await this._spill();
    return from;
  }
  // the file is removed once loaded: the session now lives on the engine and moves on from it, and
  // nothing else points at the file (a stale copy would sit on disk until evicted)
  async _loadDisk(id) {
    const st = await this.cache.get(this._key(id));
    if (!st) return false;
    this.e.importState(st);
    await this.cache.remove(this._key(id));
    return true;
  }
  // keep at most gpuSlots parked sessions on the GPU; the least recently used go to disk
  async _spill() {
    const onGpu = [...this.where].filter(([, w]) => w === "gpu").map(([id]) => id)
      .sort((a, b) => this.used.get(a) - this.used.get(b));
    while (onGpu.length > this.gpuSlots) {
      const id = onGpu.shift();
      const st = await this.e.exportSlot(this._slot(id));
      await this.cache.put(this._key(id), st, { id });
      this.e.dropSlot(this._slot(id));
      this.where.set(id, "disk");
      this.stats.spills++;
    }
  }
  // Save the active session to disk too (e.g. before the tab may close); it stays active.
  async persist() {
    if (this.active == null) return;
    await this.cache.put(this._key(this.active), await this.e.exportState(), { id: this.active });
  }
  // Forget a session everywhere. Its file goes wherever the session is: persist() also writes one
  // for the active session.
  async close(id) {
    if (this.where.get(id) === "gpu") this.e.dropSlot(this._slot(id));
    if (this.where.has(id) || this.active === id) await this.cache.remove(this._key(id));
    this.where.delete(id); this.used.delete(id);
    if (this.active === id) { this.active = null; this.e.reset(); }
  }
  list() {
    return [...new Set([...this.where.keys(), ...(this.active != null ? [this.active] : [])])]
      .map((id) => ({ id, where: id === this.active ? "active" : this.where.get(id) }));
  }
}
