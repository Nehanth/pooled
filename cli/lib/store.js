// The Responses API's stored responses (docs/design/serve.md section 7): what previous_response_id,
// item_reference and GET / DELETE /v1/responses/{id} read. In this process's memory only, bounded:
// at most 256 responses and 64 MB of JSON, each dropped after an hour without use (least recently
// used first when over a bound). Nothing goes to the room or to disk.
//
// An entry, per response id:
//   { id, response,        // the response object as the client got it (GET returns it)
//     history: Msg[],      // the conversation up to and including this response's input, as the
//                          // adapter parsed it (instructions excluded: they are not inherited)
//     output: Msg[],       // this response's output as conversation messages
//     inputItems: [],      // this response's own input items, with ids (GET …/input_items)
//     bytes, at }

export const STORE_LIMITS = { responses: 256, bytes: 64 << 20, ttlMs: 3600000 };

export class ResponseStore {
  constructor({ max = STORE_LIMITS.responses, bytes = STORE_LIMITS.bytes, ttlMs = STORE_LIMITS.ttlMs, now = () => Date.now() } = {}) {
    this.max = max; this.maxBytes = bytes; this.ttlMs = ttlMs; this.now = now;
    this.map = new Map();    // id -> entry, least recently used first
    this.items = new Map();  // item id -> { rid, item } (for item_reference)
    this.bytes = 0;
  }
  get size() { return this.map.size; }
  // the entry, or null (unknown, deleted or expired); a hit counts as use
  get(id) {
    const e = this.map.get(id);
    if (!e) return null;
    if (this.now() - e.at > this.ttlMs) { this.delete(id); return null; }
    e.at = this.now();
    this.map.delete(id); this.map.set(id, e);
    return e;
  }
  has(id) { return !!this.get(id); }
  // an item stored with any live response (output or input), or null
  item(itemId) {
    const x = this.items.get(itemId);
    if (!x || !this.get(x.rid)) return null;
    return x.item;
  }
  put({ id, response, history = [], output = [], inputItems = [] }) {
    this.delete(id);
    this.sweep();
    const bytes = JSON.stringify(response).length + JSON.stringify(history).length + JSON.stringify(output).length + JSON.stringify(inputItems).length;
    if (bytes > this.maxBytes) return false;   // one response bigger than the whole store: not kept
    const e = { id, response, history, output, inputItems, bytes, at: this.now() };
    this.map.set(id, e);
    this.bytes += bytes;
    for (const it of [...(response.output || []), ...inputItems]) if (it && typeof it.id === "string") this.items.set(it.id, { rid: id, item: it });
    while (this.map.size > this.max || this.bytes > this.maxBytes) this.delete(this.map.keys().next().value);
    return true;
  }
  delete(id) {
    const e = this.map.get(id);
    if (!e) return false;
    this.map.delete(id);
    this.bytes -= e.bytes;
    for (const it of [...(e.response.output || []), ...e.inputItems]) if (it && this.items.get(it.id)?.rid === id) this.items.delete(it.id);
    return true;
  }
  // drop everything idle for longer than the TTL
  sweep() {
    const t = this.now();
    for (const [id, e] of this.map) if (t - e.at > this.ttlMs) this.delete(id);
  }
}
