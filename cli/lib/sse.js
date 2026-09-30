// Server-sent events: the framing both APIs stream in. Every event is written (and so flushed by
// node:http) as soon as it exists; a client that went away is noticed through res "close".

export const sseData = (obj) => `data: ${typeof obj === "string" ? obj : JSON.stringify(obj)}\n\n`;
export const sseEvent = (type, obj) => `event: ${type}\ndata: ${JSON.stringify(obj)}\n\n`;
export const sseComment = (text) => `: ${text}\n\n`;

export const SSE_HEADERS = {
  "content-type": "text/event-stream; charset=utf-8",
  "cache-control": "no-cache",
  connection: "keep-alive",
  "x-accel-buffering": "no",   // proxies (nginx) must not buffer the stream
};

// One response's stream. start() sends the headers once; write() is a no-op after the client left.
export class SSEWriter {
  constructor(res) {
    this.res = res;
    this.started = false;
    this.closed = false;
    this.last = Date.now();   // when a byte last went out (for keep-alives on silence)
    res.on("close", () => { this.closed = true; });
  }
  start() {
    if (this.started || this.closed) return;
    this.started = true;
    this.res.writeHead(200, SSE_HEADERS);
    this.res.flushHeaders?.();
  }
  write(chunk) {
    if (!chunk || this.closed) return;
    this.start();
    this.res.write(chunk);
    this.last = Date.now();
  }
  end(chunk) {
    if (this.closed) return;
    this.start();
    if (chunk) this.res.write(chunk);
    this.res.end();
    this.closed = true;
  }
}

// Events with a strictly increasing sequence_number (the Responses API's stream):
//   const ev = sseSequence(); ev("response.created", { response }) -> 'event: response.created\ndata: {"type":...,"sequence_number":0,...}'
export function sseSequence(start = 0) {
  let n = start;
  const ev = (type, obj = {}) => sseEvent(type, { type, sequence_number: n++, ...obj });
  ev.next = () => n;
  return ev;
}
