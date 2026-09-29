// harness/preview-frame.js relayUrl / siteOf / relayProblem and run-js.js runJsAvailable: which
// preview-origin values turn on the relay (and so run_js), and which are ignored.
import { relayUrl, siteOf, relayProblem } from "../../harness/preview-frame.js";
import { runJsAvailable } from "../../harness/run-js.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a document at `href` with the preview-origin meta set to `meta` (undefined: no meta)
const page = (href, meta) => {
  const u = new URL(href);
  return {
    defaultView: { location: { protocol: u.protocol, hostname: u.hostname, port: u.port } },
    querySelector: (sel) => (meta !== undefined && sel === 'meta[name="preview-origin"]' ? { content: meta } : null),
  };
};
const quiet = (f) => { const w = console.warn; console.warn = () => {}; try { return f(); } finally { console.warn = w; } };

Deno.test("siteOf: last two labels, three under a two-part suffix; IPs and single labels as they are", () => {
  eq(siteOf("pooled.run"), "pooled.run");
  eq(siteOf("preview.pooled.run"), "pooled.run");
  eq(siteOf("A.B.Pooled.Run."), "pooled.run");
  eq(siteOf("x.example.co.uk"), "example.co.uk");
  eq(siteOf("pooled-git-x.vercel.app"), "pooled-git-x.vercel.app");
  eq(siteOf("127.0.0.1"), "127.0.0.1");
  eq(siteOf("localhost"), "localhost");
});

Deno.test("relayProblem: another site over https is fine; same site, http from https, non-URLs are not", () => {
  const prod = { protocol: "https:", hostname: "pooled.run" };
  eq(relayProblem("https://pooled-preview.dev", prod), "");
  eq(relayProblem("https://sandbox.pooled-preview.dev/", prod), "");
  ok(/same site/.test(relayProblem("https://preview.pooled.run", prod)));
  ok(/same site/.test(relayProblem("https://pooled.run", prod)));
  ok(/https/.test(relayProblem("http://pooled-preview.dev", prod)));
  ok(/not a URL/.test(relayProblem("pooled-preview.dev", prod)));
  ok(/http\(s\)/.test(relayProblem("javascript:alert(1)", prod)));
  eq(relayProblem("http://127.0.0.1:8080", { protocol: "http:", hostname: "localhost" }), "", "plain http on a local http page");
});

Deno.test("relayUrl and runJsAvailable: a good meta turns run_js on, a bad one leaves it off", () => {
  const good = page("https://pooled.run/room", "https://pooled-preview.dev/");
  eq(relayUrl(good), "https://pooled-preview.dev/harness/preview-relay.html");
  ok(runJsAvailable(good));
  // production today: the meta is there but empty
  eq(relayUrl(page("https://pooled.run/room", "")), null);
  ok(!runJsAvailable(page("https://pooled.run/room", "")));
  ok(!runJsAvailable(page("https://pooled.run/room")));
  // a subdomain would share the room's process: ignored, with a warning
  let warned = "";
  const w = console.warn; console.warn = (m) => { warned = m; };
  try { eq(relayUrl(page("https://pooled.run/room", "https://preview.pooled.run")), null); } finally { console.warn = w; }
  ok(/preview-origin ignored: same site/.test(warned), warned);
  eq(quiet(() => relayUrl(page("https://pooled.run/room", "http://pooled-preview.dev"))), null);
  ok(!quiet(() => runJsAvailable(page("https://pooled.run/room", "https://preview.pooled.run"))));
  // development: the other loopback name, no meta needed
  eq(relayUrl(page("http://localhost:8080/room")), "http://127.0.0.1:8080/harness/preview-relay.html");
  eq(relayUrl(page("http://127.0.0.1:8080/room")), "http://localhost:8080/harness/preview-relay.html");
  eq(relayUrl(null), null);
});
