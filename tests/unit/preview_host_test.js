// Code mode's preview origin (roadmap 29): the preview-origin meta's per-host list
// (harness/preview-frame.js previewOrigin, relayUrl), and the relay's headers in both Vercel configs
// (vercel.json for pooled.run, preview-host/vercel.json for the second site).
import { previewEntry, previewOrigin, relayProblem, relayUrl, siteOf } from "../../harness/preview-frame.js";
import { CSP } from "../../harness/preview-build.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const LIST = "pooled.run=https://pooled-preview.vercel.app pooled-dev.vercel.app=https://pooled-preview-dev.vercel.app";

Deno.test("previewOrigin: empty or missing content is local mode", () => {
  eq(previewOrigin("", "pooled.run"), null);
  eq(previewOrigin("   ", "pooled.run"), null);
  eq(previewOrigin(undefined, "pooled.run"), null);
});

Deno.test("previewOrigin: a bare origin applies to every host (the old single-value form)", () => {
  eq(previewOrigin("https://pooled-preview.vercel.app", "pooled.run"), "https://pooled-preview.vercel.app");
  eq(previewOrigin(" https://pooled-preview.vercel.app/ ", "pooled-dev.vercel.app"), "https://pooled-preview.vercel.app");
  eq(previewOrigin("https://pooled-preview.vercel.app/some/path", "pooled.run"), "https://pooled-preview.vercel.app", "path dropped");
});

Deno.test("previewOrigin: host entries pick per host; unlisted hosts get nothing", () => {
  eq(previewOrigin(LIST, "pooled.run"), "https://pooled-preview.vercel.app");
  eq(previewOrigin(LIST, "pooled-dev.vercel.app"), "https://pooled-preview-dev.vercel.app");
  eq(previewOrigin(LIST, "POOLED.RUN"), "https://pooled-preview.vercel.app", "host is case-insensitive");
  eq(previewOrigin(LIST, "pooled-git-feat-x.vercel.app"), null, "a branch preview is not listed");
  eq(previewOrigin(LIST.replace(" ", ", "), "pooled-dev.vercel.app"), "https://pooled-preview-dev.vercel.app", "commas separate too");
});

Deno.test("previewOrigin: an exact host entry wins over a bare origin", () => {
  const c = "https://fallback.example pooled.run=https://pooled-preview.vercel.app";
  eq(previewOrigin(c, "pooled.run"), "https://pooled-preview.vercel.app");
  eq(previewOrigin(c, "other.example"), "https://fallback.example");
});

Deno.test("previewOrigin: refuses insecure, malformed and same-site origins", () => {
  eq(previewOrigin("http://pooled-preview.example", "pooled.run"), null, "plain http off loopback");
  eq(previewOrigin("javascript:alert(1)", "pooled.run"), null);
  eq(previewOrigin("not a url", "pooled.run"), null);
  eq(previewOrigin("pooled.run=", "pooled.run"), null, "empty value");
  eq(previewOrigin("https://pooled.run", "pooled.run"), null, "the page's own host");
  eq(previewOrigin("https://preview.pooled.run", "pooled.run"), null, "a subdomain shares the site");
  eq(previewOrigin("https://pooled.run", "app.pooled.run"), null, "a parent");
  eq(previewOrigin("https://preview.pooled.run", "www.pooled.run"), null, "a sibling subdomain shares the site");
  eq(previewOrigin("https://pooled-preview.vercel.app", "pooled-dev.vercel.app"), "https://pooled-preview.vercel.app", "vercel.app names are separate sites");
  eq(previewOrigin("https://x.pooled-dev.vercel.app", "pooled-dev.vercel.app"), null, "a child of a vercel.app name");
  eq(previewOrigin("http://127.0.0.1:8080", "localhost", "http:"), "http://127.0.0.1:8080", "http on loopback is fine from an http page");
  eq(previewOrigin("http://127.0.0.1:8080", "pooled.run", "https:"), null, "not from an https page");
});

Deno.test("siteOf: last two labels, three under a two-part suffix; IPs and single labels as they are", () => {
  eq(siteOf("preview.pooled.run"), "pooled.run");
  eq(siteOf("A.B.Pooled.Run."), "pooled.run");
  eq(siteOf("x.example.co.uk"), "example.co.uk");
  eq(siteOf("pooled-git-x.vercel.app"), "pooled-git-x.vercel.app");
  eq(siteOf("127.0.0.1"), "127.0.0.1");
  eq(siteOf("localhost"), "localhost");
});

const fakeDoc = (href, content) => ({
  defaultView: { location: new URL(href) },
  querySelector: (s) => (s === 'meta[name="preview-origin"]' && content != null ? { content } : null),
});

Deno.test("relayUrl: uses the meta's origin for the page's host, else loopback swap, else null", () => {
  eq(relayUrl(fakeDoc("https://pooled.run/room", LIST)), "https://pooled-preview.vercel.app/harness/preview-relay.html");
  eq(relayUrl(fakeDoc("https://pooled-dev.vercel.app/r/ABCD", LIST)), "https://pooled-preview-dev.vercel.app/harness/preview-relay.html");
  eq(relayUrl(fakeDoc("https://pooled-git-x.vercel.app/room", LIST)), null);
  eq(relayUrl(fakeDoc("https://pooled.run/room", "")), null);
  eq(relayUrl(fakeDoc("http://localhost:8080/p2p.html", "")), "http://127.0.0.1:8080/harness/preview-relay.html");
  eq(relayUrl(fakeDoc("http://127.0.0.1:8080/p2p.html", null)), "http://localhost:8080/harness/preview-relay.html");
  eq(relayUrl(null), null);
});

Deno.test("relayUrl: a refused entry for this host warns once and leaves local mode (loopback swap in development)", () => {
  const w = console.warn, seen = [];
  console.warn = (m) => seen.push(m);
  try {
    const bad = "pooled.run=https://preview.pooled.run localhost=http://192.168.1.5:8080";
    for (let i = 0; i < 3; i++) eq(relayUrl(fakeDoc("https://pooled.run/room", bad)), null, "a subdomain entry is refused");
    eq(relayUrl(fakeDoc("https://pooled-dev.vercel.app/room", bad)), null, "no entry for this host: no warning");
    eq(relayUrl(fakeDoc("http://localhost:8080/p2p.html", bad)), "http://127.0.0.1:8080/harness/preview-relay.html", "refused in development: the loopback relay");
  } finally { console.warn = w; }
  eq(seen.length, 2, "one warning per refused value: " + JSON.stringify(seen));
  ok(/preview-origin ignored: same site/.test(seen[0]), seen[0]);
  ok(/only for loopback/.test(seen[1]), seen[1]);
});

Deno.test("previewEntry and relayProblem agree with previewOrigin", () => {
  eq(previewEntry(LIST, "pooled.run"), "https://pooled-preview.vercel.app");
  eq(previewEntry(LIST, "other.example"), null);
  eq(relayProblem("https://pooled-preview.vercel.app", { protocol: "https:", hostname: "pooled.run" }), "");
  ok(/only for loopback/.test(relayProblem("http://192.168.1.5:8080", { protocol: "http:", hostname: "192.168.1.4" })));
  ok(/http\(s\)/.test(relayProblem("ftp://pooled-preview.example", { protocol: "https:", hostname: "pooled.run" })));
});

Deno.test("p2p.html: a filled preview-origin has an entry for pooled.run", async () => {
  const html = await Deno.readTextFile(new URL("../../p2p.html", import.meta.url));
  const m = /<meta name="preview-origin" content="([^"]*)">/.exec(html);
  ok(m, "meta present");
  // a filled value must parse for pooled.run, or production silently runs previews in the room tab
  if (m[1].trim()) ok(previewOrigin(m[1], "pooled.run"), "preview-origin has no usable entry for pooled.run: " + m[1]);
});

const relayHeaders = async (file) => {
  const cfg = JSON.parse(await Deno.readTextFile(new URL("../../" + file, import.meta.url)));
  const all = cfg.headers.filter((h) => new RegExp("^" + h.source.replace(/\(\.\*\)/g, ".*") + "$").test("/harness/preview-relay.html"));
  return { cfg, all: all.flatMap((h) => h.headers), own: cfg.headers.find((h) => h.source === "/harness/preview-relay.html")?.headers };
};

Deno.test("relay headers: the same strict CSP in both Vercel configs, frameable from pooled.run and staging", async () => {
  const main = await relayHeaders("vercel.json"), host = await relayHeaders("preview-host/vercel.json");
  ok(main.own && host.own, "both configs have a rule for the relay");
  eq(main.own, host.own, "vercel.json and preview-host/vercel.json agree");
  const csp = Object.fromEntries(host.own.map((h) => [h.key, h.value]))["Content-Security-Policy"];
  ok(csp, "a CSP");
  // the app's document is a srcdoc child of the relay and inherits this policy: it must be the
  // document's own CSP (harness/preview-build.js) plus only what concerns the relay itself
  ok(csp.startsWith(CSP + ";"), "starts with the preview document's CSP");
  const fa = /frame-ancestors ([^;]*)/.exec(csp)?.[1].split(" ") || [];
  for (const o of ["https://pooled.run", "https://pooled-dev.vercel.app"]) ok(fa.includes(o), "frame-ancestors lets " + o + " frame the relay");
  ok(!fa.includes("*") && !fa.includes("https:"), "frame-ancestors is not open to any site");
  for (const { all } of [main, host]) {
    ok(!all.some((h) => /^x-frame-options$/i.test(h.key)), "no X-Frame-Options on the relay (it must be framed from another site)");
    ok(all.filter((h) => /^content-security-policy$/i.test(h.key)).length === 1, "exactly one CSP applies to the relay");
  }
});

Deno.test("preview-host builds only the relay", async () => {
  const { cfg } = await relayHeaders("preview-host/vercel.json");
  ok(/cp \.\.\/harness\/preview-relay\.html public\/harness\//.test(cfg.buildCommand), cfg.buildCommand);
  eq(cfg.outputDirectory, "public");
  ok(!cfg.rewrites && !cfg.redirects, "nothing else is routed");
});
