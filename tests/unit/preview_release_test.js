import { validatePreviewOrigin } from "../../scripts/check-preview-origin.mjs";

const eq = (a, b, message) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error((message || "mismatch") + ": " + JSON.stringify(a) + " != " + JSON.stringify(b)); };
const ok = (value, message) => { if (!value) throw new Error(message || "assertion failed"); };
const html = (content) => `<meta name="preview-origin" content="${content}">`;
const config = (hosts = "https://pooled.run https://pooled-dev.vercel.app http://localhost:* http://127.0.0.1:*") => ({
  headers: [{ source: "/harness/preview-relay.html", headers: [{ key: "Content-Security-Policy", value: `default-src 'none'; frame-ancestors ${hosts}` }] }],
});
const check = (content, options = {}) => validatePreviewOrigin({ html: html(content), relayConfig: config(), ...options });

Deno.test("preview release: an empty or missing production meta fails with deployment instructions", () => {
  for (const content of ["", "   "]) {
    const result = check(content);
    eq(result.origin, null);
    ok(result.problems.some((p) => /empty.*deploy preview-host/.test(p)));
  }
  ok(validatePreviewOrigin({ html: "<title>room</title>", relayConfig: config() }).problems.some((p) => /exactly one/.test(p)));
});

Deno.test("preview release: separate HTTPS site and permitted room host pass", () => {
  eq(check("https://pooled-preview.vercel.app"), { origin: "https://pooled-preview.vercel.app", problems: [] });
  eq(check("pooled.run=https://pooled-preview.vercel.app pooled-dev.vercel.app=https://pooled-preview-dev.vercel.app").problems, []);
  eq(check("https://fallback.example,pooled.run=https://pooled-preview.vercel.app").origin, "https://pooled-preview.vercel.app");
  eq(check("pooled.run=https://pooled-preview.vercel.app localhost=http://127.0.0.1:8080").problems, []);
});

Deno.test("preview release: HTML attributes and host matching follow the room configuration", () => {
  const result = check("unused", {
    hostname: "POOLED.RUN",
    html: "<!-- <meta name='preview-origin' content=''> --><meta content='pooled.run=https://pooled-preview.vercel.app&#47;' name='preview-origin' />",
  });
  eq(result, { origin: "https://pooled-preview.vercel.app", problems: [] });
  ok(check("https://pooled-preview.vercel.app", { html: html("https://pooled-preview.vercel.app") + html("https://other.example") }).problems.some((p) => /exactly one/.test(p)));
});

Deno.test("preview release: same site, malformed URLs and insecure origins are rejected", () => {
  for (const content of ["https://pooled.run", "https://preview.pooled.run", "pooled.run=https://preview.pooled.run"]) {
    const result = check(content);
    eq(result.origin, null);
    ok(result.problems.some((p) => /same site/.test(p)));
  }
  for (const content of ["http://preview.example", "ftp://preview.example", "not-a-url", "pooled.run="]) {
    const result = check(content);
    eq(result.origin, null);
    ok(result.problems.length > 0, content);
  }
});

Deno.test("preview release: per-host mappings must cover production and match relay permissions", () => {
  ok(check("pooled-dev.vercel.app=https://pooled-preview-dev.vercel.app").problems.some((p) => /no entry for pooled.run/.test(p)));
  const unknown = check("pooled.run=https://pooled-preview.vercel.app typo.pooled.run=https://other.example");
  ok(unknown.problems.some((p) => /maps typo.pooled.run.*does not allow/.test(p)));
  ok(check("https://pooled-preview.vercel.app", { relayConfig: config("https://pooled-dev.vercel.app") }).problems.some((p) => /include https:\/\/pooled.run/.test(p)));
  ok(check("https://pooled-preview.vercel.app", { relayConfig: {} }).problems.some((p) => /frame-ancestors/.test(p)));
});

Deno.test("preview release: ambiguous duplicate mappings and invalid host keys fail", () => {
  for (const content of ["pooled.run=https://preview.example POOLED.RUN=https://other.example", "https://preview.example https://other.example"])
    ok(check(content).problems.some((p) => /duplicate/.test(p)));
  for (const content of ["https://pooled.run=https://preview.example", "pooled.run:443=https://preview.example", "*.pooled.run=https://preview.example"])
    ok(check(content).problems.some((p) => /invalid preview-origin host/.test(p)), content);
  ok(check("https://preview.example", { hostname: "https://pooled.run" }).problems.some((p) => /target must be a hostname/.test(p)));
});

Deno.test("preview release: a custom production hostname needs its own isolated origin and relay permission", () => {
  const options = { hostname: "room.example.co.uk", relayConfig: config("https://room.example.co.uk") };
  eq(check("https://isolated.example.net", options).problems, []);
  ok(check("https://preview.example.co.uk", options).problems.some((p) => /same site/.test(p)));
});
