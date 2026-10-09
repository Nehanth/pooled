#!/usr/bin/env node
// Release check only: self-hosted and development rooms may still use local previews.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { previewEntry, previewOrigin, relayProblem } from "../harness/preview-frame.js";

const attrsOf = (tag) => new Map([...tag.matchAll(/([^\s=<>/'"]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)]
  .map((m) => [m[1].toLowerCase(), m[2] ?? m[3] ?? m[4] ?? ""]));
const decode = (value) => value.replace(/&#(x[\da-f]+|\d+);|&(quot|apos|amp|lt|gt);/gi, (entity, number, name) => {
  if (number) {
    const n = number[0].toLowerCase() === "x" ? parseInt(number.slice(1), 16) : Number(number);
    return n > 0 && n <= 0x10ffff ? String.fromCodePoint(n) : entity;
  }
  return { quot: '"', apos: "'", amp: "&", lt: "<", gt: ">" }[name.toLowerCase()];
});
const hostnameOf = (host) => {
  if (!/^[a-z\d](?:[a-z\d-]*[a-z\d])?(?:\.[a-z\d](?:[a-z\d-]*[a-z\d])?)*$/i.test(host)) return null;
  try {
    const u = new URL("https://" + host);
    return u.hostname.toLowerCase() === host.toLowerCase() && !u.port && !u.username && !u.password
      && u.pathname === "/" && !u.search && !u.hash ? u.hostname : null;
  } catch { return null; }
};
const isLoopback = (host) => host === "localhost" || host === "127.0.0.1";

// Pure configuration validation; no network request and no DOM required.
// -> { origin: string | null, problems: string[] }
export function validatePreviewOrigin({ html, hostname = "pooled.run", relayConfig }) {
  const problems = [];
  const host = hostnameOf(String(hostname));
  if (!host) return { origin: null, problems: ["target must be a hostname (for example pooled.run), without a scheme, path or port"] };

  const tags = String(html ?? "").replace(/<!--[\s\S]*?-->/g, "")
    .match(/<meta\b(?:[^>"']|"[^"]*"|'[^']*')*>/gi) ?? [];
  const metas = tags.map(attrsOf).filter((attrs) => attrs.get("name") === "preview-origin");
  if (metas.length !== 1) return { origin: null, problems: ["room HTML must contain exactly one <meta name=\"preview-origin\" content=\"...\">"] };
  const content = decode(metas[0].get("content") ?? "").trim();
  if (!content) return { origin: null, problems: ["preview-origin is empty: deploy preview-host on a separate site and set the room's meta before a production release"] };

  const csp = relayConfig?.headers?.find((rule) => rule.source === "/harness/preview-relay.html")
    ?.headers?.find((header) => header.key.toLowerCase() === "content-security-policy")?.value ?? "";
  const ancestors = /(?:^|;)\s*frame-ancestors\s+([^;]*)/i.exec(csp)?.[1].trim().split(/\s+/) ?? [];
  const canFrame = (h, protocol = "https:") => ancestors.includes(`${protocol}//${h}`)
    || (isLoopback(h) && ancestors.includes(`${protocol}//${h}:*`));
  if (!canFrame(host)) problems.push(`relay Content-Security-Policy must include https://${host} in frame-ancestors (preview-host/vercel.json)`);

  // Match the runtime's whitespace/comma list and exact-host precedence. Reject ambiguous
  // duplicates and bad mappings now rather than silently choosing an earlier entry at runtime.
  const seen = new Set();
  for (const token of content.split(/[\s,]+/)) {
    const i = token.indexOf("=");
    const key = i < 0 ? "*" : token.slice(0, i).toLowerCase();
    const value = i < 0 ? token : token.slice(i + 1);
    if (seen.has(key)) problems.push(`duplicate preview-origin entry for ${key === "*" ? "the default origin" : key}: keep one entry`);
    seen.add(key);
    const mappedHost = i < 0 ? host : hostnameOf(key);
    if (!mappedHost) { problems.push(`invalid preview-origin host ${key}: use an exact hostname without a scheme, path or port`); continue; }
    const protocol = isLoopback(mappedHost) && value.startsWith("http:") ? "http:" : "https:";
    const why = value ? relayProblem(value, { hostname: mappedHost, protocol }) : "empty origin";
    if (why) problems.push(`preview-origin entry for ${mappedHost}: ${why}; use an HTTPS origin on a separate site`);
    if (i >= 0 && mappedHost !== host && !canFrame(mappedHost, protocol))
      problems.push(`preview-origin maps ${mappedHost}, but the relay's frame-ancestors does not allow ${protocol}//${mappedHost}`);
  }

  const entry = previewEntry(content, host);
  if (!entry) problems.push(`preview-origin has no entry for ${host}: add ${host}=https://<deployed-preview-host>`);
  const origin = previewOrigin(content, host);
  return { origin, problems };
}

if (import.meta.main ?? (typeof Deno === "undefined" && process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url)) {
  const args = process.argv.slice(2);
  if (args[0] === "--help" || args.length > 3) {
    console.log("usage: node scripts/check-preview-origin.mjs [room-html=p2p.html] [hostname=pooled.run] [relay-config=preview-host/vercel.json]");
    process.exit(args[0] === "--help" ? 0 : 2);
  }
  const [file = "p2p.html", hostname = "pooled.run", config = "preview-host/vercel.json"] = args;
  try {
    const result = validatePreviewOrigin({ html: readFileSync(file, "utf8"), hostname, relayConfig: JSON.parse(readFileSync(config, "utf8")) });
    for (const problem of result.problems) console.error(`preview configuration: ${problem}`);
    if (result.problems.length) process.exitCode = 1;
    else console.log(`${file}: ${hostname} uses ${result.origin}/harness/preview-relay.html; relay CSP allows the room`);
  } catch (error) {
    console.error(`preview configuration: ${error.message}`);
    process.exitCode = 1;
  }
}
