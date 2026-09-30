// The sandbox project the OpenClaw agent tasks run in (runtasks.mjs): a small Node package with two
// planted bugs (subtotal ignores qty, formatMoney breaks on negative amounts) and a ~140 KB operations
// journal (~33k tokens) with five [MAJOR] incidents spread through it for the long-context task.
//   node sandbox.mjs <dir>
import fs from "node:fs";
import path from "node:path";

const FILES = {
 "package.json": "{ \"name\": \"shopcalc\", \"version\": \"0.3.0\", \"type\": \"module\", \"private\": true,\n  \"scripts\": { \"test\": \"node --test\" } }\n",
 "README.md": "# shopcalc\n\nSmall pricing helpers for the checkout service: cart totals, money formatting and an HTTP client\nfor the tax-rate service. Run the tests with `npm test` (Node's built-in test runner).\n",
 "src/cart.js": "// Cart math. Prices are in cents (integers).\nexport function subtotal(items) {\n  // items: [{ sku, price, qty }]\n  let sum = 0;\n  for (const it of items) sum += it.price;\n  return sum;\n}\n\nexport function applyDiscount(cents, percent) {\n  if (percent < 0 || percent > 100) throw new RangeError(\"percent must be 0-100\");\n  return Math.round(cents * (100 - percent) / 100);\n}\n\nexport function total(items, { discountPercent = 0, taxRate = 0 } = {}) {\n  const afterDiscount = applyDiscount(subtotal(items), discountPercent);\n  return Math.round(afterDiscount * (1 + taxRate));\n}\n",
 "src/format.js": "// Money formatting for receipts.\nexport function formatMoney(cents, currency = \"$\") {\n  const whole = Math.floor(cents / 100);\n  const frac = String(cents % 100).padStart(2, \"0\");\n  return `${currency}${whole}.${frac}`;\n}\n",
 "src/taxclient.js": "// HTTP client for the tax-rate service.\nimport { setTimeout as sleep } from \"node:timers/promises\";\n\nconst DEFAULTS = { baseUrl: \"http://tax.internal:8080\", timeoutMs: 2500 };\n\nexport class TaxClient {\n  constructor(opts = {}) {\n    this.opts = { ...DEFAULTS, ...opts };\n    this.fetch = opts.fetch || globalThis.fetch;\n  }\n\n  async rateFor(region) {\n    const url = `${this.opts.baseUrl}/v1/rates/${encodeURIComponent(region)}`;\n    const res = await withRetries(() => this.fetch(url, { signal: AbortSignal.timeout(this.opts.timeoutMs) }));\n    if (!res.ok) throw new Error(`tax service answered ${res.status}`);\n    const body = await res.json();\n    return body.rate;\n  }\n}\n\n// Retries a failed call with exponential backoff: 200 ms, 400 ms, 800 ms, ...\nexport async function withRetries(fn, attempts = RETRY_LIMIT) {\n  let lastErr;\n  for (let i = 0; i < attempts; i++) {\n    try { return await fn(); } catch (err) { lastErr = err; await sleep(200 * 2 ** i); }\n  }\n  throw lastErr;\n}\n\nimport { RETRY_LIMIT } from \"./config.js\";\n",
 "src/config.js": "// Service settings. Keep in sync with deploy/values.yaml.\nexport const RETRY_LIMIT = 4;\nexport const DEFAULT_REGION = \"US-CA\";\nexport const CURRENCY = \"$\";\n",
 "test/cart.test.js": "import { test } from \"node:test\";\nimport assert from \"node:assert/strict\";\nimport { subtotal, applyDiscount, total } from \"../src/cart.js\";\n\nconst items = [{ sku: \"A\", price: 250, qty: 2 }, { sku: \"B\", price: 1000, qty: 1 }];\n\ntest(\"subtotal counts quantities\", () => {\n  assert.equal(subtotal(items), 1500);\n});\ntest(\"applyDiscount\", () => {\n  assert.equal(applyDiscount(1000, 10), 900);\n  assert.throws(() => applyDiscount(1000, 120), RangeError);\n});\ntest(\"total with discount and tax\", () => {\n  assert.equal(total(items, { discountPercent: 10, taxRate: 0.1 }), 1485);\n});\n",
 "test/format.test.js": "import { test } from \"node:test\";\nimport assert from \"node:assert/strict\";\nimport { formatMoney } from \"../src/format.js\";\n\ntest(\"formatMoney positive\", () => {\n  assert.equal(formatMoney(1234), \"$12.34\");\n  assert.equal(formatMoney(5), \"$0.05\");\n});\n"
};

// deterministic filler with the five planted incidents (fixed seed: the same file every time)
function journal() {
  let s = 12345; const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const svcs = ["checkout-api", "tax-rate service", "cart-cache", "receipt-renderer", "inventory-sync", "payments-gateway", "search-indexer", "auth-proxy"];
  const people = ["Priya", "Marco", "Lena", "Tomasz", "Aiko", "Dev", "Sam", "Ruth", "Omar", "Jules"];
  const acts = ["rotated the TLS certificates on", "bumped the memory limit of", "rolled out a new build of", "tuned the connection pool of", "added a dashboard panel for", "cleaned up stale feature flags in", "reviewed the alert thresholds for", "migrated the staging database behind", "ran the quarterly failover drill for", "patched the base image of"];
  const obs = ["p99 latency stayed flat at around %d ms", "error rate was below %d basis points all day", "the queue depth peaked at %d messages during the lunch rush", "CPU hovered near %d percent on the busiest pod", "the cache hit ratio settled at %d percent", "we saw %d retries against the upstream, all of them successful", "disk usage grew by %d GB, in line with the forecast"];
  const notes = ["No customer impact.", "Nothing to follow up.", "Ticket filed for the next sprint.", "Runbook updated accordingly.", "Paged nobody; handled in business hours.", "Left a note in the handover doc.", "Will revisit after the freeze."];
  const majors = [
    ["2026-01-14", "checkout-api returned HTTP 500 for 38 minutes", "an expired intermediate certificate on the tax-rate service; the client had no retry on TLS errors"],
    ["2026-03-02", "double charges on 212 orders", "a payments-gateway timeout of 2 s shorter than the provider's 5 s processing time, so the client retried a charge that had already succeeded"],
    ["2026-05-19", "cart totals were off by one cent for 4 hours", "floating-point rounding in the discount step after a refactor replaced integer cents with dollars"],
    ["2026-07-30", "search returned empty results in the EU region for 2 hours", "the search-indexer ran out of disk because log rotation was disabled by a config typo"],
    ["2026-09-11", "all receipts showed the wrong currency symbol for 55 minutes", "a feature flag default flipped during the flags cleanup, falling back to CURRENCY = '€' for every region"],
  ];
  const days = 66, out = ["# Operations journal: checkout platform, 2026", "", "Daily notes from the on-call engineers. Most days are routine. Incidents that affected customers are marked [MAJOR] and have a root cause line.", ""];
  const at = [2, 18, 34, 50, 64];
  const d0 = Date.UTC(2026, 0, 10);
  for (let i = 0; i < days; i++) {
    const date = new Date(d0 + i * 86400000 * 1.75).toISOString().slice(0, 10);
    const k = at.indexOf(i);
    out.push(`## ${k >= 0 ? majors[k][0] : date} (on call: ${pick(people)})`, "");
    const n = 3 + Math.floor(rnd() * 3);
    for (let j = 0; j < n; j++) {
      const p = [];
      for (let m = 0; m < 3 + Math.floor(rnd() * 3); m++)
        p.push(`${pick(people)} ${pick(acts)} the ${pick(svcs)}; ${pick(obs).replace("%d", 2 + Math.floor(rnd() * 400))}. ${pick(notes)}`);
      out.push(p.join(" "), "");
    }
    if (k >= 0) out.push(`**[MAJOR] Incident:** ${majors[k][1]}.`, `**Root cause:** ${majors[k][2]}.`, `**Fix:** see the follow-up items in the postmortem for ${majors[k][0]}.`, "");
  }
  return out.join("\n");
  
}

const dir = process.argv[2];
if (!dir) { console.error("usage: node sandbox.mjs <dir>"); process.exit(2); }
for (const [f, s] of Object.entries(FILES)) { fs.mkdirSync(path.dirname(path.join(dir, f)), { recursive: true }); fs.writeFileSync(path.join(dir, f), s); }
fs.mkdirSync(path.join(dir, "docs"), { recursive: true });
fs.writeFileSync(path.join(dir, "docs/ops-journal.md"), journal());
