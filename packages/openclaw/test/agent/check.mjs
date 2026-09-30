// The checks of runtasks.mjs's tasks (the agent never sees them): node check.mjs <task> <projectDir> [answer text file]
import fs from "node:fs"; import path from "node:path"; import { execSync } from "node:child_process";
const [task, dir, ansFile] = process.argv.slice(2);
const ans = ansFile && fs.existsSync(ansFile) ? fs.readFileSync(ansFile, "utf8") : "";
const imp = async (f) => import(path.join(dir, f) + "?t=" + Date.now());
const res = { task, ok: false, detail: [] };
const ok = (c, m) => { res.detail.push((c ? "PASS " : "FAIL ") + m); return c; };
try {
  if (task === "read") {
    res.ok = [ok(/\b4\b/.test(ans), "says 4"), ok(/config\.js/.test(ans), "names src/config.js"), ok(/withRetries/.test(ans), "names withRetries"), ok(/taxclient/i.test(ans), "names taxclient.js")].every(Boolean);
  } else if (task === "edit") {
    const { formatMoney } = await imp("src/format.js");
    const cases = [[1234, "$12.34"], [5, "$0.05"], [-1234, "-$12.34"], [-5, "-$0.05"], [-100, "-$1.00"], [0, "$0.00"], [100000, "$1000.00"]];
    res.ok = cases.map(([c, w]) => { let g; try { g = formatMoney(c); } catch (e) { g = "throw " + e.message; } return ok(g === w, `formatMoney(${c}) = ${JSON.stringify(g)} want ${w}`); }).every(Boolean);
  } else if (task === "tests") {
    const { subtotal, total } = await imp("src/cart.js");
    const items = [{ sku: "A", price: 250, qty: 2 }, { sku: "B", price: 1000, qty: 1 }];
    const a = ok(subtotal(items) === 1500, "subtotal 1500"), b = ok(total(items, { discountPercent: 10, taxRate: 0.1 }) === 1485, "total 1485");
    const t = fs.readFileSync(path.join(dir, "test/cart.test.js"), "utf8");
    const c = ok(/1500/.test(t) && /1485/.test(t), "tests not weakened");
    let d = false; try { execSync("node --test", { cwd: dir, stdio: "pipe" }); d = true; } catch {} ok(d, "npm test passes");
    res.ok = a && b && c && d;
  } else if (task === "feature") {
    const { bulkPrice } = await imp("src/pricing.js");
    const cases = [[100, 1, 100], [100, 9, 900], [100, 10, 900], [100, 49, 4410], [100, 50, 4000], [199, 10, 1791], [250, 60, 12000]];
    const a = cases.map(([u, q, w]) => { const g = bulkPrice(u, q); return ok(g === w, `bulkPrice(${u},${q}) = ${g} want ${w}`); }).every(Boolean);
    const b = ok(fs.existsSync(path.join(dir, "test/pricing.test.js")), "test/pricing.test.js exists");
    let d = false; try { const o = execSync("node --test", { cwd: dir, stdio: "pipe" }).toString(); d = /pricing|bulk/i.test(o) || b; } catch {} ok(d, "npm test passes");
    res.ok = a && b && d;
  } else if (task === "long") {
    const f = path.join(dir, "SUMMARY.md"); const s = fs.existsSync(f) ? fs.readFileSync(f, "utf8") : "";
    ok(s.length > 0, "SUMMARY.md written");
    const want = [["2026-01-14", /certificate/i], ["2026-03-02", /timeout|double/i], ["2026-05-19", /round|float/i], ["2026-07-30", /disk|log rotation/i], ["2026-09-11", /flag|currency|€/i]];
    const hits = want.map(([d, re]) => ok(s.includes(d) && re.test(s), `incident ${d}`));
    res.found = hits.filter(Boolean).length; res.ok = res.found === 5;
  }
} catch (e) { res.detail.push("ERROR " + e.message); }
console.log(JSON.stringify(res));
