// test-only: resolve "openclaw/..." imports to the local OpenClaw install (inside OpenClaw the host provides them)
import { register } from "node:module";
register("data:text/javascript," + encodeURIComponent(`
import { createRequire } from "node:module";
const req = createRequire(${JSON.stringify(process.env.OC_ROOT + "/package.json")});
export async function resolve(spec, ctx, next) {
  if (spec === "openclaw" || spec.startsWith("openclaw/")) {
    const pkg = JSON.parse(await import("node:fs").then((fs) => fs.readFileSync(${JSON.stringify(process.env.OC_ROOT + "/node_modules/openclaw/package.json")}, "utf8")));
    const sub = "." + spec.slice("openclaw".length);
    const e = pkg.exports[sub]; const f = typeof e === "string" ? e : e.default || e.import;
    return next(${JSON.stringify("file://" + process.env.OC_ROOT + "/node_modules/openclaw/")} + f.replace(/^\\.\\//, ""), ctx);
  }
  return next(spec, ctx);
}`));
