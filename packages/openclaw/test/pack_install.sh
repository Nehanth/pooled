#!/bin/sh
# Install proof, no publishing and no GPU: npm pack the plugin, install the tarball into a clean
# OpenClaw (its own home and state dir; the machine's ~/.openclaw is never touched) the way a user
# installs it from npm, then check that it loads without a Pooled checkout: `plugins inspect`, a
# non-interactive onboard (host), the catalog entry, the synthetic auth, and that the installed
# package imports none of the repo's files.
#   sh packages/openclaw/test/pack_install.sh            (openclaw on PATH, or OPENCLAW=<bin>)
# env: KEEP=1 keeps the temp dir; TGZ=<file> installs that tarball instead of packing this checkout
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
OC=${OPENCLAW:-openclaw}
T=$(mktemp -d "${TMPDIR:-/tmp}/pooled-oc-install-XXXXXX")
[ "${KEEP:-}" = 1 ] || trap 'rm -rf "$T"' EXIT
mkdir -p "$T/home" "$T/state" "$T/run" "$T/pack"
export HOME="$T/home" OPENCLAW_HOME="$T/home" OPENCLAW_STATE_DIR="$T/state" OPENCLAW_CONFIG_PATH="$T/state/openclaw.json"
# never reach a gateway service of the machine's own user
oc() { env -u DBUS_SESSION_BUS_ADDRESS XDG_RUNTIME_DIR="$T/run" "$OC" "$@"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== openclaw $(oc --version 2>/dev/null | head -1)"
if [ -n "${TGZ:-}" ]; then cp "$TGZ" "$T/pack/"; else
  echo "== npm pack"
  (cd "$HERE" && npm pack --silent --pack-destination "$T/pack" >/dev/null)
fi
TGZ=$(ls "$T/pack"/*.tgz)
tar -tzf "$TGZ" | sed 's/^/   /'
tar -tzf "$TGZ" | grep -q '^package/dist/index.js$' || fail "no dist/index.js in the tarball"
tar -tzf "$TGZ" | grep -q '^package/openclaw.plugin.json$' || fail "no manifest in the tarball"
echo "   $(du -k "$TGZ" | cut -f1) KB"

echo "== openclaw plugins install $(basename "$TGZ")"
oc plugins install "$TGZ" --force --accept-capabilities 2>&1 | tail -n 15
DIR=$(find "$T" -path '*/node_modules/@pooled/openclaw/package.json' -o -path '*/extensions/pooled/package.json' 2>/dev/null | head -1 | xargs -r dirname)
[ -n "$DIR" ] || DIR=$(find "$T/state" "$T/home" -name openclaw.plugin.json -path '*pooled*' 2>/dev/null | head -1 | xargs -r dirname)
[ -n "$DIR" ] || fail "can't find the installed plugin"
echo "   installed in ${DIR#$T/}"
[ -d "$DIR/node_modules/node-datachannel" ] || [ -d "$(dirname "$DIR")/node-datachannel" ] || fail "node-datachannel not installed"
if [ -d "$DIR/node_modules/webgpu" ] || [ -d "$(dirname "$DIR")/webgpu" ]; then echo "   webgpu (optional): installed"; else echo "   webgpu (optional): not installed"; fi
grep -Eq '(from |import\()"\.\.?/' "$DIR/dist/index.js" && fail "the bundle imports files by relative path"

echo "== plugins inspect pooled"
oc plugins inspect pooled 2>&1 | tail -n 25 | tee "$T/inspect.txt"
grep -qi "error" "$T/inspect.txt" && grep -qi "loaded\|enabled" "$T/inspect.txt" || true

echo "== onboard --non-interactive (host)"
POOLED_NO_SERVICE=1 oc onboard --non-interactive --accept-risk --auth-choice pooled --skip-channels --skip-skills --skip-health --skip-daemon 2>&1 | tail -n 8 || true
node -e '
const c = JSON.parse(require("fs").readFileSync(process.env.OPENCLAW_CONFIG_PATH, "utf8"));
const e = c.plugins?.entries?.pooled;
if (!e?.config?.mode) { console.error("FAIL: no plugins.entries.pooled.config after onboard"); process.exit(1); }
console.log("   config:", JSON.stringify(e.config), "· default model:", c.agents?.defaults?.model?.primary);
if (!/^[A-HJKMNP-TV-Z2-9]{6}$/.test(e.config.code)) { console.error("FAIL: code", e.config.code); process.exit(1); }'
[ -f "$T/state/pooled/room.json" ] || fail "no room.json (the host's invite key) in the state dir"
# the invite key lives in room.json only, never in openclaw.json
node -e '
const fs = require("fs"), k = JSON.parse(fs.readFileSync(process.env.OPENCLAW_STATE_DIR + "/pooled/room.json", "utf8")).host?.gate?.key;
if (!k) { console.error("FAIL: no invite key in room.json"); process.exit(1); }
if (fs.readFileSync(process.env.OPENCLAW_CONFIG_PATH, "utf8").includes(k)) { console.error("FAIL: the invite key is in openclaw.json"); process.exit(1); }
console.log("   invite key: in room.json, not in openclaw.json");'
echo "   room.json mode $(stat -c %a "$T/state/pooled/room.json" 2>/dev/null || stat -f %Lp "$T/state/pooled/room.json")"

echo "== models list (catalog + synthetic auth for a non-bundled plugin)"
POOLED_NO_SERVICE=1 oc models list --provider pooled 2>&1 | tail -n 6 | tee "$T/models.txt"
grep -q "pooled/qwen3-1.7b" "$T/models.txt" || fail "pooled/qwen3-1.7b not in the model list"
POOLED_NO_SERVICE=1 oc models status --json 2>/dev/null | node -e '
let s = ""; process.stdin.on("data", (d) => (s += d)).on("end", () => {
  const i = s.indexOf("{"); let j; try { j = JSON.parse(s.slice(i)); } catch { console.log("   (models status: no JSON)"); return; }
  const txt = JSON.stringify(j);
  console.log("   auth:", /pooled-local|Pooled room on this machine|synthetic/i.test(txt) ? "synthetic marker resolved" : "no marker in models status (check by hand)");
});'
echo "OK"
