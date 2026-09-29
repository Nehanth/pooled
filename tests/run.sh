#!/usr/bin/env bash
# GPU test runner. Usage: tests/run.sh [quick|q38|q38once|all|extra|selftest]
# quick: small-model goldens (Qwen3 0.6B / SmolLM); q38: 27B suites, one process per file;
# q38once: the same 27B checks in one process sharing one upload (run_q38_once.js); all: quick + q38.
# extra: GPU tests no other suite covers (MoE, sharding, splits, the full 27B); not part of all.
# selftest: no GPU; runs tests/fixtures/run_sh/ to prove a printed FAIL (or a nonzero exit) fails the run.
# A test fails when it exits nonzero or prints a FAIL/FAILED word, even if it exits 0.
# STRICT=1 also fails a test that prints a SKIP/SKIPPED word.
# The 27B tests load through the converted-weights cache (tests/weight_cache.js, docs/testing-fast.md):
# WEIGHT_CACHE=0 disables it, WEIGHT_CACHE=<dir> moves it.
set -uo pipefail
cd "$(dirname "$0")"
WC="${WEIGHT_CACHE:-$HOME/.cache/swarmllm-weights}"
D="deno run --unstable-webgpu --allow-read --allow-env --allow-write=$WC --allow-net"   # net: test_shard fetches from Hugging Face
quick=(test_selftest.js test_qwen.js test_smollm.js test_stream.js test_batch.js test_reset.js test_qwen_split.js test_qwen_stream.js test_batch_split.js)
extra=(test_moe.js test_shard.js test_split.js test_fuse_proj.js test_q17_split.js test_qwen4.js test_stream_engine.js test_q38_full.js)
q38=(test_q38.js test_batch_q38.js test_mtp.js test_b4.js test_twins.js test_gemm.js test_q38_split.js test_mtp_split.js test_ctx.js)

# run_one <file> [tail lines]: runs one test, prints the last lines of its output, returns 1 on a
# nonzero exit, a FAIL line, or (STRICT=1) a SKIP line. The full output is checked, not just the tail.
run_one() {
  local out code tmp
  tmp=$(mktemp "${TMPDIR:-/tmp}/pooled-run.XXXXXX")
  if [ "${2:-4}" = all ]; then $D "$1" 2>&1 | grep -v "^TU:\|^MESA" | tee "$tmp"; code=${PIPESTATUS[0]}   # streams
  else $D "$1" 2>&1 | grep -v "^TU:\|^MESA" >"$tmp"; code=${PIPESTATUS[0]}; tail -n "${2:-4}" "$tmp"; fi
  out=$(cat "$tmp"); rm -f "$tmp"
  if [ "$code" -ne 0 ]; then echo "--- $1 exited $code"; return 1; fi
  if grep -qE '(^|[^A-Za-z])FAIL(ED)?([^A-Za-z]|$)' <<<"$out"; then echo "--- $1 printed FAIL but exited 0"; return 1; fi
  if [ "${STRICT:-0}" = 1 ] && grep -qE '(^|[^A-Za-z])SKIP(PED)?([^A-Za-z]|$)' <<<"$out"; then echo "--- $1 printed SKIP (STRICT=1)"; return 1; fi
  return 0
}

# selftest: each fixture's name says what run_one must return (pass_* 0, fail_* 1, skip_* 0 or 1 under STRICT=1)
selftest() {
  local bad=0 f want got
  for f in fixtures/run_sh/*.js; do
    case "$(basename "$f")" in pass_*) want=0;; fail_*) want=1;; skip_*) want=$([ "${STRICT:-0}" = 1 ] && echo 1 || echo 0);; *) continue;; esac
    run_one "$f" 0 >/dev/null; got=$?
    if [ "$got" -eq "$want" ]; then echo "ok   $f -> $got"; else echo "BAD  $f -> $got, expected $want"; bad=1; fi
  done
  [ "$bad" -eq 0 ] && echo "RUN.SH SELFTEST PASS" || echo "RUN.SH SELFTEST BROKEN"
  return $bad
}

case "${1:-quick}" in quick) list=("${quick[@]}");; q38) list=("${q38[@]}");; all) list=("${quick[@]}" "${q38[@]}");; extra) list=("${extra[@]}");;
  q38once) run_one run_q38_once.js all; exit $?;;
  selftest) selftest; exit $?;;
  *) echo "unknown suite"; exit 2;; esac
fail=0
for t in "${list[@]}"; do
  echo "=== $t"
  run_one "$t" || fail=1
done
exit $fail
