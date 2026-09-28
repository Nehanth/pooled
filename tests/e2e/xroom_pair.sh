#!/bin/bash
# Two-machine room run (tests/e2e/xroom.mjs) driven from one machine: copy this checkout to the other
# machine, take the GPU locks, start the room host, read its code, start the guest, wait for the
# host's JSON, fetch the guest's trace, release the locks. Manual trigger only.
#
#   tests/e2e/xroom_pair.sh [--here host|guest] [--out DIR] [--no-sync] -- <xroom.mjs host args>
#   e.g. tests/e2e/xroom_pair.sh --out /tmp/xr -- --model qwen3.8-27b --prompts japan,twosum --rounds 3 --trace-rounds 2,5,8,11
#
# --here host (default): the room host runs on this machine, the guest on the other one.
# --here guest: the host (embedding, head, sampler) runs on the other machine. The signaling server
# always runs here, so the other machine only needs to reach this one.
# Everything after -- goes to the host's xroom.mjs (--model, --gb, --rounds, --prompts, --modes,
# --maxnew, --trace-rounds, --query ...); --guest-gb N sets the guest's pledge (default 12).
#
# The machine-specific parts come from the environment (nothing machine-specific lives in the repo):
#   XROOM_REMOTE       required. Runs one shell command on the other machine: $XROOM_REMOTE '<cmd>'
#                      (an ssh wrapper). Its stdout comes back here.
#   XROOM_SYNC         required unless --no-sync. Copies a checkout there: $XROOM_SYNC <dir> <remote dir name>
#                      (code only; the other machine brings its own models/ with the weights)
#   XROOM_REMOTE_DIR   the checkout's directory name on the other machine, under its home (default pooled-xroom)
#   XROOM_REMOTE_ENV   shell prefix there, e.g. 'export PATH=$HOME/.local/node/bin:$PATH;'
#   XROOM_REMOTE_PREP  shell run in that directory after the sync, e.g. linking node_modules to a
#                      directory with playwright ('[ -e node_modules ] || ln -s ~/tools/node_modules node_modules')
#   XROOM_LOCAL_IP     this machine's address as the other one sees it (the signaling server's)
#   XROOM_REMOTE_IP    the other machine's address for the ping (default: from its $SSH_CONNECTION)
#   XROOM_LOCK         optional. '$XROOM_LOCK acquire' / 'release': the other machine's GPU lock, held
#                      for the whole run and released on any exit. acquire must print LOCKED. If this
#                      script is killed while acquire is still waiting, a wait loop that runs on the
#                      other machine (over ssh) can outlive it and take the lock later: kill it there.
#   XROOM_GPURUN       optional. A wrapper the local side runs through (e.g. one that waits for this
#                      machine's GPU to be idle, then execs its arguments)
# Outputs in --out DIR: host.json (the host's result; with --trace-rounds also its traces), host.log,
# guest.log, guest-trace.json (tracing only), ping.txt (before / after), ping_during.txt (every
# 0.2 s through the run), and report.txt from xroom_report.mjs (the rounds; with tracing the splits).
set -u
HERE=host OUT="" SYNC=1 GUEST_GB=12
while [ $# -gt 0 ]; do
  case "$1" in
    --here) HERE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --guest-gb) GUEST_GB=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
HOST_ARGS=("$@")
: "${XROOM_REMOTE:?set XROOM_REMOTE (a command that runs a shell command on the other machine)}"
: "${XROOM_LOCAL_IP:?set XROOM_LOCAL_IP (the address of this machine as the other machine sees it)}"
RDIR=${XROOM_REMOTE_DIR:-pooled-xroom}
RENV=${XROOM_REMOTE_ENV:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=xroom-$$-$(date +%s)
OUT=${OUT:-$(mktemp -d -t xroom-out-XXXX)}
mkdir -p "$OUT"
SIG_PORT=9000
tracing=0; for a in "${HOST_ARGS[@]}"; do [ "$a" = "--trace-rounds" ] && tracing=1; done
remote() { $XROOM_REMOTE "$RENV $1"; }
log() { echo "[xroom_pair $(date +%T)] $*" >&2; }

LOCKED=0 PIDS=() SIGPID=""
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  [ -n "$SIGPID" ] && kill "$SIGPID" 2>/dev/null
  remote "pkill -f -- '--tag $RUN' 2>/dev/null; rm -f /tmp/$RUN-*" >/dev/null 2>&1
  if [ "$LOCKED" = 1 ] && [ -n "${XROOM_LOCK:-}" ]; then $XROOM_LOCK release >&2; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ "$SYNC" = 1 ]; then
  : "${XROOM_SYNC:?set XROOM_SYNC (copies a checkout to the other machine), or pass --no-sync}"
  log "sync $ROOT -> $RDIR"
  $XROOM_SYNC "$ROOT" "$RDIR" >&2 || { log "sync failed"; exit 1; }
fi
[ -n "${XROOM_REMOTE_PREP:-}" ] && remote "cd ~/$RDIR && $XROOM_REMOTE_PREP" >&2
if [ -n "${XROOM_LOCK:-}" ]; then
  log "taking the other machine's GPU lock"
  $XROOM_LOCK acquire | grep -q LOCKED || { log "no lock"; exit 1; }
  LOCKED=1
fi
REMOTE_IP=${XROOM_REMOTE_IP:-}
[ -z "$REMOTE_IP" ] && REMOTE_IP=$(remote 'echo $SSH_CONNECTION' | awk '{print $3}')
{ echo "# ping $REMOTE_IP before"; ping -c 20 -i 0.2 -q "$REMOTE_IP" 2>&1 | tail -2; } > "$OUT/ping.txt"
log "$(tail -1 "$OUT/ping.txt")"
GPURUN=${XROOM_GPURUN:-}
# the round trip all through the run (ping -D: unix time per reply), so each round can be read
# against what the network was doing at the time (xroom_report.mjs --ping-log)
ping -D -i 0.2 "$REMOTE_IP" > "$OUT/ping_during.txt" 2>&1 &
PIDS+=("$!")

wait_code() {   # wait_code <file> <pid> <seconds>: the host's "CODE XXXX" line, while <pid> lives
  for _ in $(seq 1 "$3"); do
    c=$(grep -m1 -o '^CODE [A-Z0-9]*' "$1" 2>/dev/null | cut -d' ' -f2)
    [ -n "$c" ] && { echo "$c"; return 0; }
    kill -0 "$2" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
wait_host() {   # the host's exit code in RC; a guest that ends first is a failure (the host would wait for it)
  while kill -0 "$HPID" 2>/dev/null; do
    if ! kill -0 "$GPID" 2>/dev/null; then log "the guest ended before the host"; kill "$HPID" 2>/dev/null; remote "pkill -f -- '--tag $RUN'" >/dev/null 2>&1; fi
    sleep 2
  done
  wait "$HPID"; RC=$?
}
# XROOM_GPURUN may give up (a wrapper with a timeout prints something and runs nothing): with the
# other machine's lock already held, keep waiting for this GPU (XROOM_GPURUN_TRIES, default 8)
# rather than drop the lock and queue for it again
gpu_wait() {
  [ -z "$GPURUN" ] && return 0
  for _ in $(seq 1 "${XROOM_GPURUN_TRIES:-8}"); do
    [ "$($GPURUN echo XROOM_GPU_FREE 2>&1 | tail -1)" = XROOM_GPU_FREE ] && return 0
    log "this GPU is busy; waiting again"
  done
  return 1
}
gpu_wait || { log "this GPU stayed busy"; exit 1; }
if [ "$HERE" = host ]; then
  # host here (it runs the signaling server), guest there
  $GPURUN node "$ROOT/tests/e2e/xroom.mjs" --role host --out "$OUT/host.json" "${HOST_ARGS[@]}" > "$OUT/host.out" 2> "$OUT/host.log" &
  HPID=$!; PIDS+=("$HPID")
  CODE=$(wait_code "$OUT/host.out" "$HPID" 1200) || { log "the host printed no code"; tail -5 "$OUT/host.log" >&2; exit 1; }
  log "room $CODE; starting the guest"
  TR=""; [ "$tracing" = 1 ] && TR="--trace-out /tmp/$RUN-guest.json"
  remote "cd ~/$RDIR && node tests/e2e/xroom.mjs --role guest --signal $XROOM_LOCAL_IP:$SIG_PORT --code $CODE --gb $GUEST_GB $TR --tag $RUN" > "$OUT/guest.out" 2> "$OUT/guest.log" &
  GPID=$!; PIDS+=("$GPID")
  wait_host
  log "host exited ($RC); waiting for the guest"
  for _ in $(seq 1 90); do kill -0 "$GPID" 2>/dev/null || break; sleep 1; done
  [ "$tracing" = 1 ] && remote "cat /tmp/$RUN-guest.json" > "$OUT/guest-trace.json"
else
  # host there, guest here; the signaling server stays here
  "$ROOT/node_modules/.bin/peerjs" --port "$SIG_PORT" --path / --host 0.0.0.0 > /dev/null 2>&1 &
  SIGPID=$!; sleep 1.5
  QARGS=$(printf ' %q' "${HOST_ARGS[@]}")
  remote "cd ~/$RDIR && node tests/e2e/xroom.mjs --role host --signal $XROOM_LOCAL_IP:$SIG_PORT --signal-server 0 --out /tmp/$RUN-host.json $QARGS --tag $RUN" > "$OUT/host.out" 2> "$OUT/host.log" &
  HPID=$!; PIDS+=("$HPID")
  CODE=$(wait_code "$OUT/host.out" "$HPID" 1200) || { log "the host printed no code"; tail -5 "$OUT/host.log" >&2; exit 1; }
  log "room $CODE; starting the guest here"
  TR=(); [ "$tracing" = 1 ] && TR=(--trace-out "$OUT/guest-trace.json")
  $GPURUN node "$ROOT/tests/e2e/xroom.mjs" --role guest --signal "127.0.0.1:$SIG_PORT" --code "$CODE" --gb "$GUEST_GB" "${TR[@]}" > "$OUT/guest.out" 2> "$OUT/guest.log" &
  GPID=$!; PIDS+=("$GPID")
  wait_host
  remote "cat /tmp/$RUN-host.json" > "$OUT/host.json"
  log "host exited ($RC); waiting for the guest"
  for _ in $(seq 1 90); do kill -0 "$GPID" 2>/dev/null || break; sleep 1; done
fi
{ echo "# ping $REMOTE_IP after"; ping -c 20 -i 0.2 -q "$REMOTE_IP" 2>&1 | tail -2; } >> "$OUT/ping.txt"
if [ -s "$OUT/host.json" ]; then
  GT=(); [ -s "$OUT/guest-trace.json" ] && GT=("$OUT/guest-trace.json" --merged "$OUT/merged.json")
  node "$ROOT/tests/e2e/xroom_report.mjs" "$OUT/host.json" "${GT[@]}" --ping "$OUT/ping.txt" --ping-log "$OUT/ping_during.txt" > "$OUT/report.txt" 2>&1 || log "report failed (see $OUT/report.txt)"
fi
log "done: $OUT"
cat "$OUT/host.out" | grep -v '^CODE'
exit "${RC:-1}"
