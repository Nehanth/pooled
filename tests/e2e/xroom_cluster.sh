#!/bin/bash
# A room of any subset of three devices, driven from one machine: "here" (this machine, headless
# Chrome, tests/e2e/xroom.mjs), "there" (the other machine, the same over XROOM_REMOTE) and "phone"
# (an iPhone's Safari over WebDriver, tests/e2e/xroom_phone.mjs, run on the machine the phone is
# USB-attached to). One of the computers hosts the room (creates it, loads the embedding/head and
# samples); the others join with its code. Manual trigger only.
#
#   tests/e2e/xroom_cluster.sh --devices here,there,phone [--host here|there] [--out DIR] [--no-sync]
#        [--guest-gb 12] [--guest-query "a=1&b=2"] [--phone-gb 0.5] [--phone-url https://pooled.run/room] [--phone-query dev=1]
#        -- <host xroom.mjs args>
#   e.g. tests/e2e/xroom_cluster.sh --devices here,phone --out /tmp/xc -- --model qwen3-1.7b --gb 13 --prompts twosum --rounds 2
#
# Signaling: with the phone in the room every device uses the public PeerJS server (the page's
# default: an https page on the phone cannot reach a plain ws:// server), so the phone page
# (--phone-url) and the computers' local checkouts must speak the same room protocol: use
# https://pooled.run/room for main, or the branch's Vercel preview for a branch. Without the phone,
# the signaling server runs here, as in xroom_pair.sh.
# Layers are dealt by memory in the room's order (the host first, then the other devices by peer id,
# which is random): a phone holds layers only when the computers cannot hold the model (--query
# phonelayers=1 in the host's args deals it layers anyway); with two devices the guest holds the last
# layers. The host's JSON has the split it used.
#
# Environment (nothing machine-specific lives in the repo): XROOM_REMOTE, XROOM_SYNC, XROOM_REMOTE_DIR,
# XROOM_REMOTE_ENV, XROOM_REMOTE_PREP, XROOM_LOCAL_IP, XROOM_REMOTE_IP, XROOM_LOCK, XROOM_GPURUN as in
# xroom_pair.sh (XROOM_LOCAL_IP only without the phone), plus
#   XROOM_PHONE_REMOTE  runs a shell command on the machine the phone is attached to (default
#                       XROOM_REMOTE); "local" runs it here. The checkout is synced there too
#                       (XROOM_REMOTE_DIR), and xroom_phone.mjs needs only node there.
#   XROOM_PHONE_LOCK    optional. '$XROOM_PHONE_LOCK acquire|release', held for the whole run (only one
#                       WebDriver session can drive the phone). acquire must print LOCKED.
#   XROOM_PHONE_WDPORT  safaridriver's port there (default 4444)
# Locks are taken in the order phone, there, here and released on any exit.
# Outputs in --out DIR: host.json / host.out / host.log, guest-here.log, guest-there.log, phone.json
# (xroom_phone.mjs's result: the layers it was dealt, its load time, status samples, page errors),
# phone.log, phone.png (the phone's screen at the end), ping.txt, report.txt (xroom_report.mjs).
set -u
DEVICES="" HOST=here OUT="" SYNC=1 GUEST_GB=12 PHONE_GB=0.5 PHONE_URL=https://pooled.run/room PHONE_QUERY=dev=1 GUEST_QUERY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --devices) DEVICES=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --guest-gb) GUEST_GB=$2; shift 2 ;;
    --guest-query) GUEST_QUERY=$2; shift 2 ;;
    --phone-gb) PHONE_GB=$2; shift 2 ;;
    --phone-url) PHONE_URL=$2; shift 2 ;;
    --phone-query) PHONE_QUERY=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
HOST_ARGS=("$@")
# tracing (--trace-rounds in the host's args): every guest traces too, the computers as in xroom_pair.sh
# (--trace-out) and the phone with xroom_phone.mjs --trace-out (the whole session); the chain report
# (tests/e2e/xroom_chain_report.mjs) then splits every traced lap per device. XROOM_PHONE_TRACE=1
# traces the phone without tracing the host (its cost, or a long run's per-lap numbers over time).
TRACE=0; for a in "${HOST_ARGS[@]}"; do [ "$a" = --trace-rounds ] && TRACE=1; done
PTRACE=$TRACE; [ "${XROOM_PHONE_TRACE:-0}" = 1 ] && PTRACE=1
has() { case ",$DEVICES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
for d in ${DEVICES//,/ }; do case "$d" in here|there|phone) ;; *) echo "unknown device $d (here, there, phone)" >&2; exit 2 ;; esac; done
case "$HOST" in here|there) ;; *) echo "--host here|there (a computer hosts; the room would make it the model host anyway)" >&2; exit 2 ;; esac
has "$HOST" || { echo "the host ($HOST) must be one of --devices" >&2; exit 2; }
N=0; for d in ${DEVICES//,/ }; do N=$((N + 1)); done
[ "$N" -ge 2 ] || { echo "--devices needs at least two devices" >&2; exit 2; }
USE_THERE=0; { has there || { has phone && [ "${XROOM_PHONE_REMOTE:-}" != local ]; }; } && USE_THERE=1
[ "$USE_THERE" = 1 ] && : "${XROOM_REMOTE:?set XROOM_REMOTE (a command that runs a shell command on the other machine)}"
RDIR=${XROOM_REMOTE_DIR:-pooled-xroom}
RENV=${XROOM_REMOTE_ENV:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=xroom-$$-$(date +%s)
OUT=${OUT:-$(mktemp -d -t xroom-out-XXXX)}
mkdir -p "$OUT"
SIG_PORT=9000
if has phone; then SIGNAL=cloud; else : "${XROOM_LOCAL_IP:?set XROOM_LOCAL_IP (this machine as the other one sees it)}"; SIGNAL=$XROOM_LOCAL_IP:$SIG_PORT; fi
remote() { $XROOM_REMOTE "$RENV $1"; }
phone_remote() { if [ "${XROOM_PHONE_REMOTE:-}" = local ]; then bash -c "$1"; else ${XROOM_PHONE_REMOTE:-$XROOM_REMOTE} "$RENV $1"; fi; }
PDIR=$ROOT; [ "${XROOM_PHONE_REMOTE:-}" = local ] || PDIR="~/$RDIR"
log() { echo "[xroom_cluster $(date +%T)] $*" >&2; }

PLOCK=0 RLOCK=0 PIDS=() SIGPID=""
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  [ -n "$SIGPID" ] && kill "$SIGPID" 2>/dev/null
  # the phone script deletes its WebDriver session on SIGTERM; give it a moment
  if has phone; then phone_remote "pkill -TERM -f -- '--tag $RUN' 2>/dev/null; sleep 3; pkill -f -- '--tag $RUN' 2>/dev/null; rm -f /tmp/$RUN-*" >/dev/null 2>&1; fi
  if has there; then remote "pkill -f -- '--tag $RUN' 2>/dev/null; rm -f /tmp/$RUN-*" >/dev/null 2>&1; fi
  if [ "$RLOCK" = 1 ]; then $XROOM_LOCK release >&2; fi
  if [ "$PLOCK" = 1 ]; then $XROOM_PHONE_LOCK release >&2; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ "$SYNC" = 1 ] && [ "$USE_THERE" = 1 ]; then
  : "${XROOM_SYNC:?set XROOM_SYNC (copies a checkout to the other machine), or pass --no-sync}"
  log "sync $ROOT -> $RDIR"
  $XROOM_SYNC "$ROOT" "$RDIR" >&2 || { log "sync failed"; exit 1; }
fi
if has there && [ -n "${XROOM_REMOTE_PREP:-}" ]; then remote "cd ~/$RDIR && $XROOM_REMOTE_PREP" >&2; fi
# locks: phone, there, here (the same order everywhere, so two runs cannot wait on each other)
if has phone && [ -n "${XROOM_PHONE_LOCK:-}" ]; then
  log "taking the phone's lock"
  $XROOM_PHONE_LOCK acquire | grep -q LOCKED || { log "no phone lock"; exit 1; }
  PLOCK=1
fi
if has there && [ -n "${XROOM_LOCK:-}" ]; then
  log "taking the other machine's GPU lock"
  $XROOM_LOCK acquire | grep -q LOCKED || { log "no lock"; exit 1; }
  RLOCK=1
fi
GPURUN=""
if has here && [ -n "${XROOM_GPURUN:-}" ]; then
  GPURUN=$XROOM_GPURUN
  ok=0
  for _ in $(seq 1 "${XROOM_GPURUN_TRIES:-8}"); do
    [ "$($GPURUN echo XROOM_GPU_FREE 2>&1 | tail -1)" = XROOM_GPU_FREE ] && { ok=1; break; }
    log "this GPU is busy; waiting again"
  done
  [ "$ok" = 1 ] || { log "this GPU stayed busy"; exit 1; }
fi
# the other machine's system clock against this one's (lowest round trip of 5; the chain report uses it
# only to put a worker's laps into the host's rounds)
if [ "$USE_THERE" = 1 ]; then
  best=""; for _ in 1 2 3 4 5; do
    a=$(date +%s%3N); m=$(remote 'python3 -c "import time; print(int(time.time() * 1000))"' 2>/dev/null | tail -1); b=$(date +%s%3N)
    [ -n "$m" ] && { r=$((b - a)); o=$((m - (a + b) / 2)); if [ -z "$best" ] || [ "$r" -lt "${best% *}" ]; then best="$r $o"; fi; }
  done
  [ -n "$best" ] && echo "mac_minus_here ${best#* } rtt ${best% *}" > "$OUT/clock.txt"
fi
REMOTE_IP=""
if has there; then
  REMOTE_IP=${XROOM_REMOTE_IP:-}
  [ -z "$REMOTE_IP" ] && REMOTE_IP=$(remote 'echo $SSH_CONNECTION' | awk '{print $3}')
  { echo "# ping $REMOTE_IP before"; ping -c 20 -i 0.2 -q "$REMOTE_IP" 2>&1 | tail -2; } > "$OUT/ping.txt"
  ping -D -i 0.2 "$REMOTE_IP" > "$OUT/ping_during.txt" 2>&1 &
  PIDS+=("$!")
fi

wait_code() {   # wait_code <file> <pid> <seconds>: the host's "CODE XXXX" line, while <pid> lives
  for _ in $(seq 1 "$3"); do
    c=$(grep -m1 -o '^CODE [A-Z0-9]*' "$1" 2>/dev/null | cut -d' ' -f2)
    [ -n "$c" ] && { echo "$c"; return 0; }
    kill -0 "$2" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
SIGOPT=(--signal "$SIGNAL")
GPIDS=()
if [ "$HOST" = here ]; then
  $GPURUN node "$ROOT/tests/e2e/xroom.mjs" --role host "${SIGOPT[@]}" --peers "$N" --out "$OUT/host.json" "${HOST_ARGS[@]}" > "$OUT/host.out" 2> "$OUT/host.log" &
  HPID=$!; PIDS+=("$HPID")
else
  if [ "$SIGNAL" != cloud ]; then
    "$ROOT/node_modules/.bin/peerjs" --port "$SIG_PORT" --path / --host 0.0.0.0 > /dev/null 2>&1 &
    SIGPID=$!; sleep 1.5
  fi
  # a guest here starts first (through XROOM_GPURUN, before anything of this run is on this GPU)
  # and waits for the host's code in a file, as in xroom_pair.sh
  if has here; then
    LSIG=$SIGNAL; [ "$SIGNAL" != cloud ] && LSIG=127.0.0.1:$SIG_PORT
    GT=(); [ "$TRACE" = 1 ] && GT=(--trace-out "$OUT/guest-here.trace.json")
    $GPURUN node "$ROOT/tests/e2e/xroom.mjs" --role guest --signal "$LSIG" --codefile "$OUT/code.txt" --gb "$GUEST_GB" --query "$GUEST_QUERY" "${GT[@]}" > "$OUT/guest-here.out" 2> "$OUT/guest-here.log" &
    GPIDS+=("$!"); PIDS+=("$!")
    for _ in $(seq 1 1200); do grep -q "waiting for the room code" "$OUT/guest-here.log" 2>/dev/null && break; kill -0 "${GPIDS[0]}" 2>/dev/null || break; sleep 1; done
    grep -q "waiting for the room code" "$OUT/guest-here.log" 2>/dev/null || { log "the guest here did not start"; cat "$OUT/guest-here.out" >&2; exit 1; }
  fi
  QARGS=$(printf ' %q' "${HOST_ARGS[@]}")
  RSIG=cloud; [ "$SIGNAL" != cloud ] && RSIG="$SIGNAL --signal-server 0"
  remote "cd ~/$RDIR && node tests/e2e/xroom.mjs --role host --signal $RSIG --peers $N --out /tmp/$RUN-host.json $QARGS --tag $RUN" > "$OUT/host.out" 2> "$OUT/host.log" &
  HPID=$!; PIDS+=("$HPID")
fi
CODE=$(wait_code "$OUT/host.out" "$HPID" 1200) || { log "the host printed no code"; tail -5 "$OUT/host.log" >&2; exit 1; }
echo "$CODE" > "$OUT/code.txt"
log "room $CODE ($DEVICES, host $HOST)"
if [ "$HOST" = here ] && has there; then
  RSIG=$SIGNAL
  GT=""; [ "$TRACE" = 1 ] && GT="--trace-out /tmp/$RUN-guest-trace.json"
  remote "cd ~/$RDIR && node tests/e2e/xroom.mjs --role guest --signal $RSIG --code $CODE --gb $GUEST_GB --query $(printf '%q' "$GUEST_QUERY") $GT --tag $RUN" > "$OUT/guest-there.out" 2> "$OUT/guest-there.log" &
  GPIDS+=("$!"); PIDS+=("$!")
fi
PPID_=""
if has phone; then
  PQ=$(printf '%q' "$PHONE_QUERY")
  PT=""; [ "$PTRACE" = 1 ] && PT="--trace-out /tmp/$RUN-phone-trace.json"
  phone_remote "cd $PDIR && node tests/e2e/xroom_phone.mjs --url $PHONE_URL --query $PQ --code $CODE --gb $PHONE_GB --wd-port ${XROOM_PHONE_WDPORT:-4444} --out /tmp/$RUN-phone.json --shot /tmp/$RUN-phone.png $PT --tag $RUN" > "$OUT/phone.out" 2> "$OUT/phone.log" &
  PPID_=$!; GPIDS+=("$PPID_"); PIDS+=("$PPID_")
fi
# the host's exit code in RC; any guest that ends first is a failure (the host would wait for it)
while kill -0 "$HPID" 2>/dev/null; do
  for g in "${GPIDS[@]}"; do
    if ! kill -0 "$g" 2>/dev/null; then log "a guest ended before the host ($( [ "$g" = "$PPID_" ] && echo phone || echo computer))"; kill "$HPID" 2>/dev/null; [ "$HOST" = there ] && remote "pkill -f -- '--tag $RUN'" >/dev/null 2>&1; break 2; fi
  done
  sleep 2
done
wait "$HPID"; RC=$?
[ "$HOST" = there ] && remote "cat /tmp/$RUN-host.json" > "$OUT/host.json"
log "host exited ($RC); waiting for the guests"
for _ in $(seq 1 90); do alive=0; for g in "${GPIDS[@]}"; do kill -0 "$g" 2>/dev/null && alive=1; done; [ "$alive" = 0 ] && break; sleep 1; done
if has phone; then
  phone_remote "cat /tmp/$RUN-phone.json" > "$OUT/phone.json" 2>/dev/null
  phone_remote "base64 < /tmp/$RUN-phone.png" 2>/dev/null | base64 -d > "$OUT/phone.png" 2>/dev/null
  [ -s "$OUT/phone.json" ] || cp "$OUT/phone.out" "$OUT/phone.json" 2>/dev/null
  [ "$PTRACE" = 1 ] && phone_remote "cat /tmp/$RUN-phone-trace.json" > "$OUT/phone.trace.json" 2>/dev/null
fi
if has there && [ "$TRACE" = 1 ] && [ "$HOST" = here ]; then remote "cat /tmp/$RUN-guest-trace.json" > "$OUT/guest-there.trace.json" 2>/dev/null; fi
if has there; then { echo "# ping $REMOTE_IP after"; ping -c 20 -i 0.2 -q "$REMOTE_IP" 2>&1 | tail -2; } >> "$OUT/ping.txt"; fi
if [ -s "$OUT/host.json" ]; then
  PING=(); [ -s "$OUT/ping.txt" ] && PING=(--ping "$OUT/ping.txt" --ping-log "$OUT/ping_during.txt")
  node "$ROOT/tests/e2e/xroom_report.mjs" "$OUT/host.json" "${PING[@]}" > "$OUT/report.txt" 2>&1 || log "report failed (see $OUT/report.txt)"
  W=(); for f in "$OUT"/guest-here.trace.json "$OUT"/guest-there.trace.json "$OUT"/phone.trace.json; do [ -s "$f" ] && W+=("$f"); done
  if [ "${#W[@]}" -gt 0 ]; then node "$ROOT/tests/e2e/xroom_chain_report.mjs" "$OUT/host.json" "${W[@]}" --clock "$OUT/clock.txt" > "$OUT/chain.txt" 2>&1 || log "chain report failed (see $OUT/chain.txt)"; fi
fi
log "done: $OUT"
grep -v '^CODE' "$OUT/host.out"
exit "${RC:-1}"
