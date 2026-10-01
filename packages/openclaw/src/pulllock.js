// One download of a model at a time: onboarding's "Download now", the gateway's background pull and
// pooled pull / host / join (@pooled/cli) write the same <dir>/<model>/<file>.part, so each takes
// <dir>/<model>/.pull.lock first ({ pid, host, at }). One implementation for both: cli/lib/cache.js
// (pullLock / tryPullLock), which this plugin already imports for pullModel. A lock whose process is
// gone is taken over; another machine's counts for a day; a stale lock is removed only if unchanged.
export { pullLockPath as lockPath, tryPullLock as tryLock, pullLock as lock } from "../../../cli/lib/cache.js";
