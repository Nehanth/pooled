// /pooled in any OpenClaw chat (the TUI, the Control UI, a messaging channel): the room this gateway
// runs, and the host's Allow / Deny for devices waiting to join. It runs in the gateway that owns the
// room (pool.js current()), for the gateway's owner only (operator.admin).
//   /pooled                 the room: invite link, devices, who is waiting, the download
//   /pooled allow [n|all]   let a waiting device in (the oldest, the n-th, or every one)
//   /pooled deny [n]        turn one away
//   /pooled link            the invite link
//   /pooled pledge <GB>     lend another amount of this machine's GPU memory (the next deal uses it)
import { current, fmtCode, fitNow, modelInfo } from "./pool.js";
import { headLine, deviceBlock, memoryLine, downloadLine } from "./ui.js";

const HELP = "`/pooled` the room · `/pooled allow [n|all]` · `/pooled deny [n]` · `/pooled link` · `/pooled pledge <GB>`";

async function room() {
  const h = current();
  if (!h) return null;
  return h.ready.catch(() => null);
}

// the room as markdown (it reads the same in the TUI, the Control UI and a channel): a header line, the
// invite link, the devices in a code block (the CLI's table), then one sentence on what to do
export function statusText(r) {
  const st = r.status();
  const host = st.mode === "host";
  const model = st.model ? modelInfo(st.model).name : null;
  const n = st.devices?.length || 0;
  const devs = `${n} device${n === 1 ? "" : "s"}`;
  const out = [];
  let state;
  if (!host && st.refused) state = "turned away";
  else if (!host && st.admission !== "in") state = "waiting to be let in";
  else if (st.online && st.degraded) state = "a device left";
  else if (st.online) state = `● online · ${devs}`;
  else if (st.download && ["waiting", "running", "checking"].includes(st.download.state)) state = `downloading · ${devs}`;
  else if (st.loading) state = `loading · ${devs}`;
  else state = `${host ? "waiting for devices" : "waiting for the host to start"} · ${devs}`;
  out.push(headLine(r.code, host ? model : st.hostName ? `${st.hostName}'s room` : "joined", state));
  if (host) out.push(`Invite link: ${r.link}`);
  const rows = st.rows?.length ? st.rows : (st.devices || []).map((d) => ({ name: d.name, gb: d.gb, self: !!d.self, gpu: "", range: null }));
  const extra = [];
  if (host && st.needGB) extra.push(memoryLine(st.pledgedGB || 0, st.needGB));
  const dl = downloadLine(st.download);
  if (dl) extra.push(dl);
  if (rows.length || st.waiting?.length) out.push("", ...deviceBlock(rows, { waiting: (st.waiting || []).map((q) => ({ name: q.name || String(q.line || "").replace(/ wants to join.*$/, "") })), extra }));
  const say = [];
  if (!host) {
    if (st.refused) say.push(`The host turned this device away: ${st.refused}. Ask them for the room's invite link and run \`openclaw onboard\` with it.`);
    else if (st.admission !== "in") say.push("Waiting for the host to let this device in. The host is asked to allow or deny it; the room's invite link skips this step (run `openclaw onboard` here and paste it).");
  }
  if (host && !st.online) {
    const fit = fitNow(r, st);
    if (!fit.fits && fit.note) say.push(fit.note.replace(/([^.])$/, "$1."));
  }
  if (st.waiting?.length) {
    const who = st.waiting.map((q, i) => `${i + 1}. ${q.line}`);
    say.push(...(st.waiting.length > 1 ? ["Waiting to join:", ...who] : [`${st.waiting[0].line}.`]),
      `\`/pooled allow\` lets ${st.waiting.length > 1 ? "the first one" : "it"} in${st.waiting.length > 1 ? " (\`/pooled allow 2\`, \`/pooled allow all\`)" : ""}; \`/pooled deny\` turns it away.`);
  }
  if (say.length) out.push("", ...say);
  return out.join("\n");
}

// args: what followed /pooled -> reply text
export async function runPooledCommand(args, { getRoom = room } = {}) {
  const [sub = "", arg = ""] = String(args || "").trim().split(/\s+/);
  const r = await getRoom();
  if (!r) return "Pooled isn't running in this gateway: run `openclaw onboard` and pick Pooled, then restart the gateway.";
  switch (sub.toLowerCase()) {
    case "": case "status": return statusText(r);
    case "link": case "invite":
      return r.s.mode === "host" ? `${headLine(r.code, "invite link")}\n\n${r.link}\n\nEvery device holding layers computes what is asked here: share it only with people you trust.`
        : `This gateway joined Pooled room ${fmtCode(r.code)}; its host has the invite link.`;
    case "allow": case "deny": {
      if (r.s.mode !== "host" || !r.node.gate) return "Only the room's host lets devices in: this gateway joined someone else's room.";
      const q = r.node.waitingJoins();
      if (!q.length) return "Nobody is waiting to join.";
      if (arg && !/^\d+$/.test(arg) && !(sub === "allow" && /^all$/i.test(arg))) return `/pooled ${sub} takes a number from the waiting list${sub === "allow" ? " or all" : ""}: ${HELP}`;
      const picks = sub === "allow" && /^all$/i.test(arg) ? q : [q[(+arg || 1) - 1]].filter(Boolean);
      if (!picks.length) return `There ${q.length === 1 ? "is 1 device" : `are ${q.length} devices`} waiting: /pooled ${sub} 1${q.length > 1 ? `…${q.length}` : ""}`;
      const done = [];
      for (const x of picks) {
        const res = sub === "allow" ? await r.node.allowJoin(x.id) : r.node.denyJoin(x.id);
        if (res) done.push(x.name);
      }
      r.persist?.();
      if (!done.length) return "That device stopped waiting.";
      return `${sub === "allow" ? "Let in" : "Turned away"}: ${done.join(", ")}`;
    }
    case "pledge": {
      const gb = +arg;
      if (!(gb > 0 && gb <= 64)) return "/pooled pledge <GB>: how much GPU memory this machine lends the room, 0.5 to 64";
      const v = r.node.setPledge(gb);
      return `This machine now lends ${v} GB${r.node.ai?.online ? " (the room uses it at its next deal)" : ""}.`;
    }
    case "help": return HELP;
    default: return `Unknown: /pooled ${sub}. ${HELP}`;
  }
}

// api.registerCommand's definition
export const pooledCommand = {
  name: "pooled",
  description: "Your Pooled room: the invite link, the devices, and Allow / Deny for devices waiting to join",
  acceptsArgs: true,
  requireAuth: true,
  requiredScopes: ["operator.admin"],
  handler: async (ctx) => ({ text: await runPooledCommand(ctx.args) }),
};
