// /pooled in any OpenClaw chat (the TUI, the Control UI, a messaging channel): the room this gateway
// runs, and the host's Allow / Deny for devices waiting to join. It runs in the gateway that owns the
// room (pool.js current()), for the gateway's owner only (operator.admin).
//   /pooled                 the room: invite link, devices, who is waiting, the download
//   /pooled allow [n|all]   let a waiting device in (the oldest, the n-th, or every one)
//   /pooled deny [n]        turn one away
//   /pooled link            the invite link
//   /pooled pledge <GB>     lend another amount of this machine's GPU memory (the next deal uses it)
import { current, fmtCode, fitNow } from "./pool.js";

const HELP = "/pooled · /pooled allow [n|all] · /pooled deny [n] · /pooled link · /pooled pledge <GB>";

async function room() {
  const h = current();
  if (!h) return null;
  return h.ready.catch(() => null);
}

export function statusText(r) {
  const st = r.status();
  const lines = [];
  const host = st.mode === "host";
  lines.push(`Pooled room ${fmtCode(r.code)} · ${host ? "this gateway hosts it" : "this gateway joined it"}${st.online ? ` · ${st.model} online` : ""}`);
  if (host) lines.push(`Invite link: ${r.link}`);
  if (!host) {
    if (st.refused) lines.push(`The host turned this device away: ${st.refused}`);
    else if (st.admission !== "in") lines.push("Waiting for the host to let this device in (the room's invite link skips this: run `openclaw onboard` and paste it)");
  }
  if (st.devices?.length) lines.push(`Devices: ${st.devices.map((d) => `${d.name}${d.self ? " (this)" : ""} ${d.gb} GB`).join(", ")} · ${st.pledgedGB} GB pledged`);
  if (host && !st.online) {
    const fit = fitNow(r, st);
    if (!fit.fits && fit.note) lines.push(fit.note);
  }
  if (st.split?.length) lines.push(`Layers: ${st.split.join(" · ")}`);
  if (st.download && st.download.state !== "done") lines.push(`Model download: ${st.download.line}`);
  if (st.waiting?.length) {
    lines.push("Waiting to join:");
    st.waiting.forEach((q, i) => lines.push(`  ${i + 1}. ${q.line}`));
    lines.push("/pooled allow lets the first one in (/pooled allow 2, /pooled allow all), /pooled deny turns it away");
  }
  return lines.join("\n");
}

// args: what followed /pooled -> reply text
export async function runPooledCommand(args, { getRoom = room } = {}) {
  const [sub = "", arg = ""] = String(args || "").trim().split(/\s+/);
  const r = await getRoom();
  if (!r) return "Pooled isn't running in this gateway: run `openclaw onboard` and pick Pooled, then restart the gateway.";
  switch (sub.toLowerCase()) {
    case "": case "status": return statusText(r);
    case "link": case "invite":
      return r.s.mode === "host" ? `Invite link for Pooled room ${fmtCode(r.code)}: ${r.link}\nEvery device holding layers computes what is asked here: share it only with people you trust.`
        : `This gateway joined Pooled room ${fmtCode(r.code)}; its host has the invite link.`;
    case "allow": case "deny": {
      if (r.s.mode !== "host" || !r.node.gate) return "Only the room's host lets devices in: this gateway joined someone else's room.";
      const q = r.node.waitingJoins();
      if (!q.length) return "Nobody is waiting to join.";
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
