// pooled join's screen at a terminal (the CLI design spec, 5.11): the room's header beside the mark,
// what this computer lends, the download or load under way, and one sentence on what to do.
// Pure (the state in, lines out), so it is unit tested; cli/lib/lendrun.js draws it in a liveRegion.
//
// s: lendrun's status state { code, phase, devices, range, model, tps, passes, pct, load, dl,
//    hostName, modelLabel, you: { name, gpu, gb, totalGB }, lobbyAt, onlineAt, L (layers in the model) }
import { header, label, progressRow, clip, I, gbNum, clock, upFor } from "./style.js";
import { loadText } from "./lend.js";

const fmtCode = (c) => (String(c || "").length === 6 ? `${c.slice(0, 3)}-${c.slice(3)}` : String(c || ""));

function statusLine(s, S, { spin, now }) {
  const host = s.hostName || "the host";
  const range = s.range ? `${s.range[0]}${S.g.dash}${s.range[1] - 1}` : "";
  if (s.dl?.state === "running") return `${spin} downloading the model` + S.ink3(" · then it loads from disk");
  switch (s.phase) {
    case "connecting": return `${spin} connecting to the room`;
    case "lobby": return `${spin} waiting for ${host} to let you in` + (s.lobbyAt ? S.ink3(` · ${clock(now - s.lobbyAt)}`) : "");
    case "waiting": return `${spin} waiting for ${host} to start the room`;
    case "loading":
      if (s.dl?.state === "running" || (s.load?.from && s.load.from !== "disk" && s.load.fetched < s.load.total)) return `${spin} getting your layers` + (range ? S.ink3(` · ${range}${s.L ? ` of ${s.L}` : ""}`) : "");
      return `${spin} loading` + (range ? S.ink3(` · layers ${range}`) : "");
    case "ready": return `${spin} layers loaded` + S.ink3(" · waiting for the rest of the room");
    case "guest": return `${S.acc(S.g.live)} in the room` + S.ink3(" · without layers");
    case "online": case "answering":
      return `${S.acc(S.g.live)} in the room` + S.ink3(`${s.devices ? ` · ${s.devices} device${s.devices === 1 ? "" : "s"}` : ""}${s.onlineAt ? ` · up ${upFor(now - s.onlineAt)}` : ""}${s.phase === "answering" ? " · answering" : ""}`);
    case "degraded": return `${spin} a device left` + S.ink3(" · waiting for it");
    case "hostgone": return `${spin} lost ${host}` + S.ink3(" · knocking");
    case "rejoining": return `${spin} rejoining` + S.ink3(s.tries ? ` · try ${s.tries}` : "");
    case "leaving": return S.ink3("leaving the room");
    default: return String(s.phase || "");
  }
}

export function joinScreen(s, { S, cols = 80, spin = "", now = Date.now() } = {}) {
  spin ||= S.spin(0);
  const W = Math.max(40, cols) - 1;
  const L = [""];
  const room = s.hostName ? `${s.hostName}'s room` : "room";
  L.push(...header(S, cols, [S.bold("pooled join"), S.pill(fmtCode(s.code)) + "  " + room + (s.modelLabel ? S.ink3(` · ${s.modelLabel}`) : ""), statusLine(s, S, { spin, now })]));
  L.push("");
  const y = s.you || {};
  const range = s.range ? ` · layers ${s.range[0]}${S.g.dash}${s.range[1] - 1}` : "";
  L.push(label(S, "you") + (y.name || "this computer") + S.ink3(`${y.gpu ? ` · ${y.gpu}` : ""}${y.gb ? ` · lends ${gbNum(y.gb)}${!range && y.totalGB ? ` of ${Math.round(y.totalGB)} GB` : ""}` : ""}${range}`));
  if (s.dl?.state === "running") {
    L.push(progressRow(S, "download", { done: s.dl.done, total: s.dl.total, bps: s.dl.bps }));
    L.push(label(S, "load") + S.ink3("waits for the download"));
  } else if (s.phase === "loading") L.push(label(S, "load") + S.ink3(loadText(s.load, s.pct)));
  if ((s.phase === "online" || s.phase === "answering") && (s.passes || s.tps)) L.push(label(S, "work") + `${s.passes || 0} passes` + S.ink3(s.tps ? ` · the room answers at ${s.tps.toFixed(1)} tok/s` : ""));
  if (s.signaling === false) L.push(label(S, "network") + S.ink3("signaling down (links still up)"));
  L.push("");
  const hint = s.phase === "lobby" ? "The host is asked to allow or deny you. An invite link skips this step."
    : s.phase === "online" || s.phase === "answering" ? "Keep this open. Closing it takes your layers out of the room."
    : s.phase === "waiting" ? "The host starts the room once enough devices are in." : "";
  if (hint) L.push(I + S.ink3(hint), "");
  L.push(I + S.keys(s.phase === "online" || s.phase === "answering" ? [["q", "leave the room"]] : [["q", "leave"]]));
  return L.map((l) => clip(l, W));
}
