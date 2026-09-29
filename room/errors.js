// Errors a person can act on (roadmap/16-actionable-errors-and-versioning.md, issue #16).
//
// Every failure the room shows should be one plain sentence that says what happened and what to do
// next, not "error: socket-error" or "model host refused range requests" for a 429. This module
// holds the wording so room.js only picks the case; it has no DOM and no globals, so the unit tests
// run it as is (tests/unit/errors_test.js).

// PeerJS `err.type` (peer.on("error")) -> what the person sees. `inRoom`: this tab is already in a
// room, so the signaling server matters only for devices that join later.
export function peerErrorText(type, { inRoom = false } = {}) {
  switch (type) {
    case "peer-unavailable":
      return inRoom ? "A device in the room can't be reached any more. If it left, re-deal the layers."
        : "No room with that code. Check the code, and that the host's page is still open.";
    case "unavailable-id":
      return "That room code is already in use by an open room. Press Join to join it, or create a new room.";
    case "browser-incompatible":
      return "This browser can't make peer-to-peer connections (no WebRTC). Open the room in a recent Chrome, Edge, Firefox or Safari.";
    case "webrtc":
      return "The browser's peer-to-peer connection failed. Reload the page and try again; if it keeps failing, try another network.";
    case "network":
    case "socket-error":
    case "socket-closed":
    case "disconnected":
    case "server-error":
    case "ssl-unavailable":
      return inRoom ? "Lost the connection to the room server. Devices already here keep working; new devices can't join until you reload this page."
        : "Can't reach the room server. Check your internet connection (a VPN or strict firewall can block it), then try again.";
    case "invalid-id":
      return "That room code isn't valid. Codes are 4 to 6 letters and digits.";
    case "invalid-key":
      return "The room server refused this page's key. Reload the page; if it keeps happening, please open an issue.";
    default:
      return `Connection error (${type || "unknown"}). Reload the page and try again.`;
  }
}
// peer errors worth a toast once this tab is in a room (the rest only go to the log): a lost
// signaling server stops new devices joining, a single unreachable device is the room's business
export function peerErrorLoud(type) {
  return type === "network" || type === "socket-error" || type === "socket-closed" || type === "disconnected"
    || type === "server-error" || type === "webrtc";
}

// A failed weight download. `status` is the HTTP status (0 when the request never got an answer:
// offline, CORS, a blocked host). The message is the sentence the person sees.
export class FetchError extends Error {
  constructor(status, url = "") {
    super(fetchErrorText(status, url));
    this.name = "FetchError";
    this.status = status;
  }
}
export function fetchErrorText(status, url = "") {
  const hf = /huggingface\.co|hf\.co/.test(url), who = hf ? "Hugging Face" : "The model host";
  if (!status) return `Can't reach ${hf ? "Hugging Face" : "the model host"} to download the model. Check your internet connection (a VPN, ad blocker or firewall can block it), then press Start again; cached layers are kept.`;
  if (status === 429) return `${who} is rate-limiting downloads (429). Wait a minute and press Start again; cached layers are kept.`;
  if (status === 502 || status === 503 || status === 504) return `${who} is busy or down right now (${status}). Wait a minute and press Start again; cached layers are kept.`;
  if (status === 401 || status === 403) return `${who} refused the download (${status}). The model may have moved or need a login; please open an issue.`;
  if (status === 404 || status === 410) return `${who} has no such file any more (${status}). The model may have moved; please open an issue.`;
  if (status === 200) return `${who} sent the whole file instead of the part asked for, so layers can't be split. Reload the page and try again; if it keeps happening, please open an issue.`;
  return `${who} answered ${status} instead of the model's layers. Wait a minute and press Start again; cached layers are kept.`;
}

// The join screen while the joiner's link to the host comes up. `ms` since the connect started,
// `ice` the link's iceConnectionState (undefined before signaling answered). Returns
// { status } to show while waiting, or { fail } to give up with.
// A room that exists answers signaling within seconds and ICE then sits in "checking" while the
// two devices look for a path; that part is given longer before calling it a failure.
export const JOIN_QUIET_MS = 5000, JOIN_TIMEOUT_MS = 15000, JOIN_ICE_MS = 40000;
export function joinStep(ms, ice) {
  if (ms < JOIN_QUIET_MS) return {};
  if (ms < JOIN_TIMEOUT_MS) return { status: "Still connecting…" };
  if (ice === "checking" && ms < JOIN_ICE_MS) return { status: "Found the room; still looking for a path between the two devices…" };
  if (ice === "checking" || ice === "failed" || ice === "disconnected")
    return { fail: "Found the room, but these two devices can't reach each other (a strict firewall or mobile network on one side). Put both on the same Wi-Fi, or try another network." };
  return { fail: "No room with that code. Check the code, and that the host's page is still open." };
}

// Room protocol mismatch (PROTOCOL in room/transport.js, carried as `v` in hello and ai-load; a
// peer from before versioning sends none, which counts as 1). Both tabs see the mismatch in each
// other's hello, so each shows `local` and sends `remote` as its `bye` reason: a tab that knows
// this module words its own message and ignores that bye, an older tab shows the bye as it is.
// `name`/`theyHost` describe the other device, `me`/`iAmHost` this one.
export function versionMismatch({ mine, theirs, name, theyHost = false, me, iAmHost = false }) {
  theirs = theirs ?? 1;
  return { local: versionLine(name, theyHost, theirs, mine), remote: versionLine(me, iAmHost, mine, theirs) };
}
// how the reader's tab (on `reader`) should read the other device (`who`, on `other`)
function versionLine(who, whoHost, other, reader) {
  const label = whoHost ? "This room's host" : who || "A device";
  const again = whoHost ? ", then join again" : "";
  const nums = `(protocol ${other}, this tab ${reader})`;
  return other < reader
    ? `${label} is on an older version of Pooled ${nums}. Ask ${whoHost ? "the host" : "them"} to reload the page${again}.`
    : `${label} is on a newer version of Pooled ${nums}. Reload this page to update${again}.`;
}
