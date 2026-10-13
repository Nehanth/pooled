// This device's copy of a disk-streamed model (room/models.js DISK_COPY): what the picker asks, answered by #368's disk
// tier (room/flashdisk.js) and room.js's handling of it. Nothing here keeps its own state or asks its own question:
//   state(key)          -> "ask" | "allow" | "deny" | "url" | "none"   (diskState: tabDiskChoice's stored answer)
//   allow(key)          the Disk setting's Allow (from "ask" or a stored no): the same answer the disk question's
//                       Allow and Room settings' switch save (saveTabDiskChoice "allow", then room.js tabDiskChanged)
//   storedBytes(key)    -> Promise<number|null>  flashDiskBytes().total (the experts' copy and the n-gram table)
//   deleteCopy(key)     -> Promise<{ removed, kept } | { refused }>  room.js freeTabDisk: Room settings' "Free disk
//                       space", with its guard (not while a load or the running model reads the copy)
// room.js connects its side (connectDiskStore) at startup.
import { tabDiskChoice, saveTabDiskChoice, flashDiskBytes, TAB_DISK_MODELS } from "./flashdisk.js";

const G_ = /** @type {any} */ (globalThis);
// room.js's side: can (this tab has a disk tier: WebGPU, OPFS, not a phone), changed (after an answer), free (the delete)
let room = { can: () => false, changed: () => {}, free: async () => ({ refused: "not connected" }) };
export function connectDiskStore(hooks) { room = { ...room, ...hooks }; }

// The picker's view of this device's answer for `key`. can: this tab has a disk tier; search: location.search;
// storage: localStorage. ?tabdisk=0 is "url" (no for this page; the URL can never say yes: tabDiskChoice)
export function diskState(key, { can = false, search = "", storage = null } = {}) {
  if (!TAB_DISK_MODELS.includes(key) || !can) return "none";
  if (new URLSearchParams(search).get("tabdisk") === "0") return "url";
  return tabDiskChoice({ search, storage }) || "ask";
}

export const diskStore = {
  state: (key) => diskState(key, { can: room.can(), search: G_.location?.search || "", storage: G_.localStorage }),
  allow(key) {
    const st = diskStore.state(key);
    if (st !== "ask" && st !== "deny") return false;
    saveTabDiskChoice("allow", G_.localStorage);
    room.changed(key);
    return true;
  },
  storedBytes: async (_key) => { try { return (await flashDiskBytes()).total; } catch { return null; } },
  deleteCopy: (_key) => room.free(),
};
