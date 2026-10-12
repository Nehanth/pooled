// This device's copy of a disk-streamed model (room/models.js DISK_COPY): what the picker asks, in one place.
//   consent(key) -> bool                 did this device agree to keep the copy
//   setConsent(key, on)                  agree (Start) or take it back (Delete stored copy)
//   storedBytes(key) -> Promise<number|null>   bytes stored now (null: unknown)
//   deleteCopy(key) -> Promise<bool>     remove the stored copy (and the consent)
//   freeBytes() -> Promise<number|null>  free disk for this browser (null: unknown)
// STUB until PR #368 (feat/flash-tab-disk) lands: consent lives in localStorage; the stored size, the delete and the
// free disk are #368's (its OPFS disk tier: room/flashdisk.js, engine/expert_opfs.js removeExperts, the n-gram
// table's NGRAM_OPFS_DIR). Replace the bodies marked #368 with its functions; the picker calls only this object.
const KEY = "pooled-disk-consent";
const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch { return {}; } };
const write = (o) => { try { localStorage.setItem(KEY, JSON.stringify(o)); } catch { /* private mode: asks again next time */ } };

export const diskStore = {
  consent: (key) => !!read()[key],
  setConsent(key, on) { const o = read(); if (on) o[key] = Date.now(); else delete o[key]; write(o); },
  // #368: the OPFS copy's bytes (the experts' manifest + the n-gram table) for this model
  storedBytes: async (_key) => null,
  // #368: removeExperts for this model's copies and the n-gram table's directory
  async deleteCopy(key) { this.setConsent(key, false); return false; },
  // #368: its disk estimate (Chrome's quota alone reads ~10 GB on a fresh origin, so it is not shown as free disk)
  freeBytes: async () => null,
};
