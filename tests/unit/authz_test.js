// room/authz.js: only the room's creator names the model host (#17). A device follows Start only from the
// creator and takes layers only from the creator or the model host it named; a member's Start lets the
// creator pick between the member and itself, never a third device; a creator from before (no ha) keeps
// the old behaviour.
import { HOST_AUTH_V, strictRoom, followStart, loadAllowed, creatorPicks } from "../../room/authz.js";
import { pickModelHost } from "../../room/plan.js";

function eq(a, b, m) { const x = JSON.stringify(a), y = JSON.stringify(b); if (x !== y) throw new Error(`${m ? m + ": " : ""}${x} !== ${y}`); }
const C = "pooled-room-4TKG9P", M = "peer-mallory", L = "peer-laptop";

Deno.test("authz: a creator with ha makes the room strict; one without keeps the old rules", () => {
  eq(HOST_AUTH_V >= 1, true);
  eq(strictRoom(HOST_AUTH_V), true);
  eq(strictRoom(0), false); eq(strictRoom(undefined), false); eq(strictRoom("x"), false);
});

Deno.test("authz: Start is followed only from the creator in a strict room", () => {
  eq(followStart({ from: C, creator: C, strict: true }), true);
  eq(followStart({ from: M, creator: C, strict: true }), false, "a member can't name the model host");
  eq(followStart({ from: M, creator: C, strict: false }), true, "an older creator: as before");
});

Deno.test("authz: layers come from the creator or the model host it named, dealt as itself", () => {
  eq(loadAllowed({ from: C, host: C, creator: C, dealer: null, strict: true }), true);
  eq(loadAllowed({ from: L, host: L, creator: C, dealer: L, strict: true }), true, "the named model host");
  eq(loadAllowed({ from: M, host: M, creator: C, dealer: L, strict: true }), false, "a member making itself host");
  eq(loadAllowed({ from: M, host: M, creator: C, dealer: null, strict: true }), false, "nobody named yet");
  eq(loadAllowed({ from: L, host: M, creator: C, dealer: L, strict: true }), false, "a host deals as itself");
  eq(loadAllowed({ from: C, host: L, creator: C, dealer: L, strict: true }), false, "even the creator deals as itself");
  eq(loadAllowed({ from: L, host: undefined, creator: C, dealer: L, strict: true }), true, "an older host's ai-load without host");
  eq(loadAllowed({ from: M, host: M, creator: C, dealer: null, strict: false }), true, "an older creator: as before");
});

Deno.test("authz: on a member's Start the creator picks between the member and itself", () => {
  const me = { webgpu: true, contribGB: 8, ua: "Mac" }, big = { webgpu: true, contribGB: 24, ua: "Linux" }, small = { webgpu: true, contribGB: 2, ua: "Win" };
  const pick = (meta) => () => pickModelHost([{ id: C, meta: me }, { id: M, meta }]);
  eq(creatorPicks({ from: M, creator: C, busy: false, isMember: true, pick: pick(small) }), C, "the creator is stronger");
  eq(creatorPicks({ from: M, creator: C, busy: false, isMember: true, pick: pick(big) }), M, "the member is stronger: it runs it, as before");
  eq(creatorPicks({ from: M, creator: C, busy: true, isMember: true, pick: pick(big) }), null, "a model loads or runs: ignored");
  eq(creatorPicks({ from: M, creator: C, busy: false, isMember: false, pick: pick(big) }), null, "not a member (an API client, a lobby link)");
  eq(creatorPicks({ from: C, creator: C, busy: false, isMember: true, pick: pick(big) }), null);
  eq(creatorPicks({ from: M, creator: C, busy: false, isMember: true, pick: () => null }), C, "no pick: the creator");
});
