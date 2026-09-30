// harness/prefix.js: longest reusable checkpoint.
import { PrefixIndex, isPrefix, pinSplit } from "../../harness/prefix.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

Deno.test("PrefixIndex picks the longest strict prefix", () => {
  const P = new PrefixIndex();
  P.add([1, 2, 3], "sys");
  P.add([1, 2, 3, 4, 5, 6], "turn1");
  P.add([1, 2, 9, 9], "other");
  eq(P.best([1, 2, 3, 4, 5, 6, 7, 8]), { key: "turn1", n: 6 });
  eq(P.best([1, 2, 3, 4, 5]), { key: "sys", n: 3 }, "turn1 is longer than the prompt");
  eq(P.best([1, 2, 3, 4, 5, 6]), { key: "sys", n: 3 }, "an exact match leaves nothing to run: not usable");
  eq(P.best([7, 1, 2, 3]), null);
  P.remove("turn1");
  eq(P.best([1, 2, 3, 4, 5, 6, 7]), { key: "sys", n: 3 });
});

Deno.test("PrefixIndex keeps the most recently used entries", () => {
  const P = new PrefixIndex(2);
  P.add([1], "a"); P.add([1, 2], "b"); P.add([1, 2, 3], "c");
  eq(P.items.length, 2);
  P.add([1, 2], "b");
  eq(P.items.map((x) => x.key).sort(), ["b", "c"]);
});

Deno.test("PrefixIndex: equal-length matches go to the first one added, and that counts as its use", () => {
  const P = new PrefixIndex();
  P.add([1, 2], "a");
  P.add([1, 2], "b");               // the same tokens under another key
  eq(P.best([1, 2, 3]), { key: "a", n: 2 });
  const t = Object.fromEntries(P.items.map((x) => [x.key, x.t]));
  if (!(t.a > t.b)) throw new Error("best() must touch the entry it returns: " + JSON.stringify(t));
  P.remove("a");
  eq(P.best([1, 2, 3]), { key: "b", n: 2 });
});

Deno.test("PrefixIndex: a limit below the item count keeps the most recently used, best() included", () => {
  const P = new PrefixIndex(2);
  P.add([1], "a"); P.add([1, 2], "b");
  eq(P.best([1, 9]), { key: "a", n: 1 });   // a is now newer than b
  P.add([1, 2, 3], "c");                    // over the limit: b goes
  eq(P.items.map((x) => x.key).sort(), ["a", "c"]);
  const P1 = new PrefixIndex(1);
  for (const k of ["a", "b", "c", "d"]) P1.add([1, k.charCodeAt(0)], k);
  eq(P1.items.map((x) => x.key), ["d"]);
});

Deno.test("PrefixIndex: add() with a key already present replaces its tokens, never duplicates", () => {
  const P = new PrefixIndex();
  P.add([1, 2, 3], "k");
  P.add([7, 8], "k");
  eq(P.items.length, 1);
  eq(P.best([1, 2, 3, 4]), null, "the old tokens are gone");
  eq(P.best([7, 8, 9]), { key: "k", n: 2 });
});

Deno.test("PrefixIndex: add() copies the ids (a later push to the caller's array does not move the checkpoint)", () => {
  const P = new PrefixIndex();
  const fed = [1, 2];
  P.add(fed, "k");
  fed.push(3);
  eq(P.best([1, 2, 3]), { key: "k", n: 2 });
  P.add(Uint32Array.of(5, 6), "t");          // typed arrays are accepted and stored as plain arrays
  eq(P.best([5, 6, 7]), { key: "t", n: 2 });
});

Deno.test("PrefixIndex: empty index, empty prompt, empty checkpoint, remove of an unknown key", () => {
  const P = new PrefixIndex();
  eq(P.best([1, 2]), null);
  P.remove("nope");
  P.add([], "empty");
  eq(P.best([]), null, "nothing left to run");
  // an empty checkpoint matches with n 0; room.js ckptResume ignores any n <= what it already
  // reuses, and ckptSave never saves an empty fed, so this is harmless, but it is pinned here
  eq(P.best([1]), { key: "empty", n: 0 });
});

// ---- the pinned checkpoint (system prompt + tools, issue #73)

Deno.test("PrefixIndex: a pinned entry survives the limit, the most recently used of the rest stay", () => {
  const P = new PrefixIndex(2);
  P.add([1, 2], "sys", { pin: true });
  P.add([1, 2, 3], "a"); P.add([1, 2, 3, 4], "b"); P.add([1, 2, 3, 4, 5], "c");
  eq(P.items.map((x) => x.key).sort(), ["c", "sys"]);
  eq(P.pinned().map((x) => x.key), ["sys"]);
  eq(P.unpinned().map((x) => x.key), ["c"]);
});

Deno.test("PrefixIndex: the pinned entry is found when the middle of the prompt changed", () => {
  const P = new PrefixIndex();
  P.add([1, 2, 3], "sys", { pin: true });
  P.add([1, 2, 3, 10, 11, 12], "turn");
  eq(P.best([1, 2, 3, 10, 11, 12, 13]), { key: "turn", n: 6 }, "the prompt grew: the longest wins");
  eq(P.best([1, 2, 3, 99, 11, 12, 13]), { key: "sys", n: 3 }, "compacted in the middle: back to the system prompt");
  eq(P.best([4, 2, 3, 10]), null, "another system prompt");
  P.add([1, 2, 3], "sys", { pin: false });   // re-adding a key replaces its flag too
  eq(P.pinned(), []);
});

Deno.test("isPrefix", () => {
  eq(isPrefix([], []), true);
  eq(isPrefix([], [1]), true);
  eq(isPrefix([1, 2], [1, 2]), true);
  eq(isPrefix([1, 2], [1, 2, 3]), true);
  eq(isPrefix([1, 3], [1, 2, 3]), false);
  eq(isPrefix([1, 2, 3], [1, 2]), false);
});

Deno.test("pinSplit: cut at the system prompt only when the caches hold less and a token follows", () => {
  const cases = [
    // [reused, pin, total, want]
    [0, 5, 10, 5],        // fresh prompt: cut after the system prompt
    [3, 5, 10, 5],        // a shorter checkpoint was resumed: still cut
    [5, 5, 10, 0],        // the system prompt is already cached
    [8, 5, 10, 0],        // more than it is cached
    [0, 10, 10, 0],       // nothing follows the system prompt: the last token goes through the head
    [0, 9, 10, 9],        // exactly one token follows
    [0, 0, 10, 0],        // no system prompt
    [0, 12, 10, 0],       // longer than the prompt (another system prompt was measured)
    [0, undefined, 10, 0],
    [0, 2.5, 10, 0],
  ];
  for (const [r, p, n, want] of cases) eq(pinSplit(r, p, n), want, `pinSplit(${r}, ${p}, ${n})`);
});
