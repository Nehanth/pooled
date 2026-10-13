// room/flashdisk.js on an in-memory OPFS: the n-gram table's copy (keyed by content, one ingest of a directory at a
// time, the first failure stops the rest), and the free-disk-space helpers.
import { ingestNgram, ngramContentSig, ngramDir, NGRAM_OPFS_DIR, flashDiskBytes, removeFlashCopy } from "../../room/flashdisk.js";
import { EXPERT_OPFS_DIR } from "../../engine/expert_opfs.js";
import { ngramFileName } from "../../engine/ngram.js";
import { FakeDir } from "../helpers/fake-opfs.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a 4-head table, 8 rows of 8 bytes per head, row r of head h filled with h * 16 + r
const cfg = { nHeads: 4, vocab: [8, 8, 8, 8], rowBytes: 8, offsets: [0, 8, 16, 24], nRows: 32, shard: 0, byteOffset: 64, ggmlType: 2 };
const table = new Uint8Array(64 + 32 * 8);
for (let r = 0; r < 32; r++) table.fill((r >> 3) * 16 + (r & 7), 64 + r * 8, 64 + r * 8 + 8);
function source({ failAt = -1, slow = 0 } = {}) {
  const st = { reads: 0, aborted: 0, after: 0, failed: false };
  const readAt = async (_f, off, len, { signal } = {}) => {
    if (st.failed) st.after++;
    st.reads++;
    if (slow) await new Promise((r) => setTimeout(r, slow));
    if (signal?.aborted) { st.aborted++; throw Object.assign(new Error("aborted"), { name: "AbortError" }); }
    if (failAt >= 0 && off >= 64 + failAt * 64 && off < 64 + (failAt + 1) * 64) { st.failed = true; throw new Error("network down"); }
    return table.slice(off, off + len);
  };
  return { readAt, st };
}
const storageOf = (root) => ({ getDirectory: async () => root, estimate: async () => ({ quota: 2 ** 40, usage: 0 }) });
const headFile = async (root, dir, h) => { let d = root; for (const p of dir) d = await d.getDirectoryHandle(p); return (await d.getFileHandle(ngramFileName(h))).data; };

Deno.test("ingestNgram: copies every head into the table's own directory, once", async () => {
  const root = new FakeDir(), storage = storageOf(root), dir = ngramDir("aaaa0001");
  const { readAt, st } = source();
  eq(await ingestNgram(cfg, readAt, { dir, storage }), 256);
  for (let h = 0; h < 4; h++) { const d = await headFile(root, dir, h); eq(d.length, 64); ok(d[0] === h * 16 && d[63] === h * 16 + 7, `head ${h}`); }
  const r0 = st.reads;
  eq(await ingestNgram(cfg, readAt, { dir, storage }), 0, "all there: nothing copied");
  eq(st.reads, r0);
});

Deno.test("ingestNgram: the first failure stops the other heads (reads aborted, nothing new started, no half head kept)", async () => {
  const root = new FakeDir(), storage = storageOf(root), dir = ngramDir("aaaa0002");
  const { readAt, st } = source({ failAt: 1, slow: 2 });
  let err = null;
  try { await ingestNgram({ ...cfg }, readAt, { dir, storage, parallel: 2 }); } catch (e) { err = e; }
  ok(/network down/.test(err?.message), String(err));
  ok(st.after <= 2, `${st.after} reads after the failure (at most the other head's one in flight)`);
  // heads 2 and 3 were never started; head 1 left no file content
  for (const h of [1, 2, 3]) { let d = null; try { d = await headFile(root, dir, h); } catch { /* none */ } ok(!d || d.length === 0, `head ${h}: ${d?.length}`); }
  // the next run copies what is missing
  const ok2 = source();
  ok(await ingestNgram(cfg, ok2.readAt, { dir, storage }) >= 192);
});

Deno.test("ingestNgram: two ingests of one directory run one after the other (the second copies nothing)", async () => {
  const root = new FakeDir(), storage = storageOf(root), dir = ngramDir("aaaa0003");
  const { readAt, st } = source({ slow: 1 });
  const [a, b] = await Promise.all([ingestNgram(cfg, readAt, { dir, storage }), ingestNgram(cfg, readAt, { dir, storage })]);
  eq([a, b].sort(), [0, 256]);
  eq(st.reads, 4, "each head read once (chunkRows covers a head)");
});

Deno.test("ingestNgram: a new table's copy removes the old ones (another signature, the unkeyed layout)", async () => {
  const root = new FakeDir(), storage = storageOf(root);
  let d = root; for (const p of NGRAM_OPFS_DIR) d = await d.getDirectoryHandle(p, { create: true });
  await d.getFileHandle(ngramFileName(0), { create: true });   // the layout before ngramDir
  await d.getDirectoryHandle("old00000", { create: true });
  await ingestNgram(cfg, source().readAt, { dir: ngramDir("new00000"), storage });
  const left = []; for await (const k of d.keys()) left.push(k);
  eq(left, ["new00000"]);
});

Deno.test("ngramContentSig: same table, same signature; other bytes, size or file, another", async () => {
  const sig = (c, rd, url = "u@rev1") => ngramContentSig(c, rd, { url });
  const a = await sig(cfg, source().readAt);
  eq(await sig(cfg, source().readAt), a);
  ok(await sig(cfg, source().readAt, "u@rev2") !== a, "another revision");
  ok(await sig({ ...cfg, vocab: [8, 8, 8, 7] }, source().readAt) !== a, "another size");
  const other = async (f, off, len) => { const b = table.slice(off, off + len); b[0] ^= 1; return b; };
  ok(await sig(cfg, other) !== a, "other bytes");
});

Deno.test("flashDiskBytes / removeFlashCopy: what the copies take, and deleting them", async () => {
  const root = new FakeDir(), storage = storageOf(root);
  await ingestNgram(cfg, source().readAt, { dir: ngramDir("aaaa0004"), storage });
  const ex = await (await root.getDirectoryHandle(EXPERT_OPFS_DIR, { create: true })).getDirectoryHandle("Model-1234-abcd", { create: true });
  (await ex.getFileHandle("L000.bin", { create: true })).data = new Uint8Array(1000);
  eq(await flashDiskBytes({ storage }), { experts: 1000, ngram: 256, total: 1256 });
  // removeFlashCopy deletes the experts through removeExperts (navigator.storage): point it at this root
  const nav = Object.getOwnPropertyDescriptor(navigator, "storage");
  Object.defineProperty(navigator, "storage", { configurable: true, value: storage });
  try { eq(await removeFlashCopy({ storage }), { removed: 1256, kept: 0 }); }
  finally { if (nav) Object.defineProperty(navigator, "storage", nav); else delete navigator.storage; }
  eq(await flashDiskBytes({ storage }), { experts: 0, ngram: 0, total: 0 });
});
