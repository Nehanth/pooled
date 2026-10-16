import { tokenBytes, sha256, makeRecord, firstDifference } from "../../benchmarks/bench_room_record.js";

const eq = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${JSON.stringify(a)} != ${JSON.stringify(b)}`); };
const ok = (v) => { if (!v) throw new Error("assertion failed"); };
const manifest = () => ({ source: { commit: "a".repeat(40), treeSha256: "b".repeat(64) }, harnessSha256: "c".repeat(64),
  model: { name: "qwen3-1.7b", ...Object.fromEntries(["gguf", "config", "tokenizer"].map((p) => [p, { sha256: "d".repeat(64), revision: "pinned" }])) },
  devices: [{ alias: "R", os: "test", runtime: "test", backend: "test", adapter: {}, limits: {}, features: [] }],
  cache: { weights: "resident", shaders: "warm", source: "local" } });
const request = () => ({ run: "r", attempt: "a", trial: "measured", promptName: "hello", host: "R", manifest: manifest(),
  sample: { status: "pass", before: { chain: [] }, prompt: "hello", inputIds: [1, 256], observed: [7], result: { tokens: [7] } } });

Deno.test("record: portable canonical token encoding and known SHA-256", async () => {
  eq([...tokenBytes([1, 256, 0xffffffff])], [1, 0, 0, 0, 0, 1, 0, 0, 255, 255, 255, 255]);
  eq(await sha256(tokenBytes([])), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  for (const ids of [[-1], [1.5], [2 ** 32], [NaN]]) { let error; try { tokenBytes(ids); } catch (e) { error = e; } ok(error); }
  eq(firstDifference([1, 2], [1, 3]), { index: 1, expected: 2, actual: 3 });
  eq(firstDifference([1], [1, 2]), { index: 1, expected: null, actual: 2 });
  eq(firstDifference([1], [1]), null);
});

Deno.test("record: keeps failed partial samples and rejects bad timing separately", async () => {
  const r = request(); r.sample.status = "failed"; r.sample.result = null;
  r.captures = { R: { attempt: "wrong", epoch: 1, dropped: 0, records: [] } };
  const record = await makeRecord(r);
  eq([record.version, record.status, record.timing.status], [1, "failed", "incomplete"]);
  eq(record.sample.observed, [7]); eq(record.captures, r.captures);
  eq(record.hashes.output, await sha256(tokenBytes([7])));
  const passed = await makeRecord(request()); eq(passed.timing.status, "disabled");
});

Deno.test("record: requires reproducible source, model, cache and device identities", async () => {
  for (const corrupt of [(r) => r.manifest.source.commit = "main", (r) => delete r.manifest.model.tokenizer,
    (r) => delete r.manifest.cache, (r) => r.sample.before.chain.push("unknown")]) {
    const r = request(); corrupt(r); let error; try { await makeRecord(r); } catch (e) { error = e; } ok(error);
  }
});
