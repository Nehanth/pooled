// CPU check of Qwen4ExpBatch's state guards (engine/qwen4exp_batch.js) on a stub engine: accept() takes only the
// verify pass right before it (any step that moved the engine since makes it refuse), and a failed pass or
// Qwen4ExpEngine.reset() forgets which layer the expert store's region holds. The passes themselves (_run) and the
// state copies are stubbed; the GPU checks are tests/test_q38flash_batch.js.
// No GPU.   deno test --no-check tests/unit/q38flash_batch_state_test.js
import { Qwen4ExpBatch } from "../../engine/qwen4exp_batch.js";
import { Qwen4ExpEngine } from "../../engine/qwen4exp.js";

const assert = (c, m) => { if (!c) throw new Error(m); };
const rejects = async (p, re) => {
  let err = null;
  try { await p; } catch (e) { err = e; }
  assert(err && re.test(err.message), `expected a rejection matching ${re}, got ${err ? err.message : "none"}`);
};

// an engine with what the batch's bookkeeping touches; reset / forwardToken bump _posGen as the real engine does
function stub({ region = false } = {}) {
  const eng = Object.create(Qwen4ExpEngine.prototype);
  Object.assign(eng, {
    pos: 0, _posGen: 0, _failed: null, maxSeq: 4096, layers: [],
    device: { createCommandEncoder: () => ({ clearBuffer() {}, finish: () => ({}) }), queue: { submit() {} } },
    ngs: { rolled: 0, reset() {}, rollback(n) { this.rolled += n; } },
    experts: region ? { region: { layer: -1 } } : null,
  });
  eng.forwardToken = async (id) => { eng._check(id); eng._posGen++; eng.pos++; return {}; };   // the real one's bookkeeping
  const pb = Object.create(Qwen4ExpBatch.prototype);
  const runs = [];
  Object.assign(pb, { eng, maxCols: 64, verifyCols: 8, cfg: { hDim: 4 }, _pending: null, fail: null, runs });
  pb._save = () => {}; pb._restore = () => { pb.restored = (pb.restored || 0) + 1; };
  pb._run = async (ids, opts) => {
    runs.push({ pos: eng.pos, n: ids.length, head: opts.head });
    if (pb.fail) { if (eng.experts) eng.experts.region.layer = 7; throw pb.fail; }   // a region load encoded, never run
    eng.pos += ids.length;
    return opts.head === "all" ? { ids: ids.map((x) => x + 1) } : {};
  };
  return { eng, pb, runs };
}

Deno.test("accept right after verify: keeps n columns (restore, roll back, re-run n)", async () => {
  const { eng, pb, runs } = stub();
  await pb.prefill([1, 2, 3], { head: "none" });
  await pb.verify([4, 5, 6, 7]);
  assert(eng.pos === 7, `pos ${eng.pos}`);
  await pb.accept(2);
  assert(eng.pos === 5 && pb.restored === 1 && eng.ngs.rolled === 4, `pos ${eng.pos}, restored ${pb.restored}, rolled ${eng.ngs.rolled}`);
  assert(runs.at(-1).pos === 3 && runs.at(-1).n === 2, "the re-run starts at the verify's position with 2 columns");
  await rejects(pb.accept(1), /without a verify pass/);   // one accept per verify
});

Deno.test("accept(all) right after verify: nothing to undo", async () => {
  const { eng, pb } = stub();
  await pb.verify([1, 2, 3]);
  await pb.accept(3);
  assert(eng.pos === 3 && !pb.restored, "no restore");
});

Deno.test("accept after forwardToken refuses (stale verify snapshot)", async () => {
  const { eng, pb } = stub();
  await pb.verify([1, 2, 3]);
  await eng.forwardToken(9);
  await rejects(pb.accept(1), /moved past its verify pass/);
  assert(eng.pos === 4 && !pb.restored, "the state is left alone");
});

Deno.test("accept after a prefill refuses", async () => {
  const { pb } = stub();
  await pb.verify([1, 2, 3]);
  await pb.prefill([4, 5], { head: "none" });
  await rejects(pb.accept(1), /without a verify pass/);
});

Deno.test("accept after reset() refuses, also when the engine is back at the same position", async () => {
  const { eng, pb } = stub();
  await pb.verify([1, 2, 3]);
  eng.reset();
  await rejects(pb.accept(1), /moved past its verify pass/);
  await pb.verify([1, 2, 3]);
  eng.reset();
  for (const t of [1, 2, 3]) await eng.forwardToken(t);   // pos == pos0 + 3 again, a different history
  await rejects(pb.accept(1), /moved past its verify pass/);
});

Deno.test("a failed verify leaves nothing to accept", async () => {
  const { pb } = stub();
  await pb.verify([1, 2]);
  pb.fail = new Error("boom");
  await rejects(pb.verify([3, 4]), /boom/);
  await rejects(pb.accept(1), /without a verify pass/);
});

Deno.test("a failed pass forgets the region's layer; so does reset()", async () => {
  const { eng, pb } = stub({ region: true });
  pb.fail = new Error("mapAsync failed");
  await rejects(pb.prefill([1, 2, 3], { head: "none" }), /mapAsync failed/);
  assert(eng.experts.region.layer === -1, `region.layer ${eng.experts.region.layer} after a failed pass`);
  assert(eng._failed, "the engine refuses until reset");
  eng.experts.region.layer = 5;
  eng.reset();
  assert(eng.experts.region.layer === -1 && !eng._failed && eng.pos === 0, "reset clears the region's layer and the failure");
});
