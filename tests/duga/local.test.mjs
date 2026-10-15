import { test } from 'node:test';
import assert from 'node:assert/strict';
import { localRangeFetcher } from '../../app/local.js';
const url = 'https://huggingface.co/model/weights.gguf';
const bytes = [1,2,3,4];
test('model transport sends only byte ranges, omits credentials and caches validated bytes', async () => {
  let requests = 0, stored;
  const cache = { match:async () => stored?.clone(), put:async (_,v) => { stored = v; } };
  const read = localRangeFetcher({ cache, fetchImpl:async (u,opts) => {
    requests++; assert.equal(u,url); assert.equal(opts.headers.Range,'bytes=0-3'); assert.equal(opts.credentials,'omit'); assert.equal(opts.body,undefined);
    return new Response(new Uint8Array(bytes), { status:206 });
  } });
  assert.deepEqual([...new Uint8Array(await (await read(url,0,3)).arrayBuffer())], bytes);
  await read(url,0,3); assert.equal(requests,1);
  await read(url,0,3,true); assert.equal(requests,2);
});
test('truncated or non-range download never enters the model cache', async () => {
  let writes = 0;
  for (const response of [new Response('whole file'), new Response(new Uint8Array([1]),{status:206})]) {
    const read = localRangeFetcher({ cache:{ match:async () => null, put:async () => { writes++; } }, fetchImpl:async () => response });
    await assert.rejects(read(url,0,3));
  }
  assert.equal(writes,0);
});
test('unavailable cache is optional; corrupt cache is refetched; cancellation stops before fetching', async () => {
  const bad = { match:async () => new Response('x'), put:async () => { throw Error('quota'); } };
  let requests = 0;
  const ctrl = new AbortController();
  const read = localRangeFetcher({ cache:bad, signal:ctrl.signal, fetchImpl:async () => { requests++; return new Response(new Uint8Array(bytes),{status:206}); } });
  assert.equal((await (await read(url,0,3)).arrayBuffer()).byteLength,4);
  ctrl.abort(); await assert.rejects(read(url,0,3)); assert.equal(requests,1);
  await assert.rejects(localRangeFetcher()(url,-1,2));
});
test('Android Local selects the native bridge before touching WebGPU', async () => {
  const {loadLocal}=await import('../../app/local.js');const calls=[];
  globalThis.DugaNativeRuntime={call:async type=>{calls.push(type);throw Error('native oracle gate refused this device');}};
  try {await assert.rejects(loadLocal(),/native oracle gate/);assert.equal(calls[0],'load');}
  finally {delete globalThis.DugaNativeRuntime;}
});
