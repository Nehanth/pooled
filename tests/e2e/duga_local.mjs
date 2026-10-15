// Real WebGPU loader + generation on a tiny Qwen3 fixture. No real-model download or network inference.
import assert from 'node:assert/strict';
import path from 'node:path';
import { buildDense } from './synth_dense.mjs';
import { MODELS } from '../../room/models.js';
import { chromium } from 'playwright';
import { serveRepo, GPU_ARGS } from './engine_synth.mjs';
const root = path.resolve(new URL('../..', import.meta.url).pathname);
const server = serveRepo(0, { '/':path.join(root,'index.html') });
await new Promise(r=>server.listening?r():server.once('listening',r));
const base = `http://127.0.0.1:${server.address().port}`;
const fixture = buildDense({ layers:2,dim:128,inter:256,nH:4,nKV:2,hd:64 });
// The production header probe requests 12 MiB. Padding does not change tensor offsets.
const data = Buffer.alloc(Math.max(fixture.gguf.length,12*2**20)); data.set(fixture.gguf);
const M = MODELS['qwen3-0.6b'];
let browser;
try {
  browser = await chromium.launch({args:GPU_ARGS});
  const page = await browser.newPage(); const network=[];
  await page.route('**/*',async route=>{
    const req = route.request(), u = req.url();
    if(u.startsWith(base)) return route.continue();
    network.push({url:u,method:req.method(),body:req.postData()});
    if(u === M.cfg) return route.fulfill({json:fixture.config});
    if(u === M.tok) return route.fulfill({json:fixture.tokenizer});
    if(u === M.gguf) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(req.headers().range||''); assert.ok(m);
      const lo=+m[1],hi=+m[2];
      return route.fulfill({status:206,headers:{'Content-Range':`bytes ${lo}-${hi}/${data.length}`,'Content-Type':'application/octet-stream'},body:data.subarray(lo,hi+1)});
    }
    throw Error(`Unexpected remote request: ${u}`);
  });
  await page.goto(base);
  const supported = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (!supported) {
    if (process.env.E2E_REQUIRE_GPU === '1') throw new Error('A WebGPU adapter is required for this test.');
    console.log('SKIP real Local generation: this runner has no WebGPU adapter (set E2E_REQUIRE_GPU=1 to require it).');
  } else {
  const result = await page.evaluate(async()=>{
    const {loadLocal} = await import('/app/local.js');
    const progress=[]; const local = await loadLocal({onProgress:(done,total)=>progress.push([done,total])});
    let text='',chunks=0;
    try {
      for await(const delta of local.generate({system:'Be concise.',turns:[{role:'user',text:'Hello'}]})) { text+=delta; if(++chunks===4) break; }
      return {text,chunks,progress:progress.length};
    } finally {local.dispose();}
  });
  assert.ok(result.text.length>0); assert.equal(result.chunks,4); assert.ok(result.progress>0);
  assert.ok(network.length>2); assert.ok(network.every(r=>r.method==='GET'&&!r.body));
  assert.ok(network.every(r=>[M.cfg,M.tok,M.gguf].includes(r.url)));
  console.log('PASS real Local loader, Qwen3 dense WebGPU generation, cancellation via iterator close and model-only GET transport (synthetic weights; not a quality/performance evaluation)');
  }
} finally {await browser?.close();await new Promise(r=>server.close(r));}
