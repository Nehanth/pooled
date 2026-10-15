import {test} from 'node:test';
import assert from 'node:assert/strict';
import {buildDense} from '../e2e/synth_dense.mjs';
import {makeTokenizer} from '../../engine/tokenizer.js';
import {buildIds,specials} from '../../room/conversation.js';
import {loadAndroidLocal} from '../../app/android-local.js';

test('Android prefill preserves exact prompt order, bounds bridge work and stops on cancellation',async()=>{
  const fixture=buildDense({layers:1,dim:128,inter:256,nH:4,nKV:2,hd:64});
  const tok=makeTokenizer(fixture.tokenizer),S=specials(tok),turns=[{role:'user',text:'Hello'}];
  const ids=buildIds(tok,{turns,thinking:false}),answer=tok.encode('A')[0];
  const originalFetch=globalThis.fetch,originalRuntime=globalThis.DugaNativeRuntime;
  const calls=[],statuses=[];let evaluated=0,abort;
  globalThis.fetch=async url=>{assert.equal(url,'/models/tokenizer.json');return new Response(JSON.stringify(fixture.tokenizer));};
  globalThis.DugaNativeRuntime={call:async(type,data)=>{
    calls.push({type,data});
    if(type==='load')return {context:1024};
    if(type==='evaluate'){
      evaluated++;
      if(abort&&evaluated===2)abort.abort();
      return {token:evaluated===ids.length?answer:S.imEnd};
    }
    return {};
  }};
  try {
    const model=await loadAndroidLocal();let text='';
    for await(const delta of model.generate({turns,onStatus:s=>statuses.push(s)}))text+=delta;
    assert.equal(text,tok.decode([answer]));
    const prefill=calls.filter(c=>c.type==='evaluate').slice(0,ids.length);
    assert.deepEqual(prefill.map(c=>c.data.ids),ids.map(id=>[id]));
    assert.deepEqual(prefill.map(c=>c.data.reset),ids.map((_,i)=>i===0));
    assert.equal(statuses.length,ids.length+1);
    calls.length=0;evaluated=0;abort=new AbortController();
    await assert.rejects(async()=>{for await(const delta of model.generate({turns,signal:abort.signal})){void delta;}},{name:'AbortError'});
    assert.equal(calls.filter(c=>c.type==='evaluate').length,2);
    assert.ok(calls.some(c=>c.type==='cancel'));
    model.dispose();
  }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
});
