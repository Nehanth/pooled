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
  const calls=[],statuses=[];let evaluated=0,abort,context=1024;
  globalThis.fetch=async url=>{assert.equal(url,'/models/tokenizer.json');return new Response(JSON.stringify(fixture.tokenizer));};
  globalThis.DugaNativeRuntime={call:async(type,data)=>{
    calls.push({type,data});
    if(type==='load')return {context};
    if(type==='evaluate'){
      evaluated++;
      if(abort&&evaluated===2)abort.abort();
      return {token:evaluated===ids.length?answer:S.imEnd};
    }
    return {};
  }};
  try {
    const model=await loadAndroidLocal();let text='',published;
    for await(const delta of model.generate({turns,onStatus:s=>statuses.push(s),onTokens:ids=>{published=ids;}}))text+=delta;
    assert.equal(text,tok.decode([answer]));
    assert.deepEqual(published,[answer]);
    const prefill=calls.filter(c=>c.type==='evaluate').slice(0,ids.length);
    assert.deepEqual(prefill.map(c=>c.data.ids),ids.map(id=>[id]));
    assert.deepEqual(prefill.map(c=>c.data.reset),ids.map((_,i)=>i===0));
    assert.equal(statuses.length,ids.length+1);
    calls.length=0;evaluated=0;abort=new AbortController();
    await assert.rejects(async()=>{for await(const delta of model.generate({turns,signal:abort.signal})){void delta;}},{name:'AbortError'});
    assert.equal(calls.filter(c=>c.type==='evaluate').length,2);
    assert.ok(calls.some(c=>c.type==='cancel'));
    model.dispose();
    context=ids.length+2;calls.length=0;
    const limited=await loadAndroidLocal();
    await assert.rejects(async()=>{for await(const delta of limited.generate({turns})){void delta;}},/context/);
    assert.equal(calls.filter(c=>c.type==='evaluate').length,0,'Do not prefill a prompt with no answer budget');
    limited.dispose();
  }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
});
test('a reloaded Android model replays the exact stored assistant IDs, including noncanonical tokenization',async()=>{
  const fixture=buildDense({layers:1,dim:128,inter:256,nH:4,nKV:2,hd:64});
  const tok=makeTokenizer(fixture.tokenizer),S=specials(tok);
  const split=fixture.tokenizer.model.merges.map(([a,b])=>[tok.vocab[a],tok.vocab[b]]).find(ids=>tok.encode(tok.decode(ids)).length<ids.length);
  assert.ok(split,'Fixture must offer two tokenizations of the same text');
  const text=tok.decode(split),first=[{role:'user',text:'Hello'}];
  const originalFetch=globalThis.fetch,originalRuntime=globalThis.DugaNativeRuntime;
  let expected=buildIds(tok,{turns:first,thinking:false}),output=split,received=[];
  globalThis.fetch=async()=>new Response(JSON.stringify(fixture.tokenizer));
  globalThis.DugaNativeRuntime={call:async(type,data)=>{
    if(type==='load')return {context:1024};
    if(type==='evaluate'){
      received.push(...data.ids);const generated=received.length-expected.length;
      return {token:generated<0?S.imEnd:output[generated]??S.imEnd};
    }
    return {};
  }};
  try{
    const firstModel=await loadAndroidLocal();let stored;
    for await(const delta of firstModel.generate({turns:first,onTokens:ids=>{stored=ids;}})){void delta;}
    assert.deepEqual(stored,split);firstModel.dispose();
    const turns=[...first,{role:'assistant',text,ids:stored},{role:'user',text:'Continue'}];
    expected=buildIds(tok,{turns,thinking:false});output=[];received=[];
    const nextModel=await loadAndroidLocal();
    for await(const delta of nextModel.generate({turns})){void delta;}
    assert.deepEqual(received,expected);assert.notDeepEqual(stored,tok.encode(text));
    nextModel.dispose();
  }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
});
