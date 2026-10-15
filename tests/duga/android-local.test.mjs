import {test} from 'node:test';
import assert from 'node:assert/strict';
import {buildDense} from '../e2e/synth_dense.mjs';
import {makeTokenizer} from '../../engine/tokenizer.js';
import {buildIds,specials} from '../../room/conversation.js';
import {loadAndroidLocal} from '../../app/android-local.js';

test('an already cancelled Android preparation makes no native or tokenizer requests',async()=>{
  const originalFetch=globalThis.fetch,originalRuntime=globalThis.DugaNativeRuntime;
  const calls=[],abort=new AbortController();abort.abort();
  globalThis.fetch=async()=>{calls.push('fetch');throw Error('Unexpected tokenizer request');};
  globalThis.DugaNativeRuntime={call:async type=>{calls.push(type);return {context:1024};}};
  try{
    await assert.rejects(loadAndroidLocal({signal:abort.signal}),{name:'AbortError'});
    assert.deepEqual(calls,[]);
  }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
});

test('cancelled native opening closes the model before preparation rejects',async()=>{
  const originalFetch=globalThis.fetch,originalRuntime=globalThis.DugaNativeRuntime;
  const calls=[],abort=new AbortController();let finishOpen;
  globalThis.fetch=async()=>{calls.push('fetch');throw Error('Unexpected tokenizer request');};
  globalThis.DugaNativeRuntime={call:async type=>{
    calls.push(type);
    if(type==='load')return new Promise(resolve=>{finishOpen=resolve;});
    if(type==='close')throw Error('Cleanup failure must not hide cancellation');
    return {};
  }};
  try{
    const loading=loadAndroidLocal({signal:abort.signal});
    abort.abort();
    assert.deepEqual(calls,['load','cancel']);
    // NativeVulkan.open can publish its handle after cancel has already run.
    finishOpen({context:1024});
    await assert.rejects(loading,{name:'AbortError'});
    assert.deepEqual(calls,['load','cancel','cancel','close']);
    abort.signal.dispatchEvent(new Event('abort'));
    assert.equal(calls.length,4,'Preparation must remove its abort listener');
  }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
});

test('tokenizer preparation errors and cancellation release the native model',async t=>{
  const fixture=buildDense({layers:1,dim:128,inter:256,nH:4,nKV:2,hd:64});
  for(const failure of ['fetch','json','abort'])await t.test(failure,async()=>{
    const originalFetch=globalThis.fetch,originalRuntime=globalThis.DugaNativeRuntime;
    const calls=[],abort=new AbortController(),error=Error('Tokenizer failed');
    globalThis.DugaNativeRuntime={call:async type=>{calls.push(type);return {context:1024};}};
    globalThis.fetch=async(url,options)=>{
      assert.equal(url,'/models/tokenizer.json');assert.equal(options?.signal,abort.signal);
      if(failure==='fetch')return new Response('',{status:404});
      return {ok:true,json:async()=>{
        if(failure==='json')throw error;
        abort.abort();return fixture.tokenizer;
      }};
    };
    try{
      await assert.rejects(loadAndroidLocal({signal:abort.signal}),failure==='abort'?{name:'AbortError'}:failure==='json'?error:/Native tokenizer unavailable/);
      assert.equal(calls[0],'load');assert.equal(calls.at(-1),'close');
      assert.ok(calls.includes('cancel'));
      const count=calls.length;abort.signal.dispatchEvent(new Event('abort'));
      assert.equal(calls.length,count,'Preparation must remove its abort listener');
    }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
  });
});

test('Android decoding flushes an incomplete UTF-8 suffix when generation ends',async t=>{
  const fixture=buildDense({layers:1,dim:128,inter:256,nH:4,nKV:2,hd:64});
  // One token holds a complete Cyrillic character and the start of another,
  // reproducing the pinned tokenizer's token 2226 without external model files.
  const incomplete=Math.max(...Object.values(fixture.tokenizer.model.vocab),...fixture.tokenizer.added_tokens.map(t=>t.id))+1;
  fixture.tokenizer.model.vocab['Ð¾Ð']=incomplete;
  const tok=makeTokenizer(fixture.tokenizer),S=specials(tok),turns=[{role:'user',text:'Hello'}];
  assert.equal(tok.decode([incomplete]),'о\uFFFD');
  const prompt=buildIds(tok,{turns,thinking:false}),a=tok.encode('A')[0],continuation=tok.encode('о')[1];
  for(const {name,output,budget} of [
    {name:'one-token budget',output:[incomplete],budget:1},
    {name:'suffix after emitted text',output:[a,incomplete],budget:2},
    {name:'stop token',output:[incomplete],budget:3},
    {name:'completed UTF-8',output:[incomplete,continuation],budget:2},
  ])await t.test(name,async()=>{
    const originalFetch=globalThis.fetch,originalRuntime=globalThis.DugaNativeRuntime;
    let evaluated=0,stored;const chunks=[];
    globalThis.fetch=async()=>new Response(JSON.stringify(fixture.tokenizer));
    globalThis.DugaNativeRuntime={call:async type=>{
      if(type==='load')return {context:prompt.length+budget+2};
      if(type==='evaluate'){const generated=++evaluated-prompt.length;return {token:generated<0?S.imEnd:output[generated]??S.imEnd};}
      return {};
    }};
    try{
      const model=await loadAndroidLocal();
      try{
        for await(const delta of model.generate({turns,onTokens:ids=>{stored=ids;}}))chunks.push(delta);
        assert.equal(chunks.join(''),tok.decode(output));
        assert.deepEqual(stored,output);
        if(name==='completed UTF-8')assert.deepEqual(chunks,['оо'],'Do not emit replacement characters before decoding finishes');
      }finally{model.dispose();}
    }finally{globalThis.fetch=originalFetch;if(originalRuntime===undefined)delete globalThis.DugaNativeRuntime;else globalThis.DugaNativeRuntime=originalRuntime;}
  });
});

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
