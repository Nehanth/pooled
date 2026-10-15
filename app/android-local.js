// Android-only adapter: JS handles the unchanged tokenizer/template; JNI dispatches native Vulkan.
// The bridge is injected into bundled Android pages only. There is no WebGPU or remote inference fallback.
import { buildIds, specials } from '../room/conversation.js';
import { makeTokenizer } from '../engine/tokenizer.js';
import { OwnIds } from '../harness/model-common.js';
export async function loadAndroidLocal({ signal,onStatus=()=>{},onProgress=()=>{} }={}) {
  signal?.throwIfAborted();
  const call=globalThis.DugaNativeRuntime.call;
  const stop=()=>{call('cancel').catch(()=>{});};
  signal?.addEventListener('abort',stop,{once:true});
  let info,tok;
  try {
    info=await call('load',{},p=>{onStatus(p.status||'Preparing native Vulkan…');if(p.total)onProgress(p.done,p.total);});signal?.throwIfAborted();
    const response=await fetch('/models/tokenizer.json',{signal});if(!response.ok)throw Error('Native tokenizer unavailable.');
    tok=makeTokenizer(await response.json());signal?.throwIfAborted();
  }
  catch(e){stop();await call('close').catch(()=>{});throw e;}
  finally { signal?.removeEventListener('abort',stop); }
  const own=new OwnIds(),S=specials(tok),stops=new Set([S.imEnd,S.eot].filter(Number.isInteger));
  let disposed=false;
  return {
    dispose(){if(disposed)return;disposed=true;stop();call('close').catch(()=>{});},
    async *generate({system='',turns,signal,onStatus=()=>{},onTokens=()=>{}}){
      if(disposed)throw Error('Native model has been released.');
      own.prune(turns.filter(t=>t.role==='assistant').map(t=>t.text));
      const rendered=turns.map(t=>t.role==='assistant'?{role:t.role,ids:t.ids??own.get(t.text)??tok.encode(t.text)}:t);
      const ids=buildIds(tok,{system,turns:rendered,thinking:false});
      if(ids.length+2>=info.context)throw Error(`This native preview has a ${info.context}-token context. Shorten the question or start a new conversation.`);
      signal?.throwIfAborted();signal?.addEventListener('abort',stop,{once:true});let text='';const output=[];
      try {
        // Serial prefill matches Dense.forwardToken. Bound each bridge request so a slow
        // device can report progress and cancel without timing out a whole prompt batch.
        let result;
        for(let i=0;i<ids.length;i++){
          signal?.throwIfAborted();
          onStatus(`Reading the question on this device · ${i+1}/${ids.length} tokens`);
          result=await call('evaluate',{ids:[ids[i]],reset:i===0});
        }
        onStatus('Writing the answer on this device…');
        const max=Math.min(384,info.context-ids.length-2);
        for(let i=0;i<max&&!signal?.aborted&&!stops.has(result.token);i++){
          output.push(result.token);const now=tok.decode(output);if(now.length>text.length&&!now.endsWith('\uFFFD')){yield now.slice(text.length);text=now;}
          if(i+1<max)result=await call('evaluate',{ids:[result.token],reset:false});
        }
        const all=tok.decode(output);
        if(all.length>text.length)yield all.slice(text.length);
        own.set(all,output.slice());
        // Session history outlives the GPU model when Android pauses the app.
        onTokens(output.slice());
      } finally {signal?.removeEventListener('abort',stop);}
    },
  };
}
