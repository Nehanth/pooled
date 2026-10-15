// Included only in bundled Android documents. The native object is origin-scoped, never a broad JS interface.
(() => {
  const bridge=window.DugaAndroid;if(!bridge)return;
  // Native distributed workers are not ported yet. Room tabs are chat clients; do not start WebGPU in the APK.
  if(location.pathname==='/room'||location.pathname==='/p2p.html'||location.pathname.startsWith('/r/')){
    try{Object.defineProperty(navigator,'gpu',{value:undefined,configurable:true});}catch{}
  }
  const pending=new Map();
  bridge.onmessage=e=>{let m;try{m=JSON.parse(e.data);}catch{return;}const p=pending.get(m.id);if(!p)return;if(m.progress){p.progress?.(m.progress);return;}pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(m.error)):p.resolve(m.result||{});};
  const call=(type,data={},progress)=>new Promise((resolve,reject)=>{
    const id=crypto.randomUUID(),timer=setTimeout(()=>{pending.delete(id);reject(Error('Android request timed out. Try again.'));},type==='load'?20*60*1000:2*60*1000);
    pending.set(id,{resolve,reject,progress,timer});bridge.postMessage(JSON.stringify({id,type,...data}));
  });
  window.DugaNativeRuntime={call};
  try{Object.defineProperty(navigator,'clipboard',{value:{writeText:text=>call('copy',{text:String(text)})},configurable:true});}catch{}
  try{Object.defineProperty(navigator,'share',{value:data=>call('share',{text:String(data.text||''),url:String(data.url||'')}),configurable:true});Object.defineProperty(navigator,'canShare',{value:data=>!data.files,configurable:true});}catch{}
  const original=HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click=function(){
    if(this.download&&this.href.startsWith('blob:')){
      fetch(this.href).then(r=>r.blob()).then(async blob=>{
        if(blob.size>8*1024*1024)throw Error('Android exports are limited to 8 MiB.');
        const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=reject;reader.readAsDataURL(blob);});
        await call('export',{filename:this.download,mime:blob.type||'application/octet-stream',base64:data});
      }).catch(e=>alert(e.message));return;
    }
    original.call(this);
  };
  addEventListener('DOMContentLoaded',()=>{
    const roomNote=document.querySelector('.jfoot');if(roomNote)roomNote.textContent='This Android port joins rooms as a chat client. Run the room model on another trusted device. Native Vulkan workers for distributed rooms are not available yet.';
    const label=document.querySelector('.model-name');if(label)label.textContent='Qwen3 0.6B · Native Vulkan preview';
    const note=document.querySelector('.bootstrap-note');if(note)note.textContent='Android Local uses native Vulkan. First use downloads about 640 MB; a validation gate runs before loading. This preview has a 1,024-token context. No WebGPU or hosted fallback.';
    const support=document.getElementById('gpu-status');if(support)support.textContent='Native Vulkan preview · device validation runs when you start.';
  });
  addEventListener('duga:android-pause',()=>{
    call('cancel').catch(()=>{});
    const f=document.getElementById('workspace');if(f){f.removeAttribute('src');f.hidden=true;const open=document.getElementById('workspace-open');if(open)open.hidden=false;}
    dispatchEvent(new Event('duga:pause'));
  });
})();
