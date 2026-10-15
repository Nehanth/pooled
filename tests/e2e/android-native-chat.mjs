// Manual real-model Android UI check. Requires Node 22, a dedicated emulator,
// a debug APK and an original-WebGPU chat oracle from check-model.mjs.
// No inference is mocked. WebGPU is disabled in the WebView for this test.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';

const serial=process.env.DUGA_ANDROID_SERIAL || 'emulator-5554';
if(!/^emulator-\d+$/.test(serial))throw Error('Use a dedicated Android emulator; this test restarts its Duga app.');
const sdk=process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT;
const adb=process.env.ADB || (sdk?path.join(sdk,'platform-tools/adb'):'adb');
const oracle=JSON.parse(fs.readFileSync(process.env.DUGA_CHAT_ORACLE || 'native/vulkan/build/chat-oracle.json','utf8'));
if(typeof oracle.prompt!=='string'||typeof oracle.text!=='string'||!oracle.text)throw Error('Generate a non-empty original WebGPU chat oracle first.');
const timeout=Number(process.env.DUGA_ANDROID_CHAT_TIMEOUT_MS || 20*60*1000);
if(!Number.isFinite(timeout)||timeout<1000||timeout>2*60*60*1000)throw Error('Invalid chat-test timeout');
function command(...args){
  const result=spawnSync(adb,['-s',serial,...args],{encoding:'utf8',timeout:30000});
  if(result.status!==0)throw Error(result.stderr || result.stdout || result.error);
  return result.stdout.trim();
}
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
command('shell','am','force-stop','app.duga.android');
command('shell','am','start','-n','app.duga.android/app.duga.android.MainActivity');
const pid=command('shell','pidof','app.duga.android');
if(!/^\d+$/.test(pid))throw Error('Duga did not start');
const port=command('forward','tcp:0',`localabstract:webview_devtools_remote_${pid}`);
if(!/^\d+$/.test(port))throw Error('Could not forward the debug WebView');
let ws,finished=false;
try {
  let target;
  for(let i=0;i<60;i++){
    try{target=(await(await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(p=>p.url.startsWith('https://appassets.androidplatform.net/'));}catch{}
    if(target)break;await delay(500);
  }
  if(!target)throw Error('Install a debug APK with WebView debugging enabled');
  ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let seq=0;const pending=new Map(),errors=[],external=[];
  ws.onmessage=e=>{
    const m=JSON.parse(e.data);
    if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}
    else if(m.method==='Runtime.exceptionThrown')errors.push(m.params);
    else if(m.method==='Network.requestWillBeSent'&&!m.params.request.url.startsWith('https://appassets.androidplatform.net/'))external.push(m.params.request.url);
  };
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{
    const id=++seq,timer=setTimeout(()=>{pending.delete(id);reject(Error(`DevTools timeout: ${method}`));},30000);
    pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
  });
  const js=async expression=>{
    const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});
    if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;
  };
  await cdp('Runtime.enable');await cdp('Network.enable');
  for(let i=0;i<60;i++){if(await js("!!window.DugaNativeRuntime && !!document.querySelector('#prompt')"))break;await delay(500);}
  assert.equal(await js('!!window.DugaNativeRuntime'),true);
  await js("Object.defineProperty(navigator,'gpu',{value:undefined,configurable:true})");
  assert.equal(await js('!!navigator.gpu'),false);
  await js(`document.querySelector('#prompt').value=${JSON.stringify(oracle.prompt)};document.querySelector('#prompt').closest('form').requestSubmit();`);
  command('shell','uiautomator','dump','/sdcard/duga-test-window.xml');
  const xml=command('exec-out','cat','/sdcard/duga-test-window.xml');
  const node=xml.match(/<node\b[^>]*text="[^"]*Validate &amp; download[^"]*"[^>]*>/i)?.[0];
  if(!node)throw Error('Native experimental-backend confirmation is missing');
  const bounds=node.match(/bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"/).slice(1).map(Number);
  command('shell','input','tap',String(Math.round((bounds[0]+bounds[2])/2)),String(Math.round((bounds[1]+bounds[3])/2)));
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline){
    const state=await js("({status:document.querySelector('#run-status').textContent,reply:document.querySelector('.message.assistant .message-body')?.textContent,failed:!!document.querySelector('.message.failed')})");
    console.log(JSON.stringify(state));
    if(state.failed)throw Error(state.reply);
    if(state.status.startsWith('Finished')){assert.equal(state.reply,oracle.text);finished=true;break;}
    await delay(1000);
  }
  assert.ok(finished,'Real native-model chat did not finish');assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  const shot=await cdp('Page.captureScreenshot',{format:'png'});
  const screenshot=process.env.DUGA_ANDROID_SCREENSHOT || 'native/vulkan/build/android-chat.png';fs.writeFileSync(screenshot,Buffer.from(shot.data,'base64'));
  console.log('PASS Android native Vulkan UI answer equals original WebGPU; WebView WebGPU disabled; no WebView remote requests');
}finally{ws?.close();command('forward','--remove',`tcp:${port}`);}
