import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import { buildDense } from '../../tests/e2e/synth_dense.mjs';
import { parseGGUFHeader,ggufWeights } from '../../engine/gguf.js';
import { DenseEngine } from '../../engine/dense.js';
import { makeTokenizer } from '../../engine/tokenizer.js';
import { buildIds,specials } from '../../room/conversation.js';
import { systemPrompt } from '../../app/state.js';
const argmax=logits=>{let best=0;for(let i=1;i<logits.length;i++)if(logits[i]>logits[best])best=i;return best;};
const {create,globals}=await import(pathToFileURL(path.resolve(process.env.DUGA_WEBGPU_MODULE)).href);Object.assign(globalThis,globals);
function runModel(file,tokens,out) {
 const runner=process.env.DUGA_NATIVE_RUNNER || 'native/vulkan/build/duga-vk-model';
 const serial=process.env.DUGA_ANDROID_SERIAL;
 if(!serial)return spawnSync(runner,[file,tokens,out],{encoding:'utf8'});
 const adb=process.env.ADB || 'adb',remote='/data/local/tmp/duga-model-check';
 const call=args=>{
  const result=spawnSync(adb,['-s',serial,...args],{encoding:'utf8',timeout:300000});
  if(result.status!==0)throw Error(result.stderr||result.stdout||result.error);
  return result;
 };
 call(['shell','mkdir','-p',remote]);
 call(['push',runner,`${remote}/runner`]);
 call(['shell','chmod','755',`${remote}/runner`]);
 call(['push',file,`${remote}/model.gguf`]);
 call(['push',tokens,`${remote}/tokens.txt`]);
 try {
  const result=call(['shell',`${remote}/runner`,`${remote}/model.gguf`,`${remote}/tokens.txt`,`${remote}/actual.bin`]);
  call(['pull',`${remote}/actual.bin`,out]);
  return result;
 } finally {call(['shell','rm','-rf',remote]);}
}
const gpu=create(['backend=vulkan',...(process.env.DUGA_ORACLE_ADAPTER?[`adapter=${process.env.DUGA_ORACLE_ADAPTER}`]:[])]);let failures=0;
for(const untied of (process.env.DUGA_REAL_GGUF?[false]:[false,true])){
 const fixture=process.env.DUGA_REAL_GGUF?{gguf:fs.readFileSync(process.env.DUGA_REAL_GGUF),config:JSON.parse(fs.readFileSync(process.env.DUGA_REAL_CONFIG,'utf8'))}:buildDense({layers:2,dim:128,inter:256,nH:4,nKV:2,hd:64,untied});const dir='native/vulkan/build';fs.mkdirSync(dir,{recursive:true});const file=`${dir}/dense-${untied?'untied':'tied'}.gguf`;fs.writeFileSync(file,fixture.gguf);
 const G=parseGGUFHeader(fixture.gguf.buffer.slice(fixture.gguf.byteOffset,fixture.gguf.byteOffset+fixture.gguf.byteLength));
 const weights=await ggufWeights(G,async i=>fixture.gguf.slice(i.byteOffset,i.byteOffset+i.byteLength),{lo:0,hi:fixture.config.num_hidden_layers,hasEmbed:true,hasHead:true});
 const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice({requiredLimits:{maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}});const errs=[];device.addEventListener('uncapturederror',e=>errs.push(e.error.message));
 const engine=await DenseEngine.create({device,cfg:fixture.config,weights,layerRange:[0,fixture.config.num_hidden_layers],maxSeq:1024,coopWG:256,coopRows:4,attnFast:false,fuse:false,mergeQKV:false});
 const chat=process.env.DUGA_CHAT_TOKENIZER;
 if(chat&&!process.env.DUGA_REAL_GGUF)throw Error('A real model is required for the chat oracle.');
 const tok=chat?makeTokenizer(JSON.parse(fs.readFileSync(chat,'utf8'))):null;
 const prompt=process.env.DUGA_CHAT_PROMPT || 'Reply with OK only.';
 const ids=chat?buildIds(tok,{system:systemPrompt(),turns:[{role:'user',text:prompt}],thinking:false}):process.env.DUGA_REAL_GGUF?[33,72,9]:[33,72,9,201,111,62,45,99];
 if(ids.length+32>1024)throw Error('Oracle prompt exceeds the native context.');
 let expected=[];for(const id of ids)expected.push(Float32Array.from(await engine.forwardToken(id)));
 if(chat){
  const S=specials(tok),stops=new Set([S.imEnd,S.eot]),output=[];
  for(let i=0;i<32;i++){
   const next=argmax(expected.at(-1));if(stops.has(next))break;
   output.push(next);ids.push(next);expected.push(Float32Array.from(await engine.forwardToken(next)));
  }
  const result={prompt,text:tok.decode(output),tokens:output,promptTokens:ids.length-output.length,modelSHA256:crypto.createHash('sha256').update(fixture.gguf).digest('hex')};
  fs.writeFileSync(`${dir}/chat-oracle.json`,JSON.stringify(result,null,2));console.log('Original WebGPU chat oracle:',JSON.stringify(result));
 }
 await device.queue.onSubmittedWorkDone();device.destroy();if(errs.length)throw Error(errs.join('\n'));
 fs.writeFileSync(`${dir}/tokens.txt`,ids.join('\n'));const run=runModel(file,`${dir}/tokens.txt`,`${dir}/actual.bin`);if(run.status!==0)throw Error(run.stderr||run.error);process.stdout.write(run.stdout);
 const actual=fs.readFileSync(`${dir}/actual.bin`),oracle=Buffer.concat(expected.map(v=>Buffer.from(v.buffer)));if(actual.length!==oracle.length)throw Error('Logit byte count mismatch');let mismatches=0,maxAbs=0;for(let i=0;i<actual.length;i+=4){if(actual.readUInt32LE(i)!==oracle.readUInt32LE(i))mismatches++;maxAbs=Math.max(maxAbs,Math.abs(actual.readFloatLE(i)-oracle.readFloatLE(i)));}
 if(!process.env.DUGA_REAL_GGUF && !mismatches){
  const fixtures=`${dir}/fixtures`,name=`dense-${untied?'untied':'tied'}`;fs.mkdirSync(fixtures,{recursive:true});
  fs.copyFileSync(file,`${fixtures}/${name}.gguf`);
  fs.writeFileSync(`${fixtures}/${name}.expected.bin`,oracle);
  const greedy=expected.map(argmax);
  fs.writeFileSync(`${fixtures}/${name}.json`,JSON.stringify({tokens:ids,greedy,vocab:fixture.config.vocab_size},null,2));
 }
 console.log(`${mismatches?'FAIL':'PASS'} dense ${untied?'untied':'tied'}: ${ids.length} positions, all ${actual.length/4} logits, bitMismatch=${mismatches}, maxAbs=${maxAbs}`);if(mismatches)failures++;
}
process.exit(failures?2:0);
