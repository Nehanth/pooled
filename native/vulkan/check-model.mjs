import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
import { buildDense } from '../../tests/e2e/synth_dense.mjs';
import { parseGGUFHeader,ggufWeights } from '../../engine/gguf.js';
import { DenseEngine } from '../../engine/dense.js';
const {create,globals}=await import(pathToFileURL(path.resolve(process.env.DUGA_WEBGPU_MODULE)).href);Object.assign(globalThis,globals);
const gpu=create(['backend=vulkan',...(process.env.DUGA_ORACLE_ADAPTER?[`adapter=${process.env.DUGA_ORACLE_ADAPTER}`]:[])]);let failures=0;
for(const untied of (process.env.DUGA_REAL_GGUF?[false]:[false,true])){
 const fixture=process.env.DUGA_REAL_GGUF?{gguf:fs.readFileSync(process.env.DUGA_REAL_GGUF),config:JSON.parse(fs.readFileSync(process.env.DUGA_REAL_CONFIG,'utf8'))}:buildDense({layers:2,dim:128,inter:256,nH:4,nKV:2,hd:64,untied});const dir='native/vulkan/build';fs.mkdirSync(dir,{recursive:true});const file=`${dir}/dense-${untied?'untied':'tied'}.gguf`;fs.writeFileSync(file,fixture.gguf);
 const G=parseGGUFHeader(fixture.gguf.buffer.slice(fixture.gguf.byteOffset,fixture.gguf.byteOffset+fixture.gguf.byteLength));
 const weights=await ggufWeights(G,async i=>fixture.gguf.slice(i.byteOffset,i.byteOffset+i.byteLength),{lo:0,hi:fixture.config.num_hidden_layers,hasEmbed:true,hasHead:true});
 const adapter=await gpu.requestAdapter(),device=await adapter.requestDevice({requiredLimits:{maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}});const errs=[];device.addEventListener('uncapturederror',e=>errs.push(e.error.message));
 const engine=await DenseEngine.create({device,cfg:fixture.config,weights,layerRange:[0,fixture.config.num_hidden_layers],maxSeq:1024,coopWG:256,coopRows:4,attnFast:false,fuse:false,mergeQKV:false});
 const ids=process.env.DUGA_REAL_GGUF?[33,72,9]:[33,72,9,201,111,62,45,99];let expected=[];for(const id of ids)expected.push(Float32Array.from(await engine.forwardToken(id)));await device.queue.onSubmittedWorkDone();device.destroy();if(errs.length)throw Error(errs.join('\n'));
 fs.writeFileSync(`${dir}/tokens.txt`,ids.join('\n'));const run=spawnSync(`${dir}/duga-vk-model`,[file,`${dir}/tokens.txt`,`${dir}/actual.bin`],{encoding:'utf8'});if(run.status!==0)throw Error(run.stderr||run.error);process.stdout.write(run.stdout);
 const actual=fs.readFileSync(`${dir}/actual.bin`),oracle=Buffer.concat(expected.map(v=>Buffer.from(v.buffer)));if(actual.length!==oracle.length)throw Error('Logit byte count mismatch');let mismatches=0,maxAbs=0;for(let i=0;i<actual.length;i+=4){if(actual.readUInt32LE(i)!==oracle.readUInt32LE(i))mismatches++;maxAbs=Math.max(maxAbs,Math.abs(actual.readFloatLE(i)-oracle.readFloatLE(i)));}
 console.log(`${mismatches?'FAIL':'PASS'} dense ${untied?'untied':'tied'}: ${ids.length} positions, all ${actual.length/4} logits, bitMismatch=${mismatches}, maxAbs=${maxAbs}`);if(mismatches)failures++;
}
process.exit(failures?2:0);
