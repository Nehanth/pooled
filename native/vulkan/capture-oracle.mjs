// Capture actual original WGSL execution. No CPU expectations or tolerance-based pass criteria.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import crypto from 'node:crypto';
import { SOURCE } from './shaders.mjs';
import { pruneWGSL } from '../../engine/wgsl/prune.js';
const modulePath=process.env.DUGA_WEBGPU_MODULE;
if(!modulePath)throw Error('Set DUGA_WEBGPU_MODULE to the installed Dawn webgpu package entry point.');
const { create,globals }=await import(pathToFileURL(path.resolve(modulePath)).href);
Object.assign(globalThis,globals);
const gpu=create(['backend=vulkan',...(process.env.DUGA_ORACLE_ADAPTER?[`adapter=${process.env.DUGA_ORACLE_ADAPTER}`]:[]),...(process.env.DUGA_DAWN_REFERENCE_EXPORT==='1'?['enable-dawn-features=dump_shaders,disable_symbol_renaming','disable-dawn-features=use_vulkan_memory_model,use_spirv_1_4,use_spirv_reconvergence_mode']:[])]);
const adapter=await gpu.requestAdapter();if(!adapter)throw Error('WebGPU oracle adapter unavailable');
const device=await adapter.requestDevice();
const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
const cfg=new ArrayBuffer(48),u=new Uint32Array(cfg),f=new Float32Array(cfg);u.set([256,128,4,2,64,512,96,16]);f[8]=1e-6;f[9]=1e6;u[10]=256;
const frame=new Uint32Array([3,4,0,0]);
let seed=123;const rand=n=>Float32Array.from({length:n},()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return ((seed>>>8)/2**24-.5)*.2;});
const words=new Uint32Array(96*256/4);for(let i=0;i<words.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;words[i]=seed;}
const scales=new Uint32Array(96*256/64).fill(0x24002400);
const shape=new Uint32Array([96,256,0,0]);const zero=n=>new Float32Array(n);const uniform=n=>new Uint32Array([n,0,0,0]);
const cases=[
 ['matvec',[rand(96*256),rand(256),zero(96),shape],2,2],
 ['matvec_q8',[words,scales,rand(256),zero(96),shape],3,2],
 ['matvec_coop',[rand(96*256),rand(256),zero(96),shape],2,24],
 ['matvec_q8_coop',[words,scales,rand(256),zero(96),shape],3,24],
 ['rmsnorm',[rand(256),rand(256),zero(256),uniform(256)],2,1],
 ['head_norm',[rand(256),rand(64),uniform(4)],0,1],
 ['rope',[rand(256),uniform(4)],0,2],
 ['attn_scores',[rand(256),rand(16*128),zero(4*16)],2,1],
 ['attn_softmax',[rand(4*16)],0,4],
 ['attn_out',[rand(4*16),rand(16*128),zero(256)],2,4],
 ['silu_mul',[rand(512),rand(512)],0,8],
 ['add_res',[rand(256),rand(256)],0,4],
];
const chunks=[];const uint=n=>{const b=Buffer.alloc(4);b.writeUInt32LE(n);chunks.push(b);};const bytes=b=>{const data=ArrayBuffer.isView(b)?Buffer.from(b.buffer,b.byteOffset,b.byteLength):Buffer.from(b);uint(data.length);chunks.push(data);};
uint(0x314b5644);uint(cases.length);
const buffer=(data,usage)=>{const bytes=ArrayBuffer.isView(data)?data.byteLength:data.byteLength;const b=device.createBuffer({size:Math.max(16,Math.ceil(bytes/4)*4),usage:usage|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(b,0,data);return b;};
for(const [name,inputs,out,x] of cases){
 const source=pruneWGSL(SOURCE,name);let mask=0;for(const m of source.matchAll(/@group\(1\)\s*@binding\((\d+)\)\s*var<([^>]+)>/g))if(m[2].trim()==='uniform')mask|=1<<+m[1];
 const g0=device.createBindGroupLayout({entries:[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}}))});
 const g1=device.createBindGroupLayout({entries:inputs.map((_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:mask&(1<<binding)?'uniform':binding===out?'storage':'read-only-storage'}}))});
 const pipe=device.createComputePipeline({layout:device.createPipelineLayout({bindGroupLayouts:[g0,g1]}),compute:{module:device.createShaderModule({code:source}),entryPoint:name}});
 const common=[buffer(cfg,GPUBufferUsage.UNIFORM),buffer(frame,GPUBufferUsage.UNIFORM)],bs=inputs.map((v,i)=>buffer(v,mask&(1<<i)?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE));
 const group=(layout,buffers)=>device.createBindGroup({layout,entries:buffers.map((b,binding)=>({binding,resource:{buffer:b}}))});
 const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipe);pass.setBindGroup(0,group(g0,common));pass.setBindGroup(1,group(g1,bs));pass.dispatchWorkgroups(x);pass.end();
 const stage=device.createBuffer({size:inputs[out].byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});enc.copyBufferToBuffer(bs[out],0,stage,0,stage.size);device.queue.submit([enc.finish()]);await stage.mapAsync(GPUMapMode.READ);const expected=Buffer.from(new Uint8Array(stage.getMappedRange()).slice());stage.unmap();
 if(errors.length)throw Error(errors.join('\n'));
 bytes(Buffer.from(name));uint(x);uint(1);uint(1);bytes(cfg);bytes(frame);uint(inputs.length);for(const b of inputs)bytes(b);uint(out);bytes(expected);
 for(const b of [...common,...bs,stage])b.destroy();
}
const out=process.argv[2]||'native/vulkan/build/oracle.dvk';fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,Buffer.concat(chunks));
fs.writeFileSync(out+'.json',JSON.stringify({sourceCommit:process.env.DUGA_SOURCE_COMMIT||null,backend:'Dawn WebGPU / Vulkan',adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},shaderSourceSHA256:crypto.createHash('sha256').update(SOURCE).digest('hex'),cases:cases.map(c=>c[0]),bitExactRequired:true},null,2));
console.log(`Captured ${cases.length} original WebGPU kernel outputs to ${out}`);device.destroy();process.exit(0);
