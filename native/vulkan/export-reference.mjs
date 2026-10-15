// Export the actual Tint programs used by original WebGPU execution. Only SPIR-V
// is embedded in Android; Dawn, WGSL compilation and SPIR-V Tools are build-only.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {KERNELS} from './shaders.mjs';

const out=path.resolve('native/vulkan/build/shaders');
if(!process.env.DUGA_WEBGPU_MODULE)throw Error('DUGA_WEBGPU_MODULE is required.');
const packageInfo=JSON.parse(fs.readFileSync(path.join(path.dirname(fs.realpathSync(process.env.DUGA_WEBGPU_MODULE)),'package.json'),'utf8'));
if(packageInfo.name!=='webgpu'||packageInfo.version!=='0.6.2')throw Error('Reference export requires the pinned webgpu 0.6.2 package.');
const assembler=process.env.SPIRV_AS || 'spirv-as',validator=process.env.SPIRV_VAL || 'spirv-val';
const captured=spawnSync(process.execPath,['native/vulkan/capture-oracle.mjs','native/vulkan/build/reference-capture.dvk'],{
  env:{...process.env,DUGA_DAWN_REFERENCE_EXPORT:'1'},encoding:'utf8',maxBuffer:4*1024*1024,timeout:120000,
});
if(captured.status!==0)throw Error(captured.stderr || captured.error);
const log=captured.stderr+'\n'+captured.stdout;
fs.writeFileSync(path.join(out,'reference-export.log'),log);
const blocks=log.split('/* Dumped SPIRV disassembly */').slice(1);
if(blocks.length!==KERNELS.length)throw Error('Dawn did not export exactly twelve reference shaders.');
const metadata=JSON.parse(fs.readFileSync(path.join(out,'manifest.json'),'utf8'));
let header='#pragma once\n#include "shaders.hpp"\nnamespace duga {\n';const entries=[],manifest=[];
for(let i=0;i<KERNELS.length;i++){
  const name=KERNELS[i],block=blocks[i];
  const end=block.lastIndexOf('OpFunctionEnd');if(end<0)throw Error('Truncated SPIR-V export');
  const assembly=block.slice(0,end+'OpFunctionEnd'.length).trim()+'\n';
  const entry=/OpEntryPoint GLCompute %([\w]+) "([^"\n]+)"/.exec(assembly);
  if(!entry || entry[1]!==name || entry[2]!=='dawn_entry_point')throw Error(`Unexpected reference entry for ${name}`);
  if(!assembly.includes('; Version: 1.3') || !assembly.includes('OpMemoryModel Logical GLSL450') || /OpExtension|PushConstant/.test(assembly) || [...assembly.matchAll(/OpCapability (\w+)/g)].some(m=>m[1]!=='Shader'))throw Error(`Reference ${name} is not portable Vulkan 1.1 SPIR-V.`);
  for(const match of assembly.matchAll(/OpDecorate %\w+ DescriptorSet (\d+)/g))if(+match[1]>1)throw Error('Unexpected hidden reference bind group');
  const input=path.join(out,`${name}.reference.spvasm`),output=path.join(out,`${name}.reference.spv`);
  fs.writeFileSync(input,assembly);
  for(const [tool,args] of [[assembler,['--target-env','vulkan1.1',input,'-o',output]],[validator,['--target-env','vulkan1.1',output]]]){
    const result=spawnSync(tool,args,{encoding:'utf8'});if(result.status!==0)throw Error(result.stderr || result.error);
  }
  const code=fs.readFileSync(output),words=Array.from({length:code.length/4},(_,at)=>code.readUInt32LE(at*4));
  header+=`inline constexpr uint32_t reference_${name}[] = {${words.map(v=>'0x'+v.toString(16)).join(',')}};\n`;
  const original=metadata.kernels.find(k=>k.entry===name);if(!original)throw Error('Missing original shader metadata');
  entries.push(`{"${name}",reference_${name},sizeof(reference_${name}),${original.bindings},${original.uniforms},"dawn_entry_point"}`);
  manifest.push({entry:name,sourceSHA256:original.sourceSHA256,spirvSHA256:crypto.createHash('sha256').update(code).digest('hex')});
}
header+=`inline constexpr Shader ORACLE_SHADERS[] = {${entries.join(',\n')}};\n}\n`;
fs.writeFileSync(path.join(out,'oracle-shaders.hpp'),header);
fs.writeFileSync(path.join(out,'reference-manifest.json'),JSON.stringify({compiler:'Dawn webgpu 0.6.2 / Tint',target:'Vulkan 1.1 / SPIR-V 1.3',kernels:manifest},null,2));
console.log('Exported and validated twelve independently compiled original WebGPU reference programs.');
