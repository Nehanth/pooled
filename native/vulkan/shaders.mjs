// Compile original WGSL ahead of time. Android runs direct Vulkan, never WebGPU.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { WGSL } from '../../engine/wgsl/base.js';
import { coopWGSL } from '../../engine/wgsl/coop.js';
import { pruneWGSL } from '../../engine/wgsl/prune.js';
export const KERNELS = ['matvec','matvec_q8','matvec_coop','matvec_q8_coop','rmsnorm','head_norm','rope','attn_scores','attn_softmax','attn_out','silu_mul','add_res'];
export const SOURCE = WGSL + coopWGSL(256,4,64,4,4,false);
export function compile(out, compiler = process.env.NAGA || 'naga') {
  const version=spawnSync(compiler,['--version'],{encoding:'utf8'});
  if(version.status!==0||version.stdout.trim()!=='30.0.1')throw Error('The native port requires naga-cli 30.0.1.');
  fs.mkdirSync(out,{recursive:true});
  let header = '#pragma once\n#include <cstddef>\n#include <cstdint>\nnamespace duga {\nstruct Shader { const char* name; const uint32_t* code; size_t bytes; uint32_t bindings; uint32_t uniforms; const char* entryPoint = nullptr; };\n';
  const entries = [], manifest = [];
  for(const entry of KERNELS) {
    const source = pruneWGSL(SOURCE,entry);
    const input = path.join(out,`${entry}.wgsl`),output=path.join(out,`${entry}.spv`);
    fs.writeFileSync(input,source);
    const r=spawnSync(compiler,[input,output,'--spirv-version','1.3','--buffer-bounds-check-policy','ReadZeroSkipWrite'],{encoding:'utf8'});
    if(r.status!==0) throw Error(`naga failed for ${entry}: ${r.stderr||r.error}`);
    let bindings=0,uniforms=0;
    for(const m of source.matchAll(/@group\(1\)\s*@binding\((\d+)\)\s*var<([^>]+)>/g)) {const b=+m[1];bindings=Math.max(bindings,b+1);if(m[2].trim()==='uniform')uniforms|=1<<b;}
    const code=fs.readFileSync(output),words=Array.from({length:code.length/4},(_,i)=>code.readUInt32LE(i*4));
    header+=`inline constexpr uint32_t spv_${entry}[] = {${words.map(v=>'0x'+v.toString(16)).join(',')}};\n`;
    entries.push(`{"${entry}",spv_${entry},sizeof(spv_${entry}),${bindings},${uniforms}}`);
    manifest.push({entry,bindings,uniforms,sourceSHA256:crypto.createHash('sha256').update(source).digest('hex'),spirvSHA256:crypto.createHash('sha256').update(code).digest('hex')});
  }
  header+=`inline constexpr Shader SHADERS[] = {${entries.join(',\n')}};\n}\n`;
  fs.writeFileSync(path.join(out,'shaders.hpp'),header);
  fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify({compiler:'naga-cli 30.0.1',kernels:manifest},null,2));
  console.log(`Compiled ${entries.length} original WGSL kernels to Vulkan SPIR-V.`);
}
if(process.argv[1]===new URL(import.meta.url).pathname)compile(path.resolve(process.argv[2]||'native/vulkan/build/shaders'));
