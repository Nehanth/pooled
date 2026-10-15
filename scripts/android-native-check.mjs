// Development/CI oracle runner. The distributed APK never loads WebGPU or these executables.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const sdk=process.env.ANDROID_HOME || process.env.ANDROID_SDK_ROOT;
if(!sdk)throw Error('ANDROID_HOME is required.');
const adb=process.env.ADB || path.join(sdk,'platform-tools/adb');
const serial=process.env.DUGA_ANDROID_SERIAL || process.env.ANDROID_SERIAL || 'emulator-5554';
function run(command,args,env=process.env) {
  const result=spawnSync(command,args,{env,stdio:'inherit',timeout:15*60*1000});
  if(result.status!==0)throw Error(`${path.basename(command)} failed: ${result.error || result.status}`);
}
const query=spawnSync(adb,['-s',serial,'shell','getprop','ro.product.cpu.abi'],{encoding:'utf8'});
if(query.status!==0)throw Error(query.stderr || 'Android device unavailable');
const abi=query.stdout.trim(),triple={'x86_64':'x86_64','arm64-v8a':'aarch64'}[abi];
if(!triple)throw Error(`Unsupported Android ABI: ${abi}`);
const compiler=path.join(sdk,`ndk/27.3.13750724/toolchains/llvm/prebuilt/linux-x86_64/bin/${triple}-linux-android29-clang++`);
const out=path.resolve('native/vulkan/build/android',abi);fs.mkdirSync(out,{recursive:true});
const flags=['-std=c++17','-O2','-ffp-contract=off','-static-libstdc++','-Wl,-z,max-page-size=16384','-Inative/vulkan/build/shaders'];
for(const [name,sources] of [
  ['oracle',['vulkan.cpp','oracle.cpp']],
  ['model',['vulkan.cpp','gguf.cpp','dense.cpp','model-run.cpp']],
])run(compiler,[...flags,...sources.map(s=>`native/vulkan/${s}`),'-lvulkan','-o',`${out}/duga-vk-${name}`]);
const remote='/data/local/tmp/duga-kernel-check';
run(adb,['-s',serial,'shell','mkdir','-p',remote]);
try {
  run(adb,['-s',serial,'push',`${out}/duga-vk-oracle`,`${remote}/oracle`]);
  run(adb,['-s',serial,'push','native/vulkan/build/oracle.dvk',`${remote}/oracle.dvk`]);
  run(adb,['-s',serial,'shell','chmod','755',`${remote}/oracle`]);
  run(adb,['-s',serial,'shell',`${remote}/oracle`,`${remote}/oracle.dvk`]);
  run(adb,['-s',serial,'shell',`${remote}/oracle`,`${remote}/oracle.dvk`,'--reference']);
  run(adb,['-s',serial,'shell',`${remote}/oracle`,`${remote}/oracle.dvk`,'--on-device']);
} finally {run(adb,['-s',serial,'shell','rm','-rf',remote]);}
run(process.execPath,['native/vulkan/check-model.mjs'],{...process.env,ADB:adb,DUGA_ANDROID_SERIAL:serial,DUGA_NATIVE_RUNNER:`${out}/duga-vk-model`});
