// Explicit, repeatable APK assets. Native sources, secrets, tests, models and repo metadata are excluded.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
export function syncAndroidAssets(output) {
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.resolve(output);
  if(!out.endsWith(`${path.sep}generated${path.sep}web-assets`))throw Error('Output must be a generated/web-assets directory.');
  fs.mkdirSync(out,{recursive:true});
  // Only remove our own generated directory after the strict suffix check.
  fs.rmSync(out,{recursive:true,force:true});fs.mkdirSync(out,{recursive:true});
  const roots=['index.html','p2p.html','room.js','favicon.svg','site.webmanifest','LICENSE','app','engine','room','harness','site'];
  const manifest=[];
  function copy(relative){const src=path.join(root,relative),s=fs.lstatSync(src);if(s.isSymbolicLink())throw Error('Symlink excluded from Android assets');if(s.isDirectory()){for(const n of fs.readdirSync(src).sort())copy(path.join(relative,n));return;}
    if(!/\.(html|js|css|svg|png|woff2|json|webmanifest)$/.test(relative)&&relative!=='LICENSE')return;
    const dest=path.join(out,relative);fs.mkdirSync(path.dirname(dest),{recursive:true});let data=fs.readFileSync(src);
    if(relative==='index.html'||relative==='p2p.html')data=Buffer.from(data.toString().replace('</head>','<script src="/android-host.js"></script>\n</head>'));
    fs.writeFileSync(dest,data);manifest.push({path:relative.replaceAll('\\','/'),sha256:crypto.createHash('sha256').update(data).digest('hex')});
  }
  for(const r of roots)copy(r);
  fs.copyFileSync(path.join(root,'android/web/android-host.js'),path.join(out,'android-host.js'));
  const oracle=path.join(root,'native/vulkan/build/oracle.dvk');if(!fs.existsSync(oracle))throw Error('Capture the WebGPU oracle before packaging the native backend.');
  fs.copyFileSync(oracle,path.join(out,'oracle.dvk'));
  fs.writeFileSync(path.join(out,'android-asset-manifest.json'),JSON.stringify(manifest,null,2));
  console.log(`Synced ${manifest.length} Duga assets and the native kernel oracle.`);return manifest;
}
if(process.argv[1]===fileURLToPath(import.meta.url)){const i=process.argv.indexOf('--out');if(i<0)throw Error('--out required');syncAndroidAssets(process.argv[i+1]);}
