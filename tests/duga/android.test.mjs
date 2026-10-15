import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {androidVersion} from '../../scripts/android-version.mjs';
import {syncAndroidAssets} from '../../scripts/android-assets.mjs';
test('tag versions are validated before signing or publication',()=>{
  assert.deepEqual(androidVersion('android-v1.2.3',14),{name:'1.2.3',code:14,prerelease:false});
  assert.deepEqual(androidVersion('android-v0.1.0-preview.1',15),{name:'0.1.0-preview.1',code:15,prerelease:true});
  for(const tag of ['v1.2.3','android-v01.2.3','android-v1.2','android-v1.2.3\nDUGA_KEY=x','android-v1.2.3;echo'])assert.throws(()=>androidVersion(tag,1));
  for(const code of [0,-1,2.5,2100000001,'bad'])assert.throws(()=>androidVersion('android-v1.2.3',code));
});
test('APK bundling includes only runtime assets and preserves the browser sources',()=>{
  if(!fs.existsSync('native/vulkan/build/oracle.dvk'))return; // Native CI produces the trace before this test.
  const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'duga-android-assets-'));
  const dest=path.join(tmp,'generated/web-assets');
  try{
    const original=fs.readFileSync('index.html','utf8');const manifest=syncAndroidAssets(dest);
    assert.ok(manifest.some(f=>f.path==='app/android-local.js'));assert.ok(manifest.some(f=>f.path==='room.js'));
    assert.ok(fs.readFileSync(path.join(dest,'index.html'),'utf8').includes('/android-host.js'));
    assert.equal(fs.readFileSync('index.html','utf8'),original);
    for(const forbidden of ['.git','android','native','models','tests','.aws'])assert.equal(fs.existsSync(path.join(dest,forbidden)),false);
    assert.ok(fs.existsSync(path.join(dest,'oracle.dvk')));
    assert.throws(()=>syncAndroidAssets(tmp));
  }finally{fs.rmSync(tmp,{recursive:true,force:true});}
});
