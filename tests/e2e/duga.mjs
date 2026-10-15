// UI integration: real app + scripted adapter intercepted by Playwright, never a production hook.
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import path from 'node:path';
import { serveRepo } from './engine_synth.mjs';
const root = path.resolve(new URL('../..', import.meta.url).pathname);
const server = serveRepo(0, { '/':path.join(root,'index.html'), '/room':path.join(root,'p2p.html'), '/r/ABC234':path.join(root,'p2p.html') });
await new Promise(r => server.listening ? r() : server.once('listening',r));
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({ args:['--no-sandbox'] });
  const context = await browser.newContext({ reducedMotion:'reduce', acceptDownloads:true });
  const page = await context.newPage();
  const errors = []; page.on('pageerror',e => errors.push(e.message));
  const network = []; page.on('request',r => network.push(r.url()));
  await page.route('**/app/local.js', route => route.fulfill({ contentType:'text/javascript', body:`
    export async function loadLocal({signal,onStatus,onProgress}) {
      window.testLoadCount = (window.testLoadCount || 0) + 1;
      onStatus('Loading test model'); onProgress(1,2); signal.throwIfAborted();
      return { dispose(){ window.testDisposed = true; }, async *generate(args) {
        window.testArgs = args;
        const last = args.turns.at(-1).text;
        if (last === 'fail') throw Error('Injected model failure');
        for (const word of (last === 'slow' ? ['one ','two ','three ','four '] : ['Hello ','from ','Local.'])) {
          await new Promise(r => setTimeout(r,last === 'slow' ? 500 : 30));
          if (args.signal.aborted) return;
          yield word;
        }
      }};
    }
  ` }));
  await page.goto(base);
  await page.getByRole('heading',{name:'What’s on your mind?'}).waitFor();
  assert.equal(await page.locator('iframe[src]').count(),0);
  assert.equal(network.some(u => /huggingface|peerjs|\/api\//.test(u)),false);
  await page.getByRole('button',{name:'Local',exact:false}).first().click();
  assert.match(await page.locator('#mode-dialog').innerText(),/Private Net[\s\S]*Not available[\s\S]*Global Net/);
  await page.locator('#mode-dialog [data-close]').click();
  await page.getByRole('button',{name:'Voice',exact:true}).click();
  assert.match(await page.locator('#voice-dialog').innerText(),/has not requested microphone permission/);
  await page.locator('#voice-dialog button').last().click();
  await page.locator('[data-view="memories"]').click();
  await page.locator('#memory-text').fill('<img src=x onerror=alert(1)> 日本語で短く');
  await page.locator('#memory-save').click();
  assert.equal(await page.locator('.memory-item img').count(),0);
  assert.match(await page.locator('#memory-status').innerText(),/Saved/);
  await page.reload();
  await page.locator('.memory-item').waitFor();
  await page.locator('.memory-item button').first().click();
  await page.locator('#memory-text').fill('Use concise Japanese.');
  await page.locator('#memory-save').click();
  assert.equal(await page.locator('.memory-item p').innerText(),'Use concise Japanese.');
  const exported = page.waitForEvent('download'); await page.locator('#memory-export').click(); assert.equal((await exported).suggestedFilename(),'duga-notes.json');
  await page.locator('[data-view="chat"]').first().click();
  await page.locator('#prompt').fill('hello'); await page.locator('#send').click();
  await page.waitForFunction(() => document.querySelector('#run-status').textContent.startsWith('Finished'));
  assert.equal(await page.locator('.message.assistant .message-body').innerText(),'Hello from Local.');
  assert.equal(await page.evaluate(() => window.testArgs.system.includes('concise Japanese')),false);
  await page.locator('#use-memories').check();
  await page.locator('#prompt').fill('with notes'); await page.locator('#send').click();
  await page.waitForFunction(() => document.querySelector('#run-status').textContent.startsWith('Finished'));
  assert.equal(await page.evaluate(() => window.testArgs.system.includes('concise Japanese')),true);
  await page.locator('#prompt').fill('slow'); await page.locator('#send').click();
  await page.waitForFunction(() => document.querySelectorAll('.message.assistant .message-body')[2].textContent.length > 0);
  await page.locator('#stop').click(); await page.locator('#stop').waitFor({state:'hidden'});
  assert.match(await page.locator('#run-status').innerText(),/Stopped/);
  await page.locator('#prompt').fill('fail'); await page.locator('#send').click();
  await page.locator('.message.failed').waitFor();
  assert.equal(await page.locator('#prompt').inputValue(),'fail');
  assert.match(await page.locator('#run-status').innerText(),/No automatic fallback/);
  await page.locator('[data-view="activity"]').click();
  assert.doesNotMatch(await page.locator('#activity-list').innerText(),/concise Japanese|with notes/);
  await page.locator('[data-view="devices"]').click();
  await page.locator('#invite').fill('https://evil.example/r/ABC234'); await page.locator('#join-form button').click();
  assert.match(await page.locator('#invite-error').innerText(),/Use an invite/);
  assert.equal(await page.locator('iframe[src]').count(),0);
  const secret = 'abcdefghijklmnopqrstuv';
  await page.route('**/r/ABC234',r => r.fulfill({contentType:'text/html',body:'<title>Room</title><p>Room adapter fixture</p>'}));
  await page.locator('#invite').fill(`${base}/r/ABC234#k=${secret}`); await page.locator('#join-form button').click();
  assert.equal(await page.locator('#workspace').getAttribute('src'),`/r/ABC234#k=${secret}`);
  await page.locator('#workspace-disconnect').waitFor({state:'visible'});
  assert.equal(network.some(u => u.split('#')[0].includes(secret)),false);
  assert.equal(await page.evaluate(() => localStorage.getItem('duga.memories.v1').includes('abcdefghijkl')),false);
  await page.locator('[data-view="chat"]').first().click();
  await page.locator('#prompt').fill('should not run'); await page.locator('#send').click();
  assert.equal(await page.locator('#view-title').innerText(),'Code workspace');
  await page.locator('#workspace-disconnect').click(); await page.locator('#confirm-delete').click();
  assert.equal(await page.locator('#workspace').getAttribute('src'),null);
  await page.locator('[data-view="settings"]').click(); await page.locator('#data-delete').click();
  await page.locator('#confirm-cancel').click();
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('duga.memories.v1')).length),1);
  await page.locator('#data-delete').click(); await page.locator('#confirm-delete').click();
  assert.equal(await page.evaluate(() => localStorage.getItem('duga.memories.v1')),null);
  await page.locator('[data-view="chat"]').first().click();
  await page.getByRole('heading',{name:'What’s on your mind?'}).waitFor();
  await page.screenshot({path:'/tmp/duga-desktop.png',fullPage:true});
  for (const width of [390,768,1440]) {
    await page.setViewportSize({width,height:900});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),`overflow at ${width}`);
    if (width === 390) {
      await page.locator('#nav-toggle').click(); assert.equal(await page.locator('#nav-toggle').getAttribute('aria-expanded'),'true');
      await page.locator('[data-view="memories"]').click(); assert.equal(await page.locator('#nav-toggle').getAttribute('aria-expanded'),'false');
      await page.locator('#nav-toggle').click(); await page.locator('[data-view="chat"]').first().click();
      await page.screenshot({path:'/tmp/duga-mobile.png',fullPage:true});
    }
  }
  assert.deepEqual(errors,[]);
  // A second, unmocked context proves lack of WebGPU fails safely, with no hosted call.
  const unsupported = await browser.newContext();
  await unsupported.addInitScript(() => Object.defineProperty(navigator,'gpu',{value:undefined}));
  const p = await unsupported.newPage(); const requests=[]; p.on('request',r=>requests.push(r.url()));
  await p.goto(base); await p.locator('#prompt').fill('private prompt'); await p.locator('#send').click();
  await p.locator('.message.failed').waitFor(); assert.match(await p.locator('.message.failed').innerText(),/WebGPU/);
  assert.equal(requests.some(u=>!u.startsWith(base)),false);
  await unsupported.close();
  console.log('PASS Duga desktop/mobile UI, streaming, stop, failure, explicit memory, XSS-safe notes, export/delete, invite privacy, placement and unsupported-device behavior');
} finally { await browser?.close(); await new Promise(r=>server.close(r)); }
