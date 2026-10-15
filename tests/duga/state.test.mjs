import { test } from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { MODES, requireLocal, invitePath, MemoryStore, systemPrompt } from '../../app/state.js';
globalThis.crypto ||= webcrypto;
const origin = 'https://duga.app';
const key = 'abcdefghijklmnopqrstuv';
const storage = () => { const data = new Map(); return { getItem:k => data.get(k) || null, setItem:(k,v) => data.set(k,v), removeItem:k => data.delete(k) }; };
test('unavailable placement cannot silently fall back or execute', () => {
  requireLocal('local');
  for (const mode of ['private', 'global', 'hosted', undefined]) assert.throws(() => requireLocal(mode));
  assert.deepEqual(Object.keys(MODES).filter(k => MODES[k].available), ['local']);
});
test('invites preserve bearer secrets in the fragment, never in a request query', () => {
  assert.equal(invitePath('abc-234', origin), '/r/ABC234');
  assert.equal(invitePath('K7QX', origin), '/r/K7QX');
  const path = invitePath(`https://pooled.run/r/ABC234#k=${key}`, origin);
  assert.equal(path, `/r/ABC234#k=${key}`);
  assert.equal(new URL(path, origin).search, '');
  assert.equal(new URL(path, origin).pathname.includes(key), false);
  for (const input of ['ABCDE', 'javascript:alert(1)', 'https://evil.example/r/ABC234', 'https://duga.app.evil.example/r/ABC234', `https://duga.app/r/ABC234?k=${key}`, 'https://user:pw@duga.app/r/ABC234', 'https://duga.app/r/ABC234/extra']) assert.throws(() => invitePath(input, origin), input);
});
test('explicit notes survive reload, edits, export-shaped data and deletion', () => {
  const db = storage(), store = new MemoryStore(db);
  const saved = store.save('Short answers, please.');
  assert.equal(new MemoryStore(db).items[0].text, saved.text);
  store.save('日本語で答えて', saved.id);
  assert.equal(store.items.length, 1);
  assert.equal(store.items[0].text, '日本語で答えて');
  assert.equal(systemPrompt().includes('日本語'), false);
  assert.equal(systemPrompt(store.items).includes('日本語'), true);
  store.remove(saved.id); assert.equal(new MemoryStore(db).items.length, 0);
  store.save('x'); store.clear(); assert.deepEqual(new MemoryStore(db).items, []);
});
test('blocked storage and corrupt data do not report successful saves', () => {
  const store = new MemoryStore({ getItem:() => '{invalid', setItem:() => { throw Error('quota'); } });
  assert.ok(store.error); assert.deepEqual(store.items, []);
  assert.throws(() => store.save('hello')); assert.deepEqual(store.items, []);
  const clean = new MemoryStore(storage());
  for (const value of ['', ' ', 'a'.repeat(2001)]) assert.throws(() => clean.save(value));
  assert.throws(() => clean.save('text', 'missing'));
});
