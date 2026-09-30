// Assembles what Vercel serves: the static site from the repo root, unchanged, plus the built docs
// at /docs. Run from the repo root after `npm --prefix docs-site run build` (vercel.json does both).
//
//   <repo>/            -> dist/          (every top-level entry except the ones skipped below)
//   docs-site/dist/    -> dist/docs/
//
// On Vercel the upload already has .vercelignore applied, so the skip list only matters locally,
// where it applies the same .vercelignore names so a local build matches the deployed one.
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const out = path.join(root, 'dist');
const docs = path.join(root, 'docs-site', 'dist');

// Never part of the site: build output, the docs project itself, the contributor notes in docs/
// (the URL /docs is the built docs), git and tool state.
const ALWAYS_SKIP = new Set(['dist', 'docs-site', 'docs', '.git', '.vercel', 'node_modules', '.astro']);

// Top-level patterns from .vercelignore (plain names and simple `*` globs; negations are ignored,
// they only re-include files the docs build reads, like CHANGELOG.md).
const globs = fs
	.readFileSync(path.join(root, '.vercelignore'), 'utf8')
	.split('\n')
	.map((l) => l.trim())
	.filter((l) => l && !l.startsWith('#') && !l.startsWith('!') && !l.slice(0, -1).includes('/'))
	.map((l) => new RegExp('^' + l.replace(/\/$/, '').replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '[^/]*') + '$'));
const skip = (name) => ALWAYS_SKIP.has(name) || globs.some((re) => re.test(name));

if (!fs.existsSync(path.join(docs, 'index.html'))) {
	console.error('build-vercel: docs-site/dist/index.html is missing; run `npm --prefix docs-site run build` first');
	process.exit(1);
}

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out);
const copied = [];
for (const name of fs.readdirSync(root)) {
	if (skip(name)) continue;
	fs.cpSync(path.join(root, name), path.join(out, name), { recursive: true });
	copied.push(name);
}
fs.cpSync(docs, path.join(out, 'docs'), { recursive: true });

console.log(`build-vercel: dist/ = ${copied.join(', ')} + docs/ (${fs.readdirSync(docs).length} entries)`);
