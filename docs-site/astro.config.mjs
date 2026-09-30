// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { readFile, writeFile } from 'node:fs/promises';

// CHANGELOG.md is imported into reference/changelog and links repo files relatively (docs/protocol.md).
// On the site those resolve under /docs/reference/changelog/ and 404, so after the build they point at
// GitHub instead. (A rehype plugin can't do it: markdown.rehypePlugins needs the old unified processor.)
const repoLinks = {
	name: 'pooled-changelog-repo-links',
	hooks: {
		'astro:build:done': async ({ dir }) => {
			const file = new URL('reference/changelog/index.html', dir);
			const html = await readFile(file, 'utf8');
			await writeFile(file, html.replace(/href="((?:docs|tests|scripts|cli|room|engine|packages)\/[^"]+)"/g, 'href="https://github.com/Nehanth/pooled/blob/main/$1"'));
		},
	},
};

// Served by the main Vercel project at https://pooled.run/docs (static files, no hosted service).
export default defineConfig({
	site: 'https://pooled.run',
	base: '/docs',
	trailingSlash: 'never', // vercel.json has "trailingSlash": false
	outDir: './dist', // build-vercel.mjs copies this to <site>/docs next to the static site
	build: { format: 'directory' },
	// Pages merged into others: old links still work (pooled.run/docs links are already shared).
	redirects: {
		'/rooms/errors': '/docs/troubleshooting',
		'/rooms/at-work': '/docs/rooms/networks',
		'/rooms/serve-api-page': '/docs/serve',
		'/code/templates': '/docs/code#templates',
		'/code/share': '/docs/code#download',
		'/serve/recipes/codex': '/docs/serve/recipes#codex-cli',
		'/serve/recipes/claude-code': '/docs/serve/recipes#claude-code',
		'/serve/recipes/opencode': '/docs/serve/recipes#opencode',
		'/serve/recipes/continue': '/docs/serve/recipes#continue',
		'/serve/recipes/open-webui': '/docs/serve/recipes#open-webui',
		'/serve/recipes/sdks': '/docs/serve/recipes#openai-and-anthropic-sdks',
		'/serve/recipes/litellm': '/docs/serve/recipes#litellm',
		'/serve/recipes/curl': '/docs/serve/recipes#curl',
	},
	integrations: [
		repoLinks,
		starlight({
			title: 'Pooled docs',
			description: 'Peer-to-peer LLM inference in the browser. Pool your devices to run big open models.',
			favicon: '/favicon.svg',
			logo: { light: './src/assets/wordmark-light.svg', dark: './src/assets/wordmark-dark.svg', replacesTitle: true },
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/Nehanth/pooled' }],
			editLink: { baseUrl: 'https://github.com/Nehanth/pooled/edit/main/docs-site/' },
			lastUpdated: false,
			customCss: ['./src/styles/pooled.css'],
			head: [
				// Fonts and tokens come from the main site, same origin (/site/...), cached a year.
				{ tag: 'link', attrs: { rel: 'preload', href: '/site/fonts/geist-latin.woff2', as: 'font', type: 'font/woff2', crossorigin: '' } },
				{ tag: 'link', attrs: { rel: 'stylesheet', href: '/site/css/fonts.css' } },
			],
			components: {
				Header: './src/components/Header.astro',
				PageTitle: './src/components/PageTitle.astro',
			},
			// Code blocks (Expressive Code) are configured in ec.config.mjs (it holds functions).
			sidebar: [
				{
					label: 'Getting started',
					items: ['index', 'start-a-room', 'invite', 'models', 'phones'],
				},
				{
					label: 'Use it',
					items: [
						'chat',
						'rooms/settings',
						'code',
						{ slug: 'serve', label: 'Serve API' },
						{ slug: 'serve/recipes', label: 'Connect your tools' },
						'openclaw',
						'terminal',
					],
				},
				{ label: 'Help', items: ['troubleshooting', 'faq', 'rooms/networks', 'browsers', 'privacy'] },
				{
					label: 'Reference',
					items: [
						'reference/url-options',
						'reference/cli',
						'reference/environment',
						'reference/endpoints',
						'reference/models',
						'reference/changelog',
						{
							label: 'Serve API reference',
							collapsed: true,
							items: [
								'serve/options',
								'serve/chat-completions',
								'serve/responses',
								'serve/messages',
								'serve/tools',
								'serve/structured-output',
								'serve/limits',
							],
						},
					],
				},
				{ label: 'Self-hosting', collapsed: true, items: [{ autogenerate: { directory: 'self-host' } }] },
				{ label: 'Contributing', collapsed: true, items: [{ autogenerate: { directory: 'contributing' } }, 'roadmap'] },
				{
					label: 'Internals',
					collapsed: true,
					items: [
						{ autogenerate: { directory: 'internals' } },
						{
							label: 'How rooms work',
							items: ['rooms/split', 'rooms/devices', 'rooms/downloads', 'rooms/recovery', 'code/agent', 'code/previews'],
						},
					],
				},
			],
		}),
	],
});
