// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Served by the main Vercel project at https://pooled.run/docs (static files, no hosted service).
export default defineConfig({
	site: 'https://pooled.run',
	base: '/docs',
	trailingSlash: 'never', // vercel.json has "trailingSlash": false
	outDir: './dist', // build-vercel.mjs copies this to <site>/docs next to the static site
	build: { format: 'directory' },
	integrations: [
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
					items: ['index', 'start-a-room', 'invite', 'models', 'phones', 'browsers', 'privacy'],
				},
				{ label: 'Rooms', items: [{ autogenerate: { directory: 'rooms' } }] },
				{
					label: 'Serve API',
					items: [
						'serve',
						'serve/options',
						'serve/chat-completions',
						'serve/responses',
						'serve/messages',
						'serve/tools',
						'serve/structured-output',
						'serve/limits',
						{ label: 'Client recipes', collapsed: false, items: [{ autogenerate: { directory: 'serve/recipes' } }] },
					],
				},
				{ label: 'Code mode', items: [{ autogenerate: { directory: 'code' } }] },
				{ label: 'Self-hosting', items: [{ autogenerate: { directory: 'self-host' } }] },
				{ label: 'How it works', items: [{ autogenerate: { directory: 'internals' } }] },
				{ label: 'Reference', items: [{ autogenerate: { directory: 'reference' } }] },
				{ label: 'FAQ and troubleshooting', items: ['faq', 'troubleshooting'] },
				{ label: 'Contributing', items: [{ autogenerate: { directory: 'contributing' } }, 'roadmap'] },
			],
		}),
	],
});
