# Pooled docs

The docs at [pooled.run/docs](https://pooled.run/docs), built with [Astro Starlight](https://starlight.astro.build) into static pages. Search is [Pagefind](https://pagefind.app), built with the pages. No hosted service.

```sh
cd docs-site
npm ci
npm run dev      # http://localhost:4321/docs, live reload
npm run build    # docs-site/dist
```

Pages are Markdown/MDX in `src/content/docs/`; the path is the URL (`serve/tools.mdx` is `/docs/serve/tools`). Frontmatter: `title`, `description` (the lede under the title), `eyebrow` (the small label above it) and `sidebar.order`. The sidebar groups are in `astro.config.mjs`.

The theme follows `docs/design/language.md`: `src/styles/pooled.css` (tokens, light and dark), `src/components/Header.astro` (top bar) and `src/components/PageTitle.astro` (eyebrow, title, Copy page, lede). Code block colors are in `ec.config.mjs`. Fonts come from the site's `/site/fonts`.

## Deploy

Vercel builds this on every deploy (`vercel.json`): `npm --prefix docs-site run build && node docs-site/build-vercel.mjs`. The second step copies the static site from the repo root into `dist/` unchanged and the built docs into `dist/docs/`. To try the whole site locally:

```sh
npm --prefix docs-site run build && node docs-site/build-vercel.mjs
cp serve.json dist/ && npx serve dist   # serve.json has the /room and /r/:code rewrites
```
