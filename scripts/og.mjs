// Renders site/og.png, the 1200x630 social preview (og:image, twitter:image), from the landing page itself:
// the Duga assistant welcome. Rerun it when the UI changes, and bump
// the ?v= on the og:image and twitter:image URLs in index.html and p2p.html so the sites that cache it refetch.
//   npm run serve            (in another terminal)
//   node scripts/og.mjs [--url http://localhost:8080/] [--out site/og.png]
import { chromium } from "playwright";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const url = arg("--url", "http://localhost:8080/"), out = arg("--out", "site/og.png");
const W = 1200, H = 630;

const browser = await chromium.launch();
// Stable rendering without motion for social previews.
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2, reducedMotion: "reduce" });
await page.goto(url, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.addStyleTag({ content: `
  .sidebar,.topbar,.composer-area,.conversation{display:none!important}
  .app{display:block!important}
  .chat-panel{min-height:${H}px!important;max-width:1000px!important;padding:0 48px!important}
  .welcome{padding:30px 0!important;margin:auto!important}
  .welcome h1{font-size:46px!important}
  .welcome::before{content:"Duga";display:block;font:500 28px Geist,sans-serif;margin-bottom:24px}
` });
const shot = await page.screenshot();
// rendered at 2x, then drawn down to 1200x630 in the browser, for crisp text at the size the sites ask for
const small = await browser.newPage({ viewport: { width: W, height: H } });
await small.setContent(`<body style="margin:0"><img style="display:block;width:${W}px;height:${H}px" src="data:image/png;base64,${shot.toString("base64")}"></body>`);
await small.waitForFunction(() => document.images[0].complete);
await small.screenshot({ path: out });
await browser.close();
console.log("wrote", out);
