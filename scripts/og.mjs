// Renders site/og.png, the 1200x630 social preview (og:image, twitter:image), from the landing page itself:
// the wordmark, the headline, and the demo's finished Tetris frame. Rerun it when the demo changes, and bump
// the ?v= on the og:image and twitter:image URLs in index.html and p2p.html so the sites that cache it refetch.
//   npm run serve            (in another terminal)
//   node scripts/og.mjs [--url http://localhost:8080/] [--out site/og.png]
import { chromium } from "playwright";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const url = arg("--url", "http://localhost:8080/"), out = arg("--out", "site/og.png");
const W = 1200, H = 630;

const browser = await chromium.launch();
// reduced motion: demo.js shows the finished story, frozen, instead of playing it
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2, reducedMotion: "reduce" });
await page.goto(url, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.addStyleTag({ content: `
  .nav,.hero .sub,.modes,.stepbar,.closer,.footer,.hd-inv{display:none!important}
  .hero{padding:30px 0 22px!important}
  .og-brand{display:flex;justify-content:center;margin-bottom:14px}
  .hero h1{font-size:46px!important;max-width:none!important}
  .wrap{max-width:none!important;padding:0 44px!important}
  :root{--body-h:${H - 184 - 26}px!important}   /* the window runs to 26px above the bottom edge */
` });
await page.evaluate(() => {
  const b = document.querySelector(".nav .brand").cloneNode(true);
  const p = document.createElement("div"); p.className = "og-brand"; p.append(b);
  document.querySelector(".hero-in").prepend(p);
});
// then Tetris, at its finished frame, laid out at this size (step 7 first: it freezes the story, so no frame is
// left to run on past the end)
await page.evaluate(() => { window.__demo.setApp("tetris"); window.__demo.step(6); window.__demo.step(7); });
await page.waitForTimeout(800);
const shot = await page.screenshot();
// rendered at 2x, then drawn down to 1200x630 in the browser, for crisp text at the size the sites ask for
const small = await browser.newPage({ viewport: { width: W, height: H } });
await small.setContent(`<body style="margin:0"><img style="display:block;width:${W}px;height:${H}px" src="data:image/png;base64,${shot.toString("base64")}"></body>`);
await small.waitForFunction(() => document.images[0].complete);
await small.screenshot({ path: out });
await browser.close();
console.log("wrote", out);
