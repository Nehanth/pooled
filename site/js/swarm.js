/* The swarms. Hero: a few hundred faint dots drifting as one loose swarm, a handful of blue peers
   linked by hairlines, every so often pooling toward the middle and letting go. Closer: the same
   swarm, and when it scrolls into view part of it gathers into the dots logo, then the logo settles
   crisp while the rest keeps drifting. ~30 fps, paused off screen or in a hidden tab, a single
   still frame for reduced motion. */
(() => {
  "use strict";
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const TAU = Math.PI * 2;
  const ease = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
  // cubic-bezier(x1,y1,x2,y2), as in CSS: solve x(u) = x for u, return y(u)
  const bezier = (x1, y1, x2, y2) => x => {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bx = u => 3 * x1 * u * (1 - u) * (1 - u) + 3 * x2 * u * u * (1 - u) + u * u * u;
    const by = u => 3 * y1 * u * (1 - u) * (1 - u) + 3 * y2 * u * u * (1 - u) + u * u * u;
    let lo = 0, hi = 1, u = x;
    for (let i = 0; i < 20; i++) { u = (lo + hi) / 2; if (bx(u) < x) lo = u; else hi = u; }
    return by(u);
  };
  const gatherEase = bezier(.2, .7, .2, 1);   // the site's --ease: most of the way there by 40%
  // the mark: 3 x 3 dots on a 24-unit grid; the last one is blue
  const LOGO = [[3.4, 3.4, 1.8], [10.2, 3.4, 1.99], [18.5, 3.4, 2.38], [3.4, 10.2, 1.99], [10.2, 10.2, 2.38],
    [18.5, 10.2, 2.94], [3.4, 18.5, 2.38], [10.2, 18.5, 2.94], [18.5, 18.5, 3.9]];

  function Swarm(c, o) {
    const ctx = c.getContext("2d");
    let W = 0, H = 0, P = [], t = 0, raf = 0, last = 0, visible = false;
    let gather = -1;            // closer: the time the logo started to gather (-1: not yet)
    let slot = null;            // closer: logo centre and size, in canvas px

    function seed() {
      const n = Math.round((W < 640 ? 150 : W < 1100 ? 230 : 300) * (o.density || 1));
      let s = o.seed || 7;
      const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
      P = [];
      for (let i = 0; i < n; i++) {
        const r = Math.sqrt(rnd()) * (.2 + rnd() * .9);
        const z = rnd();
        P.push({
          a: rnd() * TAU, r, w: (.025 + rnd() * .045) * (rnd() < .5 ? -1 : 1) * (1.2 - r * .6),
          ph: rnd() * TAU, f: .35 + rnd() * .7, z, rad: .55 + z * 1.25, al: .35 + z * .65,
          blue: i % 23 === 0, x: 0, y: 0, vx: 0, vy: 0, L: -1, lx: 0, ly: 0, d: rnd()
        });
      }
      if (o.logo) {
        // assign dots to the nine logo circles, by area; blue dots go to the blue one
        const area = LOGO.map(l => l[2] * l[2]), tot = area.reduce((a, b) => a + b, 0);
        const want = Math.round(n * .36);
        const order = P.map((_, i) => i);
        for (let i = order.length - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }
        let k = 0;
        LOGO.forEach((l, li) => {
          const m = Math.max(6, Math.round(want * area[li] / tot));
          for (let j = 0; j < m && k < n; j++, k++) {
            const p = P[order[k]];
            const rr = Math.sqrt(rnd()) * .9, aa = rnd() * TAU;
            p.L = li; p.lx = l[0] + Math.cos(aa) * l[2] * rr; p.ly = l[1] + Math.sin(aa) * l[2] * rr;
            p.blue = li === 8;
          }
        });
      }
      P.forEach(p => { const h = home(p, t); p.x = h[0]; p.y = h[1]; });
    }
    // 0 = loose, 1 = pooled: a slow gather every ~16 s, held briefly, then released
    const pool = tt => {
      if (o.logo) return 0;
      const k = (tt % 16) / 16;
      return k < .55 ? 0 : k < .72 ? ease((k - .55) / .17) : k < .8 ? 1 : 1 - ease((k - .8) / .2);
    };
    // how far a dot has gone into the logo (0..1)
    const into = p => {
      if (!o.logo || p.L < 0 || gather < 0) return 0;
      return gatherEase((t - gather - p.d * 1.1 - (p.L % 3) * .1) / 2.4);
    };
    const crisp = () => (!o.logo || gather < 0) ? 0 : ease((t - gather - 3.4) / 1.1);
    function home(p, tt) {
      const g = pool(tt), spread = 1 - g * .6;
      const a = p.a + p.w * tt * (1 + g * 1.5);
      const cx = W / 2 + Math.sin(tt * .07) * W * .04, cy = H * (o.cy || .5) + Math.cos(tt * .09) * H * .05;
      const wob = 9 + 8 * (1 - g);
      let x = cx + Math.cos(a) * p.r * W * .5 * spread + Math.sin(tt * p.f + p.ph) * wob;
      let y = cy + Math.sin(a) * p.r * H * .52 * spread + Math.cos(tt * p.f * .8 + p.ph) * wob;
      const k = into(p);
      if (k > 0 && slot) {
        const u = slot.s / 24, j = 1.1 * (1 - crisp() * .6);
        const lx = slot.x + (p.lx - 12) * u + Math.sin(tt * 1.3 + p.ph) * j, ly = slot.y + (p.ly - 12) * u + Math.cos(tt * 1.1 + p.ph) * j;
        x += (lx - x) * k; y += (ly - y) * k;
      }
      return [x, y];
    }
    function measure() {
      if (!o.logo) return;
      const el = document.getElementById(o.logo), r = el && el.getBoundingClientRect(), cr = c.getBoundingClientRect();
      if (r && r.width) slot = { x: r.left - cr.left + r.width / 2, y: r.top - cr.top + r.height / 2, s: r.width };
    }
    function size() {
      const r = c.getBoundingClientRect();
      if (!r.width || !r.height) return false;
      const d = Math.min(devicePixelRatio || 1, 2);
      const nw = Math.round(r.width), nh = Math.round(r.height);
      measure();
      if (nw !== W || nh !== H) {
        W = nw; H = nh; c.width = Math.round(W * d); c.height = Math.round(H * d);
        ctx.setTransform(d, 0, 0, d, 0, 0); seed();
      }
      return true;
    }
    function step(dt) {
      t += dt;
      const k = 1 - Math.pow(.02, dt), damp = Math.pow(.2, dt);
      for (const p of P) {
        const h = home(p, t);
        if (into(p) > 0) { p.x += (h[0] - p.x) * Math.min(1, dt * 9); p.y += (h[1] - p.y) * Math.min(1, dt * 9); continue; }
        p.vx = (p.vx + (h[0] - p.x) * k * 2.2) * damp;
        p.vy = (p.vy + (h[1] - p.y) * k * 2.2) * damp;
        p.x += p.vx * dt * 6; p.y += p.vy * dt * 6;
      }
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const g = pool(t), base = (o.alpha || .22) * (1 - g * .25), cr = crisp();
      // hairlines between nearby blue peers
      const peers = P.filter(p => p.blue && into(p) < .2);
      ctx.lineWidth = .8;
      for (let i = 0; i < peers.length; i++) for (let j = i + 1; j < peers.length; j++) {
        const a = peers[i], b = peers[j], dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy, R = W < 640 ? 90 : 130;
        if (d2 > R * R) continue;
        const al = (1 - Math.sqrt(d2) / R) * .12;
        ctx.strokeStyle = `rgba(42,69,224,${al.toFixed(3)})`;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      // ink dots, in three depth bands (few fill calls)
      for (const band of [0, 1, 2]) {
        ctx.fillStyle = `rgba(20,22,29,${(base * (.45 + band * .3)).toFixed(3)})`;
        ctx.beginPath();
        for (const p of P) {
          if (p.blue || Math.min(2, (p.z * 3) | 0) !== band) continue;
          const k = into(p); if (k > 0) continue;
          ctx.moveTo(p.x + p.rad, p.y); ctx.arc(p.x, p.y, p.rad, 0, TAU);
        }
        ctx.fill();
      }
      ctx.fillStyle = `rgba(42,69,224,${(.5 - g * .1).toFixed(3)})`;
      ctx.beginPath();
      for (const p of P) if (p.blue && into(p) === 0) { const r = p.rad * 1.35; ctx.moveTo(p.x + r, p.y); ctx.arc(p.x, p.y, r, 0, TAU); }
      ctx.fill();
      if (o.logo && gather >= 0) {
        // the dots that gathered: darker as they arrive, fading back as the solid mark settles
        for (const blue of [false, true]) {
          ctx.beginPath();
          for (const p of P) {
            const k = into(p); if (k <= 0 || p.blue !== blue) continue;
            const r = p.rad * (1 - k * .15);
            ctx.moveTo(p.x + r, p.y); ctx.arc(p.x, p.y, r, 0, TAU);
          }
          const al = .28 + .5 * Math.min(1, (t - gather) / 1.5) - cr * .5;
          ctx.fillStyle = blue ? `rgba(42,69,224,${(al + .15).toFixed(3)})` : `rgba(20,22,29,${al.toFixed(3)})`;
          ctx.fill();
        }
        if (cr > 0 && slot) {
          const u = slot.s / 24;
          LOGO.forEach((l, i) => {
            ctx.fillStyle = i === 8 ? `rgba(42,69,224,${cr.toFixed(3)})` : `rgba(20,22,29,${cr.toFixed(3)})`;
            ctx.beginPath(); ctx.arc(slot.x + (l[0] - 12) * u, slot.y + (l[1] - 12) * u, l[2] * u * (.7 + .3 * cr), 0, TAU); ctx.fill();
          });
        }
      }
    }
    function frame(now) {
      raf = 0;
      const dt = last ? Math.min(.1, (now - last) / 1000) : 1 / 30;
      if (now - last >= 31 || !last) { last = now; step(dt); draw(); }
      wake();
    }
    let paused = false;   // the demo's Pause button holds the swarms too
    const wake = () => { if (!RM && !paused && visible && !document.hidden && !raf) raf = requestAnimationFrame(frame); };
    const settle = () => { P.forEach(p => { const h = home(p, t); p.x = h[0]; p.y = h[1]; }); };

    if (RM) {
      // one frame: the closer's mark already formed and crisp (gather >= 0, long enough ago)
      const still = () => { if (!size()) return; t = o.logo ? 13 : 3; if (o.logo) gather = t - 10; settle(); draw(); };
      still(); addEventListener("resize", still);
      if ("ResizeObserver" in window) new ResizeObserver(still).observe(c);
      return { still };
    }
    size(); settle(); draw();
    addEventListener("resize", () => { size(); draw(); });
    // the canvas's box changes without a window resize too (the web font lands, the section reflows)
    if ("ResizeObserver" in window) new ResizeObserver(() => { size(); draw(); }).observe(c);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(es => {
        visible = es[es.length - 1].isIntersecting;   // the latest entry: one batch can carry [false, true]
        if (visible) { last = 0; wake(); } else if (raf) { cancelAnimationFrame(raf); raf = 0; }
      }).observe(c);
      if (o.logo) new IntersectionObserver((es, io) => {
        if (!es.some(e => e.isIntersecting)) return;
        measure(); gather = t; io.disconnect();
      }, { threshold: .3 }).observe(c.parentNode);   // the closer section, 30% on screen
    } else { visible = true; if (o.logo) gather = 0; }
    document.addEventListener("visibilitychange", () => { last = 0; wake(); });
    wake();
    return {
      get paused() { return paused; },
      set paused(v) { paused = !!v; if (paused && raf) { cancelAnimationFrame(raf); raf = 0; } else if (!paused) { last = 0; wake(); } },
      get t() { return t; },
      set t(v) { t = v; settle(); draw(); },
      gatherNow(ago) { measure(); gather = t - (ago || 0); settle(); draw(); }
    };
  }

  const hero = document.getElementById("swarm"), closer = document.getElementById("swarm2");
  window.__swarm = hero && hero.getContext ? Swarm(hero, { seed: 7 }) : null;
  window.__swarm2 = closer && closer.getContext ? Swarm(closer, { seed: 13, logo: "logoSlot", density: 1.15, cy: .5, alpha: .2 }) : null;
})();
