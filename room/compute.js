// This device's screen: a full-screen view for a device that is only lending its memory and GPU (a phone on a
// charger, a laptop in the corner). It shows which layers this device holds, and a packet of dots
// runs through the logo every time a real forward pass runs here. Pure presentation: it reads the
// room's state through `state()` and is told about passes by `pass(n, ms)`; it never touches the
// GPU. The canvas only animates while packets are in flight, the page is visible and the screen
// is open; with reduced motion only the counters move.

const $ = (id) => document.getElementById(id);
const REDUCED = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
// the mark's nine dots (site/logo/mark.svg), in a 24-unit box
const DOTS = [[3.4, 3.4, 1.8], [10.2, 3.4, 1.99], [18.5, 3.4, 2.38], [3.4, 10.2, 1.99], [10.2, 10.2, 2.38], [18.5, 10.2, 2.94], [3.4, 18.5, 2.38], [10.2, 18.5, 2.94], [18.5, 18.5, 3.9]];
const PACKET = ["#2A45E0", "#7C8FFF", "#B9C6FF"];   // --blue-500, --blue-400, --blue-200
// this device's colour in the room, for its layers here; the two dark neutrals read as light on the dark screen
// a device's colour on the black screen: dark greys and near-blacks would vanish, so they show light;
// blues keep their colour (hex swatches and the generated hsl() shades alike)
const onDark = (c) => {
  if (!c) return "";
  let r, g, b;
  const hex = /^#([0-9a-f]{6})$/i.exec(c), hsl = /^hsl\((\d+) (\d+)% (\d+)%\)$/.exec(c);
  if (hex) { const n = parseInt(hex[1], 16); r = n >> 16; g = (n >> 8) & 255; b = n & 255; }
  else if (hsl) { const S = +hsl[2] / 100, L = +hsl[3] / 100, a = S * Math.min(L, 1 - L), f = (k) => { const x = (k + +hsl[1] / 30) % 12; return 255 * (L - a * Math.max(-1, Math.min(x - 3, 9 - x, 1))); }; r = f(0); g = f(8); b = f(4); }
  else return c;
  const light = (0.299 * r + 0.587 * g + 0.114 * b) / 255, sat = (Math.max(r, g, b) - Math.min(r, g, b)) / 255;
  return (sat < 0.25 && light < 0.5) || light < 0.14 ? "#EEF0F6" : c;
};
const THEME_LIGHT = "#F6F5F1", THEME_DARK = "#000000";
const fmt = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e4 ? (n / 1e3).toFixed(1) + "k" : String(n);

const gb = (b) => b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + " GB" : Math.max(1, Math.round(b / 2 ** 20)) + " MB";

// What the screen says: one status line and one small line, from the room's state `s` (see
// computeState in room.js). An ended room wins over everything else, so the screen never sits on a
// stale "Serving" after the host has gone.
export function lendStatus(s) {
  if (s.over) return s.over.final
    ? { title: "Room over", sub: s.over.why || "The host left, so this room can't answer any more. Start a new room to lend this device again." }
    : { title: "Host reconnecting", sub: s.over.why || "The host's tab closed. Waiting a minute in case it comes back." };
  const has = s.lo != null && s.hi != null;
  const model = s.model || "the model";
  // "Layers 1–20 · 2.4 GB · Qwen": which layers, and how much of the model's weights they are
  const held = has ? `Layers ${s.lo + 1}–${s.hi}${s.bytes ? " · " + gb(s.bytes) : ""} · ${model}` : "";
  // while it works too: the hop dot and the numbers show the passes (between answers it stays "Serving")
  if (s.phase === "serving" && has) return { title: "Serving", sub: held };
  if (s.phase === "loading") return { title: s.pct != null ? `Loading ${Math.round(s.pct)}%` : "Loading", sub: has ? held : model };
  if (s.phase === "serving") return { title: "Not holding layers", sub: `The other devices run ${model}` };
  return { title: "Standing by", sub: "Waiting for the room to start a model" };
}

// The things that stop a lending device from serving, as short notes for its owner: a screen that
// can sleep, a battery that is running down, a tab that was in the background. `awayMs` is how long
// the tab was just hidden (null once the note has had its time).
export function lendNotes(s) {
  // only while this device loads or holds layers: a device the room isn't using has nothing to keep up
  if (s.over || s.phase === "idle" || (s.phase === "serving" && (s.lo == null || s.hi == null))) return [];
  const notes = [];
  if (s.awayMs >= 3000) notes.push(`This tab was in the background for ${Math.round(s.awayMs / 1000)} s, and passes can stall there. Keep it in front while lending.`);
  if (s.awake === "none") notes.push(s.ios ? "This screen can sleep and stop serving: set Auto-Lock to Never." : "This screen can sleep and stop serving: set the screen timeout to its longest.");
  const b = s.battery;
  if (b && !b.charging) notes.push(b.level != null && b.level <= 0.2
    ? `Battery at ${Math.round(b.level * 100)}% and not charging. Battery saver can pause this tab: plug in to keep serving.`
    : "Not charging. Lending uses the battery fast: plug in to keep serving.");
  return notes;
}

export function computeScreen({ state, keepAwake = () => {}, newRoom = () => {} }) {
  const root = $("compute-screen"), cv = $("cs-flow"), logo = $("cs-logo");
  if (!root || !cv || !logo) return { open() {}, close() {}, pass() {}, refresh() {}, get isOpen() { return false; } };
  logo.innerHTML = DOTS.map(([x, y, r], i) => `<circle cx="${x}" cy="${y}" r="${r}" style="--i:${i};--rc:${Math.round(x / 7) + Math.round(y / 7)}"${i === 8 ? ' class="lit"' : ""}/>`).join("");
  const circles = [...logo.querySelectorAll("circle")];
  let open = false, raf = 0, timer = 0, lastSpawn = 0, owed = 0;
  let tokens = 0, lastMs = null, statAt = 0, lastRate = null;
  const stamps = [];            // [time, tokens] of the passes over the last few seconds, for the rate
  const packets = [];           // { t0, dur }
  let W = 0, H = 0, dpr = 1, cy = 0, lx = 0, lw = 0;
  let hiddenAt = 0, awayMs = null, awayAt = 0, battery = null;

  function size() {
    const r = cv.getBoundingClientRect();
    dpr = Math.min(2, devicePixelRatio || 1);
    W = r.width; H = r.height;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const lr = logo.getBoundingClientRect();
    cy = lr.top - r.top + lr.height / 2; lx = lr.left - r.left; lw = lr.width;
  }
  // the lane: faint dots across the screen through the logo
  function drawLane(ctx) {
    ctx.fillStyle = "rgba(238,240,246,0.10)";
    for (let x = 8; x < W; x += 14) { if (x > lx - 10 && x < lx + lw + 10) continue; ctx.beginPath(); ctx.arc(x, cy, 1.3, 0, 7); ctx.fill(); }
  }
  function frame(now) {
    raf = 0;
    if (!open || document.hidden) { packets.length = 0; return; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    drawLane(ctx);
    for (let i = packets.length - 1; i >= 0; i--) {
      const p = packets[i], k = (now - p.t0) / p.dur;
      if (k >= 1) { packets.splice(i, 1); continue; }
      if (k < 0) continue;
      // in from the left, through the logo (hidden behind it), out to the right
      const e = k < 0.5 ? 1 - Math.pow(1 - k * 2, 2) : Math.pow((k - 0.5) * 2, 2);
      const x = k < 0.5 ? e * (lx + lw / 2) : lx + lw / 2 + e * (W - lx - lw / 2 + 40);
      const a = k < 0.08 ? k / 0.08 : k > 0.9 ? (1 - k) / 0.1 : 1;
      PACKET.forEach((c, j) => {
        ctx.globalAlpha = a * (1 - j * 0.18);
        ctx.fillStyle = c;
        ctx.beginPath(); ctx.arc(x - j * 13, cy, 3.4 - j * 0.5, 0, 7); ctx.fill();
      });
      // a soft trail
      ctx.globalAlpha = a * 0.22;
      const g = ctx.createLinearGradient(x - 90, 0, x, 0);
      g.addColorStop(0, "rgba(42,69,224,0)"); g.addColorStop(1, "rgba(124,143,255,0.9)");
      ctx.fillStyle = g; ctx.fillRect(x - 90, cy - 1, 90, 2);
      ctx.globalAlpha = 1;
    }
    if (packets.length) raf = requestAnimationFrame(frame);
  }
  const kick = () => { if (!raf && open && !document.hidden) raf = requestAnimationFrame(frame); };
  // the logo lights up in a wave, top-left to the blue dot, as the pass goes through it
  let waveAt = 0;
  function wave(delay) {
    const now = performance.now();
    if (now - waveAt < 560) return;   // one wave at a time: fast passes ride on the running one
    waveAt = now;
    for (const c of circles) {
      const rc = +c.style.getPropertyValue("--rc");
      c.animate([{ transform: "scale(1)", fill: c.classList.contains("lit") ? "#7C8FFF" : "#EEF0F6" }, { transform: "scale(1.22)", fill: "#B9C6FF", offset: 0.35 }, { transform: "scale(1)", fill: c.classList.contains("lit") ? "#7C8FFF" : "#EEF0F6" }],
        { duration: 520, delay: delay + rc * 55, easing: "cubic-bezier(.2,.7,.2,1)" });
    }
  }
  // the pass reaches this device: a light runs along its layers
  let ringAt = 0;
  function arrive(delay) {
    const now = performance.now();
    if (now - ringAt < 420) return;
    ringAt = now;
    const strip = $("cs-strip");
    setTimeout(() => { strip.classList.remove("sweep"); void strip.offsetWidth; strip.classList.add("sweep"); }, delay);
  }
  function spawn(now) {
    const dur = Math.max(900, Math.min(1600, W * 1.3));
    // no travelling packets behind the logo: the logo wave, the hop dot and the strip show the pass
    wave(dur * 0.36);
    arrive(dur * 0.4);
  }

  function renderStats() {
    const now = performance.now();
    while (stamps.length && now - stamps[0][0] > 4000) stamps.shift();
    if (!$("cs-live")) return;
    $("cs-tok").textContent = fmt(tokens);
    // between answers: the last run's speed, not 0; a number it has never had is left out
    if (stamps.length > 1 && now - stamps[stamps.length - 1][0] < 600) lastRate = (stamps.slice(1).reduce((t, x) => t + x[1], 0) / Math.max(.05, (stamps[stamps.length - 1][0] - stamps[0][0]) / 1000)).toFixed(1);
    $("cs-rate").textContent = lastRate ?? "0";
    $("cs-rate").parentNode.hidden = lastRate == null;
    $("cs-ms").textContent = lastMs != null ? Math.round(lastMs) + " ms" : "";
    $("cs-ms").parentNode.hidden = lastMs == null;
  }
  // a pass hopped over to this device: its dot flashes
  function hop() {
    const d = root.querySelector(".cs-hop");
    if (!d || REDUCED()) return;
    d.animate([{ opacity: 1, transform: "scale(1.6)", boxShadow: "0 0 10px var(--me)" }, { opacity: .35, transform: "scale(1)", boxShadow: "0 0 0 transparent" }], { duration: 420, easing: "cubic-bezier(.2,.7,.2,1)" });
    d.parentNode.animate([{ color: "#EEF0F6" }, { color: "var(--on-game-2)" }], { duration: 420, easing: "ease-out" });
  }
  function refresh() {
    if (!open) return;
    const s = state();
    $("cs-code").textContent = s.code || "----";
    $("cs-devs").textContent = `${s.devices} device${s.devices === 1 ? "" : "s"}`;
    const has = s.lo != null && s.hi != null;
    // one status line and one small line: this screen is for the person whose device it is
    const { title, sub } = lendStatus(s);
    root.dataset.phase = s.over ? "over" : s.phase;
    if (s.color) root.style.setProperty("--me", onDark(s.color));
    // serving, and no pass for a moment: say so, and let the logo rest
    const quiet = !s.over && s.phase === "serving" && has && (!stamps.length || performance.now() - stamps[stamps.length - 1][0] > 2500);
    root.toggleAttribute("data-quiet", quiet);
    // (between answers the title stays "Serving" too; the numbers keep the last run)
    $("cs-title").textContent = title; $("cs-sub").textContent = sub;
    if ($("cs-live")) { $("cs-live").hidden = !(!s.over && s.phase === "serving" && has); renderStats(); }
    // what could stop this device serving (sleep, battery, a background tab); a room that ended offers a new one
    if (awayAt && performance.now() - awayAt > 20000) { awayMs = null; awayAt = 0; }
    const notes = lendNotes({ ...s, awayMs, battery: battery && { charging: battery.charging, level: battery.level } });
    const ul = $("cs-notes");
    if (ul && ul.dataset.sig !== notes.join("|")) {
      ul.dataset.sig = notes.join("|");
      ul.replaceChildren(...notes.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
      ul.hidden = !notes.length;
    }
    const nb = $("cs-new");
    if (nb) {
      const show = !!s.over?.final;
      if (show && nb.hidden) { nb.hidden = false; if (root.contains(document.activeElement)) nb.focus({ preventScroll: true }); }
      else if (!show) nb.hidden = true;
    }
    // this device's slice of the model
    const strip = $("cs-strip"), total = s.total || 0;
    const n = total ? Math.min(total, 64) : 0, per = total ? total / n : 1;
    if (strip.dataset.sig !== `${n}:${s.lo}:${s.hi}:${s.phase}:${Math.round(s.pct || 0)}`) {
      strip.dataset.sig = `${n}:${s.lo}:${s.hi}:${s.phase}:${Math.round(s.pct || 0)}`;
      let html = "", j = 0;
      for (let i = 0; i < n; i++) {
        const L = i * per, mine = has && L >= s.lo && L < s.hi;
        const got = mine && (s.phase !== "loading" || (L - s.lo) / Math.max(1, s.hi - s.lo) * 100 < (s.pct || 0));
        html += `<i class="${mine ? (got ? "mine" : "mine wait") : ""}"${mine ? ` style="--j:${j++}"` : ""}></i>`;
      }
      strip.innerHTML = html;
      strip.style.setProperty("--n", n);
    }
    strip.hidden = !n || !!s.over;
    renderStats();
  }

  function show(on) {
    if (on === open) return;
    open = on;
    root.hidden = !on;
    // the browser's bar matches the screen: dark while lending, the room's off-white after
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", on ? THEME_DARK : THEME_LIGHT);
    document.documentElement.classList.toggle("computing", on);
    if (on) {
      keepAwake();   // from the tap that opened the screen: the wake lock needs a user gesture on iOS
      // on battery the screen says so (Chrome and Android; Safari and Firefox have no battery API)
      if (!battery && navigator.getBattery) navigator.getBattery().then((b) => {
        battery = b;
        for (const ev of ["chargingchange", "levelchange"]) b.addEventListener(ev, refresh);
        refresh();
      }).catch(() => {});
      requestAnimationFrame(() => { size(); kick(); });
      refresh();
      timer = setInterval(refresh, 1000);
      $("compute-exit").focus({ preventScroll: true });
    } else {
      clearInterval(timer); timer = 0;
      if (raf) cancelAnimationFrame(raf); raf = 0;
      packets.length = 0;
      $("compute-open")?.focus({ preventScroll: true });   // back to the button that opened it
    }
  }
  $("compute-exit").addEventListener("click", () => show(false));
  $("cs-new")?.addEventListener("click", () => newRoom());
  addEventListener("keydown", (e) => {
    if (!open) return;
    if (e.key === "Escape") { show(false); return; }
    // a modal screen: Tab cycles through its own buttons, never to the room hidden behind it
    if (e.key !== "Tab") return;
    const f = [...root.querySelectorAll("button")].filter((b) => !b.hidden && !b.disabled);
    if (!f.length) return;
    const i = f.indexOf(document.activeElement);
    if (e.shiftKey ? i <= 0 : i === f.length - 1 || i < 0) { e.preventDefault(); f[e.shiftKey ? f.length - 1 : 0].focus(); }
  });
  addEventListener("resize", () => { if (open) { size(); kick(); } });
  document.addEventListener("visibilitychange", () => {
    if (!open) return;
    if (document.hidden) {
      hiddenAt = performance.now();
      // the tab strip is all its owner sees now: say what this tab needs
      if (root.dataset.phase === "serving" || root.dataset.phase === "loading") document.title = "Lending \u00b7 bring this tab back \u00b7 pooled";
      return;
    }
    if (hiddenAt) { awayMs = performance.now() - hiddenAt; awayAt = performance.now(); hiddenAt = 0; }
    size(); kick(); refresh();
  });

  return {
    open: () => show(true),
    close: () => show(false),
    refresh,
    get isOpen() { return open; },
    // a forward pass ran on this device: n tokens went through it, in ms
    pass(n = 1, ms = null) {
      tokens += n;
      if (ms != null) lastMs = ms;
      const now = performance.now();
      stamps.push([now, n]);
      if (stamps.length > 400) stamps.splice(0, stamps.length - 400);
      if (!open || document.hidden) return;
      if (root.hasAttribute("data-quiet")) refresh();
      if (now - statAt > 250) { statAt = now; renderStats(); }
      hop();
      if (REDUCED()) return;
      // at most one packet per 140 ms; faster passes ride along with the next one
      if (now - lastSpawn >= 140) { lastSpawn = now; owed = 0; spawn(now); }
      else if (!owed) { owed = 1; setTimeout(() => { owed = 0; if (open) { lastSpawn = performance.now(); spawn(lastSpawn); } }, 140 - (now - lastSpawn)); }
    },
  };
}
