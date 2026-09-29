/* The demo: one window, two modes, eight steps, looping. It starts with the room, the way a visitor would.
   Chat: a laptop starts a room (it already has a friendly name), a desktop joins with the code. Once the
   devices are in, each one lends memory to the room: the laptop turns its own amount up, the desktop sends
   its share, and a friend's phone joins and adds a little, which is what lets the biggest model fit. The
   room picks that model, the layers are dealt by memory and each device fetches only its own (one from its
   cache). "what is Pooled?", and while the answer streams a hidden state travels through every layer on
   every device, once per word. The answer ends "It can chat, or write code." and, a second later, the Code tab is pressed.
   Code: the visitor asks for an app (a different one each loop: Tetris, 2048, a space shooter, Snake,
   Breakout); the files appear as the agent writes them; it serves the app on :5173 and the preview opens
   on it, running; then a change request, an edit, a reload, and the changed app plays for a moment (with
   an offer to take it over) before the story starts again from the room.
   The parts worth following (the joining, the lending, the layer split, the first answer) run slower than
   the rest.
   The HTML holds the finished state (readable without JS). This script rewinds and replays it. */
(() => {
  "use strict";
  document.documentElement.classList.add("js"); // also set early by boot.js
  const $ = id => document.getElementById(id);
  // the code's colours, as the room's editor shows them (site/js/hl.js); plain text without it
  const HL = (path, line) => window.PooledHL ? window.PooledHL(path, line) : String(line).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const demo = $("demo");
  if (!demo || !window.PooledApps) return;
  const SPEED = 1.33;              // the story runs 33% faster than its timeline's seconds
  const SLOW = 1.25;               // ...except the parts that matter, which take 25% longer than that

  const restart = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };
  const press = (el, t) => { restart(el, "press"); setTimeout(() => el.classList.remove("press"), t || 260); };
  const scene = s => { demo.dataset.scene = s; };
  const flag = (cls, on) => demo.classList.toggle(cls, on);
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
  const gbs = x => (Math.round(x * 10) / 10) + " GB";

  /* ---------- friendly names: every device gets one, and you can change it ---------- */
  const ANI = ["fox", "heron", "lynx", "otter", "panda", "robin", "koala", "falcon", "badger", "owl", "wren", "moose"];   // one lowercase word, as the room names devices
  let NAMES = ["fox", "heron", "lynx"];
  const newNames = () => {
    const b = ANI.slice().sort(() => Math.random() - .5);
    NAMES = [0, 1, 2].map(i => b[i]);
    demo.querySelectorAll("[data-name]").forEach(el => { el.textContent = NAMES[+el.dataset.name]; });
    ANSWER = `Pooled splits big models across devices, so together they run models none of them could run alone. It can chat, or write code.`;
    WORDS = ANSWER.split(" "); timeWords();
  };

  /* ---------- the apps the agent builds ---------- */
  const html = (title, extra) => ['<!doctype html>', '<html lang="en">', '<head>', '  <meta charset="utf-8">', `  <title>${title}</title>`, '  <link rel="stylesheet" href="style.css">', '</head>', '<body>', '  <canvas id="game"></canvas>', ...(extra || []), '  <script src="game.js"></script>', '</body>'];
  const css = bg => ['* { box-sizing: border-box; }', 'body {', '  margin: 0; display: grid; place-items: center;', `  min-height: 100vh; background: ${bg};`, '}', '#game { width: min(92vw, 440px); aspect-ratio: 5 / 6; }', 'body { font: 500 14px ui-monospace, monospace; color: #EEF0F6; }'];
  const APPS = {
    tetris: {
      dir: "tetris", title: "Tetris", ask: "a tetris game", prompt: "build me a tetris game",
      src: ['const COLS = 10, ROWS = 20;', 'const COLORS = ["#F08A6C", "#F2C14E", "#6CC5A1", ...];', 'const SHAPES = { I: [[0,1],[1,1],[2,1],[3,1]], O: [[1,0],[2,0],[1,1],[2,1]], ... };', 'let grid = empty(), queue = bag(), piece = spawn();', 'let score = 0, lines = 0, last = 0;', '',
        'function spawn() {', '  if (queue.length < 2) queue.push(...bag());', '  return { k: queue.shift(), x: 3, y: -1, r: 0 };', '}', '', 'function fits(p) {', '  return cells(p).every(([x, y]) =>', '    x >= 0 && x < COLS && y < ROWS && !grid[y]?.[x]);', '}', '',
        'function clearLines() {', '  const full = grid.filter(row => row.every(Boolean));', '  grid = grid.filter(row => !row.every(Boolean));', '  while (grid.length < ROWS) grid.unshift(Array(COLS).fill(0));', '  lines += full.length;', '  score += [0, 100, 300, 500, 800][full.length];', '}', '',
        'function loop(t) {', '  if (t - last > speed()) { drop(); last = t; }', '  draw(grid, piece);', '  requestAnimationFrame(loop);', '}', '', 'addEventListener("keydown", e => move(e.key));', 'requestAnimationFrame(loop);'],
      done: "Tetris is running on :5173.",
      prompt2: "make the pieces blue and add a next-piece preview",
      diff: [["del", 'const COLORS = ["#F08A6C", "#F2C14E", "#6CC5A1", ...];'], ["add", 'const COLORS = ["#2A45E0", "#6E86FF", "#A5B4FC", ...];'], ["add", 'const next = document.getElementById("next");'], ["add", 'function drawNext() { paint(next, queue[0]); }'], ["add", '// called after each spawn()']],
      done2: "Done. The pieces are blue, and the next one shows beside the board.",
      aria: "The Tetris game the agent wrote, playing itself. Play it with the arrow keys and space; Escape hands it back."
    },
    "2048": {
      dir: "2048", title: "2048", ask: "a 2048 game", prompt: "build me a 2048 game",
      src: ['const N = 4;', 'const TILE = { 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179", ... };', 'let board = Array.from({ length: N }, () => Array(N).fill(0));', 'let score = 0;', '',
        'function slide(row) {', '  const v = row.filter(Boolean), out = [];', '  for (let i = 0; i < v.length; i++) {', '    if (v[i] === v[i + 1]) { out.push(v[i] * 2); score += v[i] * 2; i++; }', '    else out.push(v[i]);', '  }', '  while (out.length < N) out.push(0);', '  return out;', '}', '',
        'function move(dir) {', '  const before = JSON.stringify(board);', '  board = rotate(board, dir).map(slide);', '  board = rotate(board, -dir);', '  if (JSON.stringify(board) !== before) spawn();', '  animate(); draw();', '}', '',
        'function spawn() {', '  const empty = cellsWhere(v => v === 0);', '  const [x, y] = empty[Math.floor(Math.random() * empty.length)];', '  board[y][x] = Math.random() < 0.9 ? 2 : 4;', '}', '',
        'const KEYS = { ArrowLeft: 0, ArrowUp: 1, ArrowRight: 2, ArrowDown: 3 };', 'addEventListener("keydown", e => e.key in KEYS && move(KEYS[e.key]));', 'spawn(); spawn(); draw();'],
      done: "2048 is running on :5173.",
      prompt2: "make the tiles blue and keep a best score",
      diff: [["del", 'const TILE = { 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179", ... };'], ["add", 'const TILE = { 2: "#DCE2FF", 4: "#C9D1F7", 8: "#A5B4FC", ... };'], ["add", 'let best = Number(localStorage.best) || 0;'], ["add", 'best = Math.max(best, score);'], ["add", 'localStorage.best = best;']],
      done2: "Done. The tiles are blue, and your best score sticks around.",
      aria: "The 2048 game the agent wrote, playing itself. Play it with the arrow keys; Escape hands it back."
    },
    shooter: {
      dir: "space-shooter", title: "Space shooter", ask: "a space shooter", prompt: "build me a space shooter",
      src: ['const ship = { x: 0.5, lives: 3 };', 'const shots = [], bombs = [], sparks = [];', 'let enemies = wave(4, 7), score = 0;', '',
        'function wave(rows, cols) {', '  const out = [];', '  for (let y = 0; y < rows; y++)', '    for (let x = 0; x < cols; x++) out.push({ x, y, alive: true });', '  return out;', '}', '',
        'function update(dt) {', '  ship.x = clamp(ship.x + input.dx * dt, 0.05, 0.95);', '  for (const s of shots) s.y -= 520 * dt;', '  for (const b of bombs) b.y += 170 * dt;', '  for (const e of enemies) if (e.alive && hit(e, shots)) {', '    e.alive = false; score += 10 * (4 - e.y);', '  }', '  if (Math.random() < dt * 1.3) enemyFires();', '  if (!enemies.some(e => e.alive)) enemies = wave(4, 7);', '}', '',
        'function draw() {', '  ctx.fillStyle = "#0B0F1F"; ctx.fillRect(0, 0, W, H);', '  enemies.forEach(drawInvader);', '  drawShip(ship.x * W, H - 34);', '  hud(score, ship.lives);', '}', '', 'loop(update, draw);'],
      done: "The shooter is running on :5173.",
      prompt2: "add a starfield and make the enemies explode",
      diff: [["add", 'const stars = makeStars(90);'], ["add", 'function drawStars(dt) { for (const s of stars) s.y += s.z * dt; }'], ["add", 'function explode(x, y, color) {'], ["add", '  for (let i = 0; i < 18; i++) sparks.push(spark(x, y, color));'], ["add", '}']],
      done2: "Done. Stars drift past, and every hit bursts into sparks.",
      aria: "The space shooter the agent wrote, playing itself. Move with the arrow keys and fire with space; Escape hands it back."
    },
    snake: {
      dir: "snake", title: "Snake", ask: "a snake game", prompt: "build me a snake game",
      src: ['const COLS = 16, ROWS = 18, TICK = 95;', 'let snake = [[5, 9], [4, 9], [3, 9]];', 'let dir = [1, 0], food = place(), score = 0;', '',
        'function step() {', '  const [hx, hy] = snake[0];', '  const head = [hx + dir[0], hy + dir[1]];', '  if (hitsWall(head) || hitsSelf(head)) return reset();', '  snake.unshift(head);', '  if (same(head, food)) { score += 10; food = place(); }', '  else snake.pop();', '}', '',
        'function draw() {', '  ctx.fillStyle = "#0E1430"; ctx.fillRect(0, 0, W, H);', '  ctx.fillStyle = "#6CC5A1"; dot(food);', '  snake.forEach((p, i) => {', '    ctx.fillStyle = "#F08A6C";', '    cell(p);', '  });', '}', '',
        'addEventListener("keydown", e => turn(e.key));', 'setInterval(() => { step(); draw(); }, TICK);'],
      done: "Snake is running on :5173.",
      prompt2: "make the snake a blue gradient and speed up as it grows",
      diff: [["del", '    ctx.fillStyle = "#F08A6C";'], ["add", '    ctx.fillStyle = mix("#2A45E0", "#A5B4FC", i / snake.length);'], ["del", 'setInterval(() => { step(); draw(); }, TICK);'], ["add", 'const tick = () => Math.max(55, 100 - snake.length);'], ["add", 'loop(() => { step(); draw(); }, tick);']],
      done2: "Done. The snake fades from blue to lavender and speeds up as it eats.",
      aria: "The Snake game the agent wrote, playing itself. Steer with the arrow keys; Escape hands it back."
    },
    breakout: {
      dir: "breakout", title: "Breakout", ask: "a breakout game", prompt: "build me a breakout game",
      src: ['const ROWS = 6, COLS = 9;', 'const COLORS = ["#B9C6FF", "#7C8FFF", "#2A45E0", "#1C33B8", "#2A45E0", "#7C8FFF"];', 'let bricks = grid(ROWS, COLS), paddle = 0.5;', 'let ball = { x: 0.5, y: 0.7, vx: 0.32, vy: -0.62 };', 'let score = 0, lives = 3;', '',
        'function update(dt) {', '  ball.x += ball.vx * dt; ball.y += ball.vy * dt;', '  if (ball.x < 0 || ball.x > 1) ball.vx *= -1;', '  if (ball.y < 0.1) ball.vy *= -1;', '  if (onPaddle(ball)) bounce(ball, paddle);', '  const b = bricks.find(b => b.alive && inside(ball, b));', '  if (b) { b.alive = false; ball.vy *= -1; score += 10; }', '  if (ball.y > 1) { lives--; serve(); }', '}', '',
        'function draw() {', '  bricks.filter(b => b.alive).forEach(drawBrick);', '  drawPaddle(paddle);', '  drawBall(ball);', '}', '',
        'addEventListener("pointermove", e => paddle = e.clientX / innerWidth);', 'loop(update, draw);'],
      done: "Breakout is running on :5173.",
      prompt2: "add a glowing trail to the ball and sparks when bricks break",
      diff: [["add", 'trail.push([ball.x, ball.y]); if (trail.length > 12) trail.shift();'], ["add", 'ctx.shadowColor = "#6E86FF"; ctx.shadowBlur = 16;'], ["add", 'function shatter(b) {'], ["add", '  burst(b.x, b.y, COLORS[b.row], 14);'], ["add", '}']],
      done2: "Done. The ball leaves a glowing trail, and bricks shatter into sparks.",
      aria: "The Breakout game the agent wrote, playing itself. Play it with the arrow keys or the pointer; Escape hands it back."
    }
  };
  // the apps in the order the loops build them: tetris last
  const ORDER = ["shooter", "snake", "breakout", "tetris"];
  Object.values(APPS).forEach(a => { a.files = { "index.html": html(a.title), "style.css": css("#0B0F1F"), "game.js": a.src }; });
  /* ---------- Chat | Code ---------- */
  const mChat = $("mChat"), mCode = $("mCode"), modes = mChat.parentNode, win = $("win");
  const mode = m => {
    if (demo.dataset.mode === m) return;
    demo.dataset.mode = m;
    [[mChat, "chat"], [mCode, "code"]].forEach(([b, k]) => { b.setAttribute("aria-selected", k === m); b.tabIndex = k === m ? 0 : -1; });
    win.setAttribute("aria-labelledby", m === "code" ? "mCode" : "mChat");
    if (m === "code") window.pooledSparkle?.(mCode);   // switching to Code sparkles (site/js/sparkle.js)
  };

  /* ---------- the room: what each device lends, and the layers each one gets ---------- */
  const LEND = [10, 12, 2];                      // GB: the laptop, the desktop, the friend's phone
  const NEEDS = [4, 17, 22.5];                   // the three models in the picker
  const TOP = 24 * 1.06;                         // the meter's scale, as the room draws it
  const SEG_GB = [9.6, 11.2, 1.7];               // 40 layers of 22.5 GB, dealt 17 / 20 / 3 by memory

  /* ---------- the timeline's shape (timeline seconds; played at SPEED, or SPEED / SLOW where it matters) ---------- */
  const S1 = 2.5, S2 = S1 + 2.7;                                   // 1: a room, 2: the desktop joins
  const LEND0 = S2 + .6, LENDI = .09, CALM = S2 + 1.9, JOIN3 = S2 + 2.6, LEND3 = S2 + 3.3;   // 3: lending; the phone joins
  const S3 = S2 + 4.2;                                             // 4: pick a model; the layers are dealt
  const HOV1 = S3 + .3, HOV2 = S3 + .55, SEL = S3 + .8, STH = S3 + 1.1, STP = S3 + 1.35;
  const DL = S3 + 1.6, DEAL = DL + .3, FILL0 = DL + 1, CACHE1 = FILL0 + .6, FILL1 = FILL0 + 2, ETA0 = FILL0 + .5;
  const CH = FILL1 + .6, A0 = CH + 1;                              // 5: chat
  let ANSWER = $("a1").textContent, WORDS = ANSWER.split(" ");
  const DUR = i => [1.2, .6, .38, .26][i] || .08;    // each word's trip; the first slow enough to follow
  let WT = [], A1 = 0, C = 0, STEPS = [], END = 0, GAME = 0, SLOWS = [], SWEEPS = [];
  // 6 to 8, from C (the switch to Code): the ask, the files, :5173 serving the app; the change, the reload
  const ASK = 2.05, FILES = 2.8, SERVED = 5.45, CHANGE = 6.8, RELOAD = 8.85, SHOWN = 9.1;
  function timeWords() {
    WT = [A0]; WORDS.forEach((_, i) => WT.push(WT[i] + DUR(i)));
    // a sweep starts with a word, at most one per lap (as room.js mapPulse): the fast words ride along
    SWEEPS = []; WT.slice(0, -1).forEach(x => { if (!SWEEPS.length || x - SWEEPS[SWEEPS.length - 1] >= .6) SWEEPS.push(x); });
    A1 = WT[WORDS.length];
    C = Math.ceil((A1 + 1.5) * 10) / 10;               // a second to read the answer, then the tab switches
    GAME = C + SHOWN;                                  // the app, changed, on screen
    STEPS = [0, S1, S2, S3, CH, C, C + FILES, C + CHANGE];
    END = GAME + 2 * SPEED;                            // about two seconds of it, then the story starts again
    SLOWS = [[0, S1 + 2.1], [S2, S3], [DL, FILL1 + .3], [A0, A1]];
  }
  timeWords();
  const c = x => C + x;                               // steps 6 to 8
  // the Code part (the switch to Code until the changed app is on screen) plays at half speed: twice as long
  const CODE_RATE = .5;
  const rate = t => (t >= C && t < GAME ? CODE_RATE : 1) * (SLOWS.some(([a, b]) => t >= a && t < b) ? SPEED / SLOW : SPEED);

  /* ---------- the caption bar ---------- */
  const dotBtns = [...demo.querySelectorAll(".sb-dots button")];
  dotBtns.forEach(b => b.removeAttribute("tabindex"));   // (the markup keeps them out of the tab order for no-JS)
  const CAPS = dotBtns.map(b => b.querySelector(".lbl").textContent);
  const sbN = $("sbN"), sbT = $("sbT");
  let shownStep = -2;
  const stepAt = t => { let k = 0; STEPS.forEach((s, i) => { if (t >= s - 1e-6) k = i; }); return k; };
  const paintBar = (t, animate) => {
    const k = stepAt(t);
    if (k !== shownStep) {
      shownStep = k;
      sbN.textContent = `${k + 1}/${STEPS.length}`;
      sbT.textContent = CAPS[k];
      if (animate && !RM) restart(sbT, "in");
      dotBtns.forEach((b, i) => {
        b.classList.toggle("done", i < k); b.classList.toggle("cur", i === k);
        if (i === k) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
      });
    }
    const span = (STEPS[k + 1] ?? END) - STEPS[k];
    const p = RM ? 100 : clamp01((t - STEPS[k]) / span) * 100;
    dotBtns[k].style.setProperty("--p", p.toFixed(1) + "%");
  };
  const labelDots = () => dotBtns.forEach((b, i) => b.setAttribute("aria-label", `Step ${i + 1} of ${STEPS.length}: ${CAPS[i]}`));

  /* ---------- 1, 2: the room forms ---------- */
  const CODE = "K7QX";
  const tabA = $("tabA"), tabB = $("tabB"), aGo = $("aGo"), bBtn = $("bBtn"), nmA = $("nmA"), nmB = $("nmB");
  const aCode = [...$("aCode").children], hdCode = [...$("hdCode").children];
  const slots = [...tabB.querySelectorAll(".slots i")];
  const setSlots = (n, cur) => slots.forEach((s, i) => {
    const ch = i < n ? CODE[i] : "";
    if (s.textContent !== ch) { s.textContent = ch; if (ch) restart(s, "lit"); }
    s.classList.toggle("cur", i === cur);
  });
  const letters = n => {
    aCode.forEach((s, i) => s.classList.toggle("off", i >= n));
    hdCode.forEach((s, i) => { s.textContent = i < n ? CODE[i] : "-"; s.classList.toggle("off", i >= n); });
  };
  // a device joins first (its chip, no memory yet); what it lends comes after
  const chips = [...$("chips").children];
  const devRows = [...$("devs").children];
  const setDevices = (n, animate) => {
    chips.forEach((ch, i) => { ch.classList.toggle("out", i >= n); if (animate && i === n - 1) restart(ch, "in"); });
    devRows.forEach((li, i) => { li.classList.toggle("out", i >= n); if (animate && i === n - 1) restart(li, "in"); });
  };

  /* ---------- 3: each device lends memory; 4: pick a model that fits ---------- */
  const model = $("model"), card = $("card"), band = $("band"), rowsEl = $("rows"), mdS = $("mdS");
  const gbSum = $("gbSum"), hdSum = gbSum.parentNode, poolGB = $("poolGB");
  const fills = [...$("meter").querySelectorAll(".mt-fill i")], ticks = [...$("meter").querySelectorAll(".mt-ticks span")];
  const rungs = [...$("rungs").children], startBtn = $("startBtn"), need = $("need");
  const lendGB = $("lendGB"), lendPlus = $("lendPlus");
  let lent = [0, 0, 0], joining = -1;
  const paintPool = (animate, who) => {
    const sum = lent.reduce((a, b) => a + b, 0);
    const txt = gbs(sum);
    if (poolGB.textContent !== txt) { poolGB.textContent = txt; if (animate && !RM) restart(poolGB, "bump"); }
    gbSum.textContent = txt; hdSum.classList.toggle("none", sum === 0);
    if (animate && !RM) restart(hdSum, "bump");
    fills.forEach((f, k) => { f.style.width = (100 * lent[k] / TOP).toFixed(2) + "%"; });
    ticks.forEach((tk, k) => tk.classList.toggle("ok", sum >= NEEDS[k]));
    const short = Math.round((NEEDS[2] - sum) * 10) / 10;
    need.lastChild.textContent = short > 0 ? `${short} GB short` : "fits"; need.classList.toggle("ok", short <= 0);
    chips.forEach((ch, k) => { ch.querySelector(".cg").textContent = lent[k] ? gbs(lent[k]) : ""; });
    devRows.forEach((li, k) => {
      const b = li.querySelector("b"), t = k === joining ? "joined" : lent[k] ? gbs(lent[k]) : "";
      if (b.textContent !== t) { b.textContent = t; if (animate && k === who && !RM) restart(b, "bump"); }
      b.classList.toggle("join", k === joining);
    });
    rungs.forEach((li, k) => {
      const ok = sum >= NEEDS[k], was = li.classList.contains("ok");
      li.classList.toggle("ok", ok);
      li.querySelector(".more").textContent = ok ? "" : `needs ${Math.round((NEEDS[k] - sum) * 10) / 10} GB more`;
      if (ok && !was && animate && sum > 0 && !RM) restart(li, "unlock");
    });
    lendGB.textContent = lent[0];
    startBtn.classList.toggle("off", !rungs.some(li => li.classList.contains("sel") && li.classList.contains("ok")));
  };
  const lend = (k, gb, animate) => { lent[k] = gb; if (k === joining) joining = -1; paintPool(animate, k); };
  const rungHover = k => rungs.forEach((li, i) => li.classList.toggle("hov", i === k));
  const rungSelect = k => { rungs.forEach((li, i) => li.classList.toggle("sel", i === k)); ticks.forEach((tk, i) => tk.classList.toggle("sel", i === k)); paintPool(false); };

  /* ---------- 4: the layers are dealt; each device downloads only its own ---------- */
  // the load card (as the room's #load-card): a percent and a bar per device, the layers strip, the bytes and time left
  const lcRows = [...$("lcRows").children], lcStrip = [...$("lcStrip").children], lcStatus = $("lcStatus");
  const LC_RANGE = ["1\u201317", "18\u201337", "38\u201340"];
  let lcKey = "";
  const segs = [0, 1, 2].map(k => [...$("cells" + k).children]);
  const cells = segs.flat();
  cells.forEach((cel, i) => cel.style.setProperty("--k", i));
  const hg = [$("hg0"), $("hg1"), $("hg2")];
  const rows = [...rowsEl.querySelectorAll(".row")];
  let filled = [-1, -1, -1], mdKey = "";
  const fillSeg = (k, n) => {
    if (n === filled[k]) return;
    segs[k].forEach((cel, i) => {
      const on = i < n;
      if (on !== cel.classList.contains("f")) { cel.classList.toggle("f", on); if (on && !RM) restart(cel, "f"); }
      lcStrip[[0, 17, 37][k] + i].classList.toggle("f", on);
    });
    filled[k] = n;
    const all = segs[k].length, gb = (SEG_GB[k] * n / all).toFixed(1);
    const done = n >= all;
    hg[k].classList.toggle("ok", done);
    // .hx and .of drop out on a narrow stage, where "3.9/5.6 GB" replaces "3.9 of 5.6 GB"
    if (k === 0) hg[k].innerHTML = done ? `${SEG_GB[0]} GB<span class="hx">, <span class="cache">from cache</span></span>` : n > 0 ? `<span class="hx"><span class="cache">from cache</span> · </span>${gb} GB` : `${SEG_GB[0]} GB`;
    else hg[k].innerHTML = done ? `${SEG_GB[k]} GB<span class="hx">, ready</span>` : n > 0 ? `${gb}<span class="of"> of </span><span class="sl">/</span>${SEG_GB[k]} GB` : `${SEG_GB[k]} GB`;
  };
  const setMd = (txt, eta) => {
    const key = txt + "|" + (eta || "");
    if (key === mdKey) return;
    const fresh = eta && !mdKey.includes("left");
    mdKey = key;
    // on a narrow stage the "Downloading · " prefix drops out and "16.9 of 22.5 GB" becomes "16.9/22.5 GB"
    const m = /^Downloading · ([\d.]+) of (.*)$/.exec(txt);
    if (m) mdS.innerHTML = `<span class="hx">Downloading · </span>${m[1]}<span class="of"> of </span><span class="sl">/</span>${m[2]}`;
    else mdS.textContent = txt;
    if (eta) { const e = document.createElement("span"); e.className = "eta" + (fresh && !RM ? " in" : ""); e.textContent = " · " + eta; mdS.append(e); }
  };
  const lcPaint = (pcts, gb, eta) => {
    const key = pcts.join() + "|" + gb + "|" + eta;
    if (key === lcKey) return;
    lcKey = key;
    lcRows.forEach((rw, k) => {
      const pct = pcts[k], done = pct >= 100;
      rw.classList.toggle("done", done);
      rw.querySelector(".fill").style.width = pct + "%";
      rw.querySelector(".pct").textContent = done ? "ready" : pct + "%";
      rw.querySelector(".lr").innerHTML = (done ? "" : '<span class="lw">downloading </span>') + "layers " + LC_RANGE[k];   // as the room: the layers under the name
      chips[k].querySelector(".cst").textContent = done || gb == null ? "" : pct + "%";
    });
    // before the bytes flow the room says what this device is doing; then the bytes, the total and the time left
    const b = +gb >= 1 ? gb + " GB" : Math.round(gb * 1024) + " MB";
    lcStatus.innerHTML = gb == null ? '<span class="src gpu">This device</span><span>Getting this device ready</span>'
      : pcts.every(x => x >= 100) ? '<span class="src">Ready</span><span class="b">22.5 GB on 3 devices</span>'   // every row loaded: not "Downloading 22.5 of 22.5"
      : `<span class="src">Downloading</span><span class="b">${b} of 22.5 GB</span>` + (eta ? `<span class="eta${/^estimating/.test(eta) ? " wait" : ""}">${eta}</span>` : "");
  };
  const loadAt = t => {
    if (t < FILL0) { [0, 1, 2].forEach(k => fillSeg(k, 0)); lcPaint([0, 0, 0], null, ""); setMd("22.5 GB · 40 layers · split 3 ways by memory"); return; }
    const p0 = clamp01((t - FILL0) / (CACHE1 - FILL0)), p = clamp01((t - FILL0) / (FILL1 - FILL0)), p2 = clamp01(p * 1.3);
    fillSeg(0, Math.ceil(17 * p0)); fillSeg(1, Math.ceil(20 * p)); fillSeg(2, Math.ceil(3 * p2));
    const pcts = [p0, p, p2].map(x => Math.round(x * 100));
    if (p >= 1) { lcPaint(pcts, "22.5", ""); setMd("Ready on 3 devices · 40 layers, split by memory"); return; }
    const gb = (SEG_GB[0] * p0 + SEG_GB[1] * p + SEG_GB[2] * p2).toFixed(1);
    // what is left, at about 50 MB/s: a believable home connection, not the demo's own pace
    const left = (SEG_GB[1] + SEG_GB[2]) * (1 - clamp01((t - ETA0) / (FILL1 - ETA0)) * .75) * .85, mins = Math.round(left / .05 / 60);
    const eta = t < ETA0 ? "estimating time" : mins >= 1 ? `about ${mins} min left` : "under a minute left";
    lcPaint(pcts, gb, t < ETA0 ? "estimating time left" : mins > 1 ? eta : "about a minute left");
    setMd(`Downloading · ${gb} of 22.5 GB`, eta);
  };

  /* ---------- 5: a hidden state, through every layer, once per word ---------- */
  const pkt = $("pkt"), a1 = $("a1"), bdLive = $("bdLive");
  // the band's pill while the room writes: a word every two seconds, as in the room ("Twinkling…", "Weaving…"),
  // counted on the story's clock so seeking shows the right one; s < 0: "Ready"
  const LIVE = ["Twinkling", "Weaving", "Shimmering", "Humming", "Stitching", "Conjuring", "Scribbling", "Whirring", "Sparkling", "Spinning"];
  let liveK = -2;
  const liveWord = s => {
    const k = s < 0 ? -1 : Math.floor(s / 2) % LIVE.length;
    if (k === liveK) return;
    const anim = liveK >= 0 && k >= 0 && !RM && !frozen;
    liveK = k; bdLive.textContent = k < 0 ? "Ready" : LIVE[k] + "\u2026";
    if (anim && bdLive.animate) bdLive.animate([{ opacity: 0, transform: "translateY(3px)" }, { opacity: 1, transform: "none" }], { duration: 260, easing: "cubic-bezier(.2,.7,.2,1)" });
  };
  let geo = null;
  const measure = () => {
    const R = rowsEl.getBoundingClientRect(), k = (R.width / rowsEl.offsetWidth) || 1;
    const r = cel => { const b = cel.getBoundingClientRect(); return { x: (b.left - R.left + b.width / 2) / k, y: (b.top - R.top + b.height / 2) / k }; };
    geo = { ends: segs.map(s => [r(s[0]), r(s[s.length - 1])]) };
  };
  // the trip, in legs weighted by how far they go: along each device's layers, a step down to the next device
  const BASE = [0, 17, 37];
  const LEGS = [["seg", 0, 17], ["hop", 0, 3], ["seg", 1, 20], ["hop", 1, 3], ["seg", 2, 3], ["end", 2, 4]];
  const LW = LEGS.reduce((a, l) => a + l[2], 0);
  let hotI = -1;
  const mini = [...demo.querySelectorAll(".bd-mini i")], bdTok = $("bdTok");
  const setHot = i => { if (i === hotI) return; if (hotI >= 0) { cells[hotI].classList.remove("hot"); mini[hotI].classList.remove("hot"); } if (i >= 0) { cells[i].classList.add("hot"); mini[i].classList.add("hot"); } hotI = i; };
  const tok = d => { chips.forEach((ch, j) => ch.classList.toggle("tok", j === d)); rows.forEach((rw, j) => rw.classList.toggle("tok", j === d)); };
  /* the room's token sweep (p2p.html #room-screen.sweep: cellhot, actflash, cardtok, minihot): a glow runs along
     the layers, lane after lane, each device's dot and chip lighting as it reaches them. Driven by the timeline,
     so seeking and freezing show it too. SW is the room's lap x .9 (a 600 ms lap), in timeline seconds */
  const SW = .54;
  const bump = (q, peak) => q <= 0 || q >= 1 ? 0 : q < peak ? q / peak : (1 - q) / (1 - peak);
  const acts = rows.map(r => r.querySelector(".act")), C0 = [0, 17, 37];
  let swOn = false;
  const sweep = tau => {
    if (tau == null || tau > 2.2 * SW) { if (!swOn) return; tau = -1; swOn = false; } else swOn = true;
    const k = v => v.toFixed(3);
    cells.forEach((cel, c) => cel.style.setProperty("--h", k(bump((tau - SW * c / 40) / SW, .3))));
    mini.forEach((m, c) => m.style.setProperty("--h", k(bump((tau - SW * (c < 17 ? 0 : c < 37 ? 1 : 2) / 3) / SW, .3))));
    acts.forEach((a, d) => a.style.setProperty("--h", k(bump((tau - SW * C0[d] / 40) / SW, .25))));
    chips.forEach((ch, d) => ch.style.setProperty("--h", k(bump((tau - SW * d / 4 / .9) / (SW / .9), .2))));
  };
  const trip = u => {
    if (!geo) measure();
    const { ends } = geo;
    let acc = 0, leg = LEGS[0], k = 0;
    for (const l of LEGS) { if (u * LW < acc + l[2]) { leg = l; k = (u * LW - acc) / l[2]; break; } acc += l[2]; leg = l; k = 1; }
    const [type, s] = leg;
    if (type === "seg") { const [a, b] = ends[s], n = segs[s].length; return { x: a.x + (b.x - a.x) * k, y: a.y, i: BASE[s] + Math.min(n - 1, Math.floor(k * n)), d: s }; }
    if (type === "hop") { const a = ends[s][1], b = ends[s + 1][0], e = k * k * (3 - 2 * k); return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * e, i: -1, d: -1, hop: true }; }
    const a = ends[2][1]; return { x: a.x, y: a.y, i: -1, d: -1, end: true };
  };
  let words = -1, flowing = false;
  const aLi = a1.parentNode;
  // the answer streams into the bubble word by word (the cursor takes no room)
  a1.textContent = ""; a1.innerHTML = '<span class="vis"></span><i class="cur" aria-hidden="true"></i><span class="ghost"></span>';
  const aVis = a1.firstChild, aGhost = a1.lastChild;
  // the bubble grows a line at a time as words are added (words are only appended, so earlier lines never
  // re-wrap); the chat follows the line being written. (Reserving the final size left an empty, clipped box.)
  const sayTo = n => { aVis.textContent = WORDS.slice(0, n).join(" "); aGhost.textContent = ""; };
  const flow = t => {
    if (t < C && demo.dataset.scene === "chat" && !aLi.classList.contains("pending")) track();
    aLi.classList.toggle("streaming", t >= A0 && t < A1);
    if ((t >= A1) !== aLi.classList.contains("done")) { aLi.classList.toggle("done", t >= A1); toBottom(); }
    const wait = t < WT[1];   // from the moment the answer's place shows until its first word: the working line
    if (wait !== aLi.classList.contains("wait")) { aLi.classList.toggle("wait", wait); follow(); }
    if (t < A0 || t >= A1) {
      if (flowing) { flowing = false; band.classList.remove("writing"); liveWord(-1); sweep(null); a1.classList.remove("cursor"); }
      const n = t < A0 ? 0 : WORDS.length;
      if (n !== words) { words = n; sayTo(n); bdTok.textContent = n ? WORDS.length : 0; }
      return;
    }
    if (!flowing) { flowing = true; band.classList.add("writing"); }
    liveWord(t - A0);
    let w = 0; while (w < WORDS.length - 1 && t >= WT[w + 1]) w++;
    if (w !== words) { words = w; sayTo(w); a1.classList.add("cursor"); bdTok.textContent = w + 1; follow(); }
    // the room's sweep: one per word while words are slow, then one per lap that the faster words ride along with
    let s0 = SWEEPS[0]; for (const x of SWEEPS) { if (x <= t) s0 = x; else break; }
    sweep(t - s0);
  };

  // Code: while the agent streams (a file, a line of its answer) a token passes through every layer, pass after pass,
  // the same glow as in the chat (the room's band sweeps on every pass in Code too)
  const PASS = 2 * SW + .15;
  let coding = false;
  const codeFlow = (t, on) => {
    if (!on) { if (coding) { coding = false; sweep(null); } return; }
    coding = true;
    bdTok.textContent = WORDS.length + Math.floor((t - C) / .12);
    sweep((t - C) % PASS);
  };

  /* ---------- chat ---------- */
  const msgs = $("msgs"), chatTyped = $("chatTyped"), chatComposer = $("chatComposer");
  msgs.addEventListener("scroll", () => msgs.classList.toggle("scrolled", msgs.scrollTop > 0), { passive: true });
  const Q1 = "what is Pooled?", q1El = demo.querySelector('[data-at="q1"]');
  // the chat follows its last line, gliding (a jump when seeking, or with reduced motion)
  const glide = top => { top = Math.max(0, Math.min(msgs.scrollHeight - msgs.clientHeight, top)); if (Math.abs(top - msgs.scrollTop) < 1) return; if (RM || frozen) msgs.scrollTop = top; else msgs.scrollTo({ top, behavior: "smooth" }); };
  // scroll down just enough to show an element's bottom (never up)
  const showBottom = (el, pad) => { const need = el.getBoundingClientRect().bottom + pad - msgs.getBoundingClientRect().bottom; if (need >= 1) glide(msgs.scrollTop + need); };
  const toBottom = () => showBottom(aLi, 6);
  // the answer's place keeps the finished answer's height from the start (an invisible reserve; the bubble itself
  // grows a line at a time), so nothing below it moves and the numbers line appears in place
  const reserve = () => {
    aLi.style.minHeight = "";
    const cls = aLi.className, vis = aVis.textContent;
    aLi.classList.remove("wait"); aVis.textContent = ""; aGhost.textContent = WORDS.join(" ");
    const h = aLi.getBoundingClientRect().height;
    aLi.className = cls; aVis.textContent = vis; aGhost.textContent = "";
    aLi.style.minHeight = h + "px";
  };
  // every frame the chat eases down toward the line being written (the working line, then the cursor, then the
  // numbers): it goes down as the answer is written, never ahead of it and never up, and never in jumps
  const track = () => {
    const c = a1.querySelector(".cur");
    const ref = aLi.classList.contains("wait") ? aLi.querySelector(".wkb") : aLi.classList.contains("done") ? aLi : c && c.getClientRects().length ? c : aVis;
    if (!ref) return;
    const need = ref.getBoundingClientRect().bottom + (ref === aLi ? 6 : 18) - msgs.getBoundingClientRect().bottom;
    if (need < .5) return;
    const max = msgs.scrollHeight - msgs.clientHeight;
    msgs.scrollTop = Math.min(max, msgs.scrollTop + (RM || frozen ? need : Math.max(.5, need * .22)));
  };
  const follow = track;
  const typeInto = (el, text, t0, t1, t) => {
    const n = Math.max(0, Math.min(text.length, Math.ceil((t - t0) / (t1 - t0) * text.length)));
    if (el.textContent.length !== n) el.textContent = text.slice(0, n);
  };
  const show = k => {
    const el = demo.querySelector(`[data-at="${k}"]`); if (!el) return null;
    el.classList.remove("pending"); if (!RM) restart(el, "enter"); return el;
  };

  /* ---------- 6 to 8: Code ---------- */
  const log = $("log"), codeTyped = $("codeTyped"), codeComposer = $("codeComposer");
  const files = demo.querySelector(".files"), ftree = $("ftree");
  const fItems = Object.fromEntries([...ftree.children].map(li => [li.dataset.f, li]));
  const lv = $("lv"), lvPre = $("lvPre"), lvF = $("lvF"), lvN = $("lvN"), lvNm = lv.querySelector(".nm"), pvRev = $("pvRev");
  const app = $("app"), brLoad = $("brLoad"), game = $("game"), canvas = $("appc");
  const at = k => demo.querySelector(`[data-at="${k}"]`);
  const saysText = new Map();
  let APP = "tetris", A = APPS.tetris, inst = null, lineOut = {};
  // what a real app of each kind runs to; the live pane shows a few of those lines, the counts are the real size
  const LINES = { tetris: 213, "2048": 148, shooter: 236, snake: 122, breakout: 184 };
  const useApp = name => {
    APP = name; A = APPS[name];
    $("fDir").textContent = A.dir;
    lineOut = { "index.html": 24, "style.css": 60, "game.js": LINES[name] };
    const add = A.diff.filter(d => d[0] === "add").length, del = A.diff.length - add;
    lineOut.edit = lineOut["game.js"] + add - del;
    at("c-q").querySelector("p").textContent = A.prompt;
    at("c-q2").querySelector("p").textContent = A.prompt2;
    saysText.set(at("c-s1"), "I'll write three files and serve them on :5173.");
    saysText.set(at("c-s2"), A.done); saysText.set(at("c-s3"), A.done2);
    const t4 = at("c-t4");
    t4.querySelector(".add").textContent = "+" + add; t4.querySelector(".del").textContent = "-" + del; t4.querySelector(".del").hidden = !del;
    // each write_file card shows the new file's first lines, as the room's diff does
    ["c-t0", "c-t1", "c-t2"].forEach(k => {
      const card = at(k), f = card.querySelector(".br").textContent, rs = card.querySelector(".rs");
      card.querySelector(".nl").textContent = lineOut[f]; rs.textContent = "";
      A.files[f].filter(Boolean).slice(0, 3).forEach(txt => { const sp = document.createElement("span"); sp.className = "r add"; sp.innerHTML = HL(f, txt); rs.append(sp); });
    });
    const diff = $("diff"); diff.textContent = "";
    A.diff.forEach(([k, txt]) => { const sp = document.createElement("span"); sp.className = "r " + k; sp.innerHTML = HL("game.js", txt); diff.append(sp); });
    game.setAttribute("aria-label", A.aria);
    CAPS[5] = `Switch to Code and ask for ${A.ask}`; labelDots();
    if (inst) inst.destroy();
    inst = window.PooledApps.make(name, canvas, { seed: 11 });
  };
  const logBottom = () => { log.scrollTop = log.scrollHeight; };
  let liveSrc = [], liveShown = -1, writing = null;
  const liveTo = k => {
    if (k === liveShown) return;
    if (k < liveShown || liveShown < 0) { lvPre.textContent = ""; liveShown = 0; }
    for (let i = liveShown; i < k; i++) { const sp = document.createElement("span"); sp.innerHTML = liveSrc[i] ? HL(lvF.textContent, liveSrc[i]) : " "; lvPre.append(sp); }
    while (lvPre.children.length > 6) lvPre.firstChild.remove();
    liveShown = k;
    const n = writing && liveSrc.length ? Math.round(writing[5] * k / liveSrc.length) : k;
    lvN.textContent = `${n} line${n === 1 ? "" : "s"}`;
  };
  const fileState = (name, st, lines) => {
    const li = fItems[name];
    if (st === "hide") { li.classList.add("pending"); li.classList.remove("w", "mod"); return; }
    if (li.classList.contains("pending")) { li.classList.remove("pending"); if (!RM) restart(li, "new"); }
    li.classList.toggle("w", st === "w"); if (st === "mod") li.classList.add("mod");
    if (lines != null) li.querySelector(".fm").textContent = lines;
    files.classList.toggle("none", ftree.querySelectorAll("li:not(.pending)").length === 0);
  };
  const liveStart = (name, key, t0, t1) => {
    liveSrc = key === "edit" ? A.diff.filter(d => d[0] === "add").map(d => d[1]) : A.files[key];
    lvF.textContent = name; lvNm.textContent = key === "edit" ? "editing" : "writing"; liveShown = -1; liveTo(0);
    lvPre.style.setProperty("--lvn", Math.min(6, liveSrc.length));
    lv.classList.remove("pending"); if (!RM) restart(lv, "enter");
    writing = [t0, t1, liveSrc.length, name, key, key === "edit" ? liveSrc.length : lineOut[key]];
    fileState(name, "w", key === "edit" ? null : 0);
    lv.parentNode.append(lv); logBottom();
  };
  const liveEnd = () => { const w = writing; lv.classList.add("pending"); writing = null; if (w) fileState(w[3], "done", lineOut[w[4]]); };
  const tool = k => { const el = show(k); if (el) { log.append(el); logBottom(); } return el; };
  let streams = [];
  const reveal = (el, rate) => streams.push({ el: el.querySelector(".say"), text: saysText.get(el), t0: tl.t, rate });
  const setRun = (k, run) => { const st = at(k).querySelector(".st"); st.textContent = run ? "running" : "done"; st.classList.toggle("run", run); };
  const openPreview = () => { flag("served", true); flag("app-on", true); };
  const reload = () => { if (!RM) restart(brLoad, "go"); };
  const WARM_S = { tetris: 16, "2048": 40, shooter: 7, snake: 22, breakout: 9 };
  const runGame = (warm, v2) => { if (!inst.human) { inst.auto(); inst.reset(11); inst.v2(v2); inst.warm(WARM_S[APP] * warm); } else inst.v2(v2); if (!RM) inst.start(); else inst.draw(); };

  const tl = { t: 0, fired: 0, done: false, started: false };
  const EVENTS = () => [
    // 1: the laptop has a name already; it starts a room, and is the room's first device
    [.2, () => nmA.classList.add("fresh")],
    [.9, () => nmA.classList.remove("fresh")],
    [1.1, () => press(aGo, 260)],
    [1.3, () => { tabA.classList.add("done"); setDevices(1, true); }],
    ...[0, 1, 2, 3].map(i => [1.4 + i * .1, () => letters(i + 1)]),
    // 2: the desktop (named too) wakes, types the code and joins; the two tabs fold into one room
    [S1, () => tabB.classList.remove("idle")],
    [S1 + .1, () => nmB.classList.add("fresh")],
    [S1 + .55, () => nmB.classList.remove("fresh")],
    [S1 + .6, () => setSlots(0, 0)],
    [S1 + .75, () => setSlots(1, 1)], [S1 + .9, () => setSlots(2, 2)], [S1 + 1.05, () => setSlots(3, 3)],
    [S1 + 1.2, () => { setSlots(4, -1); bBtn.classList.add("ready"); }],
    [S1 + 1.45, () => press(bBtn, 260)],
    [S1 + 1.65, () => { tabB.classList.add("done"); tabA.classList.add("met"); setDevices(2, true); }],
    [S1 + 2.1, () => flag("merge", true)],
    // 3: now they are in, each lends memory: this laptop turns its amount up, the desktop sends its share
    [S2, () => { scene("pool"); card.dataset.face = "pool"; }],
    ...Array.from({ length: LEND[0] }, (_, i) => [LEND0 + i * LENDI, () => { press(lendPlus, 100); lend(0, i + 1, i === LEND[0] - 1); }]),
    [CALM, () => lend(1, LEND[1], true)],
    // a friend's phone joins (its chip, its row), then adds its share: now the 35B model fits
    [JOIN3, () => { joining = 2; setDevices(3, true); paintPool(false); }],
    [LEND3, () => lend(2, LEND[2], true)],
    // 4: pick the model that now fits; start it; the layers are dealt by memory
    [S3, () => { card.dataset.face = "pick"; }],
    [HOV1, () => rungHover(1)], [HOV2, () => rungHover(2)],
    [SEL, () => { rungHover(-1); rungSelect(2); }],
    [STH, () => startBtn.classList.add("hover")],
    [STP, () => { startBtn.classList.remove("hover"); press(startBtn, 240); }],
    [DL, () => { scene("split"); model.classList.add("dl"); }],
    [DEAL, () => rowsEl.classList.add("dealt")],
    // 5: chat
    [CH, () => scene("chat")],
    [CH + .3, () => { geo = null; chatComposer.classList.add("hot"); }],
    [CH + .95, () => { chatTyped.textContent = ""; chatComposer.classList.remove("hot"); show("q1"); show("a1"); sayTo(0); reserve(); measure(); }],
    // the answer ends on "It can chat, or write code.": a second later the Code tab is pressed, as if clicked, and the story goes on there
    [C - .3, () => press(mCode, 300)],
    // 6: Code
    [C, () => mode("code")],
    [c(.2), () => scene("code")],
    [c(.55), () => codeComposer.classList.add("hot")],
    [c(ASK), () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); codeComposer.classList.add("run"); tool("c-q"); }],
    [c(ASK + .2), () => reveal(tool("c-s1"), .035)],
    // 7: it writes three files and serves them; the preview opens on the running app
    [c(FILES), () => liveStart("index.html", "index.html", c(FILES), c(FILES + .35))],
    [c(FILES + .4), () => { liveEnd(); tool("c-t0"); }],
    [c(FILES + .5), () => liveStart("style.css", "style.css", c(FILES + .5), c(FILES + .8))],
    [c(FILES + .85), () => { liveEnd(); tool("c-t1"); }],
    [c(FILES + .95), () => liveStart("game.js", "game.js", c(FILES + .95), c(FILES + 2.2))],
    [c(FILES + 2.3), () => { liveEnd(); tool("c-t2"); }],
    [c(FILES + 2.4), () => { setRun("c-t3", true); tool("c-t3"); }],
    [c(SERVED), () => { setRun("c-t3", false); app.classList.remove("blank"); openPreview(); reload(); runGame(.6, false); }],
    [c(SERVED + .15), () => reveal(tool("c-s2"), .035)],
    [c(SERVED + .6), () => { tool("c-st2"); codeComposer.classList.remove("run"); }],
    // 8: a change, an edit, a reload: the app, with the change, for a moment, and an offer to play it.
    //    (on a phone the preview covers the agent, so it steps aside while the change is asked for and made)
    [c(CHANGE), () => { flag("app-on", false); codeComposer.classList.add("hot"); }],
    [c(CHANGE + 1.25), () => { codeTyped.textContent = ""; codeComposer.classList.remove("hot"); codeComposer.classList.add("run"); tool("c-q2"); }],
    // the served app frosts over while the agent edits it, as the room's preview does
    [c(CHANGE + 1.4), () => { liveStart("game.js", "edit", c(CHANGE + 1.4), c(CHANGE + 1.8)); flag("editing", true); }],
    [c(CHANGE + 1.9), () => { liveEnd(); fileState("game.js", "mod"); tool("c-t4"); }],
    // the reload happens under the frost; the frost lifts as the changed app comes up (no dark flash between)
    [c(RELOAD), () => { pvRev.textContent = "rev 2"; app.classList.add("blank"); openPreview(); reload(); }],
    [GAME, () => { flag("editing", false); app.classList.remove("blank"); runGame(.6, true); ask(true); }],
    [GAME + .25, () => reveal(tool("c-s3"), .035)],
    [GAME + .9, () => { tool("c-st3"); codeComposer.classList.remove("run"); }],
    [GAME + 1.5 * SPEED, () => ask(false)],
  ].sort((a, b) => a[0] - b[0]);
  let EV = EVENTS();

  const toastB = $("toastB"), toastC = $("toastC"), NARROW = matchMedia("(max-width: 640px)");
  // a phone shows one device at a time: "heron joined" waits until the two screens have folded into the room
  const frame = t => {
    // "heron joined", "lynx joined": the room's toast, for a few seconds after each device comes in
    [[toastB, NARROW.matches ? S1 + 2.1 : S1 + 1.65], [toastC, JOIN3]].forEach(([el, t0]) => {
      const st = t >= t0 && t < t0 + 3.8 ? "on" : t >= t0 + 3.8 && t < t0 + 4.2 ? "out" : "";
      if (el.dataset.st !== st) { el.dataset.st = st; el.classList.remove("on", "out"); if (st) el.classList.add(st); }
    });   // Invite shows only while devices are joining (steps 2 and 3)
    if (t >= DL) loadAt(t);
    if (t > CH + .3 && t < CH + .95) typeInto(chatTyped, Q1, CH + .35, CH + .8, t);
    if (t >= CH) flow(t);
    // Code: the room's band says Writing while the agent writes
    if (t >= C) { const on = !!writing || streams.length > 0; if (on !== band.classList.contains("writing")) { band.classList.toggle("writing", on); if (!on) liveWord(-1); } if (on) liveWord(t - C); codeFlow(t, on); }
    if (t > c(.55) && t < c(ASK)) typeInto(codeTyped, A.prompt, c(.65), c(1.85), t);
    if (t > c(CHANGE) && t < c(CHANGE + 1.25)) typeInto(codeTyped, A.prompt2, c(CHANGE + .1), c(CHANGE + 1.1), t);
    if (writing) {
      const [a, b, n] = writing;
      liveTo(Math.max(0, Math.min(n, Math.ceil((t - a) / (b - a) * n))));
      fileState(writing[3], "w", writing[4] === "edit" ? null : Math.round(writing[5] * liveShown / Math.max(1, writing[2])));
      logBottom();
    }
    streams = streams.filter(s => {
      const w = s.text.split(" "), n = Math.min(w.length, Math.floor((t - s.t0) / s.rate) + 1);
      s.el.textContent = w.slice(0, n).join(" "); s.el.classList.toggle("cursor", n < w.length);
      logBottom();
      return n < w.length;
    });
  };

  const LOGKEYS = ["c-q", "c-s1", "c-t0", "c-t1", "c-t2", "c-t3", "c-s2", "c-st2", "c-q2", "c-t4", "c-s3", "c-st3"];
  tl.reset = () => {
    tl.t = 0; tl.fired = 0; tl.done = false; streams = []; writing = null;
    scene("tabs"); mode("chat"); ["merge", "served", "app-on", "ask", "done", "editing"].forEach(k => flag(k, false));
    codeComposer.classList.remove("run"); pvRev.textContent = "rev 1";
    tabA.classList.remove("done", "met"); tabB.classList.remove("done"); tabB.classList.add("idle"); letters(0);
    nmA.classList.remove("fresh"); nmB.classList.remove("fresh");
    setSlots(0, -1); bBtn.classList.remove("ready", "press"); aGo.classList.remove("press");
    setDevices(0); chips.forEach(ch => ch.classList.remove("in")); devRows.forEach(li => li.classList.remove("in"));
    lent = [0, 0, 0]; joining = -1; card.dataset.face = "pool";
    rungHover(-1); rungSelect(-1); rungs.forEach(li => li.classList.remove("unlock")); startBtn.classList.remove("hover", "press");
    model.classList.remove("dl"); rowsEl.classList.remove("dealt"); filled = [-1, -1, -1]; mdKey = ""; loadAt(0); geo = null;
    words = -1; flowing = true; coding = false; flow(0);
    demo.querySelectorAll("[data-at]").forEach(el => el.classList.add("pending"));
    chatTyped.textContent = ""; chatComposer.classList.remove("hot");
    codeTyped.textContent = ""; codeComposer.classList.remove("hot");
    Object.keys(fItems).forEach(n => fileState(n, "hide")); files.classList.add("none");
    lv.classList.add("pending"); liveShown = -1; liveSrc = []; liveTo(0);
    saysText.forEach((txt, el) => { const s = el.querySelector(".say"); if (s) { s.textContent = txt; s.classList.remove("cursor"); } });
    LOGKEYS.forEach(k => log.append(at(k)));
    app.classList.add("blank"); brLoad.classList.remove("go");
    if (!inst.human) { inst.stop(); inst.v2(false); }
    aLi.style.minHeight = ""; msgs.scrollTop = 0; log.scrollTop = 0;
    paintBar(0, true);
  };
  tl.final = () => {
    streams = []; writing = null; tl.done = true; tl.t = END; tl.fired = EV.length;
    scene("code"); mode("code"); flag("merge", true); flag("served", true); flag("app-on", true); flag("ask", false); flag("done", true);
    flag("editing", false); codeComposer.classList.remove("run"); pvRev.textContent = "rev 2";
    tabA.classList.add("done", "met"); tabB.classList.add("done"); tabB.classList.remove("idle"); letters(4); setSlots(4, -1);
    setDevices(3); joining = -1; lent = LEND.slice(); rungHover(-1); rungSelect(2); card.dataset.face = "pick";
    model.classList.add("dl"); rowsEl.classList.add("dealt"); loadAt(FILL1 + 1);
    words = -1; flowing = true; coding = false; flow(END);
    bdTok.textContent = WORDS.length + Math.floor((GAME + .6 - C) / .12);   // the Code run's tokens too, about where the live run ends (flow() counts the chat's only)
    demo.querySelectorAll("[data-at]").forEach(el => el.classList.remove("pending", "enter"));
    LOGKEYS.forEach(k => log.append(at(k)));   // in story order: a partly played run has moved the ones it showed to the end
    saysText.forEach((txt, el) => { const s = el.querySelector(".say"); if (s) { s.textContent = txt; s.classList.remove("cursor"); } });
    chatTyped.textContent = ""; codeTyped.textContent = ""; chatComposer.classList.remove("hot"); codeComposer.classList.remove("hot");
    Object.keys(fItems).forEach(n => fileState(n, "done", lineOut[n])); fileState("game.js", "mod", lineOut.edit);
    lv.classList.add("pending"); sweep(null);
    app.classList.remove("blank"); runGame(1, true);
    toBottom(); logBottom();
    paintBar(END);
  };
  tl.advance = dt => {
    tl.t += dt;
    while (tl.fired < EV.length && EV[tl.fired][0] <= tl.t) { EV[tl.fired][1](); tl.fired++; }
    frame(tl.t);
    paintBar(Math.min(tl.t, END), true);
    if (tl.t >= END && tl.fired >= EV.length) nextLoop();
  };
  // a new loop: new names, the next app. Tetris comes last in the round (it is what the demo video builds)
  const nextLoop = () => {
    newNames(); EV = EVENTS();
    useApp(ORDER[(ORDER.indexOf(APP) + 1) % ORDER.length]);
    tl.reset();
  };

  /* ---------- driver: one rAF, only while the window is on screen, the page is visible and nobody is playing ---------- */
  let raf = 0, last = 0, visible = false, frozen = false, playing = false;
  const needs = () => !frozen && !playing && visible && !document.hidden && tl.started && !tl.done;
  function loop(now) {
    if (sbT.getAttribute("aria-live") !== "off") sbT.setAttribute("aria-live", "off");
    const dt = last ? Math.min(.1, (now - last) / 1000) : .016; last = now;
    tl.advance(dt * rate(tl.t));
    raf = needs() ? requestAnimationFrame(loop) : 0;
  }
  const wake = () => { if (needs() && !raf) { last = 0; raf = requestAnimationFrame(loop); } };
  const halt = () => { if (raf) { cancelAnimationFrame(raf); raf = 0; } };
  const begin = () => {
    if (tl.started) { wake(); return; }
    tl.started = true;
    if (RM) { tl.final(); return; }
    wake();
  };

  // jump to a moment; with reduced motion, to a step's finished state, frozen
  const seek = (s, freeze) => {
    stopPlay(true);
    halt(); tl.started = true; frozen = !!freeze;
    tl.reset();
    while (tl.t < s - 1e-6) tl.advance(Math.min(1 / 30, s - tl.t));
    inst.draw(); wake();
  };
  // the caption is announced when the visitor moves to a step, not on every step the autoplay reaches
  const goStep = k => {
    sbT.setAttribute("aria-live", "polite");
    if (RM) {
      if (k >= STEPS.length - 1) { stopPlay(true); tl.final(); return; }
      // step 2 ends before the tabs merge (S1 + 2.1): after that the window is blank until the pool scene
      const ends = [S1 - .3, S1 + 2.0, S3 - .3, CH - .3, A1 + .5, c(FILES - .1), c(CHANGE - .1)];
      seek(ends[k], true); tl.done = true; flow(A1 + 1); paintBar(STEPS[k]); return;
    }
    seek(STEPS[k]);
  };
  dotBtns.forEach((b, i) => b.addEventListener("click", () => goStep(i)));
  $("replay").addEventListener("click", () => goStep(0));
  mChat.addEventListener("click", () => { if (demo.dataset.mode !== "chat" || tl.done) goStep(0); });
  mCode.addEventListener("click", () => { if (demo.dataset.mode !== "code" || tl.done) goStep(5); });
  modes.addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const b = e.key === "ArrowLeft" ? mChat : mCode; b.focus(); b.click();
  });

  /* ---------- the app: it plays itself; for a moment (or once the story is over) the visitor can take it over.
     Then the story waits, the keys (or, on a touch screen, the pad) play it, and leaving it lets the story go on ---------- */
  const pvBack = $("pvBack"), pad = $("pad"), pv = $("pv");
  const COARSE = matchMedia("(pointer: coarse)");
  let lastPointer = "";
  const touchy = () => lastPointer ? lastPointer !== "mouse" : COARSE.matches;
  const ask = on => flag("ask", on);   // (no "Click to play" offer any more; clicking the game still plays it)
  const canPlay = () => demo.classList.contains("served") && demo.classList.contains("app-on") && (demo.classList.contains("ask") || tl.done);
  const startPlay = () => {
    if (playing || !canPlay()) return false;
    playing = true; halt();
    flag("playing", true); flag("touch", touchy()); flag("ask", false);
    inst.play(); inst.v2(true); inst.start();
    game.focus({ preventScroll: true });
    return true;
  };
  function stopPlay(quiet) {
    if (!playing) return;
    playing = false; flag("playing", false); flag("touch", false);
    pad.querySelectorAll(".on").forEach(b => b.classList.remove("on"));
    inst.auto(); inst.reset(11); inst.v2(true); inst.warm(WARM_S[APP]);
    if (!RM) inst.start();
    if (!quiet && document.activeElement === game) game.blur();
    wake();
  }
  pvBack.addEventListener("click", () => stopPlay());
  game.addEventListener("pointerdown", e => { lastPointer = e.pointerType; if (!playing) startPlay(); });
  game.addEventListener("keydown", e => {
    if (e.key === "Escape") { stopPlay(); return; }
    if (!playing && !startPlay()) return;
    if (inst.key(e, true)) e.preventDefault();
  });
  game.addEventListener("keyup", e => { if (playing) inst.key(e, false); });
  game.addEventListener("pointermove", e => { if (!playing) return; const r = game.getBoundingClientRect(); inst.pointer((e.clientX - r.left) / r.width); });
  // leaving it: a tap or click anywhere outside the preview
  document.addEventListener("pointerdown", e => { if (playing && !pv.contains(e.target)) stopPlay(); });
  // the pad: each button holds its key down while pressed
  pad.querySelectorAll("button").forEach(b => {
    const k = { key: b.dataset.key };
    const up = () => { if (!b.classList.contains("on")) return; b.classList.remove("on"); inst.key(k, false); };
    b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture?.(e.pointerId); b.classList.add("on"); inst.key(k, true); });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach(ev => b.addEventListener(ev, up));
    b.addEventListener("contextmenu", e => e.preventDefault());
  });

  /* ---------- the hero and the whole demo share the first screen: the stage takes the height that is left ---------- */
  const bodyEl = $("body"), stepbar = $("stepbar");
  let fitW = 0, fitH = 0;
  const fit = force => {
    const w = innerWidth, h = innerHeight;
    // phones: the browser's bars come and go as the page scrolls; only a real change of size refits
    if (!force && w === fitW && (Math.abs(h - fitH) < 1 || (COARSE.matches && Math.abs(h - fitH) < 120))) return;
    fitW = w; fitH = h;
    const b = bodyEl.getBoundingClientRect(), s = stepbar.getBoundingClientRect();
    const top = b.top + scrollY, below = s.bottom - b.bottom, phone = w <= 640;
    const px = Math.round(Math.min(phone ? 520 : 580, Math.max(300, h - top - below - (phone ? 10 : 16))));
    document.documentElement.style.setProperty("--body-h", px + "px");
    geo = null;
  };
  // the load card keeps to the stage: on a short one it scales down whole, so its title and status line never clip
  const ldc = $("ldc");
  const fitLdc = () => {
    const h = ldc.offsetHeight, room = bodyEl.clientHeight - 16;
    ldc.style.setProperty("--ldc-k", h > room && room > 0 ? (room / h).toFixed(3) : "1");
  };
  if (window.ResizeObserver) { const ro = new ResizeObserver(fitLdc); ro.observe(ldc); ro.observe(bodyEl); }
  fit(true);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => fit(true));
  addEventListener("resize", () => { fit(); geo = null; if (demo.dataset.scene === "chat") { measure(); if (aLi.style.minHeight) reserve(); } });

  /* ---------- start when 30% visible ---------- */
  new IntersectionObserver(es => {
    visible = es[es.length - 1].isIntersecting;
    if (visible) begin();
    wake();
  }, { threshold: .3 }).observe(win);
  document.addEventListener("visibilitychange", wake);

  useApp(ORDER[0]); labelDots();
  if (RM) tl.final(); else tl.reset();
  demo.dataset.ready = "1";   // the static page shows the finished Code scene; with JS it starts at Chat, step 1

  // tests and screenshots: jump to a moment, pick an app
  window.__demo = {
    seek(s, freeze) { seek(s, freeze); },
    step: goStep,
    setApp(n) { useApp(n); seek(0); },
    play: () => startPlay(), stop: () => stopPlay(),
    set frozen(v) { frozen = v; wake(); }, get t() { return tl.t; }, get app() { return APP; }, get playing() { return playing; },
    get STEPS() { return STEPS; }, get END() { return END; }, get C() { return C; }, get A1() { return A1; }, get GAME() { return GAME; },
    T: { S1, S2, LEND0, CALM, JOIN3, LEND3, S3, SEL, DL, DEAL, FILL0, FILL1, CH, A0, ASK, FILES, SERVED, CHANGE, RELOAD, SHOWN }, APPS: Object.keys(APPS),
    // real seconds between two moments of the timeline, at the speeds it plays at
    real(a, b) { let s = 0; for (let t = a; t < b; t += .01) s += Math.min(.01, b - t) / rate(t); return s; }
  };
})();
