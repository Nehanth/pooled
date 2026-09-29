// Starter templates for a new Code project: a small app that already runs, so the first request is
// a change ("add a high score") instead of a whole build. The small models do much better editing a
// working file than writing one from nothing. Each template is plain HTML, CSS and one JS module,
// every file well under the ~100 lines the agent writes at once (harness/code-prompt.js), no
// network, and nothing the preview sandbox blocks: no <form> submit (the frame has no allow-forms,
// so a submit event never fires; a button's click and Enter do the work), and localStorage is the
// preview's in-memory shim.
//
//   TEMPLATES                      [{ id, label, blurb, next, files: { path: text } }]
//   templateById(id) -> template | null     "" and "blank" are the empty project: null
//   applyTemplate(ws, id) -> [paths]        writes the files into a workspace, [] for blank
//
// tests/unit/templates_test.js builds each one with buildPreviewDoc (no missing files) and parses
// its script.

const page = (title, body, { bg = "#f6f7f9" } = {}) => `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${title}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body style="background: ${bg}">
${body}
  <script type="module" src="app.js"></script>
</body>
</html>
`;

// ---------------------------------------------------------------- game
const GAME_JS = `// Catch the stars: move the basket with the arrow keys, the mouse or a finger.
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height;
const scoreEl = document.getElementById("score"), livesEl = document.getElementById("lives");

let basket = { x: W / 2, w: 70 }, stars = [], score = 0, lives = 3, over = false, keys = {};

function spawn() { stars.push({ x: 15 + Math.random() * (W - 30), y: -10, v: 2 + Math.random() * 2 + score / 20 }); }

function step() {
  if (over) return;
  if (keys.ArrowLeft) basket.x -= 7;
  if (keys.ArrowRight) basket.x += 7;
  basket.x = Math.max(basket.w / 2, Math.min(W - basket.w / 2, basket.x));
  if (Math.random() < 0.03) spawn();
  for (const s of stars) s.y += s.v;
  for (const s of stars.filter((s) => s.y > H - 30)) {
    if (Math.abs(s.x - basket.x) < basket.w / 2) score++; else lives--;
    stars.splice(stars.indexOf(s), 1);
  }
  if (lives <= 0) over = true;
  scoreEl.textContent = score; livesEl.textContent = Math.max(0, lives);
}

function draw() {
  ctx.fillStyle = "#101828"; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#facc15";
  for (const s of stars) { ctx.beginPath(); ctx.arc(s.x, s.y, 8, 0, Math.PI * 2); ctx.fill(); }
  ctx.fillStyle = "#38bdf8"; ctx.fillRect(basket.x - basket.w / 2, H - 22, basket.w, 12);
  if (over) {
    ctx.fillStyle = "#fff"; ctx.font = "bold 28px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Game over", W / 2, H / 2); ctx.font = "16px sans-serif";
    ctx.fillText("press Space or tap to play again", W / 2, H / 2 + 30);
  }
}

function restart() { stars = []; score = 0; lives = 3; over = false; }

document.addEventListener("keydown", (e) => { keys[e.key] = true; if (e.key === " " && over) restart(); });
document.addEventListener("keyup", (e) => { keys[e.key] = false; });
canvas.addEventListener("pointermove", (e) => { basket.x = (e.offsetX / canvas.clientWidth) * W; });
canvas.addEventListener("pointerdown", () => { if (over) restart(); });

function loop() { step(); draw(); requestAnimationFrame(loop); }
loop();
`;
const GAME_CSS = `body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: system-ui, sans-serif; color: #e5e7eb; }
main { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 16px; }
h1 { margin: 0; font-size: 22px; }
.hud { display: flex; gap: 20px; font-size: 15px; }
canvas { width: min(400px, 92vw); aspect-ratio: 400 / 500; border-radius: 12px; touch-action: none; }
`;
const game = {
  id: "game", label: "Game", blurb: "catch the falling stars, on a canvas",
  next: "Add a high score that stays after a reload",
  files: {
    "index.html": page("Catch the stars", `  <main>
    <h1>Catch the stars</h1>
    <div class="hud">Score <b id="score">0</b> Lives <b id="lives">3</b></div>
    <canvas id="game" width="400" height="500"></canvas>
  </main>`, { bg: "#0b1220" }),
    "style.css": GAME_CSS,
    "app.js": GAME_JS,
  },
};

// ---------------------------------------------------------------- dashboard
const DASH_JS = `// A small dashboard: stat cards, a bar chart drawn on a canvas and a table, all from DATA.
const DATA = [
  { month: "Jan", visits: 1200, signups: 80 },
  { month: "Feb", visits: 1500, signups: 95 },
  { month: "Mar", visits: 1350, signups: 90 },
  { month: "Apr", visits: 1800, signups: 130 },
  { month: "May", visits: 2100, signups: 160 },
  { month: "Jun", visits: 2400, signups: 190 },
];

const sum = (key) => DATA.reduce((n, d) => n + d[key], 0);

function cards() {
  const visits = sum("visits"), signups = sum("signups");
  const stats = [["Visits", visits.toLocaleString()], ["Sign-ups", signups.toLocaleString()], ["Conversion", (signups / visits * 100).toFixed(1) + "%"]];
  document.getElementById("cards").innerHTML = stats.map(([k, v]) => \`<div class="card"><span>\${k}</span><b>\${v}</b></div>\`).join("");
}

function chart() {
  const canvas = document.getElementById("chart"), ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 30, max = Math.max(...DATA.map((d) => d.visits));
  const bw = (W - pad * 2) / DATA.length;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "12px system-ui"; ctx.textAlign = "center";
  DATA.forEach((d, i) => {
    const h = (d.visits / max) * (H - pad * 2), x = pad + i * bw + bw * 0.15;
    ctx.fillStyle = "#6366f1"; ctx.fillRect(x, H - pad - h, bw * 0.7, h);
    ctx.fillStyle = "#475569"; ctx.fillText(d.month, x + bw * 0.35, H - pad + 16);
  });
}

function table() {
  document.querySelector("#table tbody").innerHTML = DATA.map((d) => \`<tr><td>\${d.month}</td><td>\${d.visits}</td><td>\${d.signups}</td></tr>\`).join("");
}

cards(); chart(); table();
`;
const DASH_CSS = `body { margin: 0; font-family: system-ui, sans-serif; color: #0f172a; }
main { max-width: 760px; margin: 0 auto; padding: 20px 16px; display: grid; gap: 16px; }
h1 { margin: 0; font-size: 22px; }
#cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.card, .panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; }
.card span { display: block; font-size: 13px; color: #64748b; }
.card b { font-size: 24px; }
canvas { width: 100%; height: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e2e8f0; }
`;
const dashboard = {
  id: "dashboard", label: "Dashboard", blurb: "stat cards, a bar chart and a table",
  next: "Add a line for sign-ups to the chart",
  files: {
    "index.html": page("Dashboard", `  <main>
    <h1>Dashboard</h1>
    <section id="cards"></section>
    <section class="panel"><canvas id="chart" width="700" height="260"></canvas></section>
    <section class="panel">
      <table id="table"><thead><tr><th>Month</th><th>Visits</th><th>Sign-ups</th></tr></thead><tbody></tbody></table>
    </section>
  </main>`),
    "style.css": DASH_CSS,
    "app.js": DASH_JS,
  },
};

// ---------------------------------------------------------------- form app
const FORM_JS = `// A sign-up form: checks the fields, then adds the entry to a list saved in localStorage.
// (No form element: the preview's sandbox blocks a form submit. The button and Enter call add().)
const nameEl = document.getElementById("name"), emailEl = document.getElementById("email"), planEl = document.getElementById("plan");
const list = document.getElementById("list"), error = document.getElementById("error");
let entries = JSON.parse(localStorage.getItem("entries") || "[]");

function render() {
  list.innerHTML = "";
  for (const [i, e] of entries.entries()) {
    const li = document.createElement("li");
    li.textContent = \`\${e.name} <\${e.email}> · \${e.plan}\`;
    const del = document.createElement("button");
    del.textContent = "Remove"; del.type = "button";
    del.onclick = () => { entries.splice(i, 1); save(); };
    li.append(del);
    list.append(li);
  }
  document.getElementById("count").textContent = entries.length;
}

function save() { localStorage.setItem("entries", JSON.stringify(entries)); render(); }

function add() {
  const name = nameEl.value.trim(), email = emailEl.value.trim(), plan = planEl.value;
  if (name.length < 2) { error.textContent = "Please enter your name."; return; }
  if (!/^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$/.test(email)) { error.textContent = "Please enter a valid email."; return; }
  error.textContent = "";
  entries.push({ name, email, plan });
  nameEl.value = emailEl.value = "";
  save();
}

document.getElementById("add").addEventListener("click", add);
for (const el of [nameEl, emailEl]) el.addEventListener("keydown", (e) => { if (e.key === "Enter") add(); });

render();
`;
const FORM_CSS = `body { margin: 0; font-family: system-ui, sans-serif; color: #111827; }
main { max-width: 440px; margin: 0 auto; padding: 24px 16px; display: grid; gap: 16px; }
h1 { margin: 0; font-size: 22px; }
#form { display: grid; gap: 10px; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; }
label { display: grid; gap: 4px; font-size: 14px; }
input, select { font: inherit; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; }
button { font: inherit; padding: 8px 14px; border: 0; border-radius: 8px; background: #16a34a; color: #fff; cursor: pointer; }
#error { margin: 0; min-height: 1.2em; color: #dc2626; font-size: 14px; }
ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
li { display: flex; justify-content: space-between; align-items: center; gap: 8px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 10px; font-size: 14px; }
li button { background: #e5e7eb; color: #111827; padding: 4px 10px; }
`;
const form = {
  id: "form", label: "Form app", blurb: "a sign-up form with checks and a saved list",
  next: "Add a phone number field and a search box over the list",
  files: {
    "index.html": page("Sign up", `  <main>
    <h1>Sign up</h1>
    <div id="form">
      <label>Name <input id="name" autocomplete="name"></label>
      <label>Email <input id="email" type="email" autocomplete="email"></label>
      <label>Plan <select id="plan"><option>Free</option><option>Pro</option><option>Team</option></select></label>
      <p id="error" role="alert"></p>
      <button id="add" type="button">Sign up</button>
    </div>
    <h2>Signed up (<span id="count">0</span>)</h2>
    <ul id="list"></ul>
  </main>`),
    "style.css": FORM_CSS,
    "app.js": FORM_JS,
  },
};

// ---------------------------------------------------------------- landing page
const LANDING_JS = `// A landing page: a menu button on small screens, and a sign-up box that thanks the visitor.
const nav = document.getElementById("nav");
document.getElementById("menu").addEventListener("click", () => nav.classList.toggle("open"));

const email = document.getElementById("email");
function join() {
  const v = email.value.trim();
  document.getElementById("thanks").textContent = v.includes("@") ? \`Thanks! We will write to \${v}.\` : "Please enter your email.";
  email.value = "";
}
document.getElementById("join").addEventListener("click", join);
email.addEventListener("keydown", (e) => { if (e.key === "Enter") join(); });
`;
const LANDING_CSS = `body { margin: 0; font-family: system-ui, sans-serif; color: #1f2937; }
header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: #fff; border-bottom: 1px solid #e5e7eb; }
header b { font-size: 18px; }
#nav { display: flex; gap: 18px; }
#nav a { color: inherit; text-decoration: none; }
#menu { display: none; font: inherit; background: none; border: 1px solid #d1d5db; border-radius: 8px; padding: 4px 10px; }
.hero { text-align: center; padding: 64px 20px 40px; }
.hero h1 { margin: 0 0 12px; font-size: clamp(28px, 6vw, 44px); }
.hero p { margin: 0 auto 24px; max-width: 520px; color: #4b5563; }
#signup { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
#signup input { font: inherit; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 8px; min-width: 220px; }
#signup button { font: inherit; padding: 10px 16px; border: 0; border-radius: 8px; background: #7c3aed; color: #fff; }
.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; max-width: 900px; margin: 0 auto; padding: 20px; }
.features div { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; }
@media (max-width: 600px) {
  #menu { display: block; }
  #nav { display: none; position: absolute; top: 56px; right: 20px; flex-direction: column; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
  #nav.open { display: flex; }
}
`;
const landing = {
  id: "landing", label: "Landing page", blurb: "a hero, features and an email sign-up",
  next: "Add a pricing section with three plans",
  files: {
    "index.html": page("Brightside", `  <header>
    <b>Brightside</b>
    <button id="menu" type="button" aria-label="Menu">Menu</button>
    <nav id="nav"><a href="#features">Features</a><a href="#signup">Sign up</a></nav>
  </header>
  <section class="hero">
    <h1>Plan your week in minutes</h1>
    <p>Brightside turns your to-dos into a calm, simple plan. Free while in beta.</p>
    <div id="signup"><input id="email" type="email" placeholder="you@example.com" aria-label="Email"><button id="join" type="button">Get early access</button></div>
    <p id="thanks" role="status"></p>
  </section>
  <section class="features" id="features">
    <div><h3>Fast</h3><p>Add a task in one line.</p></div>
    <div><h3>Focused</h3><p>See only what matters today.</p></div>
    <div><h3>Shared</h3><p>Plan with friends and family.</p></div>
  </section>`),
    "style.css": LANDING_CSS,
    "app.js": LANDING_JS,
  },
};

export const TEMPLATES = [game, dashboard, form, landing];

export const templateById = (id) => TEMPLATES.find((t) => t.id === id) || null;

// -> the paths written, in order; nothing for blank or an unknown id
export async function applyTemplate(ws, id) {
  const t = templateById(id);
  if (!t) return [];
  const paths = Object.keys(t.files);
  for (const p of paths) await ws.write(p, t.files[p]);
  return paths;
}
