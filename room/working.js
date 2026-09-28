// The "working" line for the wait before a model's first token, in Chat and in Code: the Pooled
// dots in their loading wave and a verb that changes every two seconds (no clock: the wait is not
// counted out). Presentation only: whoever shows it removes it when output arrives, and its
// timer stops by itself once it is off the page. With reduced motion the dots hold still and the
// words change without the fade.

const VERBS = ["Pondering", "Ruminating", "Noodling", "Tinkering", "Percolating", "Mulling", "Brewing", "Musing",
  "Simmering", "Cogitating", "Puzzling", "Whittling", "Marinating", "Deliberating", "Churning", "Conjuring"];
// the mark's nine dots (site/logo/mark.svg); --rc is the diagonal, for the wave
const MARK = '<svg class="wk-mk" width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">'
  + [[3.4, 3.4, 1.8], [10.2, 3.4, 1.99], [18.5, 3.4, 2.38], [3.4, 10.2, 1.99], [10.2, 10.2, 2.38], [18.5, 10.2, 2.94], [3.4, 18.5, 2.38], [10.2, 18.5, 2.94], [18.5, 18.5, 3.9]]
    .map(([x, y, r], i) => `<circle cx="${x}" cy="${y}" r="${r}" style="--rc:${Math.round(x / 7) + Math.round(y / 7)}"${i === 8 ? ' class="lit"' : ""}/>`).join("")
  + "</svg>";
// the mark alone at another size (the edit overlay over the preview uses it)
export const markSVG = (size = 12) => MARK.replace('width="12" height="12"', `width="${size}" height="${size}"`);
const reduced = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

// a random order of the verbs, never the same one twice in a row
export function verbs(rand = Math.random) {
  let last = null, bag = [];
  return () => {
    if (!bag.length) bag = VERBS.filter((v) => v !== last).sort(() => rand() - 0.5);
    last = bag.pop();
    return last;
  };
}

// a <span class="working">; since: the performance.now() the wait started
export function working({ since = performance.now(), label = "Working" } = {}) {
  const el = document.createElement("span");
  el.className = "working";
  el.innerHTML = `${MARK}<span class="wk-v" aria-hidden="true"></span><span class="sr-only">${label}</span>`;
  const v = el.querySelector(".wk-v"), next = verbs();
  let word = -1, seen = false;
  const born = performance.now();
  const tick = () => {
    if (el.isConnected) seen = true;
    else if (seen || performance.now() - born > 10000) { clearInterval(timer); return; }
    const t = performance.now() - since, w = Math.floor(t / 2000);
    if (w !== word) {
      word = w;
      v.textContent = next() + "…";
      if (w > 0 && !reduced() && v.animate) v.animate([{ opacity: 0, transform: "translateY(3px)" }, { opacity: 1, transform: "none" }], { duration: 260, easing: "cubic-bezier(.2,.7,.2,1)" });
    }
  };
  const timer = setInterval(tick, 250);
  tick();
  return el;
}

// The band's pill while the room is writing: a word that changes every two seconds, like the working
// line's verbs ("Twinkling…", "Weaving…"); "Ready" otherwise. liveWords(el) returns set(on).
export const LIVE = ["Twinkling", "Weaving", "Shimmering", "Humming", "Stitching", "Conjuring", "Scribbling", "Whirring", "Sparkling", "Spinning"];
export function liveWords(el) {
  let timer = 0, k = 0;
  const show = (anim) => {
    el.textContent = LIVE[k % LIVE.length] + "\u2026";
    if (anim && !reduced() && el.animate) el.animate([{ opacity: 0, transform: "translateY(3px)" }, { opacity: 1, transform: "none" }], { duration: 260, easing: "cubic-bezier(.2,.7,.2,1)" });
  };
  return (on) => {
    if (on === !!timer) return;
    if (on) { k = 0; show(false); timer = setInterval(() => { k++; show(true); }, 2000); }
    else { clearInterval(timer); timer = 0; el.textContent = "Ready"; }
  };
}
