// Toasts: the CSS fade and the JS removal timer agree, so an error toast really stays ~8 s
// (not faded at 4 s while still holding its space) and every toast frees its slot once faded.
const html = await Deno.readTextFile(new URL("../../p2p.html", import.meta.url));
const js = await Deno.readTextFile(new URL("../../room.js", import.meta.url));

// the toastout [duration, delay] in seconds from a `.toast...{ animation: ... }` rule
function fadeOut(selector) {
  const m = html.match(new RegExp("\\n\\s*" + selector.replace(/\./g, "\\.") + "\\s*\\{[^}]*animation:([^;}]*)"));
  if (!m) throw new Error(`no animation rule for ${selector}`);
  const part = m[1].split(",").find((p) => p.includes("toastout"));
  if (!part) throw new Error(`${selector} has no toastout`);
  const times = [...part.matchAll(/([\d.]+)s\b/g)].map((x) => parseFloat(x[1]));
  return { dur: times[0], delay: times[1] ?? 0 };
}
const timers = () => {
  const m = js.match(/setTimeout\(\(\) => t\.remove\(\), kind === "error" \? (\d+) : (\d+)\)/);
  if (!m) throw new Error("toast removal timer not found in room.js");
  return { error: +m[1] / 1000, plain: +m[2] / 1000 };
};

Deno.test("toast: an error toast stays visible about 8 s, not the plain 4 s", () => {
  const plain = fadeOut(".toast"), err = fadeOut(".toast.error");
  if (err.delay < 7) throw new Error(`error toast fades at ${err.delay}s`);
  if (err.delay < plain.delay * 1.8) throw new Error("error toast should stay about twice as long");
});

Deno.test("toast: each toast is removed right after its fade ends (no invisible toast holding a slot)", () => {
  const t = timers();
  for (const [sel, timer] of [[".toast", t.plain], [".toast.error", t.error]]) {
    const f = fadeOut(sel), gone = f.delay + f.dur;
    if (timer < gone) throw new Error(`${sel} removed at ${timer}s before its fade ends at ${gone}s`);
    if (timer - gone > 0.5) throw new Error(`${sel} stays invisible ${timer - gone}s after fading`);
  }
  if (!/animationend[\s\S]{0,80}toastout[\s\S]{0,40}t\.remove\(\)/.test(js)) throw new Error("toast not removed on its toastout animationend");
});
