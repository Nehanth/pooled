/* The Code tab: small stars twinkle inside it while it is chosen or hovered, and hovering turns it black.
   Shared by the landing demo and the room. Nothing moves under reduced motion. */
(function () {
  const STAR = "M8 0C8.6 4.6 11.4 7.4 16 8 11.4 8.6 8.6 11.4 8 16 7.4 11.4 4.6 8.6 0 8 4.6 7.4 7.4 4.6 8 0Z";
  // clicking Code has no splash: the black pill just slides over (callers still call pooledSparkle)
  function sparkle() {}
  window.pooledSparkle = sparkle;

  // [data-twinkle] tabs: small stars that twinkle inside the tab while it is selected (the page's CSS
  // turns the selected Code pill black). Kept to the edges so they never sit on the word.
  const SPOTS = [[9, 28, 5], [17, 72, 4], [30, 18, 3], [74, 20, 4], [86, 66, 5], [91, 30, 3], [66, 80, 3]];
  const css = document.createElement("style");
  css.textContent = ".pooled-tw{position:absolute;inset:0;border-radius:inherit;overflow:hidden;pointer-events:none;opacity:0;transition:opacity .35s ease}"
    + "[data-twinkle][aria-selected=\"true\"]>.pooled-tw{opacity:1}"
    + ".pooled-tw svg{position:absolute;animation:pooledtw 1.9s ease-in-out infinite}"
    + "@keyframes pooledtw{0%,100%{opacity:0;transform:scale(.3) rotate(0)}50%{opacity:1;transform:scale(1) rotate(45deg)}}"
    // hovering the tab before it is chosen: a half-dark wash (white word, stars twinkling), a hint of the black it
    // turns when chosen, so sweeping the pointer across the switch looks natural; a smooth fade, no border or glow
    + "[data-twinkle]{transition:background-color .28s cubic-bezier(.2,.7,.2,1),color .28s cubic-bezier(.2,.7,.2,1)}"
    + ":is(#mode-bar,.modes) [data-twinkle]:not([aria-selected=\"true\"]):hover,[data-twinkle]:not([aria-selected=\"true\"]):hover{background:rgba(11,13,20,.45);color:#fff;box-shadow:none}"
    + "[data-twinkle]:not([aria-selected=\"true\"]):hover>.pooled-tw{opacity:1}"
    + "@media (prefers-reduced-motion:reduce){.pooled-tw svg{animation:none;opacity:.8}}";
  function twinkles(el) {
    if (el.querySelector(".pooled-tw")) return;
    const box = document.createElement("span");
    box.className = "pooled-tw";
    box.setAttribute("aria-hidden", "true");
    SPOTS.forEach(([x, y, s], i) => {
      box.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 16 16" style="left:calc(${x}% - ${s / 2}px);top:calc(${y}% - ${s / 2}px);width:${s}px;height:${s}px;animation-delay:${(i * .27).toFixed(2)}s"><path d="${STAR}" fill="${i % 3 ? "#FFFFFF" : "#A5B4FC"}"/></svg>`);
    });
    el.appendChild(box);
  }
  window.pooledTwinkles = twinkles;   // the room's picker: a disk model's row, picked, twinkles like the Code tab
  const init = () => { document.head.appendChild(css); document.querySelectorAll("[data-twinkle]").forEach(twinkles); };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
