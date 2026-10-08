/* Copy buttons: any button with [data-copy] puts that text on the clipboard, shows "Copied" for a moment
   and says so to a screen reader (#copy-status). Falls back to a hidden textarea where the async
   clipboard isn't there (an http preview, an older in-app browser). */
(() => {
  "use strict";
  const btns = document.querySelectorAll("[data-copy]");
  if (!btns.length) return;
  const status = document.getElementById("copy-status");
  const legacy = text => {
    const t = document.createElement("textarea");
    t.value = text; t.setAttribute("readonly", ""); t.style.cssText = "position:fixed;top:0;left:0;opacity:0;pointer-events:none";
    document.body.appendChild(t); t.select(); t.setSelectionRange(0, text.length);
    let ok = false;
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    t.remove();
    return ok;
  };
  const copy = text => navigator.clipboard && window.isSecureContext
    ? navigator.clipboard.writeText(text).then(() => true, () => legacy(text))
    : Promise.resolve(legacy(text));
  btns.forEach(b => {
    let timer = 0;
    b.addEventListener("click", () => {
      const text = b.dataset.copy;
      copy(text).then(ok => {
        if (!ok) { if (status) status.textContent = "Couldn't copy. Select the command and copy it by hand."; return; }
        b.classList.add("copied");
        if (status) status.textContent = "Copied to the clipboard";
        clearTimeout(timer);
        timer = setTimeout(() => { b.classList.remove("copied"); if (status) status.textContent = ""; }, 1800);
      });
    });
  });
})();
