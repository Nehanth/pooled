// The terminal banner of pooled serve and what it prints as the room changes (bin/pooled.js).
// The agent settings depend on the room's context, which only counts once the model is ready: a host
// that has not started the model yet can still carry an old or default ctx in its hello.
import { agentSettings } from "./http.js";

// the room's context in tokens, when the host said it (a v2 host: hello meta / ai-ready-all)
export const roomCtx = (bridge) => { const c = +bridge.hostMeta?.ctx; return Number.isInteger(c) && c > 0 ? c : null; };
export const roomLabel = (bridge) => (bridge.ready ? `${bridge.modelLabel || bridge.model}${roomCtx(bridge) ? ` · ${roomCtx(bridge)} tokens of context` : ""}` : "model not ready yet");

// the settings block for the room as it is now, or null (model not ready, or no context known)
export function settingsFor(bridge, port) {
  const ctx = bridge.ready ? roomCtx(bridge) : null;
  if (!ctx) return null;
  return agentSettings(ctx, port, { model: bridge.model ? `pooled/${bridge.model}` : null, label: bridge.modelLabel || bridge.model }).join("\n").trimEnd();
}

// print the banner, then follow the room: when the model becomes ready (or its context changes)
// print the settings for it; log when it stops being ready and when the host disconnects us
export function showRoom(bridge, { code, port, token, print, log }) {
  print(`pooled serve · room ${code} · ${roomLabel(bridge)}
  OpenAI     http://127.0.0.1:${port}/v1         (OPENAI_BASE_URL, any API key: chat/completions, responses)
  Anthropic  http://127.0.0.1:${port}            (ANTHROPIC_BASE_URL: messages)
  bound to 127.0.0.1 only · ${token ? "token required" : "no token (set POOLED_TOKEN to require one)"}
  prompts go to the room's host and may be shown to everyone in the room`);
  let shown = settingsFor(bridge, port);
  if (shown) print("\n" + shown);
  let wasReady = bridge.ready;
  const onState = () => {
    if (bridge.ready && !wasReady) log(`the room's model is ready: ${roomLabel(bridge)}`);
    else if (!bridge.ready && wasReady && !bridge.kicked) log("the room's model is not ready (a device left or the host is re-dealing)");
    wasReady = bridge.ready;
    const now = settingsFor(bridge, port);
    if (now && now !== shown) print("\n" + now);
    if (now) shown = now;
    if (bridge.kicked) log(`disconnected by the host: ${bridge.kicked}; requests now get 503`, "error");
  };
  bridge.on("state", onState);
  return () => bridge.off("state", onState);
}
