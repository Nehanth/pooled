// @pooled/openclaw: pick "Pooled" in OpenClaw and this machine becomes a device in a Pooled room. It
// holds its share of the model's layers on the local GPU with Pooled's own engine (WebGPU via Dawn,
// in this process: no browser tab, no `pooled serve`), can host the room, and other devices (another
// machine with this plugin, `pooled join`, or a browser tab / phone on pooled.run) join over WebRTC,
// so together they run a model none of them can alone. OpenClaw's requests are answered through a
// custom StreamFn that calls the room directly (no HTTP hop). /pooled in the chat shows the room and
// lets devices waiting to join in.
import path from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { PROVIDER, AUTH_MARKER, runSetup, nonInteractive, providerConfig } from "./src/setup.js";
import { roomSettings, ensureRoom, ensureOnline, current } from "./src/pool.js";
import { pooledCommand } from "./src/commands.js";
import { initStateDir, pooledDir, writeJson, joinState } from "./src/state.js";
import { prewarm } from "./src/prewarm.js";

const RETRY_MS = 30000;
// the room as this gateway sees it, for `cat` and support: <state dir>/pooled/status.json (0600: the
// invite link is in it)
function writeState(r) {
  try { writeJson(path.join(pooledDir(), "status.json"), { at: new Date().toISOString(), pid: process.pid, ...r.status(), events: r.events.slice(-10) }); } catch {}
}

export default definePluginEntry({
  id: PROVIDER,
  name: "Pooled",
  description: "Run a model across your own devices: this machine joins a Pooled room and holds part of the model on its GPU.",
  register(api) {
    if (process.env.POOLED_DEBUG) console.error(`[pooled] register mode=${api.registrationMode} pid=${process.pid}`);
    const log = (m) => { try { api.logger?.info?.(`[pooled] ${m}`); } catch {} if (process.env.POOLED_DEBUG) console.error(`[pooled] ${m}`); const r = current(); r?.ready?.then(writeState, () => {}); };
    const cfgOf = (config) => config?.plugins?.entries?.[PROVIDER]?.config || api.pluginConfig || {};
    let lastConfig = null, tick = null;   // tick: the state file's writer while the service runs
    // the StreamFn and OpenClaw's stream helpers, loaded on the first model call
    const loadStream = async () => {
      const [{ createPooledStream }, llm, transport] = await Promise.all([import("./src/stream.js"),
        import("openclaw/plugin-sdk/llm"), import("openclaw/plugin-sdk/provider-transport-runtime")]);
      const sdk = { createAssistantMessageEventStream: llm.createAssistantMessageEventStream,
        createEmptyTransportUsage: transport.createEmptyTransportUsage, failTransportStream: transport.failTransportStream };
      return { createPooledStream: (o) => createPooledStream({ ...o, sdk }) };
    };
    api.registerProvider({
      id: PROVIDER,
      label: "Pooled",
      docsPath: "https://pooled.run/docs/openclaw/",
      auth: [{
        id: "room",
        label: "Pooled (run a model across your devices)",
        hint: "start a room on this device, or join one with its invite link",
        kind: "custom",
        run: (ctx) => runSetup(ctx),
        runNonInteractive: async (ctx) => nonInteractive(ctx.config),
      }],
      catalog: {
        order: "late",
        run: async (ctx) => {
          const provider = ctx.config.models?.providers?.[PROVIDER];
          if (provider) return { provider };
          const s = roomSettings(cfgOf(ctx.config));
          if (!s.mode) return null;
          await initStateDir();
          return { provider: providerConfig(s, s.mode === "join" && s.code ? joinState(s.code) : {}) };
        },
      },
      resolveSyntheticAuth: ({ providerConfig: pc }) => pc ? { apiKey: AUTH_MARKER, source: "Pooled room on this machine", mode: "api-key" } : undefined,
      buildMissingAuthMessage: () => "Run `openclaw onboard` and pick Pooled to start or join a room. No API key is needed.",
      createStreamFn: (ctx) => {
        lastConfig = ctx.config || lastConfig;
        return async (...args) => (await loadStream()).createPooledStream({ getPluginConfig: () => cfgOf(lastConfig), log })(...args);
      },
    });
    if (api.registrationMode !== "full") return;
    api.registerCommand?.(pooledCommand);
    // the room lives with the gateway: open it at startup so other devices can join before the first
    // question, download the model if it is not in the cache yet, load it as soon as the room has the
    // devices and memory it needs, then warm up OpenClaw's system prompt and tools
    let running = false, retry = null;
    api.registerService({
      id: "pooled-room",
      start: async (ctx) => {
        if (process.env.POOLED_DEBUG) console.error("[pooled] service start");
        const s = roomSettings(cfgOf(ctx?.config || api.config));
        if (!s.mode || process.env.POOLED_NO_SERVICE) return;
        await initStateDir();
        lastConfig = ctx?.config || api.config || lastConfig;
        running = true;
        // a room that can't open yet (its host isn't up, the code is still registered after a crash)
        // is tried again every 30 s while the gateway runs; a setup problem is not
        const open = async () => {
          retry = null;
          let r;
          try { r = await ensureRoom(s, log); }
          catch (err) {
            const again = running && err.code !== "setup" && err.code !== "install";
            log(`${err.message}${again ? "; trying again in 30 s" : ""}`);
            if (again) { retry = setTimeout(open, RETRY_MS); retry.unref?.(); }
            return;
          }
          writeState(r);
          clearInterval(tick); tick = setInterval(() => { writeState(r); r.persist?.(); }, 5000); tick.unref?.();
          if (!r.node.hosting()) return;
          ensureOnline(r, { waitPull: true, waitMs: Infinity, onWait: () => writeState(r) })
            .then(() => { writeState(r); if (s.prewarm && !r.asked) return prewarm(r, { log: r.note }).catch((e) => r.note(`warm-up failed: ${e.message}`)); })
            .catch((err) => { log(`room not online yet: ${err.message}`); writeState(r); });
        };
        await open();
      },
      stop: async () => { running = false; clearTimeout(retry); clearInterval(tick); tick = null; const h = current(); if (h) { const r = await h.ready.catch(() => null); await r?.close(); } },
    });
  },
});
