// @pooled/openclaw: pick "Pooled" in OpenClaw and this machine becomes a device in a Pooled room. It
// holds its share of the model's layers on the local GPU with Pooled's own engine (WebGPU via Dawn,
// in this process: no browser tab, no `pooled serve`), can host the room, and other devices (another
// machine with this plugin, or a browser tab / phone on pooled.run) join over WebRTC, so together they
// run a model none of them can alone. OpenClaw's requests are answered through a custom StreamFn that
// calls the room directly (no HTTP hop).
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { PROVIDER, AUTH_MARKER, runSetup, setupFromEnv, applyToConfig, providerConfig } from "./src/setup.js";
import { roomSettings, ensureRoom, ensureOnline, current } from "./src/pool.js";

const stateFile = () => path.join(process.env.OPENCLAW_STATE_DIR || path.join(os.homedir(), ".openclaw"), "pooled-room.json");
function writeState(r) { try { fs.writeFileSync(stateFile(), JSON.stringify({ at: new Date().toISOString(), pid: process.pid, ...r.status(), events: r.events.slice(-10) }, null, 1)); } catch {} }

export default definePluginEntry({
  id: PROVIDER,
  name: "Pooled",
  description: "Run a model across your own devices: this machine joins a Pooled room and holds part of the model",
  register(api) {
    if (process.env.POOLED_DEBUG) console.error(`[pooled] register mode=${api.registrationMode} pid=${process.pid}`);
    const log = (m) => { try { api.logger?.info?.(`[pooled] ${m}`); } catch {} if (process.env.POOLED_DEBUG) console.error(`[pooled] ${m}`); const r = current(); r?.ready?.then(writeState, () => {}); };
    const cfgOf = (config) => config?.plugins?.entries?.[PROVIDER]?.config || api.pluginConfig || {};
    let lastConfig = null;
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
      docsPath: "https://pooled.run",
      auth: [{
        id: "room",
        label: "Pooled (run a model across your devices)",
        hint: "start a room on this device, or join one with its code",
        kind: "custom",
        run: (ctx) => runSetup(ctx),
        runNonInteractive: async (ctx) => applyToConfig(ctx.config, setupFromEnv()),
      }],
      catalog: {
        order: "late",
        run: async (ctx) => {
          const provider = ctx.config.models?.providers?.[PROVIDER];
          if (provider) return { provider };
          const s = roomSettings(cfgOf(ctx.config));
          return s.mode ? { provider: providerConfig(s) } : null;
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
    // the room lives with the gateway: open it at startup so other devices can join before the first
    // question, and load the model as soon as the room has the devices and memory it needs
    api.registerService({
      id: "pooled-room",
      start: async (ctx) => {
        if (process.env.POOLED_DEBUG) console.error("[pooled] service start");
        const s = roomSettings(cfgOf(ctx?.config || api.config));
        if (!s.mode || process.env.POOLED_NO_SERVICE) return;
        lastConfig = ctx?.config || api.config || lastConfig;
        const r = await ensureRoom(s, log);
        writeState(r);
        const tick = setInterval(() => writeState(r), 5000); tick.unref?.();
        if (r.node.hosting()) ensureOnline(r, { onWait: () => writeState(r) }).then(() => writeState(r), (err) => { log(`room not online yet: ${err.message}`); writeState(r); });
      },
      stop: async () => { const h = current(); if (h) { const r = await h.ready.catch(() => null); await r?.close(); } },
    });
  },
});
