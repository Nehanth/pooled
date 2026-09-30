// Onboarding: the user picked Pooled. Start a room on this device (the code and link to open on the
// other Mac / phone), or join one with its code. The choice is saved in plugins.entries.pooled.config;
// the room itself runs in OpenClaw's gateway (the plugin's service), not in the wizard.
import { MODEL_CHOICES, modelInfo, roomLink, roomSettings, CODE_ABC, CODE_RE } from "./pool.js";
import { maxSeqFor } from "../../../room/models.js";

export const PROVIDER = "pooled";
export const AUTH_MARKER = "pooled-local";
export const newCode = () => Array.from(crypto.getRandomValues(new Uint32Array(4)), (x) => CODE_ABC[x % CODE_ABC.length]).join("");

// the catalog entry: one model, the room's. Native transport: the base URL is never called.
export function providerConfig(s) {
  const info = modelInfo(s.model);
  const id = s.mode === "join" ? "room" : s.model;
  const name = s.mode === "join" ? `Pooled room ${s.code}` : `${info.name} · Pooled room ${s.code}`;
  return {
    baseUrl: "http://127.0.0.1",
    api: "openai-completions",
    authHeader: false,
    timeoutSeconds: 900,
    models: [{
      id, name,
      reasoning: s.model !== "qwen3-1.7b" && s.mode !== "join",
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: s.mode === "join" ? 32768 : s.ctx ? maxSeqFor(s.model, s.ctx) : info.ctx,
      maxTokens: 4096,
      compat: { supportsTools: true, supportsDeveloperRole: false, supportsUsageInStreaming: true },
    }],
  };
}
export const modelRef = (s) => `${PROVIDER}/${s.mode === "join" ? "room" : s.model}`;

function result(s) {
  const ref = modelRef(s);
  return {
    profiles: [],
    defaultModel: ref,
    notes: s.mode === "host"
      ? [`Pooled room ${s.code} starts with OpenClaw's gateway. Open ${roomLink(s.code, s.signal)} on your other devices (or pick Pooled → "Join a room" → ${s.code} in OpenClaw there).`]
      : [`This device joins Pooled room ${s.code} when OpenClaw's gateway starts, and holds layers when the room deals them.`],
    configPatch: {
      models: { providers: { [PROVIDER]: providerConfig(s) } },
      plugins: { entries: { [PROVIDER]: { enabled: true, config: clean(s) } } },
      agents: { defaults: { models: { [ref]: { agentRuntime: { id: "openclaw" } } } } },
      // a small room model (1.7B, 16k context) cannot follow OpenClaw's full tool catalog: give it the
      // file tools only, listed directly (no tool-search meta tools). Bigger models keep the defaults.
      ...(s.model === "qwen3-1.7b" && s.mode === "host" ? { tools: { byProvider: { [PROVIDER]: { allow: ["read", "write", "edit", "ls"] } } } } : {}),
    },
  };
}
const clean = (s) => Object.fromEntries(Object.entries({ mode: s.mode, code: s.code, model: s.model, pledgeGB: s.pledgeGB, minDevices: s.minDevices, signal: s.signal, modelDir: s.modelDir }).filter(([, v]) => v != null && v !== ""));

export async function runSetup(ctx) {
  const p = ctx.prompter;
  const mode = await p.select({
    message: "Pooled runs a model across your own devices. How should this device take part?",
    options: [
      { value: "host", label: "Start a room on this device", hint: "you get a code; your other Mac, PC or phone joins with it" },
      { value: "join", label: "Join a room", hint: "another device already started one and showed you its code" },
    ],
  });
  const prev = roomSettings(ctx.config?.plugins?.entries?.[PROVIDER]?.config || {}, {});
  if (mode === "host") {
    const model = await p.select({
      message: "Which model should the room run?",
      options: MODEL_CHOICES.map((k) => { const v = modelInfo(k); return { value: k, label: v.name, hint: `needs about ${v.needGB} GB of GPU memory across the room · ${Math.round(v.ctx / 1024)}k context` }; }),
      initialValue: prev.model || "qwen3-1.7b",
    });
    const pledge = await p.text({ message: "GPU memory this device gives the room (GB)", initialValue: String(prev.pledgeGB || Math.ceil(modelInfo(model).needGB / 2) + 1),
      validate: (v) => (+v > 0 && +v <= 512 ? undefined : "a number of GB, e.g. 12") });
    const devs = await p.select({ message: "Wait for how many devices before loading the model?", options: [
      { value: 1, label: "Just this one when it has enough memory", hint: "others can still join later as askers" },
      { value: 2, label: "Two (this + one more)" }, { value: 3, label: "Three" }], initialValue: 2 });
    const s = { mode, model, code: prev.mode === "host" && prev.code ? prev.code : newCode(), pledgeGB: +pledge, minDevices: devs, signal: prev.signal, modelDir: prev.modelDir };
    await p.note([
      `Room code: ${s.code}`,
      `Link: ${roomLink(s.code, s.signal)}`,
      "",
      "On the other device, either:",
      `  - open the link in Chrome / Safari (a Mac, a PC, or a phone), set how much memory it gives, and press Join;`,
      `  - or run OpenClaw with this plugin, pick Pooled → "Join a room" and enter ${s.code}.`,
      `The model (${modelInfo(model).name}) loads once ${devs > 1 ? `${devs} devices are in the room and ` : ""}the room has about ${modelInfo(model).needGB} GB.`,
    ].join("\n"), "Pooled room");
    return result(s);
  }
  const code = (await p.text({ message: "Room code (4 letters/digits, from the device that started the room)", placeholder: "ABCD",
    validate: (v) => (CODE_RE.test(String(v).trim().toUpperCase()) ? undefined : "4 to 6 characters, like K7QX") })).trim().toUpperCase();
  const pledge = await p.text({ message: "GPU memory this device gives the room (GB)", initialValue: String(prev.pledgeGB || 8),
    validate: (v) => (+v > 0 && +v <= 512 ? undefined : "a number of GB, e.g. 12") });
  return result({ mode: "join", code, model: prev.model || "qwen3-1.7b", pledgeGB: +pledge, signal: prev.signal, modelDir: prev.modelDir });
}

// `openclaw onboard --non-interactive --auth-choice pooled`: the choice from POOLED_* env
export function setupFromEnv(env = process.env) {
  const s = roomSettings({}, env);
  if (!s.mode) s.mode = "host";
  if (s.mode === "host" && !s.code) s.code = newCode();
  if (s.mode === "host" && !CODE_RE.test(s.code)) throw new Error(`POOLED_CODE: ${s.code} is not a room code the room page opens (4 to 6 of ${CODE_ABC})`);
  if (s.mode === "join" && !CODE_RE.test(s.code || "")) throw new Error("POOLED_CODE: the room code to join (4 to 6 characters)");
  return s;
}
export function applyToConfig(cfg, s) {
  const r = result(s).configPatch, ref = modelRef(s);
  return {
    ...cfg,
    models: { ...cfg.models, providers: { ...cfg.models?.providers, ...r.models.providers } },
    plugins: { ...cfg.plugins, entries: { ...cfg.plugins?.entries, [PROVIDER]: { ...cfg.plugins?.entries?.[PROVIDER], ...r.plugins.entries[PROVIDER] } } },
    agents: { ...cfg.agents, defaults: { ...cfg.agents?.defaults, model: { ...(typeof cfg.agents?.defaults?.model === "object" ? cfg.agents.defaults.model : {}), primary: ref },
      models: { ...cfg.agents?.defaults?.models, [ref]: { ...cfg.agents?.defaults?.models?.[ref], agentRuntime: { id: "openclaw" } } } } },
  };
}
