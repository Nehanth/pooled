// Onboarding: the user picked Pooled. Start a room on this device (the invite link to open on the
// other Mac / PC / phone), or join one by pasting its link or code. The choice is saved in
// plugins.entries.pooled.config; secrets (the invite key, the pass the host gives this device) go to
// the plugin's own state file (state.js), not openclaw.json. The room itself runs in OpenClaw's
// gateway (the plugin's service), not in the wizard; joining is settled here, though: the wizard
// knocks on the room without the GPU, waits for the host's Allow if it asks, and keeps the pass, so
// the gateway walks straight in.
import { randomCode, CODE_LEN, makeGate, saveGate, validKey } from "../../../room/joingate.js";
import { Bridge } from "../../../cli/lib/room.js";
import { roomLink, roomSettings, parseRoom, fmtCode, defaultName, ROOM_ORIGIN } from "./pool.js";
import { MODEL_CHOICES, MODELS, modelInfo, modelChoices, memoryDefaults, isPulled, modelsDir, fmtBytes, shortCtxNote, isSmall, SMALL_WARNING } from "./models.js";
import { download, pullLine } from "./download.js";
import { initStateDir, savedGate, saveHostGate, joinState, saveJoinState } from "./state.js";

export const PROVIDER = "pooled";
export const AUTH_MARKER = "pooled-local";
export const newCode = () => randomCode(CODE_LEN);
export const JOIN_WAIT_MS = 180000;   // onboarding waits this long for the host's Allow
// what onboarding reaches outside itself (tests swap them): the room link and this GPU's memory
export const deps = { Bridge, memoryDefaults, waitMs: JOIN_WAIT_MS };

// the catalog entry: one model, the room's. Native transport: the base URL is never called.
// learned: what onboarding (or an earlier run) learned of a joined room's host: { model, ctx }
export function providerConfig(s, learned = {}) {
  const join = s.mode === "join";
  const hostModel = join && MODELS[learned.model] ? learned.model : null;
  const info = modelInfo(join ? hostModel || s.model : s.model, s.ctx || 0);
  const id = join ? "room" : s.model;
  const name = join ? `Pooled room ${fmtCode(s.code)}${hostModel ? ` (${info.name})` : ""}` : `${info.name} · Pooled room ${fmtCode(s.code)}`;
  return {
    baseUrl: "http://127.0.0.1",
    api: "openai-completions",
    authHeader: false,
    timeoutSeconds: 900,
    models: [{
      id, name,
      reasoning: join ? !!hostModel && hostModel !== "qwen3-1.7b" : s.model !== "qwen3-1.7b",
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: join ? (Number.isInteger(learned.ctx) && learned.ctx > 0 ? learned.ctx : hostModel ? info.ctx : 32768) : info.ctx,
      maxTokens: 4096,
      compat: { supportsTools: true, supportsDeveloperRole: false, supportsUsageInStreaming: true },
    }],
  };
}
export const modelRef = (s) => `${PROVIDER}/${s.mode === "join" ? "room" : s.model}`;

// The trimmed OpenClaw settings a small model may opt into (onboarding asks; off unless chosen). Both are
// global in OpenClaw, not per provider: tool search off puts every allowed tool in the prompt instead of
// the tool_search/tool_describe meta tools a 1.7B loops on; memory flush off stops the background
// "save memories before compaction" turn, which a 16k context triggers on nearly every turn.
export const TRIM_PATCH = { tools: { toolSearch: false }, agents: { defaults: { compaction: { memoryFlush: { enabled: false } } } } };
export const TRIM_NOTE = [
  "OpenClaw's tool search and its memory flush (a background turn that saves memories before compaction)",
  "make a small model slow: it loops on tool search, and the flush holds the room for about two minutes.",
  "This can turn both off: tools.toolSearch = false and agents.defaults.compaction.memoryFlush.enabled = false.",
  "These are global OpenClaw settings: they apply to every model and agent, not only Pooled, and stay",
  "after you switch models. Undo with `openclaw config unset tools.toolSearch` and",
  "`openclaw config unset agents.defaults.compaction.memoryFlush.enabled`.",
].join("\n");

function result(s, learned = {}) {
  const ref = modelRef(s);
  const small = s.mode === "host" ? isSmall(s.model) : isSmall(learned.model);
  return {
    profiles: [],
    defaultModel: ref,
    notes: s.mode === "host"
      ? [`Pooled room ${fmtCode(s.code)} opens with OpenClaw's gateway. Invite link for your other devices: ${roomLink(s.code, { key: learned.key, signal: s.signal })}. /pooled in the chat shows the room and who is waiting to join.`]
      : [`This device joins Pooled room ${fmtCode(s.code)} when OpenClaw's gateway starts, and holds layers when the room deals them.`],
    configPatch: {
      models: { providers: { [PROVIDER]: providerConfig(s, learned) } },
      plugins: { entries: { [PROVIDER]: { enabled: true, config: clean(s) } } },
      agents: { defaults: { models: { [ref]: { agentRuntime: { id: "openclaw" } } }, ...(s.trim ? TRIM_PATCH.agents.defaults : {}) } },
      // a small room model (1.7B, 16k context) cannot follow OpenClaw's full tool catalog: give it the
      // file tools only. Bigger models keep the defaults. s.trim: the opt-in global trims (TRIM_PATCH).
      ...(small || s.trim ? { tools: { ...(small ? { byProvider: { [PROVIDER]: { allow: ["read", "write", "edit", "ls"] } } } : {}), ...(s.trim ? TRIM_PATCH.tools : {}) } } : {}),
    },
  };
}
// what goes in openclaw.json (never the invite key or a pass; the default models folder is left out)
const clean = (s) => Object.fromEntries(Object.entries({ mode: s.mode, code: s.code, model: s.model, pledgeGB: s.pledgeGB, minDevices: s.minDevices, signal: s.signal,
  modelDir: s.modelDir && s.modelDir !== modelsDir() ? s.modelDir : null, ask: s.mode === "host" && s.ask === false ? false : null, pull: s.pull === false ? false : null })
  .filter(([, v]) => v != null && v !== ""));

// the host's invite key for code, kept (state.js) so the link shown now is the gateway's
export function hostKey(code, ask = true) {
  const saved = savedGate(code);
  if (saved && validKey(saved.key)) return saved.key;
  const g = makeGate({ ask });
  saveHostGate(code, saveGate(g));
  return g.key;
}

// Knock on a room without the GPU (the `pooled serve` bridge): the host lets this device in with the
// key, a pass from before, or its Allow; the pass it gives is kept for the gateway.
// -> { ok, host, model, ctx, pass } | { ok: false, refused | error | timeout }
export async function knock(code, key, { name = defaultName(), signal = null, waitMs = deps.waitMs, onLobby = () => {}, BridgeClass = deps.Bridge } = {}) {
  const b = new BridgeClass({ code, key, signal, name, client: "OpenClaw" });
  const prev = joinState(code).pass;
  if (validKey(prev)) b.pass = prev;
  b.on?.("lobby", () => onLobby(b));
  let timer = null;
  try {
    const out = await Promise.race([
      b.connect().then(() => ({ ok: true })),
      new Promise((res) => { timer = setTimeout(() => res({ ok: false, timeout: true }), waitMs); }),
    ]);
    if (!out.ok) return out;
    return { ok: true, host: b.hostName || null, model: b.model || b.hostMeta?.model || null, ctx: Number.isInteger(b.hostMeta?.ctx) ? b.hostMeta.ctx : null, pass: validKey(b.pass) ? b.pass : null };
  } catch (err) {
    return b.kicked ? { ok: false, refused: b.kicked } : { ok: false, error: err.message };
  } finally {
    clearTimeout(timer);
    try { if (b.connected) await b.leave(); else b.destroy(); } catch {}
  }
}

const gbPrompt = (p, mem, initial) => p.text({
  message: `This GPU: ${mem.label}. How much of its memory does this device lend the room (GB)?`,
  initialValue: String(initial),
  validate: (v) => (+v > 0 && +v <= 512 ? undefined : "a number of GB, e.g. 12"),
});

export async function runSetup(ctx) {
  await initStateDir();
  const p = ctx.prompter;
  const mode = await p.select({
    message: "Pooled runs a model across your own devices. How should this device take part?",
    options: [
      { value: "host", label: "Start a room on this device", hint: "you get an invite link; your other Mac, PC or phone joins with it" },
      { value: "join", label: "Join a room", hint: "another device already started one: paste its link or code" },
    ],
  });
  const prev = roomSettings(ctx.config?.plugins?.entries?.[PROVIDER]?.config || {}, {});
  const mem = deps.memoryDefaults();
  const dir = prev.modelDir;
  if (mode === "host") {
    const pledge = +(await gbPrompt(p, mem, prev.mode === "host" && prev.pledgeGB ? prev.pledgeGB : mem.def));
    const { rows, recommended } = modelChoices(dir, pledge);
    const model = await p.select({
      message: "Which model should the room run?",
      options: rows.map((r) => ({ value: r.key, label: r.name, hint: r.hint })),
      // the earlier choice, unless it was a small model: then the one recommended for OpenClaw
      initialValue: prev.mode === "host" && MODEL_CHOICES.includes(prev.model) && !isSmall(prev.model) ? prev.model : recommended,
    });
    const trim = isSmall(model) ? await smallModel(p) : false;
    let pull = prev.pull;
    if (!isPulled(dir, model)) pull = await getModel(p, dir, model);
    const devs = await p.select({ message: "Wait for how many devices before loading the model?", options: [
      { value: 1, label: "Just this one when it has enough memory", hint: "others can still join later as askers" },
      { value: 2, label: "Two (this + one more)" }, { value: 3, label: "Three" }], initialValue: rows.find((r) => r.key === model)?.fitsAlone ? 1 : 2 });
    const ask = await p.select({ message: "Who can join the room?", options: [
      { value: true, label: "Devices with the invite link", hint: "recommended: a device with only the code waits until you send /pooled allow" },
      { value: false, label: "Anyone with the room code", hint: "no asking" }], initialValue: prev.ask !== false });
    const code = prev.mode === "host" && prev.code ? prev.code : newCode();
    const s = { mode, model, code, pledgeGB: pledge, minDevices: devs, signal: prev.signal, modelDir: dir, ask, pull, trim };
    const key = hostKey(code, ask);
    const link = roomLink(code, { key, signal: s.signal });
    const info = modelInfo(model);
    await p.note([
      `Room code: ${fmtCode(code)}`,
      `Invite link: ${link}`,
      "",
      "On the other device, either:",
      "  - open the invite link in Chrome or Safari (a Mac, a PC or a phone), set how much memory it lends, and press Join;",
      `  - or run \`npx -p @pooled/cli -p webgpu@0.6.1 pooled join "${link}"\`;`,
      "  - or pick Pooled → \"Join a room\" in OpenClaw there and paste the link.",
      ask ? `A device with only the code ${fmtCode(code)} waits: /pooled allow (in any OpenClaw chat) lets it in.` : `Anyone with the code ${fmtCode(code)} can join.`,
      `The model (${info.name}) loads once ${devs > 1 ? `${devs} devices are in the room and ` : ""}the room has about ${info.needGB} GB.`,
      "Share the link only with people you trust: every device holding layers computes what is asked here.",
    ].join("\n"), "Pooled room");
    return result(s, { key });
  }
  // join
  const pasted = await p.text({ message: "Paste the room's invite link (or type its code)", placeholder: `${ROOM_ORIGIN}/r/4TK-G9P#k=…  or  4TK-G9P`,
    validate: (v) => (parseRoom(v).code ? undefined : "a link like https://pooled.run/r/4TKG9P#k=…, or a code like 4TK-G9P") });
  const { code, key } = parseRoom(pasted);
  const pledge = +(await gbPrompt(p, mem, prev.mode === "join" && prev.pledgeGB ? prev.pledgeGB : mem.def));
  if (key) saveJoinState(code, { key });
  const prog = p.progress(`Connecting to Pooled room ${fmtCode(code)}…`);
  const name = defaultName();
  const k = await knock(code, key || joinState(code).key || null, { signal: prev.signal, name,
    onLobby: (b) => prog.update(`Waiting for ${b.hostName || "the host"} to let this device in… (it sees "${name} wants to join"; the invite link skips this)`) });
  const learned = { model: null, ctx: null };
  if (k.ok) {
    saveJoinState(code, { key, pass: k.pass, host: k.host, model: k.model, ctx: k.ctx });
    Object.assign(learned, { model: k.model, ctx: k.ctx });
    prog.stop(`In Pooled room ${fmtCode(code)}${k.host ? ` (host: ${k.host})` : ""}${k.model && MODELS[k.model] ? `, running ${modelInfo(k.model).name}` : ""}`);
    const short = shortCtxNote(k.model, k.ctx, k.host || "the host");
    if (short) await p.note(short, "The room's context is too short for OpenClaw");
    if (isSmall(k.model)) learned.trim = await smallModel(p, k.host || "the host");
  } else if (k.refused) {
    prog.stop(`The host of room ${fmtCode(code)} turned this device away: ${k.refused}`);
    throw new Error(`Pooled: the host of room ${fmtCode(code)} turned this device away (${k.refused}). Ask for the room's invite link and run the setup again with it.`);
  } else {
    prog.stop(k.timeout ? `The host of room ${fmtCode(code)} has not let this device in yet: the gateway asks again when it starts`
      : `Couldn't reach room ${fmtCode(code)} now (${k.error}): the gateway tries again when it starts`);
  }
  const s = { mode: "join", code, model: prev.model || "qwen3-1.7b", pledgeGB: pledge, signal: prev.signal, modelDir: dir, trim: !!learned.trim };
  return result(s, learned);
}

// a small model was picked (or the joined room runs one): say what to expect, and offer the trimmed
// OpenClaw settings, off unless chosen -> whether to apply TRIM_PATCH
async function smallModel(p, host = null) {
  await p.note(host ? `${host} runs this room with a small model. ${SMALL_WARNING}` : SMALL_WARNING, "Small model");
  await p.note(TRIM_NOTE, "Optional: trim OpenClaw for a small model");
  const v = await p.select({
    message: "Turn off OpenClaw's tool search and memory flush? (global: every model and agent)",
    options: [
      { value: false, label: "No, keep OpenClaw's settings", hint: "keep them if you use other models in OpenClaw too" },
      { value: true, label: "Yes, turn both off for all of OpenClaw", hint: "tools.toolSearch = false, compaction.memoryFlush off" },
    ],
    initialValue: false,
  });
  return v === true;
}

// the model is not on disk: download it now (with progress), let the gateway do it, or stream
async function getModel(p, dir, model) {
  const info = modelInfo(model);
  const how = await p.select({
    message: `${info.name} is not downloaded yet (${fmtBytes(info.fileBytes)}, kept in ${dir.replace(process.env.HOME || "~", "~")} with \`pooled pull\`'s models)`,
    options: [
      { value: "now", label: "Download now", hint: "resumable, checked against its SHA-256" },
      { value: "later", label: "Download when the gateway starts", hint: "devices can join while it downloads" },
      { value: "stream", label: "Don't download", hint: "stream this device's layers from Hugging Face at each start" },
    ],
    initialValue: "now",
  });
  if (how === "stream") return false;
  if (how === "later") return true;
  const prog = p.progress(`Downloading ${info.name}…`);
  const st = await download(dir, model, { onChange: (x) => prog.update(`Downloading ${info.name}: ${pullLine(x)}`) });
  prog.stop(st.state === "done" ? `${info.name} downloaded (${fmtBytes(st.total)})` : `${info.name}: ${pullLine(st)}; the gateway tries again when it starts`);
  return true;
}

// `openclaw onboard --non-interactive --auth-choice pooled`: the choice from POOLED_* env
// (POOLED_LINK: a room's invite link to join)
export function setupFromEnv(env = process.env) {
  const s = roomSettings({}, env);
  if (!s.mode) s.mode = s.code && env.POOLED_LINK ? "join" : "host";
  if (s.mode === "host" && !s.code) s.code = newCode();
  if (s.mode === "host" && !/^[A-HJKMNP-TV-Z2-9]{4}$|^[A-HJKMNP-TV-Z2-9]{6}$/.test(s.code)) throw new Error(`POOLED_CODE: ${s.code} is not a room code the room page opens (6 of ABCDEFGHJKMNPQRSTVWXYZ23456789)`);
  if (s.mode === "join" && !/^[A-Z0-9]{4}$|^[A-Z0-9]{6}$/.test(s.code || "")) throw new Error("POOLED_LINK: the invite link of the room to join (or POOLED_CODE: its code)");
  return s;
}
export function applyToConfig(cfg, s, learned = {}) {
  const r = result(s, learned).configPatch, ref = modelRef(s);
  return {
    ...cfg,
    models: { ...cfg.models, providers: { ...cfg.models?.providers, ...r.models.providers } },
    plugins: { ...cfg.plugins, entries: { ...cfg.plugins?.entries, [PROVIDER]: { ...cfg.plugins?.entries?.[PROVIDER], ...r.plugins.entries[PROVIDER] } } },
    agents: { ...cfg.agents, defaults: { ...cfg.agents?.defaults, model: { ...(typeof cfg.agents?.defaults?.model === "object" ? cfg.agents.defaults.model : {}), primary: ref },
      models: { ...cfg.agents?.defaults?.models, [ref]: { ...cfg.agents?.defaults?.models?.[ref], agentRuntime: { id: "openclaw" } } } } },
    ...(r.tools ? { tools: { ...cfg.tools, byProvider: { ...cfg.tools?.byProvider, ...r.tools.byProvider } } } : {}),
  };
}
// non-interactive: keep the key from POOLED_LINK (join), or make the host's key (host), in the state file
export async function nonInteractive(cfg, env = process.env) {
  await initStateDir();
  const s = setupFromEnv(env);
  const learned = {};
  if (s.mode === "join" && s.key) saveJoinState(s.code, { key: s.key });
  if (s.mode === "host") learned.key = hostKey(s.code, s.ask);
  return applyToConfig(cfg, s, learned);
}
