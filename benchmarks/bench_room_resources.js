import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execute = promisify(execFile);
const numeric = (s) => s != null && String(s).trim() !== "" && Number.isFinite(Number(s)) ? Number(s) : null;
const field = (text, name) => new RegExp("^\\s*" + name + ":\\s*(\\S+)", "m").exec(text)?.[1];
const boolean = (s) => s === "true" ? true : s === "false" ? false : null;

export function nvidiaReading(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length !== 1) throw new Error("expected one NVIDIA adapter");
  const parts = lines[0].split(",").map((s) => s.trim());
  if (parts.length !== 7) throw new Error("unexpected NVIDIA query fields");
  const [name, driver, temperature, utilization, used, total, power] = parts;
  const metrics = { gpuTemperatureC: numeric(temperature), gpuUtilizationPercent: numeric(utilization),
    gpuMemoryUsedMiB: numeric(used), gpuMemoryTotalMiB: numeric(total), gpuPowerW: numeric(power) };
  return { source: "nvidia-smi", device: { name, driver }, metrics,
    unavailable: Object.fromEntries(Object.entries(metrics).filter(([, v]) => v === null).map(([k]) => [k, "driver did not report this metric"])),
    scope: "whole GPU, including other applications; not per-process or kernel measurements" };
}

// Samsung dumpsys output can append historical logs and device identifiers.
// Only allowlisted current scalar fields and temperature entries leave this parser.
export function androidReading({ battery = "", thermal = "", lowPower = "", brightness = "" }) {
  const level = numeric(field(battery, "level")), scale = numeric(field(battery, "scale"));
  const rawTemperature = numeric(field(battery, "temperature"));
  const metrics = { batteryPercent: level !== null && scale > 0 ? level / scale * 100 : null,
    batteryTemperatureC: rawTemperature === null ? null : rawTemperature / 10,
    thermalStatus: numeric(field(thermal, "Thermal Status")), brightnessRaw: numeric(brightness) };
  const power = numeric(lowPower);
  const conditions = { usbPowered: boolean(field(battery, "USB powered")), acPowered: boolean(field(battery, "AC powered")),
    wirelessPowered: boolean(field(battery, "Wireless powered")), batteryStatus: numeric(field(battery, "status")),
    lowPower: power === 0 ? false : power === 1 ? true : null };
  const sensors = new Map();
  const current = /^Current temperatures from HAL:\r?\n((?:[\t ]+[^\n]*(?:\n|$))*)/m.exec(thermal)?.[1] ?? "";
  for (const m of current.matchAll(/Temperature\{mValue=([^,]+), mType=(\d+), mName=([^,}]+), mStatus=(\d+)\}/g)) {
    const sensor = { name: m[3], type: +m[2], temperatureC: numeric(m[1]), status: +m[4] };
    sensors.set(JSON.stringify([sensor.name, sensor.type]), sensor);
  }
  return { source: "Android dumpsys/settings via ADB", metrics, conditions, sensors: [...sensors.values()],
    unavailable: Object.fromEntries(Object.entries({ ...metrics, ...conditions }).filter(([, v]) => v === null)
      .map(([k]) => [k, "Android did not report this field"])),
    scope: "battery temperature is not GPU temperature; thermal sensors retain Android type/status codes; brightness is a raw setting" };
}

// Commands have explicit arguments, no shell, finite time and output bounds.
// Prefix arguments can select an already-authorised remote command or ADB server.
export function resourceReader(spec) {
  if (!["nvidia", "android"].includes(spec.kind)) throw new Error("unknown resource source");
  const timeoutMs = spec.timeoutMs ?? 5000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) throw new Error("invalid resource timeout");
  const command = spec.command || (spec.kind === "nvidia" ? "nvidia-smi" : "adb");
  const prefix = spec.args ?? [];
  if (!Array.isArray(prefix) || prefix.some((s) => typeof s !== "string")) throw new Error("invalid resource arguments");
  const run = async (args) => (await execute(command, [...prefix, ...args], { timeout: timeoutMs, maxBuffer: 262144, windowsHide: true })).stdout;
  if (spec.kind === "nvidia") {
    const gpu = String(spec.gpu ?? 0);
    if (!/^\d+$/.test(gpu)) throw new Error("resource GPU selector must be an index");
    return async () => nvidiaReading(await run(["--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
      "--format=csv,noheader,nounits", "--id=" + gpu]));
  }
  return async () => {
    const names = ["battery", "thermal", "lowPower", "brightness"];
    const commands = [["shell", "dumpsys", "battery"], ["shell", "dumpsys", "thermalservice"],
      ["shell", "settings", "get", "global", "low_power"], ["shell", "settings", "get", "system", "screen_brightness"]];
    const results = await Promise.allSettled(commands.map(run));
    const value = androidReading(Object.fromEntries(results.map((r, i) => [names[i], r.status === "fulfilled" ? r.value : ""])));
    value.readErrors = results.flatMap((r, i) => r.status === "rejected" ? [{ source: names[i], error: commandError(r.reason) }] : []);
    return value;
  };
}

function commandError(e) {
  // execFile's full error includes the command and possibly a private serial/address.
  return e?.killed ? "resource command timed out" : `resource command failed (${e?.code ?? "unknown"})`;
}

// Polls do not overlap. Timestamps bracket each external command on the controller
// clock and are never subtracted from a phone, browser or native engine clock.
export function resourceMonitor(readers, { intervalMs = 1000, capacity = 4096, sources = null } = {}) {
  if (!Number.isSafeInteger(intervalMs) || intervalMs < 1 || !Number.isSafeInteger(capacity) || capacity < 1 || !Object.keys(readers).length) {
    throw new Error("invalid resource sampling configuration");
  }
  let active = null;
  async function poll(state, phase) {
    const entries = await Promise.all(Object.entries(readers).map(async ([device, read]) => {
      const startMs = performance.now() - state.started;
      let value = null, error = null;
      try { value = await read(); } catch (e) { error = commandError(e); }
      return { device, phase, startMs, endMs: performance.now() - state.started, value, error };
    }));
    for (const entry of entries) {
      if (state.samples.length < capacity) state.samples.push(entry); else state.dropped++;
    }
  }
  function schedule(state) {
    state.timer = setTimeout(() => {
      state.pending = poll(state, "during").finally(() => { if (!state.stopping) schedule(state); });
    }, intervalMs);
  }
  return {
    async begin(attempt) {
      if (active) throw new Error("resource attempt already active");
      if (typeof attempt !== "string" || !attempt) throw new Error("resource attempt needs an id");
      const state = active = { attempt, started: performance.now(), samples: [], dropped: 0, stopping: false, pending: null, timer: null };
      await poll(state, "before");
      if (!state.stopping) schedule(state);
    },
    async end() {
      if (!active) throw new Error("no active resource attempt");
      const state = active; state.stopping = true; clearTimeout(state.timer);
      await state.pending; await poll(state, "after"); active = null;
      return { attempt: state.attempt, configuration: { intervalMs, capacity, devices: Object.keys(readers), sources },
        clock: "controller monotonic milliseconds since resource begin; brackets the attempt, not just GPU work",
        samples: state.samples, dropped: state.dropped,
        status: state.dropped || state.samples.some((s) => s.error || s.value?.readErrors?.length || Object.keys(s.value?.unavailable ?? {}).length)
          ? "incomplete" : "pass" };
    },
  };
}
