import test from "node:test";
import assert from "node:assert/strict";
import { nvidiaReading, androidReading, resourceReader, resourceMonitor } from "../../../benchmarks/bench_room_resources.js";

test("resource parsers retain explicit units and unavailable values without vendor history", () => {
  const gpu = nvidiaReading("NVIDIA GeForce RTX 3090, 591.86, 41, 5, 8330, 24576, [N/A]\n");
  assert.equal(gpu.metrics.gpuMemoryUsedMiB, 8330); assert.equal(gpu.metrics.gpuPowerW, null);
  assert.ok(gpu.unavailable.gpuPowerW); assert.equal(gpu.metrics.gpuTemperatureC, 41);
  assert.throws(() => nvidiaReading("one\ntwo"), /one NVIDIA/);
  const phone = androidReading({ battery: "USB powered: true\nAC powered: false\nWireless powered: false\nstatus: 5\nlevel: 75\nscale: 100\ntemperature: 305\n[EventLogBuffer]\nprivate-history-and-identifiers",
    thermal: "Thermal Status: 2\nCurrent temperatures from HAL:\n\tTemperature{mValue=32.0, mType=0, mName=AP, mStatus=2}\n\tTemperature{mValue=32.0, mType=0, mName=AP, mStatus=2}", lowPower: "1\n", brightness: "133\n" });
  assert.equal(phone.metrics.batteryTemperatureC, 30.5); assert.equal(phone.metrics.thermalStatus, 2);
  assert.equal(phone.conditions.lowPower, true); assert.equal(phone.conditions.usbPowered, true);
  assert.equal(phone.sensors.length, 1); assert.ok(!JSON.stringify(phone).includes("private-history"));
  const missing = androidReading({ battery: "permission denied" });
  assert.equal(missing.metrics.batteryPercent, null); assert.equal(missing.conditions.lowPower, null);
  assert.ok(missing.unavailable.thermalStatus);
});

test("Android sensors use current HAL readings, never cached or historical temperatures", () => {
  const thermal = "Thermal Status: 0\r\nCached temperatures:\r\n\tTemperature{mValue=80, mType=0, mName=AP, mStatus=4}\r\n" +
    "Current temperatures from HAL:\r\n\tTemperature{mValue=32.5, mType=0, mName=AP, mStatus=0}\r\n" +
    "Current cooling devices from HAL:\r\nHistory:\r\n\tTemperature{mValue=90, mType=0, mName=AP, mStatus=5}\r\n";
  assert.deepEqual(androidReading({ thermal }).sensors, [{ name: "AP", type: 0, temperatureC: 32.5, status: 0 }]);
  assert.deepEqual(androidReading({ thermal: "Thermal Status: 0\nHistory:\n\tTemperature{mValue=90, mType=0, mName=AP, mStatus=5}" }).sensors, []);
});

test("resource monitor brackets work, does not overlap polls, drains and stops", async () => {
  let active = 0, maxActive = 0, count = 0;
  let during;
  const seenDuring = new Promise((resolve) => { during = resolve; });
  const read = async () => {
    maxActive = Math.max(maxActive, ++active); count++;
    await new Promise((resolve) => setTimeout(resolve, 2)); active--;
    if (count === 2) during();
    return { metrics: { gpuTemperatureC: 40 } };
  };
  const monitor = resourceMonitor({ R: read }, { intervalMs: 1 });
  await monitor.begin("one"); await seenDuring;
  const result = await monitor.end();
  assert.equal(maxActive, 1); assert.equal(active, 0); assert.equal(result.status, "pass");
  assert.equal(result.samples[0].phase, "before"); assert.equal(result.samples.at(-1).phase, "after");
  assert.ok(result.samples.some((s) => s.phase === "during"));
  assert.ok(result.samples.every((s) => s.startMs <= s.endMs));
  const ended = count; await new Promise((resolve) => setTimeout(resolve, 10)); assert.equal(count, ended);
  await assert.rejects(monitor.end(), /no active/);
});

test("resource bounds and failures remain explicit, with no private command details", async () => {
  const monitor = resourceMonitor({ R: async () => { throw Object.assign(new Error("ssh private-host serial=secret"), { code: 1 }); } }, { capacity: 1 });
  await monitor.begin("failed"); await assert.rejects(monitor.begin("again"), /already active/);
  const r = await monitor.end(); assert.equal(r.status, "incomplete"); assert.equal(r.dropped, 1);
  assert.equal(r.samples[0].value, null); assert.equal(r.samples[0].error, "resource command failed (1)");
  assert.ok(!JSON.stringify(r).includes("secret"));
});

test("resource command reader has a real bounded subprocess and rejects unsafe selectors", async () => {
  assert.throws(() => resourceReader({ kind: "nvidia", gpu: "0; other-command" }), /selector/);
  const read = resourceReader({ kind: "nvidia", command: process.execPath,
    args: ["-e", "console.log('fixture, test, 40, 5, 10, 100, 20')", "--"] });
  assert.equal((await read()).metrics.gpuPowerW, 20);
  const stuck = resourceReader({ kind: "nvidia", command: process.execPath, timeoutMs: 30,
    args: ["-e", "setInterval(()=>{},1000)", "--"] });
  const monitor = resourceMonitor({ R: stuck }); await monitor.begin("timeout");
  const r = await monitor.end(); assert.equal(r.status, "incomplete");
  assert.ok(r.samples.every((s) => s.error === "resource command timed out"));
});
