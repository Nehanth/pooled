// @pooled/room-node: a Pooled room device in a Node process (Dawn WebGPU + node-datachannel WebRTC).
// It runs Pooled's own engine unmodified and speaks the room protocol, so it sits in the same room as
// browser tabs and phones on pooled.run. See README.md.
//
//   import { createRoom, joinRoom } from "@pooled/room-node";
//   const room = await createRoom({ model: "qwen3.6-35b-moe", pledgeGB: 24 });   // room.code -> share it
//   await room.start({ minDevices: 2 });            // optional: wait for devices, then deal the layers
//   for await (const ev of room.ask([{ role: "user", content: "Hi" }], { maxTokens: 256 })) {
//     if (ev.type === "token") process.stdout.write(ev.text);
//   }
//   const node = await joinRoom("ABCD", { pledgeGB: 24 });   // hold layers in someone else's room
export { createRoom, joinRoom, RoomNode, PREFIX, toApiRequest, eventEncoder, nodeServers, nodeCtxFor } from "./roomnode.js";
export { PROTOCOL } from "../../room/transport.js";
export { reconnectDelay } from "../../room/signal.js";
export { MODELS, FILES, FILE_GB, NEED_GB, NEED_MIN_GB, SHAPE, PICKER, CTX, roomBytes, pickCtx, ctxChoices, ctxShortNote, ctxK } from "../../room/models.js";
// the room page's fit math (#271), so pooled host says "short" exactly when the room page would
export { roomFit, shortNote, shortBy, gbUp, dealRoom } from "../../room/plan.js";
export { pledgeGB } from "../../room/pledge.js";
export { setupNode, runtime, probeMeta } from "./env.js";
export { openModel, LOCAL } from "./source.js";
