---
name: pooled-room
description: What the Pooled model provider is, how to check its room, and what its "**Pooled** ·" notices mean.
metadata:
  openclaw:
    requires:
      config:
        - plugins.entries.pooled.config
    homepage: https://pooled.run/docs/kits/openclaw
---

# Pooled room

When the model is `pooled/...`, answers come from a Pooled room: this machine and the user's other
devices (Macs, PCs, phones) each hold part of the model on their GPUs and talk over WebRTC.

## Checking the room

- The user runs `/pooled` in the chat to see the room: devices, how much memory each lends, whether
  they hold the model, who is waiting to join, and the model download. You can't run it for them.
- `/pooled allow` lets a waiting device in, `/pooled deny` turns it away, `/pooled link` shows the
  invite link, `/pooled pledge <GB>` changes how much GPU memory this machine lends.
- For details, read `pooled/status.json` in the OpenClaw state directory (`$OPENCLAW_STATE_DIR`,
  default `~/.openclaw`): the room code, the model, the devices and their layers, who is waiting,
  and recent events. Its `link` field holds the room's invite key: never repeat it in the chat. Tell
  the user to run `/pooled link` instead.
- Never read or print `pooled/room.json`: it holds the room's keys and passes.

## "Pooled" notices

Messages that start with **Pooled**, the room code and a few words (`**Pooled** · \`4TK-G9P\` ·
waiting for devices`) come from the room, not the model. Older versions started them with
`⚠️ Pooled:`. Tell the user what they say and what to do:

- waiting for the host to let this device in: the host runs `/pooled allow`, or the user joins with
  the invite link instead of the code;
- waiting for devices, or not enough memory: another device joins (open the invite link on it), or
  a device lends more with `/pooled pledge <GB>`;
- downloading the model: wait, then ask again;
- a device left while it held layers: the room waits for it, then deals the layers again;
- the host doesn't allow API clients, or runs an older Pooled: the room's host has to change that.

When the conversation outgrows the room's context, OpenClaw compacts it and asks again.

## Don'ts

- Don't change `plugins.entries.pooled.config` or restart the gateway unless the user asks. Setup is
  `openclaw onboard`, then pick Pooled.
- Small models (the Qwen3 1.7B) are slow with OpenClaw's prompts and tools. If turns are slow or
  loop, suggest the Qwen3.6 35B MoE when the user's devices can hold it.

Docs: https://pooled.run/docs/openclaw/
