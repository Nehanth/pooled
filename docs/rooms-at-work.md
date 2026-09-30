# Rooms at work (relay)

A room links its devices directly with WebRTC, over UDP. Work, school and hotel networks often block UDP and every port except 443. Then there is no direct path between two devices, and without a relay the join fails with "Found the room, but this network blocks direct (UDP) connections…".

A TURN relay gets around that. Both devices reach the relay over TCP or TLS on port 443, which looks like ordinary web traffic, and the relay forwards between them. Once the site's owner sets up a relay provider (below), rooms use it on their own:

- Each device asks `POST /api/turn` for the relay's addresses and a credential that expires. The site holds the provider's key as a Vercel environment variable; the page, join links and QR codes never carry it.
- Links still go direct whenever they can. ICE tries direct paths first and uses the relay only when none of them works. `?relay=1` forces the relay for this device, which also hides its IP address from the other devices. `?relay=0` never asks for the site's relay.
- The room log says when a link goes through the relay, and over what (`TURN over TCP`, `TLS` or `UDP`). `pooledDebug()` shows each link's `path` (`direct` or `relay`) and `via`, and `pooledNet()` shows this tab's relay and whether UDP gets out.
- **Model weights never go through the relay.** A device can usually take its layers from another device's cache instead of downloading them, which can move several GB. On a relayed link it downloads them from the model host (Hugging Face) instead, and both sides enforce this. Only token traffic crosses the relay, a few KB per token per hop, so Cloudflare's free tier (1,000 GB a month) covers normal use.
- The site's relay is only used when nothing else is set. A relay in the Network box under the join form, `?turn=`, or `window.TURN_SERVERS` wins, and then `/api/turn` is not called.
- If a device can't reach anything (UDP blocked and no relay), it says so. The join fails after 15 s with the reason, a link to this page, and the Network box opened. A host on such a network gets a warning that only devices on the same network can join.

## Owner setup: Cloudflare Realtime TURN (recommended)

Cloudflare's TURN service is anycast, listens on UDP and TCP 3478 and TLS 443, and costs nothing for the first 1,000 GB a month ($0.05/GB after that).

1. In the Cloudflare dashboard (https://dash.cloudflare.com), open **Realtime → TURN Server** and create a TURN key. Copy the **Turn Token ID** and the **API Token**; the token is shown only once. Docs: https://developers.cloudflare.com/realtime/turn/
2. In the Vercel project (pooled), go to **Settings → Environment Variables** and add the following, for Production and Preview:
   - `TURN_KEY_ID`: the Turn Token ID
   - `TURN_KEY_API_TOKEN`: the API Token (mark it Sensitive)
3. Redeploy production. Environment variables only reach new deployments: Deployments → the latest production deployment → Redeploy.
4. Check it. `curl -si -X POST -H 'Origin: https://pooled.run' https://pooled.run/api/turn` should answer `200` with `iceServers`. A `204` means the variables are not set on this deployment. Then open a room with `?relay=1` on two devices; the room log should say "goes through the relay (TURN over …)".

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `TURN_TTL` | `21600` (6 h) | credential lifetime in seconds, 600 to 86400; the page fetches new ones at 3/4 of it, for links made later in a long session |
| `TURN_RATE` | `10` | credential requests per minute per IP, per function instance |
| `TURN_ALLOWED_ORIGINS` | none | other origins allowed to ask, comma-separated; `*` matches within one host label (`https://pooled-*.vercel.app`). The deployment's own origin is always allowed. |

## Owner setup: your own coturn

For a coturn server in "use-auth-secret" (TURN REST API) mode:

```
# /etc/turnserver.conf
listening-port=3478
tls-listening-port=443              # TLS on 443 gets through the strictest firewalls
cert=/etc/letsencrypt/live/relay.example.org/fullchain.pem
pkey=/etc/letsencrypt/live/relay.example.org/privkey.pem
use-auth-secret
static-auth-secret=<a long random string>
realm=relay.example.org
no-multicast-peers
denied-peer-ip=10.0.0.0-10.255.255.255      # plus the other private ranges: the relay must not reach your LAN
denied-peer-ip=172.16.0.0-172.31.255.255
denied-peer-ip=192.168.0.0-192.168.255.255
total-quota=200                     # concurrent allocations
bps-capacity=0
```

Then set these in Vercel instead of the Cloudflare variables:

- `TURN_SECRET`: the same `static-auth-secret`
- `TURN_HOST=relay.example.org`, which hands out `turn:relay.example.org:3478` over UDP and TCP and `turns:relay.example.org:443` over TLS. Or set `TURN_URLS` to your own comma-separated list.

The function signs `<expiry>:pooled` with HMAC-SHA1 of the secret, and coturn refuses a credential after its expiry. If both providers are set, Cloudflare wins.

## Abuse and cost

Anyone who can call `/api/turn` gets a working relay credential until it expires, and relayed bytes cost money. The endpoint only answers `POST` with an `Origin` that is this deployment's own, or one listed in `TURN_ALLOWED_ORIGINS`. That stops other websites from minting credentials in their visitors' browsers. It does not stop a script, which can forge any header, so:

- Credentials are short-lived (`TURN_TTL`).
- There is a per-IP limit (`TURN_RATE`). It is kept per function instance, so it slows abuse down but is not a hard limit. For a hard limit, add a Vercel Firewall rule: Firewall → Rules → path equals `/api/turn` → Rate limit, for example 20 requests per minute per IP.
- Set a spending cap or usage alert with the provider. Cloudflare bills relayed traffic past 1,000 GB a month.

## Signaling at work

Before devices can link, they find each other through the signaling server, which by default is the public PeerJS cloud (`wss://0.peerjs.com:443`). That is a normal TLS WebSocket on port 443, so most work networks allow it. Filters that block uncategorized hosts or WebSockets, or TLS inspection that breaks WebSockets, can still stop it. Then the join fails at once with "Can't reach the signaling server". The fix is a signaling server of your own on a host the network allows, listed in `window.POOLED_SIGNAL_SERVERS` or passed as `?signal=`. See [self-host-signaling.md](self-host-signaling.md). A room that is already running doesn't need signaling.

## What a relay can't fix

- A network that blocks TLS to the relay's host, or only allows traffic through an HTTP proxy that requires authentication. Browsers may be able to reach a TURN server over TCP/TLS through a configured proxy, but many proxies don't allow it. Try a phone hotspot.
- Speed. A relay adds a hop to every token's round trip. In PR #259's test, a room with every link on a relay decoded at 33 tok/s.

## Testing

- Unit tests: `tests/unit/turn_api_test.js` covers the function: both providers with mocked HTTP, the origin check, the rate limit and the not-configured case. `tests/unit/ice_test.js` covers the client: when the relay is asked for, validation, the timeout, refresh, the weights-over-relay decision and the UDP probe. `tests/unit/room_memory_test.js` covers weights refused over a relayed link, on both sides.
- End to end: `tests/e2e/room_relay.mjs` runs two tabs with UDP turned off in Chromium (`disable_non_proxied_udp`, set as a profile preference), `?relay=1`, credentials from the real `api/turn.mjs` handler, and a local coturn. It checks that every link is a relay candidate over TCP or TLS, that coturn logged the allocations, and, with a model, that answers come back. `--transport none` checks the failure message instead. To get coturn without root:

  ```sh
  mkdir relay && cd relay
  apt-get download coturn libpq5 libmysqlclient21 libhiredis1.1.0 libevent-extra-2.1-7t64 libevent-openssl-2.1-7t64 libevent-pthreads-2.1-7t64
  for d in *.deb; do dpkg -x $d root; done
  LD_LIBRARY_PATH=$PWD/root/usr/lib/$(uname -m)-linux-gnu node ../tests/e2e/room_relay.mjs --coturn $PWD/root/usr/bin/turnserver --transport tls
  ```
