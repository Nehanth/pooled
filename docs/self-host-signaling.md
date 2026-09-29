# Self-hosting signaling for rooms

A room needs a signaling server only so its devices can find each other: the host registers the room code, a joiner asks for it, and the two swap WebRTC offers and network candidates through it. After that the links are direct (or through TURN), and the model, the prompts and the tokens never touch the signaling server. A room that is already running keeps going when signaling drops. Until the server is back, new devices can't join, and a host that reloads can't claim its room again.

Pooled uses [PeerJS](https://peerjs.com). By default a room uses the public PeerJS cloud (`0.peerjs.com`). The cloud is free and shared, so it can be slow, down, or blocked on some networks. Run your own server if you need rooms to be reliable. It also keeps room codes and device IP addresses off a third-party server.

## Run a PeerServer

Any PeerJS server 1.x works. The repo already has one as a dev dependency (`peer`):

```sh
npx peer --port 9000 --path /               # ws://<this machine>:9000
docker run -p 9000:9000 -d peerjs/peerjs-server   # the same, in a container
```

A page served over https (pooled.run, a Vercel preview) can only open `wss://`, so put the server behind TLS. Either pass `--sslkey key.pem --sslcert cert.pem`, or use a reverse proxy and pass `--proxied`. For example, with Caddy:

```
signal.example.com {
  reverse_proxy 127.0.0.1:9000
}
```

```sh
npx peer --port 9000 --path / --proxied
```

The proxy must pass WebSocket upgrades through (Caddy does this by default; nginx needs `proxy_http_version 1.1` plus the `Upgrade` and `Connection` headers). Joiners also make one plain HTTPS request, `GET <path>peerjs/id`. The server allows any origin by default; if you restrict origins with `--cors`, include the room's origin. Keep the server's default `--expire_timeout` (5 s) or a similar value: it controls how fast a joiner learns that a server has no such room and moves on to the next server.

Check it: `curl https://signal.example.com/peerjs/id` should print a fresh id.

## Point rooms at it

**For one room**: add `?signal=` to the room URL. The host and every joiner must use the same server, because a room registered on one server doesn't exist on another. The host's invite link and QR code carry the parameter for you.

```
https://pooled.run/room?signal=signal.example.com                 wss, port 443, path /
https://pooled.run/room?signal=wss://signal.example.com:8443/pooled
https://pooled.run/room?signal=signal.example.com,cloud            yours first, then the cloud
http://127.0.0.1:8080/p2p.html?signal=127.0.0.1:9000               local dev: ws, since the page is http
```

The accepted forms are `cloud`, `host`, `host:port`, `host:port/path`, or a `wss://`, `ws://`, `https://` or `http://` URL. A bare host uses `wss` when the page is https (and `ws` when it is http) and port 443. When `?signal=` is present, it replaces every other setting: only the servers it names are tried, in order.

**For a deployment**: define an ordered list before `room.js` loads, for example in `p2p.html`:

```html
<script>window.POOLED_SIGNAL_SERVERS = ["wss://signal.example.com", "cloud"];</script>
```

Entries can also be objects: `{ host, port, path, secure, key }`. Use `key` if your server runs with a non-default `--key`. Without the list, rooms use `["cloud"]`.

## How the fallback works (room/signal.js)

- **Order.** The host tries the servers in order. The first one that registers the room code is used. A server counts as down when it refuses or drops the connection, fails TLS, or gives no answer within 8 s. The last server in the list gets 20 s, since there's nothing left to try after it. The join screen shows which server it is trying.
- **Taken codes.** "That code is already hosting a room" is an answer, not an outage, so it never triggers a fallback. Falling back there would split one room across two servers.
- **Joiners.** A joiner walks the same list. When a server answers but has no such room, the joiner tries the next one, because the host may have fallen back to it. A joiner who comes from an invite link goes straight to the host's server: whenever the host is not on the list's first server, the link includes `?signal=` (and `dev=0`, so the page doesn't open in dev mode).
- **Host reloads.** A host that reloads tries the server its room was on first, because its guests are there.
- **All servers down.** If no server answers, the join screen says the signaling server can't be reached, that running rooms are unaffected, and links to this page. The Create and Join buttons work again, so you can retry.
- **Drops during a room.** If signaling drops while a room is running, every device shows a note under the header ("Lost the signaling server… Reconnecting…"). Each device re-registers its own id with backoff (2, 4, 8, 16, then every 30 s). The note clears when the server is back.

The room protocol itself does not change: signaling isn't part of it, and devices on different versions still meet as long as they use the same server.

## Testing

- `deno test --allow-read tests/unit/signal_test.js` covers spec parsing, list order, and the fallback rules against a fake Peer.
- `node tests/e2e/signal_fallback.mjs` runs the fallback end to end in headless Chromium, without a GPU. It starts two local PeerServers, makes the cloud unreachable with a resolver rule, and uses a TCP port that never answers as a black-holed server. It covers four cases: the cloud is down; one server is down for the host only; every server is down; and a server drops and comes back while a room is running.
