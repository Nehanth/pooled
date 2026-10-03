// room/lan.js: which ?invite= origins a host's invite links may use (loopback page, private address only).
import { inviteOrigin } from "../../room/lan.js";

const eq = (a, b, m) => { if (a !== b) throw new Error((m || "mismatch") + ": " + JSON.stringify(a) + " != " + JSON.stringify(b)); };

Deno.test("inviteOrigin: private addresses from a loopback page", () => {
  eq(inviteOrigin("http://10.0.0.92:8080", "localhost"), "http://10.0.0.92:8080");
  eq(inviteOrigin("http://192.168.1.20:8080/room?x=1", "127.0.0.1"), "http://192.168.1.20:8080", "path and query dropped");
  eq(inviteOrigin("https://172.20.3.4", "[::1]"), "https://172.20.3.4");
  eq(inviteOrigin("http://100.101.102.103:8080", "localhost"), "http://100.101.102.103:8080", "CGNAT / Tailscale");
  eq(inviteOrigin("http://my-mac.local:8080", "localhost"), "http://my-mac.local:8080");
});

Deno.test("inviteOrigin: ignored off loopback, for public hosts and for junk", () => {
  eq(inviteOrigin("http://10.0.0.92:8080", "pooled.run"), null, "a deployed page never swaps its origin");
  eq(inviteOrigin("http://10.0.0.92:8080", "10.0.0.92"), null);
  for (const bad of ["https://evil.example", "http://8.8.8.8", "http://172.32.0.1", "http://192.169.1.1", "http://100.128.0.1",
    "http://10.0.0.300", "http://user:pw@10.0.0.2", "javascript:alert(1)", "ftp://10.0.0.2", "10.0.0.2:8080", "", null])
    eq(inviteOrigin(bad, "localhost"), null, "rejects " + JSON.stringify(bad));
});
