# Drive OpenClaw's interactive "models auth login --provider pooled" in a pty, as a user would:
#   python3 onboard_pty.py host              start a room: lend 4 GB, the recommended model, don't download, 1 device, link only
#   python3 onboard_pty.py join <LINK|CODE>  join: paste it, lend 4 GB, wait for the host's Allow
# env: an isolated OpenClaw (HOME, OPENCLAW_STATE_DIR, OPENCLAW_CONFIG_PATH), FAKERUN (an XDG_RUNTIME_DIR
# without the user's systemd bus), PTY_LOG. Prints the screen text it saw.
import os, re, sys, pexpect
mode = sys.argv[1]; room = sys.argv[2] if len(sys.argv) > 2 else ""
out = open(os.environ.get("PTY_LOG", "/dev/null"), "w")
env = {k: v for k, v in os.environ.items() if k != "DBUS_SESSION_BUS_ADDRESS"}
if os.environ.get("FAKERUN"): env["XDG_RUNTIME_DIR"] = os.environ["FAKERUN"]
c = pexpect.spawn("openclaw", ["models", "auth", "login", "--provider", "pooled", "--set-default"], encoding="utf-8", timeout=180, dimensions=(50, 200), env=env)
c.logfile_read = out
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
seen = []
def wait(pat, timeout=180):
    c.expect(pat, timeout=timeout); s = ANSI.sub("", c.before + c.after); seen.append(s); return s
DOWN = "\x1b[B"
wait("How should this device take part")
if mode == "host":
    c.send("\r")                                             # Start a room on this device
    wait("How much of its memory"); c.send("\x15"); c.send("4\r")
    wait("Which model"); c.send("\r")                        # the recommended one
    i = c.expect(["not downloaded yet", "Wait for how many devices"])
    if i == 0: c.send(DOWN + DOWN + "\r"); wait("Wait for how many devices")   # Don't download
    c.send("\r")
    wait("Who can join"); c.send("\r")                       # devices with the invite link
else:
    c.send(DOWN + "\r")                                      # Join a room
    wait("Paste the room"); c.send(room + "\r")
    wait("How much of its memory"); c.send("\x15"); c.send("4\r")
c.expect(pexpect.EOF, timeout=300)
seen.append(ANSI.sub("", c.before))
print("\n".join(seen)[-5000:])
