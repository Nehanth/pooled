# Drive OpenClaw's interactive "models auth login --provider pooled" in a pty, as a user would:
#   python3 onboard_pty.py host|join [CODE]   (env from ocenv.sh); prints the screen text it saw.
import os, re, sys, pexpect
mode = sys.argv[1]; code = sys.argv[2] if len(sys.argv) > 2 else ""
out = open(os.environ.get("PTY_LOG", "/dev/null"), "w")
c = pexpect.spawn("openclaw", ["models", "auth", "login", "--provider", "pooled", "--set-default"], encoding="utf-8", timeout=120, dimensions=(50, 160), env={**{k: v for k, v in os.environ.items() if k != "DBUS_SESSION_BUS_ADDRESS"}, "XDG_RUNTIME_DIR": os.environ["FAKERUN"]})
c.logfile_read = out
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
def wait(pat):
    c.expect(pat); return ANSI.sub("", c.before + c.after)
DOWN = "\x1b[B"
wait("How should this device take part")
if mode == "host":
    c.send("\r")                                  # Start a room on this device
    wait("Which model"); c.send("\r")            # Qwen3 1.7B (first)
    wait("GPU memory this device gives"); c.send("\x15"); c.send("4\r")
    wait("Wait for how many devices"); c.send("\r")   # initial: two
else:
    c.send(DOWN + "\r")                           # Join a room
    wait("Room code"); c.send(code + "\r")
    wait("GPU memory this device gives"); c.send("\x15"); c.send("4\r")
c.expect(pexpect.EOF, timeout=120)
print(ANSI.sub("", c.before)[-3000:])
