#!/usr/bin/env python3
# What killed the iPhone's Safari tab, and at what footprint (issue #207). Runs on the Mac the phone is
# attached to, next to memprobe_phone.mjs:
#   pymobiledevice3 syslog live | grep --line-buffered -iE 'memorystatus|footprint|Gigacage|Memory Limits' > syslog.txt
#   pymobiledevice3 crash pull crashes/
#   python3 memprobe_jetsam.py syslog.txt crashes/ [since, e.g. 2026-09-29-12]
# Prints every WebKit kill (process, footprint, reason), the soft limits iOS set, and the per-process
# footprint / JS ArrayBuffer (Gigacage) lines WebKit logs at each memory-pressure warning.
# What it showed (2026-09-29, iPhone 14 Pro Max, iOS 26.6.1): WebContent gets ActiveSoft 1536 MB and is
# killed once over it and the system runs low (kills seen at 1.50-3.25 GB); WebKit.Networking gets 840 MB
# and its kill closes the page (fetch bodies nobody reads pile up there).
import glob, json, os, re, sys

def syslog(path):
    for line in open(path, errors="replace"):
        t = line[11:26]
        m = re.search(r"killing_\w+ pid (\d+) \[([\w.]+)\] \((\S+).*?\) (\d+)KB", line)
        if m and "WebKit" in m.group(2):
            print(f"{t} KILLED {m.group(2)} pid {m.group(1)} at {int(m.group(4)) // 1024} MB ({m.group(3)})")
            continue
        m = re.search(r"\[([\w.]+)\] \[(\d+)\] exceeded mem limit: (\w+ \d+ MB)", line) or re.search(r"([\w.]+) \[(\d+)\] exceeded mem limit: (\w+ \d+ MB)", line)
        if m and "WebKit" in m.group(1):
            print(f"{t} over {m.group(3)}: {m.group(1)} pid {m.group(2)}")
            continue
        m = re.search(r"WebContent\[(\d+)\]\s+(phys_footprint_mb: \d+|Gigacage: \d+ MB)", line)
        if m: print(f"{t}   pid {m.group(1)} {m.group(2)}")

def reports(d, since=""):
    for f in sorted(glob.glob(os.path.join(d, "JetsamEvent*.ips"))):
        if os.path.basename(f) < "JetsamEvent-" + since: continue
        head, body = open(f).read().split("\n", 1)
        b = json.loads(body); pg = b.get("memoryStatus", {}).get("pageSize", 16384)
        print(os.path.basename(f), "largest:", b.get("largestProcess"))
        for p in b["processes"]:
            if p.get("reason") and any(k in p.get("name", "") for k in ("WebKit", "Safari")):
                print(f"   KILLED {p['name']} {p.get('rpages', 0) * pg >> 20} MB ({p['reason']}, lifetime max {p.get('lifetimeMax', 0) * pg >> 20} MB)")

if __name__ == "__main__":
    for a in sys.argv[1:]:
        if os.path.isdir(a): reports(a, sys.argv[-1] if not os.path.exists(sys.argv[-1]) else "")
        elif os.path.isfile(a): syslog(a)
