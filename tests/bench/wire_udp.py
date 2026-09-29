#!/usr/bin/env python3
"""Wire lab, below the browser: plain UDP bursts between two machines, to tell the network's share
of a hop from Chrome's (WebRTC, SCTP) share. No dependencies.

  on A (the side that accepts inbound UDP, e.g. the Linux box):  python3 wire_udp.py serve 47998
  on B:                                                          python3 wire_udp.py probe <A's ip> 47998

B sends `up` datagrams of 1200 bytes back to back (a frame cut into packets, as SCTP would); A answers
the last one with `down` datagrams. The round trip is timed from B's first send to B's last receive.
Sweeps: packets per frame each way (1, 2, 4) and the idle time between frames (2 ms .. 300 ms), which
is where Wi-Fi power save shows up.
"""
import socket
import sys
import time


def serve(port, idle_exit=120.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.settimeout(idle_exit)
    try:
        while True:
            d, a = s.recvfrom(65535)
            if len(d) >= 8 and d[6] == 1:   # last datagram of the burst: answer with d[7] datagrams
                for _ in range(max(1, d[7])):
                    s.sendto(d[:16] + b"y" * (1184 if d[7] > 1 else 0), a)
    except socket.timeout:
        pass


def q(a, f):
    a = sorted(a)
    return a[min(len(a) - 1, int(f * len(a)))] if a else float("nan")


def probe(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1.0)

    def run(up, down, gap, reps):
        rt, lost = [], 0
        for rep in range(reps):
            seq = rep.to_bytes(4, "big")
            t0 = time.perf_counter()
            for k in range(up):
                s.sendto(bytes([0]) + seq + bytes([k, 1 if k == up - 1 else 0, down]) + b"x" * 1192, (host, port))
            got = 0
            try:
                while got < down:
                    d, _ = s.recvfrom(65535)
                    if d[1:5] == seq:
                        got += 1
                rt.append((time.perf_counter() - t0) * 1000)
            except socket.timeout:
                lost += 1
            time.sleep(gap)
        return rt, lost

    for gap in (0.002, 0.01, 0.03, 0.1, 0.3):
        for up, down in ((1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (4, 4)):
            rt, lost = run(up, down, gap, max(40, min(200, int(8 / gap))))
            print(f"idle {gap * 1000:4.0f} ms  B->A {up} pkt  A->B {down} pkt  rtt p50 {q(rt, .5):.2f} p95 {q(rt, .95):.2f} "
                  f"p99 {q(rt, .99):.2f}  lost {lost}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "serve":
        serve(int(sys.argv[2]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "probe":
        probe(sys.argv[2], int(sys.argv[3]))
    else:
        print(__doc__)
