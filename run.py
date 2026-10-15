#!/usr/bin/env python
"""Single entry point.

    python run.py demo [--sim] [--port COM5]   dashboard (real device by default, --sim = backup demo)
    python run.py record [--port COM5]         dashboard opened on the RECORDER tab
    python run.py train [--source synthetic|real|mixed] [--model rf|svm]
    python run.py gen-data                     regenerate the SYNTHETIC dataset
    python run.py monitor [--port COM5]        text-mode live check (is the signal alive?)
    python run.py ports                        list serial ports
    python run.py test                         run the test-suite
"""
import argparse
import os
import sys
import threading
import time
import webbrowser


def cmd_serve(args, tab: str = "live"):
    import uvicorn
    from spine.config import CFG
    os.environ["SPINE_MODE"] = "sim" if args.sim else "serial"
    os.environ["SPINE_PORT"] = args.port or "auto"
    host, port = CFG["server"]["host"], args.http_port or CFG["server"]["port"]
    url = f"http://{host}:{port}/#{tab}"
    print(f"\n  Mode: {'SIMULATED (no hardware)' if args.sim else 'REAL DEVICE on ' + (args.port or 'auto-detected port')}")
    print(f"  Dashboard: {url}\n")
    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("spine.server.app:create_app", factory=True, host=host, port=port, log_level="warning")


def cmd_train(args):
    from spine.ml.train import train
    train(args.source, args.model)


def cmd_gen(args):
    from spine.dataset import generate_synthetic
    generate_synthetic(args.per_class)


def cmd_ports(_):
    from spine.transports.serial_transport import list_serial_ports
    print("\n".join(list_serial_ports()) or "no serial ports found")


def cmd_monitor(args):
    """Terminal view of live amplitude + taps. Use this to verify wiring before anything else."""
    from spine.coordinator import Coordinator, Listener
    from spine.transports.serial_transport import SerialTransport
    from spine.transports.sim_transport import SimTransport

    class Mon(Listener):
        def on_live(self, live):
            s = live["samples"]
            a0, a1 = abs(s[0]).max(), abs(s[1]).max()
            bar = lambda v: "#" * int(min(v, 1000) / 1000 * 40)
            print(f"mic_0 {a0:5.0f} {bar(a0):<40} | mic_1 {a1:5.0f} {bar(a1)}", flush=True)

        def on_data(self, ev):
            print(f"\n>>> TAP  {ev.final_material} / {ev.final_zone}  conf {ev.material_conf:.2f}  "
                  f"peak {abs(ev.ch).max():.0f}\n", flush=True)

        def on_service_message(self, m):
            print("node:", m["code_name"], m["text"])

    t = SimTransport() if args.sim else SerialTransport(args.port or "auto")
    c = Coordinator(t)
    c.add_listener(Mon())
    c.start()
    info = c.connect_and_start(live=True)
    if not info:
        sys.exit("No node answered. Check the port / flash / baud (921600).")
    print("node:", info)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        c.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sim", action="store_true", help="use the SIMULATED virtual node")
        p.add_argument("--port", help="serial port, e.g. COM5 (default: auto-detect)")
        p.add_argument("--http-port", type=int)
        p.add_argument("--no-browser", action="store_true")

    common(sub.add_parser("demo"))
    common(sub.add_parser("record"))
    common(sub.add_parser("monitor"))
    t = sub.add_parser("train")
    t.add_argument("--source", default="synthetic", choices=["synthetic", "real", "mixed"])
    t.add_argument("--model", default=None, choices=["rf", "svm"])
    g = sub.add_parser("gen-data")
    g.add_argument("--per-class", type=int, default=None)
    sub.add_parser("ports")
    sub.add_parser("test")
    a = ap.parse_args()
    if a.cmd == "demo":
        cmd_serve(a, "live")
    elif a.cmd == "record":
        cmd_serve(a, "rec")
    elif a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "gen-data":
        cmd_gen(a)
    elif a.cmd == "ports":
        cmd_ports(a)
    elif a.cmd == "monitor":
        cmd_monitor(a)
    elif a.cmd == "test":
        import pytest
        sys.exit(pytest.main(["-q", "tests"]))


if __name__ == "__main__":
    main()
