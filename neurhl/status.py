"""NeurHL live pipeline status — super lightweight, stdlib only.

Run in its own terminal:   python3 neurhl/status.py        (refreshes every 5s)
One-shot snapshot:         python3 neurhl/status.py --once

Reads only artifacts and logs (never touches training): checkpoint inventory,
neurhl/data/tensors/_logs/*.log, params_neurhl.json, prediction files, and the
Mac power source. Zero dependencies, zero load.
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

NRL = Path(__file__).resolve().parent
CKPT = NRL / "checkpoints"
LOGS = NRL / "data" / "tensors" / "_logs"
NOUT = NRL / "output"
TUNE_V = [2012, 2013, 2014, 2015, 2016, 2017]
RESTATE_V = list(range(2018, 2028))
EPOCH_RE = re.compile(r"epoch (\d+): train ([\d.]+)\s+val\((\d+)\) ([\d.]+) \[(\d+)s\]")
GAME_RE = re.compile(r"seed (\d) ep (\d+): val out4 ll ([\d.]+)")


def bar(done, total, width=18):
    n = int(width * done / max(total, 1))
    return "█" * n + "░" * (width - n) + f" {done}/{total}"


def power():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=3).stdout
        if "AC Power" in out:
            return "AC ✓"
        m = re.search(r"(\d+)%", out)
        return f"BATTERY {m.group(1) if m else '?'}% ⚠ (throttled — plug in!)"
    except Exception:
        return "?"


def newest_log():
    logs = sorted(LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def count(pattern, root=CKPT):
    return len(list(root.glob(pattern)))


def gates():
    p = NOUT / "params_neurhl.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("gates", {})


def render():
    lines = []
    now = datetime.now().strftime("%H:%M:%S")
    lines.append(f"NeurHL pipeline — {now}   power: {power()}")
    lines.append("─" * 62)

    snaps = count("event_lm_v*.json")
    snaps_r = sum((CKPT / f"event_lm_v{v}.json").exists() for v in RESTATE_V)
    careers = count("career_v*.json")
    embs = count("embeddings_v*.npz", NRL / "data" / "tensors")
    gseeds = count("game_T*_s*.pt")
    g = gates()
    restated = False
    proj = (NOUT / "projections_2026_27_neurhl.csv").exists()
    if (NOUT / "params_neurhl.json").exists():
        restated = "restatement" in json.loads(
            (NOUT / "params_neurhl.json").read_text())

    lines.append(f" 1 pretrain snapshots (tune)   {bar(snaps - snaps_r, 6)}")
    lines.append(f" 2 careers + embeddings        {bar(min(careers, embs), 16)}")
    lines.append(f" 3 game models (seed-runs)     {bar(gseeds, 30 if snaps_r == 0 else 80)}")
    gs = ("pending" if not g else "  ".join(
        f"{k}:{'PASS' if v.get('pass') else 'fail'}" for k, v in sorted(g.items())))
    lines.append(f" 4 gates                       {gs}")
    lines.append(f" 5 restatement 2018-2026       {'DONE' if restated else ('snapshots ' + bar(snaps_r, 10) if snaps >= 6 else 'waiting for gates')}")
    lines.append(f" 6 projection 2026-27          {'DONE ✓' if proj else 'pending'}")
    lines.append("─" * 62)

    log = newest_log()
    if log:
        age = time.time() - log.stat().st_mtime
        txt = log.read_text()[-6000:]
        epochs = EPOCH_RE.findall(txt)
        gm = GAME_RE.findall(txt)
        tail = [l for l in txt.strip().splitlines() if l.strip()][-1]
        lines.append(f" active log: {log.name}  ({int(age)}s ago"
                     + (")" if age < 900 else "  ⚠ STALLED?)"))
        if epochs and (not gm or txt.rfind("epoch") > txt.rfind("seed")):
            ep, tr, vs, val, secs = epochs[-1]
            pace = [int(e[4]) for e in epochs[-3:]]
            trend = float(epochs[0][3]) - float(val) if len(epochs) > 1 else 0
            lines.append(f" pretrain: epoch {ep}  val({vs}) {val}  "
                         f"~{sum(pace)//len(pace)}s/epoch  "
                         f"(improved {trend:.2f} this run)")
        elif gm:
            s, ep, ll = gm[-1]
            lines.append(f" game model: seed {s} epoch {ep}  val ll {ll}")
        lines.append(f" last: {tail[:70]}")
    else:
        lines.append(" no active log found")
    return "\n".join(lines)


def main():
    if "--once" in sys.argv:
        print(render())
        return
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H" + render() + "\n")
            sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
