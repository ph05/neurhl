"""NeurHL read-only verification battery (mirrors repo-root review_tests.py).

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/tests/review_tests_neurhl.py

Sections appear as their artifacts come into existence; a missing artifact is a
SKIP (printed), never a silent pass. Nothing here retunes or rewrites anything.
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, PROC, PROJ, RAW, TENSORS  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))


def skip(name: str, why: str):
    print(f"[SKIP] {name}  ({why})")


def section_1_isolation():
    print("\n== S1 isolation: src/ never imports neurhl ==")
    r = subprocess.run(["grep", "-rl", "neurhl", str(PROJ / "src")],
                       capture_output=True, text=True)
    check("no src/ file references neurhl", r.stdout.strip() == "",
          r.stdout.strip()[:120])


def section_2_tensors():
    print("\n== S2 event/game tensors ==")
    seasons = list(range(2012, 2027))
    missing = [s for s in seasons if not (TENSORS / f"events_{s}.parquet").exists()]
    if missing:
        skip("event shards", f"missing {missing}")
        return
    tot = 0
    for s in (2013, 2017, 2021, 2026):
        ev = pd.read_parquet(TENSORS / f"events_{s}.parquet",
                             columns=["game_id", "game_type", "event_type",
                                      "h_on1", "a_on1", "home_event", "period"])
        tot += len(ev)
        reg = ev[(ev.game_type == 2) & (ev.home_event >= 0) & (ev.period < 5)]
        on = ((reg.h_on1 > 0) & (reg.a_on1 > 0)).mean()
        check(f"{s}: on-ice slots populated (owned reg events)", on > 0.985,
              f"{on:.4f}")
        po = ev[(ev.game_type == 3) & (ev.home_event >= 0)]
        if len(po):
            onp = ((po.h_on1 > 0) & (po.a_on1 > 0)).mean()
            check(f"{s}: playoff on-ice slots populated", onp > 0.98, f"{onp:.4f}")
    check("excluded degenerate game absent",
          2012020660 not in set(pd.read_parquet(
              TENSORS / "events_2012.parquet", columns=["game_id"]).game_id.unique()))
    check("spot-check shard volume sane", tot > 1_300_000, f"{tot:,}")


def section_3_labels():
    print("\n== S3 label integrity (EDA-06 graduated) ==")
    cache = TENSORS / "_edacache" / "game_summary.parquet"
    if not cache.exists():
        skip("label checks", "scan cache missing")
        return
    gs = pd.read_parquet(cache)
    games = pd.read_csv(PROC / "games.csv")
    gs["home_m"] = gs.home.replace({"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"})
    gs["away_m"] = gs.away.replace({"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"})
    m = gs.merge(games[games.season_end >= 2012],
                 left_on=["date", "home_m", "away_m"],
                 right_on=["date", "home", "away"], how="left",
                 suffixes=("", "_hr"))
    check("PBP<->games.csv join complete", m.home_g.notna().all(),
          f"{m.home_g.isna().sum()} unmatched")
    ok = m.dropna(subset=["home_g", "home_score"])
    mism = ((ok.home_score != ok.home_g) | (ok.away_score != ok.away_g)).sum()
    check("final scores agree", mism == 0, f"{mism} mismatches")
    reg = ok[ok.game_type == 2]
    extra = reg.last_period_type.isin(["OT", "SO"])
    dis = (extra != (reg.went_ot.astype(bool) | reg.went_so.astype(bool))).sum()
    check("OT/SO flags agree", dis == 0, f"{dis} disagreements")


def section_4_vocab():
    print("\n== S4 vocab determinism & monotonicity ==")
    sizes = []
    for v in range(2013, 2028):
        p = TENSORS / f"vocab_{v}.json"
        if not p.exists():
            skip("vocab", f"vocab_{v}.json missing")
            return
        sizes.append(len(json.loads(p.read_text())))
    check("vocab sizes strictly increasing", all(a < b for a, b in
                                                 zip(sizes, sizes[1:])), str(sizes))
    v27 = json.loads((TENSORS / "vocab_2027.json").read_text())
    check("vocab indices contiguous 1..N",
          sorted(v27.values()) == list(range(1, len(v27) + 1)))


def section_5_baselines():
    print("\n== S5 tune-window baseline artifacts ==")
    p = NOUT / "baselines_tune.json"
    if not p.exists():
        skip("baselines_tune.json", "not yet computed")
        return
    b = json.loads(p.read_text())
    ll = b["game_logloss"]
    check("v1 tune logloss in sane band",
          0.66 < ll["pooled_2012_2017"] < 0.69, f"{ll['pooled_2012_2017']:.5f}")
    check("v1 beats constant-home on tune",
          ll["pooled_2012_2017"] < ll["constant_home"])
    s = b["season_h1_mean_2012_2017"]
    check("v1 season MAE beats uniform & regressed",
          s["v1_mae82"] < s["regressed_mae82"] < s["uniform_mae82"],
          f"{s['v1_mae82']:.2f} < {s['regressed_mae82']:.2f} < {s['uniform_mae82']:.2f}")
    if b.get("tier0"):
        t = b["tier0"]["logloss_pooled_2014_2017"]
        check("tier0 tune logloss recorded & sane", 0.6 < t < 0.72, f"{t:.5f}")
    else:
        skip("tier0", "not yet computed")


def section_6_gates():
    print("\n== S6 gate records vs prereg rules ==")
    p = NOUT / "params_neurhl.json"
    if not p.exists():
        skip("params_neurhl.json", "gates not yet run")
        return
    blob = json.loads(p.read_text())
    for gname, rec in blob.get("gates", {}).items():
        check(f"{gname} decision matches recorded rule",
              rec.get("pass") == rec.get("rule_eval"), str(rec)[:100])


def main():
    for fn in (section_1_isolation, section_2_tensors, section_3_labels,
               section_4_vocab, section_5_baselines, section_6_gates):
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
