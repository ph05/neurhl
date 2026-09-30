"""2026-27 rookie priors (PLAN_NeurHL_1_1 A15): for every skater on the
post-deadline rosters or in the call-up pool with fewer than 20 NHL regular-
season games before 2026-27, his expected goals and assists per game in a first
NHL season, translated from his last two seasons in other leagues
(eval/rookie_priors.py; league factors and shrinkage fitted on all moves and
rookies up to 2025-26).

Writes neurhl/configs/rookie_priors_2027.csv:
  player_id, pos, age, pre_gp, tg, ta, pred_g, pred_a, base_g, base_a, name, team
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import rookie_priors as RP  # noqa: E402

V = 2027
OUT = ROOT / "configs" / "rookie_priors_2027.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters-date", required=True, help="the season set's roster snapshot (YYYY-MM-DD)")
    a = ap.parse_args()
    import sim.availability_2027 as AV
    d = RP.load()
    rday, rost = AV._live("fetch_rosters").latest_rosters(a.rosters_date)
    # every skater the season set can dress: rosters, injured and reserves (PLAN_NeurHL_1_2 R4)
    R = AV.load(a.rosters_date)
    av = pd.concat([R[t].skaters[["player_id", "name", "team"]] for t in R], ignore_index=True)
    ids = set(rost.player_id) | set(av.player_id)
    nhl_gp = d[(d.lg == "NHL") & (d.season_end < V)].groupby("player_id").gp.sum()
    bio = d.groupby("player_id").agg(pos=("pos", "first"), pick=("pick", "first"))
    birth = pd.read_parquet(ROOT / "data" / "tensors" / "prenhl_seasons.parquet").groupby("player_id").birth_year.first()
    cand = [p for p in ids if nhl_gp.get(p, 0) < 20 and p in bio.index]
    te = pd.DataFrame({"player_id": cand, "season_end": V})
    te["pos"] = te.player_id.map(bio.pos)
    te["pick"] = te.player_id.map(bio.pick)
    te["age"] = V - te.player_id.map(birth)
    te["gp"], te["g"], te["a"] = 1, 0, 0                                  # unused placeholders
    rk = RP.rookies(d)
    f = RP.factors(d, V)
    tr = rk[rk.season_end < V].merge(RP.translated(d, rk[rk.season_end < V][["player_id", "season_end"]], f),
                                      on=["player_id", "season_end"])
    te = te.merge(RP.translated(d, te[["player_id", "season_end"]], f), on=["player_id", "season_end"])
    pr = RP.fit_predict(tr, te)
    nm = dict(zip(rost.player_id, rost["first"] + " " + rost["last"]))
    tm = dict(zip(rost.player_id, rost.team))
    for p, n, t in zip(av.player_id, av.name, av.team):
        nm.setdefault(p, n)
        tm.setdefault(p, t)
    pr["name"] = pr.player_id.map(nm)
    pr["team"] = pr.player_id.map(tm)
    pr = pr[["player_id", "pos", "age", "pre_gp", "tg", "ta", "pred_g", "pred_a", "base_g", "base_a", "name", "team"]]
    pr.sort_values("pred_g", ascending=False).to_csv(OUT, index=False, float_format="%.4f")
    print(f"{len(pr)} rookies (rosters {rday}) -> {OUT}")
    print(pr.assign(pts82=82 * (pr.pred_g + pr.pred_a)).sort_values("pts82", ascending=False)
          .head(15)[["name", "team", "age", "pre_gp", "pred_g", "pred_a", "pts82"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
