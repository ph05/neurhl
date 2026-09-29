"""All-player non-NHL priors for the engine tensors (candidate g1x, PLAN_NeurHL_1_1 A19).

For every season V and every skater who plays in V: his goals and assists per
game in non-NHL leagues over V-1 (weight 2) and V-2 (weight 1), translated to
NHL terms with league factors fitted only on moves into NHL seasons before V
(eval/rookie_priors.py), plus the non-NHL games behind them. International
events are pooled into one low-weight "EVENT" league (factor at half the pooled
value). Players with no non-NHL games in those two seasons get no row.
Writes data/tensors/prior_features.parquet:
  player_id, season_end, px_g, px_a, px_gp
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import rookie_priors as RP  # noqa: E402
from common import TENSORS  # noqa: E402


def load_all() -> pd.DataFrame:
    d = pd.read_parquet(TENSORS / "prenhl_seasons.parquet")
    d["lg"] = d.league.map(RP.norm_league)            # keeps "EVENT" rows this time
    d = d[d.pos_group < 2]
    return d.groupby(["player_id", "season_end", "lg"], as_index=False).agg(
        gp=("gp", "sum"), g=("g", "sum"), a=("a", "sum"), age=("age", "first"),
        pos=("pos_group", "first"), pick=("draft_overall", "first"))


def main():
    d = load_all()
    nhl_players = d[d.lg == "NHL"][["player_id", "season_end"]].drop_duplicates()
    other = d[d.lg != "NHL"]
    out = []
    for V in range(2009, 2028):
        f = RP.factors(d[d.lg != "EVENT"], V)
        players = nhl_players[nhl_players.season_end == V].player_id if V < 2027 else pd.Series(dtype=int)
        o = other[(other.season_end >= V - 2) & (other.season_end <= V - 1)]
        if V < 2027:
            o = o[o.player_id.isin(players)]
        if not len(o):
            continue
        w = o.gp * np.where(o.season_end == V - 1, 2.0, 1.0)
        fg = [f.get((l, p, "g"), f[("ALL", p, "g")] * (0.5 if l in ("OTHER", "EVENT") else 1.0)) for l, p in zip(o.lg, o.pos)]
        fa = [f.get((l, p, "a"), f[("ALL", p, "a")] * (0.5 if l in ("OTHER", "EVENT") else 1.0)) for l, p in zip(o.lg, o.pos)]
        o = o.assign(tg=np.array(fg) * o.g / o.gp, ta=np.array(fa) * o.a / o.gp, w=w)
        agg = o.groupby("player_id").apply(lambda x: pd.Series({
            "px_g": np.average(x.tg, weights=x.w), "px_a": np.average(x.ta, weights=x.w), "px_gp": x.gp.sum()}),
            include_groups=False).reset_index()
        agg["season_end"] = V
        out.append(agg)
        print(V, len(agg), flush=True)
    t = pd.concat(out, ignore_index=True)
    t.to_parquet(TENSORS / "prior_features.parquet", index=False)
    print(f"{len(t)} player-seasons -> prior_features.parquet")


if __name__ == "__main__":
    main()
