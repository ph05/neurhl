"""Walk-forward rookie features for the engine tensors (PLAN_NeurHL_1_1 A15, candidate g1rk).

For every season V (2009-2027) and every skater with fewer than 20 NHL games
before V who plays in V: his translated goals and assists per game from the two
seasons before V (eval/rookie_priors.py; league factors and shrinkage fitted
only on moves and rookies before V). Writes data/tensors/rookie_features.parquet:
  player_id, season_end, rk_pred_g, rk_pred_a, rk_pre_gp
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import rookie_priors as RP  # noqa: E402
from common import TENSORS  # noqa: E402


def main():
    d = RP.load()
    rk = RP.rookies(d)
    nhl = d[d.lg == "NHL"].sort_values(["player_id", "season_end"])
    nhl["prior_gp"] = nhl.groupby("player_id").gp.cumsum() - nhl.gp
    out = []
    for V in range(2009, 2028):
        cand = nhl[(nhl.season_end == V) & (nhl.prior_gp < 20)][["player_id", "season_end", "gp", "g", "a", "age", "pos", "pick"]]
        if not len(cand):
            continue
        f = RP.factors(d, V)
        tr = rk[rk.season_end < V]
        if len(tr) < 30:
            continue
        tr = tr.merge(RP.translated(d, tr[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        te = cand.merge(RP.translated(d, cand[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        pr = RP.fit_predict(tr, te)
        out.append(pr[["player_id", "season_end", "pred_g", "pred_a", "pre_gp"]].rename(
            columns={"pred_g": "rk_pred_g", "pred_a": "rk_pred_a", "pre_gp": "rk_pre_gp"}))
        print(V, len(pr), flush=True)
    t = pd.concat(out, ignore_index=True)
    t.to_parquet(TENSORS / "rookie_features.parquet", index=False)
    print(f"{len(t)} rookie-seasons -> rookie_features.parquet")


if __name__ == "__main__":
    main()
