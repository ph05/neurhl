"""NeurHL EDA-10 — does a player -> team -> game hierarchy beat team-level form?

The load-bearing claim of a hierarchical design is that aggregating PLAYER
performance up to a team is better than measuring the TEAM directly, because
it is roster-aware and forward-looking: it updates the moment a lineup changes
(injury, trade, call-up, goalie switch), whereas team form is backward-looking
and goes stale.

This tests that claim with the simplest possible Layer 1 (each player's own
pre-game EWMA on-ice shot rates and ice time) so the result reflects the
ARCHITECTURE rather than any particular network. If roster aggregation adds
nothing here, a neural Layer 1 will not rescue it; if it does add, a neural
Layer 1 has something real to improve on.

Feature built per game/side:
  proj_cf% = sum_i (ewma_toi_i * ewma_cf%_i) / sum_i ewma_toi_i
over the players actually dressed, all EWMAs strictly pre-game (P1-clean).

Compared against: Elo alone, Elo + backward-looking team shot form, and both.
Report-only; runs on tune seasons. Writes eda/eda_10_hierarchy_probe.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, TENSORS  # noqa: E402
from train.train_game import (SHOT_COLS, elo_features,  # noqa: E402
                              load_frames, shot_form)

ALPHA = 0.1
TRAIN_MAX, TEST = 2014, [2015, 2016, 2017]
CS = (0.01, 0.03, 0.1, 0.3, 1.0)


def player_projection(seasons) -> pd.DataFrame:
    """Layer 1 (simplest form) + Layer 2 aggregation -> projected team CF%."""
    rates = pd.concat(
        [pd.read_parquet(TENSORS / f"onice_rates_{se}.parquet").assign(season_end=se)
         for se in seasons], ignore_index=True)
    pg = pd.concat(
        [pd.read_parquet(TENSORS / f"player_games_{se}.parquet",
                         columns=["game_id", "player_id", "is_home",
                                  "pos_group", "toi_sec"]).assign(season_end=se)
         for se in seasons], ignore_index=True)
    gc, _ = load_frames(seasons)
    dates = gc.set_index("game_id").date.astype(str)
    d = pg.merge(rates, on=["game_id", "player_id", "is_home"], how="left")
    d[["cf", "ca"]] = d[["cf", "ca"]].fillna(0)
    d["date"] = d.game_id.map(dates)
    d = d[d.pos_group < 2].sort_values(["player_id", "date", "game_id"])
    g = d.groupby("player_id")
    # strictly pre-game EWMAs (shifted): this is Layer 1
    d["e_cf"] = g.cf.transform(lambda s: s.ewm(alpha=ALPHA).mean().shift(1))
    d["e_ca"] = g.ca.transform(lambda s: s.ewm(alpha=ALPHA).mean().shift(1))
    d["e_toi"] = g.toi_sec.transform(lambda s: s.ewm(alpha=ALPHA).mean().shift(1))
    d = d.dropna(subset=["e_cf", "e_ca", "e_toi"])
    d["e_cfpct"] = d.e_cf / (d.e_cf + d.e_ca).clip(lower=1e-6)
    # Layer 2: TOI-weighted aggregation over the players actually dressed
    d["w"] = d.e_toi
    d["wx"] = d.w * d.e_cfpct
    agg = d.groupby(["game_id", "is_home"])[["w", "wx"]].sum()
    agg["proj_cf"] = agg.wx / agg.w.clip(lower=1e-6)
    p = agg.reset_index().pivot(index="game_id", columns="is_home",
                                values="proj_cf")
    p.columns = ["proj_cf_a", "proj_cf_h"]
    return p


def main():
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    seasons = list(range(2008, max(TEST) + 1))
    gc, _ = load_frames(seasons)
    elo = elo_features(gc)
    sf = shot_form(gc)
    proj = player_projection(seasons)
    d = (gc.set_index("game_id").join(elo).join(sf).join(proj)
         .dropna(subset=["elo_logit", "proj_cf_h", "proj_cf_a"]))
    d["y"] = ((d.outcome4 == 0) | (d.outcome4 == 2)).astype(int)
    d["proj_diff"] = d.proj_cf_h - d.proj_cf_a
    tr = d[d.season_end <= TRAIN_MAX]
    te = d[d.season_end.isin(TEST)]

    def nll(p, y):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    yte = te.y.to_numpy()
    base = nll(1 / (1 + np.exp(-te.elo_logit.to_numpy())), yte)
    lines = ["# EDA-10 — does a player->team hierarchy beat team-level form?\n",
             f"Train <= {TRAIN_MAX} ({len(tr):,} games), test {TEST} "
             f"({len(te):,} games). Elo alone = {base.mean():.5f}.\n",
             "| features | log loss | vs Elo | t | p |", "|---|---|---|---|---|"]
    res = {}

    def run(cols, label):
        sc = StandardScaler().fit(tr[cols])
        best = (9.0, None, None)
        for C in CS:
            m = LogisticRegression(C=C, max_iter=4000).fit(
                sc.transform(tr[cols]), tr.y)
            l = nll(m.predict_proba(sc.transform(te[cols]))[:, 1], yte)
            if l.mean() < best[0]:
                best = (l.mean(), C, l)
        df = best[2] - base
        se = df.std(ddof=1) / np.sqrt(len(df))
        t = df.mean() / se
        p = stats.norm.sf(abs(t)) * 2
        res[label] = best[0]
        lines.append(f"| {label} | **{best[0]:.5f}** | {df.mean():+.5f} | "
                     f"{t:+.2f} | {p:.4f}{' **SIG**' if df.mean() < 0 and p < 0.05 else ''} |")

    E = ["elo_logit"]
    S = [c for c in SHOT_COLS if c.endswith("25")]
    run(E, "Elo (refit)")
    run(E + S, "Elo + team shot form (backward-looking)")
    run(E + ["proj_diff"], "Elo + ROSTER-PROJECTED shot share")
    run(E + S + ["proj_diff"], "Elo + team form + roster projection")
    run(["proj_diff"], "roster projection alone")

    lines.append(f"""
## Reading

The hierarchy's claim is that roster aggregation beats measuring the team
directly. Elo + roster projection = {res['Elo + ROSTER-PROJECTED shot share']:.5f} vs
Elo + team form = {res['Elo + team shot form (backward-looking)']:.5f}
(Elo alone {base.mean():.5f}).

Layer 1 here is deliberately trivial — each player's own shifted EWMA. Any
advantage is therefore attributable to the ARCHITECTURE (roster-aware,
forward-looking aggregation), not to model capacity, and is a lower bound on
what a trained neural Layer 1 could achieve.
""")
    (EDA / "eda_10_hierarchy_probe.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
