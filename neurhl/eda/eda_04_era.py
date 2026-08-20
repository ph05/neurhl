"""NeurHL EDA-04 — the era-drift dossier (PLAN_NeurHL Phase 0, parameterizes P5).

Quantifies how the NHL "meta" moved 2006->2026 so the model conditions on era
instead of averaging across it: scoring, special teams, pace, home ice, OT/SO
structure, parity, and scorer-recorded soft events (hits/blocks/give/take) whose
recording standards drift. Sources: scan_pbp cache (2012+), games.csv and
team_seasons.csv (2006+). Writes eda/eda_04_era.md + figs.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, PROC, TENSORS  # noqa: E402

CACHE = TENSORS / "_edacache"
FIGS = EDA / "figs"


def main():
    gs = pd.read_parquet(CACHE / "game_summary.parquet")
    ev = pd.read_parquet(CACHE / "events_summary.parquet")
    games = pd.read_csv(PROC / "games.csv")
    ts = pd.read_csv(PROC / "team_seasons.csv")
    lines = ["# EDA-04 — era drift dossier (parameterizes P5 era covariates)\n"]

    # ---- 2006+ from games.csv: scoring, home ice, OT share
    g = games[games.game_type == "R"].copy()
    era = g.groupby("season_end").apply(lambda d: pd.Series({
        "gpg": (d.home_g + d.away_g).mean(),
        "home_win": (d.home_g > d.away_g).mean(),
        "reg_home_win": ((d.home_g > d.away_g) & ~d.went_ot.astype(bool)).mean()
        / max((~d.went_ot.astype(bool)).mean(), 1e-9),
        "ot_share": d.went_ot.astype(bool).mean(),
        "so_share": d.went_so.astype(bool).mean() if "went_so" in d else float("nan"),
        "margin_abs": d.margin.abs().mean() if "margin" in d else float("nan"),
    }), include_groups=False)
    lines.append("## League environment by season (games.csv, regular season, 2006+)\n")
    lines.append(era.round(3).to_markdown() + "\n")

    # ---- parity: SD of points pct across teams
    parity = ts.groupby("season_end").pts_pct.std().rename("pts_pct_sd")
    lines.append("## Parity (SD of team points%, higher = more spread)\n")
    lines.append(parity.round(4).to_frame().to_markdown() + "\n")
    lines.append("\nNote: 2025-26 is the flagged parity season where uniform beat "
                 "all skill models in the house restatement.\n")

    # ---- scorer-drift events from PBP (2012+)
    reg = gs[gs.game_type == 2]
    n_games = reg.groupby("season_end").size()
    evr = ev[ev.game_type == 2].pivot_table(index="season_end", columns="event_type",
                                            values="n", aggfunc="sum").fillna(0)
    soft = ["hit", "blocked-shot", "giveaway", "takeaway", "penalty", "faceoff"]
    soft = [s for s in soft if s in evr.columns]
    per_game = evr[soft].div(n_games, axis=0)
    per_game["events_total"] = evr.sum(axis=1) / n_games
    lines.append("## Scorer-recorded event rates per game (PBP, 2012+) — recording drift\n")
    lines.append(per_game.round(2).to_markdown() + "\n")
    cv = (per_game.std() / per_game.mean()).sort_values(ascending=False)
    lines.append("\nCoefficient of variation across seasons (drift magnitude):\n")
    lines.append(cv.round(3).to_frame("cv").to_markdown() + "\n")

    # ---- proposed era covariate vector (locked shape for P5)
    lines.append("""## Proposed era_vec (8 dims, all walk-forward computable — P5)

| dim | covariate | preseason value | in-season value |
|---|---|---|---|
| 1 | league GPG | prior-season mean | expanding to-date mean |
| 2 | penalties/game | prior-season | expanding to-date |
| 3 | OT share (went_ot) | prior-season | expanding to-date |
| 4 | 3v3-OT era flag (season_end >= 2016) | static rule | static rule |
| 5 | shootout-exists flag (>= 2006) / loser point | static rule | static rule |
| 6 | COVID flag (2020, 2021) | static rule | static rule |
| 7 | scaled season index (season_end - 2006)/20 | static | static |
| 8 | parity: prior-season pts_pct SD | prior-season | prior-season |
""")

    # ---- figure
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    era.gpg.plot(ax=axes[0, 0], marker="o", ms=3, title="goals/game")
    era.home_win.plot(ax=axes[0, 1], marker="o", ms=3, title="home win %")
    era.ot_share.plot(ax=axes[0, 2], marker="o", ms=3, title="OT share")
    parity.plot(ax=axes[1, 0], marker="o", ms=3, title="parity (pts% SD)")
    if "hit" in per_game:
        per_game["hit"].plot(ax=axes[1, 1], marker="o", ms=3, title="hits/game (scorer drift)")
    if "penalty" in per_game:
        per_game["penalty"].plot(ax=axes[1, 2], marker="o", ms=3, title="penalties/game")
    for ax in axes.flat:
        ax.axvline(2016, color="gray", lw=0.6, ls="--")   # 3v3 era
        ax.axvspan(2020, 2021, color="orange", alpha=0.15)  # COVID
    fig.suptitle("Era drift 2006-2026 (dashed = 3v3 OT era start; shaded = COVID)")
    fig.tight_layout(); fig.savefig(FIGS / "eda_04_era_drift.png", dpi=110); plt.close(fig)

    (EDA / "eda_04_era.md").write_text("\n".join(lines))
    print("wrote eda_04_era.md")


if __name__ == "__main__":
    main()
