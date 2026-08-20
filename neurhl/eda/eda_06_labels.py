"""NeurHL EDA-06 — cross-source label integrity (PLAN_NeurHL Phase 0).

The training labels (final scores, OT/SO flags) must agree across independent
sources before tensorization: PBP-derived scores (scan_pbp cache) vs the
Hockey-Reference-derived games.csv spine, and shift-chart team TOI vs the
60-minute identity. Disagreements are enumerated, not averaged away. The checks
here graduate into tests/review_tests_neurhl.py. Writes eda/eda_06_labels.md.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, PROC, TENSORS  # noqa: E402

CACHE = TENSORS / "_edacache"
# games.csv is franchise-mapped; PBP abbrevs are era-actual
REMAP = {"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"}


def main():
    gs = pd.read_parquet(CACHE / "game_summary.parquet")
    games = pd.read_csv(PROC / "games.csv")
    lines = ["# EDA-06 — cross-source label integrity\n"]

    pbp = gs[gs.game_type.isin([2, 3])].copy()
    pbp["home_m"] = pbp.home.replace(REMAP)
    pbp["away_m"] = pbp.away.replace(REMAP)
    g = games[games.season_end >= 2012].copy()

    m = pbp.merge(g, left_on=["date", "home_m", "away_m"],
                  right_on=["date", "home", "away"], how="left",
                  suffixes=("", "_hr"))
    unmatched = m[m.home_g.isna()]
    # PBP gameDate can sit one day off HR local dates — retry +/- 1 day
    if len(unmatched):
        un = unmatched.copy()
        un["d"] = pd.to_datetime(un.date)
        g2 = g.copy(); g2["d"] = pd.to_datetime(g2.date)
        fixed = 0
        for shift in (1, -1):
            un["d_try"] = (un.d + pd.Timedelta(days=shift)).dt.strftime("%Y-%m-%d")
            j = un.merge(g2, left_on=["d_try", "home_m", "away_m"],
                         right_on=["date", "home", "away"], how="inner",
                         suffixes=("", "_j"))
            fixed += len(j)
        lines.append(f"Join: {len(m) - len(unmatched):,}/{len(pbp):,} exact "
                     f"(date,home,away) matches; {len(unmatched)} unmatched, of which "
                     f"{fixed} match at date±1 day (timezone recording).\n")
    else:
        lines.append(f"Join: {len(m):,}/{len(pbp):,} exact matches — full coverage.\n")

    ok = m.dropna(subset=["home_g"])
    ok = ok.dropna(subset=["home_score", "away_score"])
    score_mismatch = ok[(ok.home_score != ok.home_g) | (ok.away_score != ok.away_g)]
    lines.append(f"\n## Final-score agreement\n\nCompared: {len(ok):,} games. "
                 f"Score mismatches: **{len(score_mismatch)}** "
                 f"({len(score_mismatch) / max(len(ok), 1):.5%}).\n")
    if len(score_mismatch):
        lines.append(score_mismatch[["game_id", "date", "home", "away",
                                     "home_score", "home_g", "away_score", "away_g"]]
                     .head(20).to_markdown(index=False) + "\n")

    # OT/SO flags (regular season only; playoffs have no SO)
    # games.csv semantics: went_ot = ended IN OT, went_so = ended in SO
    # (mutually exclusive); PBP lastPeriodType gives the same partition.
    reg = ok[ok.game_type == 2].copy()
    reg["pbp_extra"] = reg.last_period_type.isin(["OT", "SO"])
    reg["pbp_so"] = reg.last_period_type == "SO"
    ot_dis = reg[reg.pbp_extra != (reg.went_ot.astype(bool)
                                   | reg.went_so.astype(bool))]
    so_dis = reg[reg.pbp_so != reg.went_so.astype(bool)]
    lines.append(f"## OT/SO flag agreement (regular season)\n\n"
                 f"went_ot disagreements: **{len(ot_dis)}** ({len(ot_dis)/max(len(reg),1):.5%}); "
                 f"went_so disagreements: **{len(so_dis)}**.\n")
    if len(ot_dis):
        lines.append(ot_dis[["game_id", "date", "home", "away", "last_period_type",
                             "went_ot", "went_so"]].head(10).to_markdown(index=False) + "\n")

    # shift TOI identity (team-season hours vs 60 min * GP)
    try:
        sts = pd.read_csv(PROC / "shift_team_seasons.csv")
        ts = pd.read_csv(PROC / "team_seasons.csv")
        j = sts.merge(ts[["season_end", "team", "gp"]], on=["season_end", "team"],
                      how="inner")
        # shift charts include goalies: ~6 players on ice x 60 min = 6 h/game,
        # minus PK 4v5 time, plus OT
        j["hrs_per_game"] = j.team_toi_hr / j.gp
        lines.append("## Shift-chart TOI identity (team on-ice player-hours per game)\n")
        lines.append(f"Expected ≈6.0 h/game (6 on-ice incl. goalie × 60 min; PK time "
                     f"subtracts, OT adds). Observed: "
                     f"mean {j.hrs_per_game.mean():.3f}, min {j.hrs_per_game.min():.3f}, "
                     f"max {j.hrs_per_game.max():.3f} across {len(j)} team-seasons.\n")
        odd = j[(j.hrs_per_game < 5.5) | (j.hrs_per_game > 6.3)]
        lines.append(f"Team-seasons outside [5.5, 6.3]: **{len(odd)}**"
                     + (f" -> {odd[['season_end','team','hrs_per_game']].round(2).to_dict('records')[:8]}"
                        if len(odd) else "") + "\n")
    except FileNotFoundError:
        lines.append("## Shift TOI identity: shift_team_seasons.csv not found (skipped)\n")

    lines.append("\nVerdict: checks above graduate to review_tests_neurhl.py with "
                 "thresholds — score mismatch = hard fail; date±1 handled in the "
                 "tensorizer join; OT-flag disagreements enumerated as exclusions.\n")
    (EDA / "eda_06_labels.md").write_text("\n".join(lines))
    print("wrote eda_06_labels.md")


if __name__ == "__main__":
    main()
