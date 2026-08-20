"""NeurHL EDA-02 — shot-coordinate quality & arena scorer bias (PLAN_NeurHL Phase 0).

The NHL's rink scorers systematically bias recorded shot locations by venue
(the classic MSG effect). Before any coordinate enters the event-LM, we must
(i) verify attacking-direction normalization is possible, (ii) measure per
venue-season location bias, and (iii) prototype the train-window arena offsets
promised by P3. Consumes scan_pbp shot_events. Writes eda/eda_02_coords.md,
figs, and a prototype offsets CSV (cache, not shipped).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, TENSORS  # noqa: E402

CACHE = TENSORS / "_edacache"
FIGS = EDA / "figs"
NET_X = 89.0


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Flip coords so the shooting team always attacks the net at +x.

    EDA FINDING: homeTeamDefendingSide exists only from season_end 2020 — it is
    NOT era-stable. Direction is therefore INFERRED per (game, shooting side,
    period) as the sign of the median x of that group's offensive-zone
    (zoneCode=O) shots, and validated against defendingSide where it exists.
    Returns (normalized df, share of shots with an inferable direction,
    agreement with defendingSide on 2020+).
    """
    d = df.dropna(subset=["x", "y"]).copy()
    oz = d[d.zone == "O"]
    grp = ["game_id", "home_event", "period"]
    sign = np.sign(oz.groupby(grp).x.median()).rename("dir")
    d = d.merge(sign, left_on=grp, right_index=True, how="left")
    coverage = float((d["dir"].isin([1.0, -1.0])).mean())
    d = d[d["dir"].isin([1.0, -1.0])]
    d["xn"] = d.x * d["dir"]
    d["yn"] = d.y * d["dir"]
    d["dist"] = np.hypot(NET_X - d.xn, d.yn)
    # validate vs defendingSide (2020+): home attacks right iff home defends left
    v = d[d.defending_side.isin(["left", "right"])]
    if len(v):
        want_right = np.where(v.home_event, v.defending_side == "left",
                              v.defending_side == "right")
        agree = float((np.where(want_right, 1.0, -1.0) == v["dir"]).mean())
    else:
        agree = float("nan")
    return d, coverage, agree


def main():
    sh = pd.read_parquet(CACHE / "shot_events.parquet")
    sh = sh[sh.game_type == 2]
    lines = ["# EDA-02 — shot coordinates & arena scorer bias\n"]

    n_all = len(sh)
    side_by_season = (sh.groupby("season_end").defending_side
                      .apply(lambda s: s.isin(["left", "right"]).mean()))
    first_side = side_by_season[side_by_season > 0.5].index.min()
    lines.append(f"Shot-family events (regular season): **{n_all:,}**; "
                 f"share with x/y: **{sh.x.notna().mean():.4f}**.\n")
    lines.append(f"**FINDING: `homeTeamDefendingSide` is populated only from "
                 f"season_end {first_side}** — it is NOT era-stable. Attacking "
                 f"direction is inferred per (game, side, period) from the median "
                 f"x of offensive-zone shots and validated against defendingSide "
                 f"where it exists.\n")

    d, dir_cov, dir_agree = normalize(sh)
    lines.append(f"Direction inference: coverage **{dir_cov:.4f}** of shots; "
                 f"agreement with defendingSide (2020+): **{dir_agree:.4f}**.\n")
    sog = d[d.event_type.isin(["shot-on-goal", "goal"])]

    # --- direction sanity: offensive-zone shots should sit at xn > 25
    oz = d[d.zone == "O"]
    ok_rate = (oz.xn > 25).mean()
    by_season = oz.groupby("season_end").apply(lambda t: (t.xn > 25).mean(),
                                               include_groups=False)
    lines.append("## Attacking-direction normalization sanity\n")
    lines.append(f"Offensive-zone (zoneCode=O) shots with normalized x>25: "
                 f"**{ok_rate:.4f}** overall.\n")
    lines.append(by_season.round(4).to_frame("oz_x_gt_25").to_markdown() + "\n")

    # --- league drift in recorded shot distance (recording standard, not hockey)
    lg = sog.groupby("season_end").agg(mean_dist=("dist", "mean"),
                                       mean_absy=("yn", lambda s: s.abs().mean()),
                                       n=("dist", "size"))
    lines.append("## League mean SOG distance by season (recording drift)\n")
    lines.append(lg.round(2).to_markdown() + "\n")

    # --- venue-season bias: venue mean minus season league mean
    vs = (sog.groupby(["season_end", "venue_team"])
          .agg(dist=("dist", "mean"), absy=("yn", lambda s: s.abs().mean()),
               n=("dist", "size")).reset_index())
    vs = vs.merge(lg.mean_dist.rename("lg_dist"), on="season_end")
    vs["dev"] = vs.dist - vs.lg_dist
    piv = vs.pivot(index="venue_team", columns="season_end", values="dev")
    worst = vs.assign(a=vs.dev.abs()).sort_values("a", ascending=False).head(15)
    lines.append("## Venue-season shot-distance bias (venue mean − league mean, ft)\n")
    lines.append("Worst 15 venue-seasons:\n")
    lines.append(worst[["season_end", "venue_team", "dev", "n"]]
                 .round(2).to_markdown(index=False) + "\n")
    persist = piv.mean(axis=1).sort_values()
    lines.append("Persistent venue bias (mean dev across seasons):\n")
    lines.append(persist.round(2).to_frame("mean_dev_ft").to_markdown() + "\n")

    # --- prototype offsets (train-window fit <= 2017 only, per P1/P3)
    train = vs[vs.season_end <= 2017]
    proto = (train.groupby("venue_team")
             .apply(lambda t: float((t.dev * t.n).sum() / t.n.sum()),
                    include_groups=False).rename("dist_offset").to_frame())
    proto["n"] = train.groupby("venue_team").n.sum()
    proto.to_csv(CACHE / "arena_offsets_train_prototype.csv")
    lines.append(f"Prototype train-window (≤2017) arena offsets written to cache "
                 f"(`arena_offsets_train_prototype.csv`); production offsets will be "
                 f"per-vantage expanding versions of the same estimator. "
                 f"Range: {proto.dist_offset.min():.2f} to {proto.dist_offset.max():.2f} ft.\n")

    # --- figures
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=6)
    ax.set_xticks(range(len(piv.columns)), piv.columns, fontsize=7, rotation=90)
    fig.colorbar(im, label="venue mean SOG distance − league (ft)")
    ax.set_title("Arena scorer bias by venue-season")
    fig.tight_layout(); fig.savefig(FIGS / "eda_02_venue_bias.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    lg.mean_dist.plot(ax=ax, marker="o", ms=3)
    ax.set_title("League mean SOG distance (ft)"); ax.set_xlabel("season_end")
    fig.tight_layout(); fig.savefig(FIGS / "eda_02_league_dist.png", dpi=110); plt.close(fig)

    (EDA / "eda_02_coords.md").write_text("\n".join(lines))
    print("wrote eda_02_coords.md")


if __name__ == "__main__":
    main()
