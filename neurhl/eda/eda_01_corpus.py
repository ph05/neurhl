"""NeurHL EDA-01 — PBP corpus integrity & coverage (PLAN_NeurHL Phase 0).

Consumes the scan_pbp.py cache. Verifies the event corpus is fit for
tensorization: event-type inventory by season, coordinate coverage, situation
code sanity, roster completeness, short/degenerate games. Writes
eda/eda_01_corpus.md + figs/eda_01_*.png (committed).
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


def main():
    ev = pd.read_parquet(CACHE / "events_summary.parquet")
    sit = pd.read_parquet(CACHE / "situation_codes.parquet")
    gs = pd.read_parquet(CACHE / "game_summary.parquet")
    lines = ["# EDA-01 — PBP corpus integrity & coverage\n"]

    # --- corpus size
    reg, po = gs[gs.game_type == 2], gs[gs.game_type == 3]
    lines.append(f"Corpus: **{len(gs):,} games** ({len(reg):,} regular, {len(po):,} playoff), "
                 f"seasons {gs.season_end.min()}–{gs.season_end.max()}, "
                 f"**{int(ev.n.sum()):,} events** "
                 f"({int(ev[ev.game_type == 2].n.sum()):,} regular).\n")

    # --- event counts by season (regular season)
    piv = (ev[ev.game_type == 2].pivot_table(index="event_type", columns="season_end",
                                             values="n", aggfunc="sum").fillna(0).astype(int))
    per_game = piv / reg.groupby("season_end").size()
    lines.append("## Event types per game by season (regular season)\n")
    lines.append(per_game.round(1).to_markdown() + "\n")

    # types present in some seasons but not others (schema drift check)
    missing = {t: sorted(set(per_game.columns[per_game.loc[t] == 0]))
               for t in per_game.index if (per_game.loc[t] == 0).any()}
    lines.append("**Schema drift check:** " +
                 ("no event type disappears in any season.\n" if not missing else
                  f"types absent in some seasons: {missing}\n"))

    # --- coordinate coverage
    cov = (ev[ev.game_type == 2].groupby(["season_end", "event_type"])
           .apply(lambda d: d.n_xy.sum() / max(d.n.sum(), 1), include_groups=False)
           .unstack("event_type"))
    core = [c for c in ("shot-on-goal", "goal", "missed-shot", "blocked-shot",
                        "faceoff", "hit", "giveaway", "takeaway") if c in cov.columns]
    lines.append("## x/y coordinate coverage (share of events with coords, regular season)\n")
    lines.append(cov[core].round(3).to_markdown() + "\n")
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in core:
        ax.plot(cov.index, cov[c], marker="o", ms=3, label=c)
    ax.set_ylim(0, 1.05); ax.set_xlabel("season_end"); ax.set_ylabel("xy coverage")
    ax.legend(fontsize=8); ax.set_title("Coordinate coverage by event type")
    fig.tight_layout(); fig.savefig(FIGS / "eda_01_xy_coverage.png", dpi=110); plt.close(fig)

    # --- situation codes
    sit["code"] = sit.situation_code.astype(str)
    ok_mask = sit.code.str.fullmatch(r"[01]\d\d[01]")
    weird = sit[~ok_mask].groupby("code").n.sum().sort_values(ascending=False)
    tot = sit.n.sum()
    lines.append("## Situation codes\n")
    top = sit.groupby("code").n.sum().sort_values(ascending=False).head(12)
    lines.append("Top codes (format = awayGoalie|awaySkaters|homeSkaters|homeGoalie):\n")
    lines.append((top / tot).round(4).to_markdown() + "\n")
    lines.append(f"Codes failing the `[01]d d[01]` pattern: "
                 f"{len(weird)} distinct, {weird.sum():,} events "
                 f"({weird.sum() / tot:.5%}). "
                 + (f"Top offenders: {dict(weird.head(5))}\n" if len(weird) else "\n"))
    skaters = sit[ok_mask].copy()
    a_sk = skaters.code.str[1].astype(int)
    h_sk = skaters.code.str[2].astype(int)
    extreme = skaters[(a_sk > 6) | (h_sk > 6) | (a_sk < 3) | (h_sk < 3)]
    lines.append(f"Codes with skater counts outside [3,6]: {extreme.n.sum():,} events "
                 f"({extreme.n.sum() / tot:.5%}).\n")

    # --- rosters
    lines.append("## Roster completeness (rosterSpots)\n")
    rc = gs.groupby("season_end").agg(
        mean_roster=("n_roster", "mean"),
        pct_not_40=("n_roster", lambda s: (s != 40).mean()),
        pct_goalies_ne_4=("n_g", lambda s: (s != 4).mean()))
    lines.append(rc.round(4).to_markdown() + "\n")
    lines.append(f"Games with roster != 36..44: {((gs.n_roster < 36) | (gs.n_roster > 44)).sum()}\n")

    # --- degenerate games
    short = gs[gs.n_plays < 200]
    nosc = gs[gs.home_score.isna() | gs.away_score.isna()]
    dup = gs[gs.game_id.duplicated(keep=False)]
    lines.append("## Degenerate-game scan\n")
    lines.append(f"- games with < 200 plays: **{len(short)}**"
                 + (f" -> {short.game_id.tolist()[:10]}" if len(short) else "") + "\n")
    lines.append(f"- games missing a final score: **{len(nosc)}**\n")
    lines.append(f"- duplicated game ids: **{len(dup)}**\n")

    # --- per-season game counts vs expectation
    cnt = reg.groupby("season_end").size()
    lines.append("## Regular-season game counts\n")
    lines.append(cnt.to_frame("games").to_markdown() + "\n")

    # heatmap figure of per-game event rates
    fig, ax = plt.subplots(figsize=(10, 5))
    hm = per_game.loc[core]
    im = ax.imshow(np.log10(hm.values + 0.1), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(hm.index)), hm.index, fontsize=8)
    ax.set_xticks(range(len(hm.columns)), hm.columns, fontsize=7, rotation=90)
    fig.colorbar(im, label="log10(events/game)")
    ax.set_title("Event rate by season (regular season)")
    fig.tight_layout(); fig.savefig(FIGS / "eda_01_event_rates.png", dpi=110); plt.close(fig)

    (EDA / "eda_01_corpus.md").write_text("\n".join(lines))
    print("wrote eda_01_corpus.md")


if __name__ == "__main__":
    main()
