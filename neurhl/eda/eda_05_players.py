"""NeurHL EDA-05 — player vocabulary, long tail, and cold-start coverage.

Sizes the embedding table (players per vantage), quantifies the appearance
long tail, counts rookies per season, and measures career-encoder coverage:
what share of dressed players (and of rookies specifically) have a
player_landing career file (drafted players only) — the population that must
fall back to position-mean embeddings is the documented cold-start null set.
Writes eda/eda_05_players.md + figs.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, RAW, TENSORS  # noqa: E402

CACHE = TENSORS / "_edacache"
FIGS = EDA / "figs"


def main():
    app = pd.read_parquet(CACHE / "player_appearances.parquet")
    app = app[app.dressed > 0]
    landing_ids = {int(p.name.split(".")[0])
                   for p in (RAW / "player_landing").glob("*.json.gz")}
    draft = json.loads((RAW / "nhl_draft_records.json").read_text())
    draft_rows = draft if isinstance(draft, list) else draft.get("data", draft)
    lines = ["# EDA-05 — player vocabulary & cold-start coverage\n"]

    # --- vocab growth by vantage
    first = app.groupby("player_id").season_end.min()
    seasons = sorted(app.season_end.unique())
    growth = pd.Series({v: int((first <= v).sum()) for v in seasons}, name="vocab")
    per = app.groupby("season_end").player_id.nunique().rename("dressed_players")
    rookies = first.value_counts().sort_index().rename("rookies")
    tab = pd.concat([per, rookies, growth], axis=1)
    lines.append("## Vocabulary by season (dressed skaters+goalies in PBP rosterSpots)\n")
    lines.append(tab.to_markdown() + "\n")
    lines.append(f"Total distinct players 2012–2026: **{app.player_id.nunique():,}** "
                 f"(embedding table size at the 2026 vantage).\n")

    # --- long tail of event mentions
    tot = app.groupby("player_id").agg(mentions=("event_mentions", "sum"),
                                       dressed=("dressed", "sum"))
    q = tot.mentions.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(0)
    lines.append("## Event-mention long tail (pretraining signal per player)\n")
    lines.append(q.to_frame("event_mentions").to_markdown() + "\n")
    lines.append(f"Players with <100 career event mentions: "
                 f"**{(tot.mentions < 100).mean():.3f}** of vocab "
                 f"({(tot.mentions < 100).sum():,} players) — these lean on the "
                 f"career encoder / position mean.\n")

    # --- career-file (landing) coverage
    app["has_landing"] = app.player_id.isin(landing_ids)
    cov = app.groupby("season_end").has_landing.mean().rename("dressed_with_career_file")
    rook = app.merge(first.rename("first_season"), on="player_id")
    rook = rook[rook.season_end == rook.first_season]
    rcov = rook.groupby("season_end").has_landing.mean().rename("rookies_with_career_file")
    lines.append("## Career-encoder coverage (player_landing files = drafted players)\n")
    lines.append(pd.concat([cov, rcov], axis=1).round(3).to_markdown() + "\n")
    lines.append(f"\nOverall: {app.has_landing.mean():.3f} of dressed player-seasons and "
                 f"{rook.has_landing.mean():.3f} of rookie debuts have a career file. "
                 f"The remainder (undrafted college/European FAs) is the cold-start "
                 f"null population (position-mean embedding, counted at report time).\n")
    lines.append(f"Draft records on disk: {len(draft_rows):,} picks "
                 f"(2006–2025, incl. never-made-NHL — no survivorship bias).\n")

    # --- figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(growth.index, growth.values, marker="o", ms=3)
    axes[0].set_title("cumulative vocab by vantage"); axes[0].set_xlabel("season_end")
    axes[1].hist(np.log10(tot.mentions.clip(lower=1)), bins=50)
    axes[1].set_title("log10 career event mentions"); axes[1].set_xlabel("log10(mentions)")
    fig.tight_layout(); fig.savefig(FIGS / "eda_05_vocab.png", dpi=110); plt.close(fig)

    (EDA / "eda_05_players.md").write_text("\n".join(lines))
    print("wrote eda_05_players.md")


if __name__ == "__main__":
    main()
