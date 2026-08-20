"""NeurHL EDA-07b — HTM-vs-JSON gold-standard cross-validation (season 2012).

Season_end 2012 exists in BOTH corpora: the api-web JSON PBP (canonical) and
the HTM backfill parse (events_htm_2012.parquet, built by tensorize_htm.py once
the crawl lands). Agreement between the two independent representations bounds
the HTM parser's error for 2008-2011, where no JSON exists. Appends results to
eda/eda_07_htm.md.

Checks per game (joined on game_id):
  - per-event-type counts (SOG/miss/block/goal/hit/give/take/penalty/faceoff)
  - goal sequences: times + scorer playerIds
  - faceoff winner playerIds at matching times
  - on-ice sets at goal events (strongest joint test of shifts vs PL grids)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, TENSORS  # noqa: E402

ON = [f"h_on{i}" for i in range(7)] + [f"a_on{i}" for i in range(7)]


def main():
    htm_p = TENSORS / "events_htm_2012.parquet"
    if not htm_p.exists():
        print("events_htm_2012.parquet missing — run tensorize_htm.py first")
        sys.exit(1)
    maps = json.loads((TENSORS / "maps.json").read_text())
    et = maps["event_type"]
    inv = {v: k for k, v in et.items()}
    h = pd.read_parquet(htm_p)
    j = pd.read_parquet(TENSORS / "events_2012.parquet")
    j = j[j.game_type == 2]
    common = sorted(set(h.game_id) & set(j.game_id))
    lines = [f"\n## 2012 HTM-vs-JSON cross-validation ({len(common)} games)\n"]

    core = ["shot-on-goal", "missed-shot", "blocked-shot", "goal", "hit",
            "giveaway", "takeaway", "penalty", "faceoff"]
    hc = (h[h.event_type.isin([et[c] for c in core]) & (h.period < 5)]
          .groupby(["game_id", "event_type"]).size())
    jc = (j[j.event_type.isin([et[c] for c in core]) & (j.period < 5)]
          .groupby(["game_id", "event_type"]).size())
    cmp = pd.concat([hc.rename("htm"), jc.rename("json")], axis=1).fillna(0)
    cmp = cmp.reset_index()
    cmp["event"] = cmp.event_type.map(inv)
    agg = cmp.groupby("event").apply(
        lambda d: pd.Series({
            "exact_game_rate": (d.htm == d.json).mean(),
            "rel_err": abs(d.htm.sum() - d.json.sum()) / max(d.json.sum(), 1)}),
        include_groups=False).round(4)
    lines.append("Per-type totals and per-game exact-count agreement:\n")
    lines.append(agg.to_markdown() + "\n")

    # goals: time + scorer identity
    gh = h[(h.event_type == et["goal"]) & (h.period < 5)][
        ["game_id", "t", "p1"]].sort_values(["game_id", "t"])
    gj = j[(j.event_type == et["goal"]) & (j.period < 5)][
        ["game_id", "t", "p1"]].sort_values(["game_id", "t"])
    m = gh.merge(gj, on=["game_id", "t"], suffixes=("_h", "_j"), how="outer",
                 indicator=True)
    matched = m[m._merge == "both"]
    lines.append(f"Goals: {len(matched)}/{max(len(gj), 1)} matched on exact "
                 f"(game, second); scorer playerId agreement "
                 f"{(matched.p1_h == matched.p1_j).mean():.4f}\n")

    # faceoff winners at matching seconds
    fh = h[h.event_type == et["faceoff"]][["game_id", "t", "p1"]]
    fj = j[j.event_type == et["faceoff"]][["game_id", "t", "p1"]]
    fm = fh.merge(fj, on=["game_id", "t"], suffixes=("_h", "_j"))
    lines.append(f"Faceoffs matched on (game, second): {len(fm)}/{len(fj)}; "
                 f"winner playerId agreement "
                 f"{(fm.p1_h == fm.p1_j).mean():.4f}\n")

    # on-ice sets at goals
    oh = h[(h.event_type == et["goal"]) & (h.period < 5)][
        ["game_id", "t"] + ON]
    oj = j[(j.event_type == et["goal"]) & (j.period < 5)][
        ["game_id", "t"] + ON]
    om = oh.merge(oj, on=["game_id", "t"], suffixes=("_h", "_j"))
    def setify(row, suf):
        return set(v for c in ON if (v := row[c + suf]) > 0)
    same = [setify(r, "_h") == setify(r, "_j") for _, r in om.iterrows()]
    lines.append(f"On-ice exact-set agreement at matched goals: "
                 f"{np.mean(same):.4f} ({len(om)} goals)\n")

    with open(EDA / "eda_07_htm.md", "a") as f:
        f.write("\n".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
