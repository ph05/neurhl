"""I5: prospect pipeline (PLAN_V4) — draft->panel name join + arrival ramps.

Join: raw nhl_draft_<year>.json picks (firstName/lastName/positionCode) to MoneyPuck
skater panels by (normalized name, F/D position group), disambiguated by debut-year
plausibility (first NHL season within draft_year..draft_year+7). Goalies excluded
(pipeline feature targets skaters; goalie development is its own problem).
Join written to data/processed/draft_join.csv for inspection.

Quality bar (prereg): among picks 1-60 of drafts 2009-2015, the match rate for players
established enough to matter must clear 80%. Verifiable proxies reported: match rate on
picks 1-15 (near-universal NHL arrival) and named spot checks.

Ramp (train-only): mean NHL points in season draft_year+y per drafted skater
(zeros included: non-arrival IS the expectation), by pick bucket x years-since-draft,
drafts <= 2013, outcome seasons <= 2017.

Feature: prospect_pipeline(V, h) = sum over own not-yet-established draftees
(career TOI < 1500 min through V, draft years V-7..V) of ramp(bucket, (V+h)-draft_year).
Gate F2 in backtest4.py decides entry, separately at h1 and h2.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import players as P

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"

BUCKETS_PICK = [(1, 10), (11, 30), (31, 60), (61, 300)]
ESTABLISHED_TOI = 1500.0
RAMP_MAX_Y = 8


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.\'\-]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def pick_bucket(overall: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS_PICK):
        if lo <= overall <= hi:
            return i
    return len(BUCKETS_PICK) - 1


def load_draft_picks() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("nhl_draft_*.json")):
        d = json.loads(f.read_text())
        year = int(d.get("draftYear", f.stem.split("_")[-1]))
        for p in d.get("picks", []):
            pos = p.get("positionCode", "?")
            first = (p.get("firstName") or {}).get("default", "")
            last = (p.get("lastName") or {}).get("default", "")
            rows.append({"draft_year": year, "overall": int(p["overallPick"]),
                         "team": p.get("teamPickHistory") or p.get("teamAbbrev"),
                         "name": f"{first} {last}".strip(), "pos": pos})
    df = pd.DataFrame(rows)
    df["team"] = df.team.astype(str).str.split().str[-1]  # pick-history "A -> B" keeps B
    df["team"] = df.team.replace(P.MP_FRAN)
    df["pos_group"] = np.where(df.pos == "D", "D", np.where(df.pos == "G", "G", "F"))
    df["nname"] = df.name.map(norm_name)
    df["bucket"] = df.overall.map(pick_bucket)
    return df[df.pos_group != "G"].copy()


def build_join(skaters: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    dest = PROC / "draft_join.csv"
    if dest.exists() and not force:
        return pd.read_csv(dest)
    picks = load_draft_picks()
    pan = skaters.groupby("playerId").agg(
        name=("name", "first"), pos_group=("pos_group", "first"),
        first_season=("season_end", "min"), career_toi=("toi_min", "sum")).reset_index()
    pan["nname"] = pan.name.map(norm_name)
    cand = picks.merge(pan, on=["nname", "pos_group"], how="left",
                       suffixes=("", "_panel"))
    # debut plausibility: first NHL season within draft_year+1 .. draft_year+8
    # (season_end convention: draft June Y -> earliest NHL season_end Y+1)
    ok = cand.first_season.between(cand.draft_year + 1, cand.draft_year + 8)
    cand.loc[~ok.fillna(False), ["playerId", "first_season", "career_toi"]] = np.nan
    # ambiguity: multiple surviving candidates for one pick -> drop the match
    n_cand = cand.groupby(["draft_year", "overall"]).playerId.transform(
        lambda s: s.notna().sum())
    ambiguous = (n_cand > 1)
    cand.loc[ambiguous, ["playerId", "first_season", "career_toi"]] = np.nan
    out = cand.drop_duplicates(["draft_year", "overall"]).copy()
    out["ambiguous"] = out.set_index(["draft_year", "overall"]).index.map(
        ambiguous.groupby([cand.draft_year, cand.overall]).any())
    out.to_csv(dest, index=False)
    return out


def join_quality(join: pd.DataFrame) -> dict:
    top15 = join[(join.overall <= 15) & join.draft_year.between(2009, 2015)]
    top60 = join[(join.overall <= 60) & join.draft_year.between(2009, 2015)]
    spot = {}
    for nm in ("connor mcdavid", "jack eichel", "auston matthews", "cale makar"):
        row = join[join.nname == nm]
        spot[nm] = bool(len(row) and row.playerId.notna().any())
    return {"match_rate_picks_1_15": round(float(top15.playerId.notna().mean()), 3),
            "match_rate_picks_1_60": round(float(top60.playerId.notna().mean()), 3),
            "n_ambiguous": int(join.ambiguous.sum()),
            "spot_checks": spot,
            "pass": bool(top15.playerId.notna().mean() >= 0.85
                         and all(spot.values()))}


def fit_ramp(join: pd.DataFrame, skaters: pd.DataFrame, drafts_max: int = 2013,
             outcomes_max: int = 2017) -> pd.DataFrame:
    """Mean points at years-since-draft y (zeros for non-arrivals), bucket x y.
    Train-frozen: drafts <= drafts_max, outcome seasons <= outcomes_max."""
    pts = skaters.set_index(["playerId", "season_end"]).points
    rows = []
    d = join[join.draft_year.between(2009, drafts_max)]
    for y in range(1, RAMP_MAX_Y + 1):
        for b in range(len(BUCKETS_PICK)):
            sub = d[(d.bucket == b) & (d.draft_year + y <= outcomes_max)]
            if not len(sub):
                continue
            vals = []
            for r in sub.itertuples():
                if pd.notna(r.playerId):
                    vals.append(float(pts.get((int(r.playerId), r.draft_year + y), 0.0)))
                else:
                    vals.append(0.0)
            rows.append({"bucket": b, "y": y, "mean_pts": float(np.mean(vals)),
                         "n": len(vals)})
    ramp = pd.DataFrame(rows)
    # extend flat beyond the last observed y per bucket (train horizon limit)
    full = []
    for b in range(len(BUCKETS_PICK)):
        rb = ramp[ramp.bucket == b].set_index("y").mean_pts
        for y in range(1, RAMP_MAX_Y + 1):
            v = rb.get(y, rb.loc[rb.index.max()] if len(rb) else 0.0)
            full.append({"bucket": b, "y": y, "mean_pts": float(v)})
    return pd.DataFrame(full)


def pipeline_feature(join: pd.DataFrame, skaters: pd.DataFrame, ramp: pd.DataFrame,
                     V: int, h: int, teams) -> pd.Series:
    """Expected points arriving in season V+h from not-yet-established own draftees."""
    career = skaters[skaters.season_end <= V].groupby("playerId").toi_min.sum()
    rmap = ramp.set_index(["bucket", "y"]).mean_pts
    d = join[join.draft_year.between(V - 7, V)]
    out = {t: 0.0 for t in teams}
    for r in d.itertuples():
        team = r.team
        if team not in out:
            continue
        toi = float(career.get(int(r.playerId), 0.0)) if pd.notna(r.playerId) else 0.0
        if toi >= ESTABLISHED_TOI:
            continue
        y = (V + h) - r.draft_year
        if y < 1:
            continue
        out[team] += float(rmap.get((r.bucket, min(y, RAMP_MAX_Y)), 0.0))
    return pd.Series(out, name="prospect_pipeline")
