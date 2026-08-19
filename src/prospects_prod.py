"""Production-weighted prospect pipeline (PLAN_V6 c4).

build_production(): flattens player-landing career rows for every draft pick
(records.nhl.com id table) into data/processed/prospect_production.csv.

prod_pipeline_feature(): same skeleton as prospects.pipeline_feature (slot-value
ramps, not-yet-established filter) but each prospect's ramp value is scaled by a
production multiplier from PRE-NHL seasons known at vantage V:

    nhle_ppg = factor(league) * pts/gp of the most recent season <= V with gp>=10
    mult     = clip(0.5 + 0.5 * nhle_ppg / bucket_mean, 0.3, 3.0)

League factors are FIXED published-style NHLe constants (declared in PLAN_V6 —
external knowledge, not tuned here). bucket_mean is computed once from TRAIN-era
picks (draft years <= 2013, the fit_ramp window), so no validation leakage.
Missing production -> mult = 1.0 (pure slot value, v4 behavior).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prospects as PR

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"

NHLE = {  # fixed translation factors (approximate published NHLe family)
    "AHL": 0.47, "KHL": 0.77, "SHL": 0.62, "Liiga": 0.54, "NL": 0.46,
    "DEL": 0.45, "Czechia": 0.55, "Czech": 0.55, "SL": 0.30, "VHL": 0.35,
    "Allsvenskan": 0.44, "Mestis": 0.30, "OHL": 0.30, "WHL": 0.29,
    "QMJHL": 0.28, "USHL": 0.25, "NCAA": 0.41, "H-East": 0.41, "NCHC": 0.41,
    "Big Ten": 0.41, "ECAC": 0.41, "CCHA": 0.41, "WCHA": 0.41, "AHAmerica": 0.41,
    "MHL": 0.25, "J20 Nationell": 0.22, "SuperElit": 0.22, "U20 SM-sarja": 0.20,
    "Russia": 0.55, "Slovakia": 0.35, "NLB": 0.25, "BCHL": 0.18, "AJHL": 0.16,
}
DEFAULT_FACTOR = 0.15
MIN_GP = 10


def build_production():
    recs = json.loads((RAW / "nhl_draft_records.json").read_text())
    rows = []
    for r in recs:
        pid = r.get("playerId")
        if not pid:
            continue
        f = RAW / "player_landing" / f"{pid}.json.gz"
        if not f.exists():
            continue
        with gzip.open(f, "rt") as fh:
            d = json.load(fh)
        for st in d.get("seasonTotals", []):
            if st.get("gameTypeId") != 2 or st.get("leagueAbbrev") == "NHL":
                continue
            gp = st.get("gamesPlayed") or 0
            if gp <= 0 or st.get("points") is None:
                continue
            rows.append({"playerId": pid, "draft_year": r["draftYear"],
                         "overall": r["overallPickNumber"],
                         "season_end": int(str(st["season"])[4:]),
                         "league": st["leagueAbbrev"], "gp": gp,
                         "pts": st["points"]})
    df = pd.DataFrame(rows).drop_duplicates(
        ["playerId", "season_end", "league"])
    df.to_csv(PROC / "prospect_production.csv", index=False)
    print(f"prospect_production: {len(df)} rows, {df.playerId.nunique()} players, "
          f"{df.league.nunique()} leagues")
    return df


class ProdPipeline:
    def __init__(self, skaters: pd.DataFrame):
        self.join = PR.build_join(skaters)
        self.sk = skaters
        self.ramp = PR.fit_ramp(self.join, skaters)
        prod = pd.read_csv(PROC / "prospect_production.csv")
        prod["factor"] = prod.league.map(NHLE).fillna(DEFAULT_FACTOR)
        prod["nhle_ppg"] = prod.factor * prod.pts / prod.gp
        self.prod = prod[prod.gp >= MIN_GP]
        # records id map covers EVERY pick (incl. never-NHL), unlike join.playerId
        recs = json.loads((RAW / "nhl_draft_records.json").read_text())
        self.recmap = {(r["draftYear"], r["overallPickNumber"]): r["playerId"]
                       for r in recs if r.get("playerId")}
        # bucket means from TRAIN-era picks only (fit_ramp window), full pick pop
        m = self.prod.copy()
        m["bucket"] = m.overall.map(PR.pick_bucket)
        m = m[m.draft_year <= 2013]
        self.bucket_mean = m.groupby("bucket").nhle_ppg.mean().to_dict()
        self._best_cache: dict = {}

    def _mult(self, pid, draft_year, bucket, V: int) -> float:
        key = (pid, V)
        if key not in self._best_cache:
            s = self.prod[(self.prod.playerId == pid)
                          & (self.prod.season_end <= V)]
            self._best_cache[key] = (
                float(s.sort_values("season_end").iloc[-1].nhle_ppg)
                if len(s) else np.nan)
        nhle = self._best_cache[key]
        base = self.bucket_mean.get(bucket)
        if not np.isfinite(nhle) or not base:
            return 1.0
        return float(np.clip(0.5 + 0.5 * nhle / base, 0.3, 3.0))

    def feature(self, V: int, h: int, teams) -> pd.Series:
        career = self.sk[self.sk.season_end <= V].groupby("playerId").toi_min.sum()
        rmap = self.ramp.set_index(["bucket", "y"]).mean_pts
        d = self.join[self.join.draft_year.between(V - 7, V)]
        out = {t: 0.0 for t in teams}
        for r in d.itertuples():
            if r.team not in out:
                continue
            toi = float(career.get(int(r.playerId), 0.0)) \
                if pd.notna(r.playerId) else 0.0
            if toi >= PR.ESTABLISHED_TOI:
                continue
            y = (V + h) - r.draft_year
            if y < 1:
                continue
            base = float(rmap.get((r.bucket, min(y, PR.RAMP_MAX_Y)), 0.0))
            pid = self.recmap.get((int(r.draft_year), int(r.overall)))
            if pid is None and pd.notna(r.playerId):
                pid = int(r.playerId)
            mult = self._mult(pid, r.draft_year, r.bucket, V) if pid else 1.0
            out[r.team] += base * mult
        return pd.Series(out, name="prospect_prod")


if __name__ == "__main__":
    build_production()
