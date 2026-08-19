"""v6 candidate features on top of FeatureBuilderV5 (PLAN_V6 F).

Ridge candidates, measured at season_end == V (walk-forward safe):
  line_cont, toi_hhi_f  from shift charts (team_seasons_v6)
  coach_new, coach_tenure  from HR coach records (one of the two may gate in)
  prospect_prod  production-weighted prospect pipeline (prospects_prod)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_v5 import FEATURES_V5_ALL, FeatureBuilderV5
from prospects_prod import ProdPipeline

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"

CAND_MAP_V6 = {"line_cont": "line_cont", "toi_hhi_f": "toi_hhi_f",
               "coach_new": "coach_new", "coach_tenure": "coach_tenure"}
FEATURES_CAND_V6 = list(CAND_MAP_V6) + ["prospect_prod"]
FEATURES_V6_ALL = FEATURES_V5_ALL + FEATURES_CAND_V6


class FeatureBuilderV6(FeatureBuilderV5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v6 = pd.read_csv(PROC / "team_seasons_v6.csv") \
            .set_index(["season_end", "team"]).sort_index()
        self._prodpipe = None

    def team_features(self, V: int, h: int) -> pd.DataFrame:
        df = super().team_features(V, h)
        try:
            cur = self.v6.xs(V, level="season_end")
        except KeyError:
            cur = None
        for feat, col in CAND_MAP_V6.items():
            if cur is not None and col in cur.columns:
                df[feat] = cur[col].reindex(df.index).astype(float)
            else:
                df[feat] = np.nan
        if self._prodpipe is None:
            self._prodpipe = ProdPipeline(self.sk)
        df["prospect_prod"] = self._prodpipe.feature(V, h, list(df.index))
        return df
