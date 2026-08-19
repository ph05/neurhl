"""v5 candidate features on top of the v4 FeatureBuilder (PLAN_V5 F).

Adds the six locked candidates from data/processed/team_seasons_v5.csv, all
measured in season_end == V (walk-forward safe; within-vantage centering and
expanding scaling happen in FeatureBuilder.feature_matrix as for every other
feature; pre-coverage seasons scale-fill to the league-average vector).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES_V4, FeatureBuilder

PROJ = Path(__file__).resolve().parents[1]
PROC = PROJ / "data" / "processed"

CAND_MAP = {                       # feature name -> team_seasons_v5 column
    "fo_dev": "fo_pct",
    "pen_diff": "pen_net60",
    "hd_share": "hd_share_f",
    "flurry_xg_dev": "flurry_xg5_pct",
    "corsi_dev": "corsi_sa_pct",
    "rush_xg_dev": "rush_xg_pct",
}
FEATURES_CAND = list(CAND_MAP)
FEATURES_V5_ALL = FEATURES_V4 + FEATURES_CAND


class FeatureBuilderV5(FeatureBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v5 = pd.read_csv(PROC / "team_seasons_v5.csv") \
            .set_index(["season_end", "team"]).sort_index()

    def team_features(self, V: int, h: int) -> pd.DataFrame:
        df = super().team_features(V, h)
        try:
            cur = self.v5.xs(V, level="season_end")
        except KeyError:
            cur = None
        for feat, col in CAND_MAP.items():
            if cur is not None and col in cur.columns:
                df[feat] = cur[col].reindex(df.index).astype(float)
            else:
                df[feat] = np.nan
        return df
