"""Team feature construction for the v2 ridge, walk-forward safe by construction.

team_features(V, h): one row per team active at vantage V, all 15 candidates.
feature_matrix(predict_seasons, h): stacked (X, y, meta) with within-season centering and
expanding-window (causal) scaling.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import players as P

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"

FEATURES = ["elo_dev", "xg_dev", "gsax_1yr", "gsax_marcel", "tandem_gsax",
            "goalie_consistency", "goalie_age", "goalie_trend", "toi_age", "share_u23",
            "share_32p", "prod_age_exp", "returning_toi", "star_share", "draft_cap",
            "player_points_proj"]
FEATURES_V3 = FEATURES + ["finishing", "st_pp", "st_pk"]
TANDEM_SHARES = np.array([0.58, 0.30, 0.12])


def vantage_drafts(V: int) -> list[int]:
    """Draft years known at vantage V (season_end V => drafts up to calendar year V)."""
    return [V - 2, V - 1, V]


class FeatureBuilder:
    def __init__(self, end_ratings: dict, ts: pd.DataFrame,
                 goalie_hp: dict | None = None, skater_delta: float = 0.8):
        self.end_r = end_ratings
        self.ts = ts.set_index(["season_end", "team"])
        self.sk, self.go, self.skt, self.got, self.bios = P.load_panels()
        self.goalie_hp = goalie_hp or {"delta": 0.7, "nu0": 4.0, "window": 4}
        self.skater_delta = skater_delta
        drafts = pd.read_csv(RAW / "nhl_drafts.csv")
        drafts["team"] = drafts.team.replace(P.MP_FRAN)
        drafts["value"] = np.exp(-(drafts.overall - 1) / 25.0)
        self.drafts = drafts
        self._fe_cache: dict = {}
        self._gproj_cache: dict = {}
        self._marcel_cache: dict = {}

    def fe_coefs(self, V: int) -> dict:
        if V not in self._fe_cache:
            self._fe_cache[V] = {pg: P.age_curve_fe(self.sk, self.bios, V, pg)
                                 for pg in ("F", "D")}
        return self._fe_cache[V]

    def goalie_proj(self, V: int) -> pd.DataFrame:
        if V not in self._gproj_cache:
            self._gproj_cache[V] = P.goalie_project(self.go, V, **self.goalie_hp)
        return self._gproj_cache[V]

    def marcel(self, V: int, h: int) -> pd.DataFrame:
        key = (V, h)
        if key not in self._marcel_cache:
            self._marcel_cache[key] = P.skater_marcel(
                self.sk, self.bios, V, delta=self.skater_delta,
                fe_coefs=self.fe_coefs(V), horizon=h)
        return self._marcel_cache[key]

    # ---------------- goalie team aggregates at vantage V ----------------
    def team_goalies(self, V: int) -> pd.DataFrame:
        gt = self.got[self.got.season_end == V].copy()
        proj = self.goalie_proj(V).set_index("playerId")
        tau_bar2 = proj.attrs["tau_bar2"]
        tau2_pop = proj.attrs["tau2_pop"]
        rows = []
        ref_age = P.age_of(self.bios, gt.playerId, V)
        gt["age"] = ref_age.to_numpy()
        for team, d in gt.groupby("team"):
            d = d.sort_values("ongoal", ascending=False).head(3)
            shares = TANDEM_SHARES[:len(d)].copy()
            shares /= shares.sum()
            theta = np.array([proj.theta.get(p, 0.0) for p in d.playerId])
            vpost = np.array([proj.V_post.get(p, tau2_pop + tau_bar2) for p in d.playerId])
            cons = np.array([proj.C.get(p, 1.0) for p in d.playerId])
            trend = np.array([proj.trend.get(p, 0.0) for p in d.playerId])
            ages = np.nan_to_num(d.age.to_numpy(), nan=28.0)
            rows.append({"team": team,
                         "tandem_gsax": float(shares @ theta),
                         "tandem_var": float((shares ** 2) @ vpost),
                         "goalie_consistency": float(shares @ cons),
                         "goalie_age": float(shares @ ages),
                         "goalie_trend": float(shares @ trend)})
        return pd.DataFrame(rows).set_index("team")

    def team_gsax_history(self, V: int) -> pd.DataFrame:
        """Season-centered team per-shot GSAx, 1yr and delta-decayed 3yr."""
        got = self.got[self.got.season_end <= V].copy()
        agg = got.groupby(["season_end", "team"]).agg(
            shots=("ongoal", "sum"), xga=("xGoals", "sum"), ga=("goals", "sum"))
        agg["rate"] = (agg.xga - agg.ga) / agg.shots
        agg["rate"] = agg.rate - agg.groupby("season_end").rate.transform("mean")
        one = agg.xs(V, level="season_end")["rate"].rename("gsax_1yr")
        rows = {}
        for team in one.index:
            num = den = 0.0
            for lag in range(3):
                s = V - lag
                if (s, team) in agg.index:
                    w = 0.7 ** lag * agg.loc[(s, team), "shots"]
                    num += w * agg.loc[(s, team), "rate"]
                    den += w
            rows[team] = num / den if den else 0.0
        return pd.concat([one, pd.Series(rows, name="gsax_marcel")], axis=1)

    # ---------------- skater/age team aggregates ----------------
    def team_age_structure(self, V: int, h: int) -> pd.DataFrame:
        st = self.skt[self.skt.season_end == V].copy()
        st["age"] = P.age_of(self.bios, st.playerId, V).to_numpy()
        st = st.dropna(subset=["age"])
        sk_v = self.sk[self.sk.season_end == V].set_index("playerId")
        fe = self.fe_coefs(V)
        pos_mean = {pg: float(np.average(d.pts60, weights=d.toi_min))
                    for pg, d in self.sk[(self.sk.season_end <= V)
                                         & (self.sk.toi_min >= 200)].groupby("pos_group")}
        rows = []
        for team, d in st.groupby("team"):
            w = d.toi_min.to_numpy()
            a = d.age.to_numpy()
            toi_age = float(np.average(a, weights=w))
            share_u23 = float(w[a < 23].sum() / w.sum())
            share_32p = float(w[a >= 32].sum() / w.sum())
            pts = d.I_F_points.to_numpy()
            pg = sk_v.pos_group.reindex(d.playerId).fillna("F").to_numpy()
            rel = np.zeros(len(d))
            for grp in ("F", "D"):
                mask = pg == grp
                if mask.any():
                    f_now = P.curve_value(fe[grp], a[mask])
                    f_fut = P.curve_value(fe[grp], a[mask] + h)
                    rel[mask] = (f_fut - f_now) / max(pos_mean.get(grp, 2.0), 0.8)
            prod_age_exp = float((pts * rel).sum() / max(pts.sum(), 1.0))
            top3 = np.sort(d.groupby("playerId").I_F_points.sum().to_numpy())[-3:].sum()
            star_share = float(top3 / max(pts.sum(), 1.0))
            rows.append({"team": team, "toi_age": toi_age, "share_u23": share_u23,
                         "share_32p": share_32p, "prod_age_exp": prod_age_exp,
                         "star_share": star_share})
        return pd.DataFrame(rows).set_index("team")

    def team_returning_toi(self, V: int) -> pd.Series:
        cur = self.skt[self.skt.season_end == V]
        prev = P.majority_team(self.skt[self.skt.season_end == V - 1])
        prev_map = prev.set_index("playerId").team
        ret = {}
        for team, d in cur.groupby("team"):
            same = d.playerId.map(prev_map) == team
            ret[team] = float(d.toi_min[same].sum() / d.toi_min.sum())
        return pd.Series(ret, name="returning_toi")

    def team_draft_cap(self, V: int, teams) -> pd.Series:
        d = self.drafts[self.drafts.draft_year.isin(vantage_drafts(V))]
        cap = d.groupby("team").value.sum()
        return cap.reindex(teams).fillna(0.0).rename("draft_cap")

    def team_player_points(self, V: int, h: int) -> pd.Series:
        m = self.marcel(V, h).set_index("playerId")
        roster = P.majority_team(self.skt[self.skt.season_end == V])
        roster = roster[roster.playerId.isin(m.index)]
        pts = m.proj_points.reindex(roster.playerId).to_numpy()
        return pd.Series(pts, index=roster.team.to_numpy()).groupby(level=0).sum() \
            .rename("player_points_proj")

    def team_finishing(self, V: int) -> pd.Series:
        """Roster expected goals-above-xG per season (EB-shrunk shooters, majority team)."""
        if not hasattr(self, "_fin_cache"):
            self._fin_cache = {}
        if V not in self._fin_cache:
            fin = P.finishing_project(self.sk, V).set_index("playerId")
            roster = P.majority_team(self.skt[self.skt.season_end == V])
            roster = roster[roster.playerId.isin(fin.index)]
            g_extra = (fin.theta_fin.reindex(roster.playerId)
                       * fin.sog82.reindex(roster.playerId)).to_numpy()
            self._fin_cache[V] = pd.Series(g_extra, index=roster.team.to_numpy()) \
                .groupby(level=0).sum().rename("finishing")
        return self._fin_cache[V]

    def team_st(self, V: int) -> pd.DataFrame:
        """Special teams process rates at vantage: PP xGF/60 (5on4), PK -xGA/60 (4on5)."""
        if not hasattr(self, "_st_cache"):
            self._st_cache = {}
        if V not in self._st_cache:
            f = RAW / f"mp_teams_{V - 1}.csv"
            mp = pd.read_csv(f, usecols=["team", "situation", "iceTime",
                                         "xGoalsFor", "xGoalsAgainst"])
            mp["team"] = mp.team.replace(P.MP_FRAN)
            pp = mp[mp.situation == "5on4"].set_index("team")
            pk = mp[mp.situation == "4on5"].set_index("team")
            out = pd.DataFrame({
                "st_pp": pp.xGoalsFor / (pp.iceTime / 3600.0),
                "st_pk": -(pk.xGoalsAgainst / (pk.iceTime / 3600.0)),
            })
            self._st_cache[V] = out
        return self._st_cache[V]

    # ---------------- assemble ----------------
    def team_features(self, V: int, h: int) -> pd.DataFrame:
        teams = sorted(self.end_r[V].keys())
        df = pd.DataFrame(index=pd.Index(teams, name="team"))
        df["elo_dev"] = [self.end_r[V][t] - E.MEAN for t in teams]
        xg = self.ts.xg_pct_all
        df["xg_dev"] = [xg.get((V, t), np.nan) - 0.5 for t in teams]
        df = df.join(self.team_gsax_history(V), how="left")
        df = df.join(self.team_goalies(V), how="left")
        df = df.join(self.team_age_structure(V, h), how="left")
        df = df.join(self.team_returning_toi(V), how="left")
        df = df.join(self.team_draft_cap(V, teams), how="left")
        df = df.join(self.team_player_points(V, h), how="left")
        df = df.join(self.team_finishing(V), how="left")
        df = df.join(self.team_st(V), how="left")
        df["goalie_age"] = df.goalie_age.fillna(28.0)
        df["toi_age"] = df.toi_age.fillna(27.0)
        return df

    def target(self, T: int) -> pd.Series | None:
        """Per-82 points deviation for season T; None if T not yet played."""
        try:
            t = self.ts.xs(T, level="season_end")
        except KeyError:
            return None
        return (t.pts_pct - t.pts_pct.mean()) * 164

    def feature_matrix(self, predict_seasons: list[int], h: int, feats: list[str] | None = None):
        """Stack pairs (V=T-h -> T). Centering within V; scaling by expanding sd over
        vantages <= V; missing (expansion debut in T) -> 0 vector."""
        FEATURES = feats or globals()["FEATURES"]
        frames = []
        for T in predict_seasons:
            V = T - h
            f = self.team_features(V, h)
            f = f - f.mean()  # within-vantage centering (also handles league drift)
            f["_V"], f["_T"] = V, T
            frames.append(f.reset_index())
        raw = pd.concat(frames, ignore_index=True)
        X_parts, metas = [], []
        for T in predict_seasons:
            V = T - h
            cur = raw[raw._T == T].set_index("team")
            hist = raw[raw._V <= V]
            scale = hist[FEATURES].std(ddof=1).replace(0, 1.0)
            y = self.target(T)
            teams_T = y.index if y is not None else cur.index  # future season: vantage teams
            yv = y.to_numpy(float) if y is not None else np.full(len(teams_T), np.nan)
            Xt = cur[FEATURES].reindex(teams_T)
            Xt = (Xt / scale).fillna(0.0)  # expansion debuts -> league-average vector
            X_parts.append(Xt.to_numpy(float))
            metas.append(pd.DataFrame({"team": teams_T, "T": T, "y": yv}))
        X = np.vstack(X_parts)
        meta = pd.concat(metas, ignore_index=True)
        return X, meta.y.to_numpy(float), meta


# ------------------------------------------------------------------ constants
def elo_per_gpg(end_ratings: dict, ts: pd.DataFrame, max_season: int) -> float:
    xs, ys = [], []
    for s in range(2010, max_season + 1):
        t = ts[ts.season_end == s]
        for row in t.itertuples(index=False):
            if row.team in end_ratings.get(s, {}):
                xs.append((row.gf - row.ga) / row.gp)
                ys.append(end_ratings[s][row.team] - E.MEAN)
    xs, ys = np.array(xs), np.array(ys)
    return float((xs @ ys) / (xs @ xs))


def pts_per_elo(om: dict, rng=None) -> float:
    """Numeric: perturb one team +20 Elo on the 84-game synthetic schedule."""
    rng = rng or np.random.default_rng(5)
    sched = E.synthetic_schedule_84(E.DIVISIONS_CURRENT, rng)
    base = {t: 1505.0 for dv in E.DIVISIONS_CURRENT.values() for t in dv}
    xp0 = E.analytic_xpts(base, sched, om)
    pert = dict(base)
    pert["BOS"] += 20.0
    xp1 = E.analytic_xpts(pert, sched, om)
    return float((xp1["BOS"] - xp0["BOS"]) / 20.0 * 82 / 84)  # per-82 convention
