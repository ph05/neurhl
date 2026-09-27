"""NeurHL-G player, goalie and team state before every game (PLAN_NeurHL4 M).

For each regular-season game, the state of every dressed skater, every goalie
and both teams is computed from games strictly before it: earlier games of the
same season and all earlier seasons. Game 50 sees games 1-49; game 2 sees
game 1; both see prior seasons.

State = exponentially discounted sums S_t = d * S_{t-1} + x_t, read BEFORE
game t, for decays d in DECAYS (1 - alpha, alpha = 0.02, 0.05, 0.15, 0.4). Rates
are ratios of discounted sums (ratio-of-sums: a 2-minute game weighs less than
a 20-minute one) and the discounted game count is the effective sample size,
so the network can learn how far to trust recent form. Season-to-date and
career totals are also carried.

Outputs per season s (data/tensors, gitignored):
  gst_sk_{s}  (game_id, player_id, is_home, pos_group, team) + state + targets
  gst_gk_{s}  per goalie-game: state + targets (started goalie flag)
  gst_tm_{s}  per (game, side): team state + game context + targets
Targets are the game's own outcomes (OUTCOME); every feature is shifted.

Usage: ... --with numba python neurhl/data/build_g_state.py
"""
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

SEASONS = list(range(2008, 2027))
DECAYS = (0.98, 0.95, 0.85, 0.6)
DTAG = ("d98", "d95", "d85", "d60")


@numba.njit(cache=True)
def disc_sums(X, grp, decay):
    """Discounted sums read BEFORE each row; reset at each new group."""
    n, k = X.shape
    out = np.zeros((n, k))
    S = np.zeros(k)
    for i in range(n):
        if i == 0 or grp[i] != grp[i - 1]:
            S[:] = 0.0
        out[i] = S
        S = decay * S + X[i]
    return out


def game_dates() -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        g = pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet")
        g = g[g.game_type == 2].assign(season_end=s)
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)
    g["date"] = pd.to_datetime(g.date.astype(str))
    return g


# ----------------------------------------------------------------- skaters
SK_X = ["toi_all", "toi_ev", "toi_pp", "toi_sh", "isog", "iatt", "ixg_ev",
        "ixg_pp", "ixg_all", "g", "a", "oi_xgf_ev", "oi_xga_ev", "oi_xgf_pp",
        "oi_xga_sh", "cf", "ca", "pen_taken", "pen_drawn", "blk", "fo_w",
        "fo_l", "pp1", "gp"]
# rate = numerator / denominator (minutes basis), both discounted sums
SK_RATES = {"isog60": ("isog", "toi_all"), "iatt60": ("iatt", "toi_all"),
            "ixg_ev60": ("ixg_ev", "toi_ev"), "ixg_pp60": ("ixg_pp", "toi_pp"),
            "g60": ("g", "toi_all"), "a60": ("a", "toi_all"),
            "xgf_ev60": ("oi_xgf_ev", "toi_ev"), "xga_ev60": ("oi_xga_ev", "toi_ev"),
            "xgf_pp60": ("oi_xgf_pp", "toi_pp"), "xga_sh60": ("oi_xga_sh", "toi_sh"),
            "cf60": ("cf", "toi_all"), "ca60": ("ca", "toi_all"),
            "pent60": ("pen_taken", "toi_all"), "pend60": ("pen_drawn", "toi_all"),
            "blk60": ("blk", "toi_all"), "fo_pct": ("fo_w", None),
            "shpct": ("g", "isog")}
SK_TOI = {"toi_pg": ("toi_all", "gp"), "toi_ev_pg": ("toi_ev", "gp"),
          "toi_pp_pg": ("toi_pp", "gp"), "toi_sh_pg": ("toi_sh", "gp"),
          "pp1_share": ("pp1", "gp")}


def skater_rows(dates: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        pg = pd.read_parquet(TENSORS / f"player_games_{s}.parquet")
        pg = pg[(pg.game_type == 2) & (pg.pos_group != 2)]
        u = pd.read_parquet(TENSORS / f"usage_{s}.parquet",
                            columns=["game_id", "player_id", "team", "ev_toi",
                                     "pp_toi", "sh_toi", "other_toi", "pp_rank",
                                     "rank5"])
        oi = pd.read_parquet(TENSORS / f"onice_rates_{s}.parquet",
                             columns=["game_id", "player_id", "cf", "ca"])
        d = pg.merge(u, on=["game_id", "player_id"], how="left") \
              .merge(oi, on=["game_id", "player_id"], how="left")
        px = TENSORS / f"pgx_{s}.parquet"
        if px.exists():
            x = pd.read_parquet(px).drop(columns=["is_home", "pos_group"])
            d = d.merge(x, on=["game_id", "player_id"], how="left")
            d["has_xg"] = 1.0
        else:
            d["has_xg"] = 0.0
        d["season_end"] = s
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d.merge(dates[["game_id", "date"]], on="game_id", how="inner")
    # Position from the bio table: the HTM-era player_games (2008-2011) code
    # every skater as pos_group 0, with no defencemen. Bio positions agree
    # 100% with player_games wherever both exist (checked on 2016).
    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    d["pos_group"] = d.player_id.map(bios.pos_group).fillna(d.pos_group).clip(0, 1)
    m = 1 / 60.0
    d["toi_all"] = d.toi_sec * m
    d["toi_ev"] = (d.ev_toi.fillna(d.toi_sec) + d.other_toi.fillna(0)) * m
    d["toi_pp"] = d.pp_toi.fillna(0) * m
    d["toi_sh"] = d.sh_toi.fillna(0) * m
    d["isog"], d["iatt"], d["a"], d["blk"] = d.sog, d.att, d.assists, d.blocks
    d["g"] = d.goals
    d["pp1"] = d.pp_rank.between(1, 5).astype(float)
    d["gp"] = 1.0
    for c in SK_X:
        if c not in d:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0).astype(float)
    return d.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)


def add_state(d: pd.DataFrame, cols: list, rates: dict, levels: dict,
              key: str = "player_id") -> pd.DataFrame:
    grp = pd.factorize(d[key])[0]
    X = d[cols].to_numpy(float)
    idx = {c: i for i, c in enumerate(cols)}
    feats = {}
    for dec, tag in zip(DECAYS, DTAG):
        S = disc_sums(X, grp, dec)
        for name, (num, den) in rates.items():
            if den is None:        # faceoff pct: wins / (wins + losses)
                w, l_ = S[:, idx["fo_w"]], S[:, idx["fo_l"]]
                feats[f"{name}_{tag}"] = np.where(w + l_ > 0, w / np.maximum(w + l_, 1e-9), np.nan)
            else:
                nu, de = S[:, idx[num]], S[:, idx[den]]
                scale = 60.0 if name.endswith("60") else 1.0
                feats[f"{name}_{tag}"] = np.where(de > 1e-6, scale * nu / np.maximum(de, 1e-9), np.nan)
        for name, (num, den) in levels.items():
            nu, de = S[:, idx[num]], S[:, idx[den]]
            feats[f"{name}_{tag}"] = np.where(de > 1e-6, nu / np.maximum(de, 1e-9), np.nan)
        feats[f"neff_{tag}"] = S[:, idx["gp"]]
    # season-to-date (reset each season) and career
    sgrp = pd.factorize(d[key].astype(str) + "_" + d.season_end.astype(str))[0]
    Ss = disc_sums(X, sgrp, 1.0)
    Sc = disc_sums(X, grp, 1.0)
    feats["gp_season"] = Ss[:, idx["gp"]]
    feats["gp_career"] = Sc[:, idx["gp"]]
    for name, (num, den) in list(rates.items())[:6]:
        if den is None:
            continue
        nu, de = Sc[:, idx[num]], Sc[:, idx[den]]
        feats[f"{name}_career"] = np.where(de > 1e-6, 60 * nu / np.maximum(de, 1e-9), np.nan)
    return pd.concat([d, pd.DataFrame(feats, index=d.index)], axis=1)


def skater_state(dates: pd.DataFrame) -> pd.DataFrame:
    d = skater_rows(dates)
    d = add_state(d, SK_X, SK_RATES, SK_TOI)
    g = d.groupby("player_id")
    d["days_since"] = (d.date - g.date.shift(1)).dt.days.fillna(365).clip(0, 365)
    team_prev = g.team.shift(1)
    d["team_change"] = (team_prev.notna() & (team_prev != d.team)).astype(float)
    tg = (d.team != team_prev).cumsum()
    d["gp_with_team"] = d.groupby([d.player_id, tg]).cumcount().astype(float)
    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    d["age"] = (d.season_end - d.player_id.map(bios.birth_year)).clip(17, 45)
    d["height"] = d.player_id.map(bios.height_in)
    d["weight"] = d.player_id.map(bios.weight_lb)
    d["draft_overall"] = d.player_id.map(bios.draft_overall).fillna(260)
    # as-of RAPM prior: season s reads rapm_prior_{s} (fit on seasons < s)
    rp = []
    for s in SEASONS:
        p = TENSORS / f"rapm_prior_{s}.parquet"
        if p.exists():
            r = pd.read_parquet(p, columns=["player_id", "cf_off", "cf_def",
                                            "g_off", "g_def", "cf_off_se",
                                            "cf_def_se", "is_replacement"])
            rp.append(r.assign(season_end=s))
    rp = pd.concat(rp, ignore_index=True)
    rp.columns = ["player_id"] + [f"rapm_{c}" for c in rp.columns[1:-1]] + ["season_end"]
    d = d.merge(rp, on=["player_id", "season_end"], how="left")
    return d


# ----------------------------------------------------------------- goalies
GK_X = ["toi_m", "sf", "ga", "xga", "start", "gp"]


def goalie_state(dates: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for s in SEASONS:
        g = pd.read_parquet(TENSORS / f"goalie_games_{s}.parquet")
        parts.append(g.assign(season_end=s))
    g = pd.concat(parts, ignore_index=True)
    g = g.merge(dates[["game_id", "date"]], on="game_id", how="inner")
    g["toi_m"] = g.toi_sec / 60.0
    g["xga"] = g.xgf.fillna(0.0)            # goalie_games.xgf = xG FACED
    g["start"] = g.goalie_start.astype(float)
    g["gp"] = (g.toi_sec > 0).astype(float)
    for c in GK_X:
        g[c] = g[c].fillna(0.0).astype(float)
    g = g.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    grp = pd.factorize(g.player_id)[0]
    X = g[GK_X].to_numpy(float)
    feats = {}
    for dec, tag in zip(DECAYS, DTAG):
        S = disc_sums(X, grp, dec)
        sf, ga, xga, toi = S[:, 1], S[:, 2], S[:, 3], S[:, 0]
        feats[f"gk_svpct_{tag}"] = np.where(sf > 0, 1 - ga / np.maximum(sf, 1e-9), np.nan)
        feats[f"gk_gsax_rate_{tag}"] = np.where(xga > 0, (xga - ga) / np.maximum(xga, 1e-9), np.nan)
        feats[f"gk_xga60_{tag}"] = np.where(toi > 0, 60 * xga / np.maximum(toi, 1e-9), np.nan)
        feats[f"gk_start_share_{tag}"] = np.where(S[:, 5] > 0, S[:, 4] / np.maximum(S[:, 5], 1e-9), np.nan)
        feats[f"gk_neff_{tag}"] = S[:, 5]
    Sc = disc_sums(X, grp, 1.0)
    feats["gk_shots_career"] = Sc[:, 1]
    # shrunk GSAx per xG toward a replacement prior (n0 = 150 xG faced)
    feats["gk_gsax_shrunk"] = (Sc[:, 3] - Sc[:, 2]) / (Sc[:, 3] + 150.0)
    g = pd.concat([g, pd.DataFrame(feats, index=g.index)], axis=1)
    g["gk_days_since"] = (g.date - g.groupby("player_id").date.shift(1)).dt.days.fillna(365).clip(0, 365)
    return g


# ------------------------------------------------------------------ teams
TM_X = ["xgf_ev", "xga_ev", "xgf_pp", "xga_sh", "xgf_all", "xga_all", "gf",
        "ga", "sogf", "soga", "attf", "atta", "pp_opps", "pk_opps", "ev_m",
        "pp_m", "sh_m", "gp"]


def team_state(dates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        p = TENSORS / f"tgx_{s}.parquet"
        g = dates[dates.season_end == s]
        if p.exists():
            t = pd.read_parquet(p)
        else:                    # 2008: no stream xG; goals only
            t = pd.concat([pd.DataFrame({"game_id": g.game_id, "is_home": h})
                           for h in (1, 0)], ignore_index=True)
        t = t.merge(g[["game_id", "date", "home_idx", "away_idx", "home_g",
                       "away_g", "season_end"]], on="game_id", how="inner")
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["team"] = np.where(t.is_home == 1, t.home_idx, t.away_idx)
    t["gf"] = np.where(t.is_home == 1, t.home_g, t.away_g).astype(float)
    t["ga"] = np.where(t.is_home == 1, t.away_g, t.home_g).astype(float)
    opp = t.set_index(["game_id", "is_home"])
    t["pk_opps"] = t.set_index(["game_id", "is_home"]).index.map(
        lambda k: opp.pp_opps.get((k[0], 1 - k[1]), np.nan) if "pp_opps" in opp else np.nan)
    for c in ("pp_toi", "sh_toi"):
        if c not in t:
            t[c] = np.nan
    t["pp_m"] = t.pp_toi / 60.0
    t["sh_m"] = t.sh_toi / 60.0
    t["ev_m"] = 60.0 - t.pp_m.fillna(0) - t.sh_m.fillna(0)
    t["gp"] = 1.0
    for c in TM_X:
        if c not in t:
            t[c] = np.nan
    t["has_xg"] = t.xgf_all.notna().astype(float)
    X = t[TM_X].fillna(0.0).to_numpy(float)
    t = t.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    X = t[TM_X].fillna(0.0).to_numpy(float)
    grp = pd.factorize(t.team)[0]
    feats = {}
    for dec, tag in zip(DECAYS, DTAG):
        S = disc_sums(X, grp, dec)
        i = {c: j for j, c in enumerate(TM_X)}
        per60 = lambda a, b: np.where(S[:, i[b]] > 1e-6, 60 * S[:, i[a]] / np.maximum(S[:, i[b]], 1e-9), np.nan)
        pergm = lambda a: np.where(S[:, i["gp"]] > 0, S[:, i[a]] / np.maximum(S[:, i["gp"]], 1e-9), np.nan)
        feats[f"tm_xgf_ev60_{tag}"] = per60("xgf_ev", "ev_m")
        feats[f"tm_xga_ev60_{tag}"] = per60("xga_ev", "ev_m")
        feats[f"tm_xgf_pp60_{tag}"] = per60("xgf_pp", "pp_m")
        feats[f"tm_xga_sh60_{tag}"] = per60("xga_sh", "sh_m")
        for c in ("gf", "ga", "sogf", "soga", "xgf_all", "xga_all", "pp_opps", "pk_opps"):
            feats[f"tm_{c}_pg_{tag}"] = pergm(c)
        feats[f"tm_neff_{tag}"] = S[:, i["gp"]]
    t = pd.concat([t, pd.DataFrame(feats, index=t.index)], axis=1)
    t["tm_gp_season"] = t.groupby(["team", "season_end"]).cumcount().astype(float)
    return t


def main():
    dates = game_dates()
    print(f"{len(dates):,} regular-season games {dates.season_end.min()}-{dates.season_end.max()}")
    sk = skater_state(dates)
    gk = goalie_state(dates)
    tm = team_state(dates)
    for s in SEASONS:
        sk[sk.season_end == s].to_parquet(TENSORS / f"gst_sk_{s}.parquet", index=False)
        gk[gk.season_end == s].to_parquet(TENSORS / f"gst_gk_{s}.parquet", index=False)
        tm[tm.season_end == s].to_parquet(TENSORS / f"gst_tm_{s}.parquet", index=False)
    print(f"skater rows {len(sk):,} ({sk.shape[1]} cols), goalie rows {len(gk):,}, "
          f"team rows {len(tm):,}")
    chk = sk[sk.season_end == 2024]
    print("2024 means: toi_pg_d85", round(chk.toi_pg_d85.mean(), 2),
          " ixg_ev60_d85", round(chk.ixg_ev60_d85.mean(), 3),
          " neff_d85", round(chk.neff_d85.mean(), 2),
          " rapm coverage", round(chk.rapm_cf_off.notna().mean(), 3))


if __name__ == "__main__":
    main()
