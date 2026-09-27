"""Assemble the NeurHL-G master tensor: one record per regular-season game.

Per game (seasons 2009-2026) and side (0 = home, 1 = away):
  SK   (N,2,20,Fs)  skater pre-game state (dressed skaters, ordered by
                    expected ice time; mask marks real rows)
  SKB  (N,2,20,Bs)  shrunk skater baselines the residual heads start from
  SKY  (N,2,20,Ts)  skater targets (the game's own outcomes)
  GK   (N,2,Fg)     starting goalie state;  GKB baselines;  GKY targets
  TM   (N,2,Ft)     team state + side schedule context
  TMY  (N,2,Tt)     team targets
  CTX  (N,Fc)       game context (home Elo logit, era vector, days in)
  meta              game_id, season, date, outcome4, regulation goals, ids

Baselines shrink each player's discounted history toward a position prior
fitted on burn-in seasons 2008-2011 only, so no baseline reads a scored
season's outcomes. Everything else is already pre-game (build_g_state.py).

Usage: ... python neurhl/data/build_g_tensors.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

SEASONS = list(range(2009, 2027))
NS = 20
SK_TGT = ["toi_ev", "toi_pp", "toi_sh", "isog", "iatt", "ixg_all", "g", "a",
          "oi_xgf_all", "oi_xga_all"]
GK_TGT = ["toi_m", "sf", "ga", "xga"]
TM_TGT = ["xgf_ev", "xgf_pp", "xgf_sh", "xgf_all", "sogf", "attf", "gf_reg",
          "pp_opps", "pp_m"]
CTX_COLS = ["elo_logit", "prior_gpg", "prior_ot_share", "prior_so_share",
            "prior_margin_abs", "prior_parity", "flag_3v3", "flag_covid",
            "season_scaled", "days_in"]
SIDE_CTX = ["rest", "b2b", "km3d", "dtz"]
# baseline: (level column at decay tag, prior by position [F, D], pseudo-games)
SK_BASE = {
    "toi_ev": ("toi_ev_pg_d85", None, 3.0),
    "toi_pp": ("toi_pp_pg_d85", None, 3.0),
    "toi_sh": ("toi_sh_pg_d85", None, 3.0),
    "isog60": ("isog60_d95", None, 8.0),
    "iatt60": ("iatt60_d95", None, 8.0),
    "ixg60": ("ixg_ev60_d95", None, 8.0),
    "g_per_sog": ("shpct_d98", None, 30.0),
    "a60": ("a60_d95", None, 10.0),
    "xgf_ev60": ("xgf_ev60_d95", None, 8.0),
    "xga_ev60": ("xga_ev60_d95", None, 8.0),
}


def season_frames():
    sk = pd.concat([pd.read_parquet(TENSORS / f"gst_sk_{s}.parquet") for s in SEASONS],
                   ignore_index=True)
    gk = pd.concat([pd.read_parquet(TENSORS / f"gst_gk_{s}.parquet") for s in SEASONS],
                   ignore_index=True)
    tm = pd.concat([pd.read_parquet(TENSORS / f"gst_tm_{s}.parquet") for s in SEASONS],
                   ignore_index=True)
    return sk, gk, tm


def position_priors(sk: pd.DataFrame) -> dict:
    burn = sk[sk.season_end <= 2011]
    pri = {}
    for name, (col, _, _) in SK_BASE.items():
        pri[name] = [float(np.nanmedian(burn.loc[burn.pos_group == p, col]))
                     for p in (0, 1)]
    return pri


def shrink(sk: pd.DataFrame, pri: dict) -> pd.DataFrame:
    out = {}
    pos = sk.pos_group.clip(0, 1).to_numpy()
    for name, (col, _, k) in SK_BASE.items():
        tag = col.rsplit("_", 1)[1]
        n = sk[f"neff_{tag}"].to_numpy()
        x = sk[col].to_numpy()
        p = np.array(pri[name])[pos]
        x = np.where(np.isfinite(x), x, p)
        out[f"b_{name}"] = (n * x + k * p) / (n + k)
    return pd.DataFrame(out, index=sk.index)


def reg_goals() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        ev = pd.read_parquet(TENSORS / f"stream_{s}.parquet",
                             columns=["game_id", "game_type", "event_type",
                                      "period", "home_event"])
        g = ev[(ev.game_type == 2) & (ev.event_type == 7) & (ev.period <= 3)]
        c = g.groupby(["game_id", "home_event"]).size().unstack(fill_value=0)
        rows.append(pd.DataFrame({"gh_reg": c.get(1, 0), "ga_reg": c.get(0, 0)}))
    return pd.concat(rows)


def main():
    from train.train_game import elo_features
    sk, gk, tm = season_frames()
    pri = position_priors(sk)
    sk = pd.concat([sk, shrink(sk, pri)], axis=1)

    gc = pd.concat([pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet").assign(season_end=s)
                    for s in SEASONS], ignore_index=True)
    gc = gc[gc.game_type == 2].copy()
    elo = elo_features(gc)
    gc = gc.merge(elo[["elo_logit"]], left_on="game_id", right_index=True, how="inner")
    gc = gc.merge(reg_goals(), left_on="game_id", right_index=True, how="left")
    gc = gc.sort_values(["date", "game_id"]).reset_index(drop=True)
    gc["date"] = gc.date.astype(str)
    N = len(gc)
    gidx = {g: i for i, g in enumerate(gc.game_id)}

    skip = set(SK_TGT) | {"game_id", "player_id", "is_home", "pos_group", "team",
                          "season_end", "date", "game_type"}
    raw = {"toi_sec", "goalie_start", "goals", "sog", "att", "blocks", "pen",
           "fo_w", "fo_l", "assists", "ev_toi", "pp_toi", "sh_toi", "other_toi",
           "pp_rank", "rank5", "cf", "ca", "toi_all", "blk", "pp1", "gp", "has_xg",
           "pen_taken", "pen_drawn", "oi_sogf_all", "oi_soga_all", "iatt_all",
           "isog", "iatt", "ixg_ev", "ixg_pp", "ixg_sh", "g_ev", "g_pp", "g_sh",
           "g_all", "isog_ev", "isog_pp", "isog_sh", "isog_all", "oi_xgf_ev",
           "oi_xga_ev", "oi_xgf_pp", "oi_xga_pp", "oi_xgf_sh", "oi_xga_sh", "a"}
    sk_feat = [c for c in sk.columns if c not in skip and c not in raw
               and not c.startswith("b_") and sk[c].dtype.kind in "fi"]
    sk_base = [c for c in sk.columns if c.startswith("b_")]
    gk_feat = [c for c in gk.columns if c.startswith("gk_")]
    tm_feat = [c for c in tm.columns if c.startswith("tm_")]

    sk = sk[sk.game_id.isin(gidx)].copy()
    sk["gi"] = sk.game_id.map(gidx)
    sk["side"] = 1 - sk.is_home.astype(int)
    sk = sk.sort_values(["gi", "side", "b_toi_ev"], ascending=[True, True, False])
    sk["slot"] = sk.groupby(["gi", "side"]).cumcount()
    over = int((sk.slot >= NS).sum())
    sk = sk[sk.slot < NS]

    SK = np.full((N, 2, NS, len(sk_feat)), np.nan, np.float32)
    SKB = np.full((N, 2, NS, len(sk_base)), np.nan, np.float32)
    SKY = np.full((N, 2, NS, len(SK_TGT)), np.nan, np.float32)
    SKM = np.zeros((N, 2, NS), np.float32)
    SKP = np.zeros((N, 2, NS), np.int8)
    SKID = np.zeros((N, 2, NS), np.int64)
    ii, ss, kk = sk.gi.to_numpy(), sk.side.to_numpy(), sk.slot.to_numpy()
    SK[ii, ss, kk] = sk[sk_feat].to_numpy(np.float32)
    SKB[ii, ss, kk] = sk[sk_base].to_numpy(np.float32)
    SKY[ii, ss, kk] = sk[SK_TGT].to_numpy(np.float32)
    SKM[ii, ss, kk] = 1.0
    SKP[ii, ss, kk] = sk.pos_group.clip(0, 1).to_numpy()
    SKID[ii, ss, kk] = sk.player_id.to_numpy()

    st = gk[(gk.start == 1) & gk.game_id.isin(gidx)].copy()
    st["gi"] = st.game_id.map(gidx)
    st["side"] = 1 - st.is_home.astype(int)
    st = st.drop_duplicates(["gi", "side"])
    GK = np.full((N, 2, len(gk_feat)), np.nan, np.float32)
    GKY = np.full((N, 2, len(GK_TGT)), np.nan, np.float32)
    GKID = np.zeros((N, 2), np.int64)
    GK[st.gi, st.side] = st[gk_feat].to_numpy(np.float32)
    GKY[st.gi, st.side] = st[GK_TGT].to_numpy(np.float32)
    GKID[st.gi, st.side] = st.player_id.to_numpy()

    t = tm[tm.game_id.isin(gidx)].copy()
    t["gi"] = t.game_id.map(gidx)
    t["side"] = 1 - t.is_home.astype(int)
    g2 = gc.set_index("game_id")
    h = t.side == 0
    t["rest"] = np.where(h, t.game_id.map(g2.home_rest), t.game_id.map(g2.away_rest))
    t["km3d"] = np.where(h, t.game_id.map(g2.home_km3d), t.game_id.map(g2.away_km3d))
    t["dtz"] = np.where(h, t.game_id.map(g2.home_dtz), t.game_id.map(g2.away_dtz))
    t["b2b"] = (t.rest <= 1).astype(float)
    t["gf_reg"] = np.where(h, t.game_id.map(g2.gh_reg), t.game_id.map(g2.ga_reg))
    tmf = tm_feat + SIDE_CTX
    TM = np.full((N, 2, len(tmf)), np.nan, np.float32)
    TMY = np.full((N, 2, len(TM_TGT)), np.nan, np.float32)
    TM[t.gi, t.side] = t[tmf].to_numpy(np.float32)
    for j, c in enumerate(TM_TGT):
        if c in t:
            TMY[t.gi, t.side, j] = t[c].to_numpy(np.float32)
    CTX = gc[CTX_COLS].to_numpy(np.float32)

    meta = gc[["game_id", "season_end", "date", "outcome4", "home_g", "away_g",
               "gh_reg", "ga_reg", "home_idx", "away_idx"]].copy()
    np.savez_compressed(TENSORS / "g_master.npz", SK=SK, SKB=SKB, SKY=SKY, SKM=SKM,
                        SKP=SKP, SKID=SKID, GK=GK, GKY=GKY, GKID=GKID, TM=TM,
                        TMY=TMY, CTX=CTX)
    meta.to_parquet(TENSORS / "g_meta.parquet", index=False)
    names = {"sk_feat": sk_feat, "sk_base": sk_base, "sk_tgt": SK_TGT,
             "gk_feat": gk_feat, "gk_tgt": GK_TGT, "tm_feat": tmf,
             "tm_tgt": TM_TGT, "ctx": CTX_COLS, "priors": pri}
    (TENSORS / "g_names.json").write_text(json.dumps(names, indent=1))
    print(f"games {N:,} | skater feats {len(sk_feat)}, baselines {len(sk_base)}, "
          f"goalie feats {len(gk_feat)}, team feats {len(tmf)}, ctx {len(CTX_COLS)}")
    print(f"skater slots filled {SKM.sum(-1).mean():.2f} per side (over {NS}: {over}); "
          f"starters found {np.isfinite(GK[:, :, 0]).mean():.3f}; "
          f"reg goals found {meta.gh_reg.notna().mean():.3f}")


if __name__ == "__main__":
    main()
