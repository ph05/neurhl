"""X1 tasks 1-3: dressed lineups, walk-forward player values, expected lineup.

1. ``dressed()``: every skater who dressed in each regular-season game
   (fastRhockey box scores, 2011-2024; 2020 and 2024 partial), keyed by the
   ORR game row ``gid`` and side ('h' | 'a').
2. ``player_values(V)``: each dressed player's PRESEASON value for season V
   from ``orr.players.project(V)`` (seasons < V only):
     v_xgf  = tpg_ev * rel_xgf60 / 60     5v5 on-ice xGF impact per game
     v_xga  = tpg_ev * rel_xga60 / 60     5v5 on-ice xGA impact (+ = bad)
     v_ppf  = tpg_pp * pp_onice_rel * league PP xGF per minute
     v_pka  = tpg_sh * sh_onice_rel * league PK xGA per minute (+ = bad)
     toi    = projected minutes per game, all situations (tpg_* summed)
   (rel_* are shrunk toward 0, so a player with no NHL history is average.)
3. ``lineup_table(halflife)``: per game and side the dressed sum L of each
   value, the EXPECTED sum E = the dress-probability-weighted sum over the
   team's players, with each player's dress probability his exponentially
   weighted share of the team's earlier games this season (half-life in
   team games; E of a team's first game with box data = its own lineup, so
   that game carries no lineup signal), and the delta L - E. Every E uses
   only games dated before the game. No first-10-games roster is used.

Live the dressed lineups would come from the pregame lineup files; here the
box score of the game itself supplies who dressed (known before puck drop).
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from orr import data as D
from orr import players as PL
from orr import structural as S

COMPS = ("v_xgf", "v_xga", "v_ppf", "v_pka", "toi")
SEASONS = range(2011, 2025)


@functools.lru_cache(maxsize=None)
def dressed() -> pd.DataFrame:
    """gid, season_end, side ('h'|'a'), team, player_id, pos for every dressed
    skater of every regular-season game with a box score."""
    frames = []
    for y in SEASONS:
        f = D.FR / f"player_box_{y}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["player_id", "position_code", "home_away", "game_id"])
        p = p[(p.position_code != "G") & ((p.game_id // 10000) % 100 == 2)]
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    g = S.game_frame()
    g = g[g.game_id.notna()][["gid", "game_id", "season_end", "date", "home", "away"]].copy()
    g["game_id"] = g.game_id.astype("int64")
    p = p.merge(g, on="game_id", how="inner")
    p["side"] = np.where(p.home_away.str.lower() == "home", "h", "a")
    p["team"] = np.where(p.side == "h", p.home, p.away)
    p["pos"] = np.where(p.position_code == "D", "D", "F")
    return p[["gid", "season_end", "date", "side", "team", "player_id", "pos"]].reset_index(drop=True)


def _ref_season(V: int) -> int:
    return int(PL.project_league(V).name)


@functools.lru_cache(maxsize=None)
def league_xg_pg(V: int) -> float:
    """League xGF per team-game (all situations) in the reference season of
    V (the last intact season before V, as players.project_league)."""
    t = D.team_seasons()
    t = t[t.season_end == _ref_season(V)]
    return float(t.xgf_all.sum() / t.gp.sum())


@functools.lru_cache(maxsize=None)
def player_values(V: int) -> pd.DataFrame:
    """Preseason values (seasons < V only) of every skater who dressed in V."""
    prm = PL.load_params()
    ids = sorted(set(dressed().query("season_end == @V").player_id))
    pj = PL.project(V, prm, ids=ids)
    out = pd.DataFrame({"player_id": pj.player_id})
    out["v_xgf"] = pj.tpg_ev * pj.rel_xgf60 / 60.0
    out["v_xga"] = pj.tpg_ev * pj.rel_xga60 / 60.0
    ell_pp = pj.pp_onice_60 / (1.0 + pj.pp_onice_rel) / 60.0       # league PP xGF per minute
    ell_sh = pj.sh_onice_60 / (1.0 + pj.sh_onice_rel) / 60.0
    out["v_ppf"] = pj.tpg_pp * pj.pp_onice_rel * ell_pp
    out["v_pka"] = pj.tpg_sh * pj.sh_onice_rel * ell_sh
    out["toi"] = sum(pj[f"tpg_{k}"] for k in PL.SITS)
    out["has_hist"] = pj.has_hist.to_numpy()
    out["season_end"] = V
    # dressed players absent from the projection (none expected) are average
    miss = sorted(set(ids) - set(out.player_id))
    if miss:
        z = pd.DataFrame({"player_id": miss, "season_end": V, "has_hist": False})
        for c in COMPS:
            z[c] = 0.0
        out = pd.concat([out, z], ignore_index=True)
    return out.fillna({c: 0.0 for c in COMPS})


@functools.lru_cache(maxsize=None)
def lineup_sums() -> pd.DataFrame:
    """gid, season_end, date, side, team, n_sk and the dressed sums L_<comp>."""
    d = dressed()
    vals = pd.concat([player_values(V) for V in sorted(set(d.season_end))], ignore_index=True)
    d = d.merge(vals, on=["player_id", "season_end"], how="left")
    for c in COMPS:
        d[c] = d[c].fillna(0.0)
    s = d.groupby(["gid", "season_end", "date", "side", "team"], as_index=False).agg(
        n_sk=("player_id", "size"), **{f"L_{c}": (c, "sum") for c in COMPS})
    return s


@functools.lru_cache(maxsize=None)
def lineup_table(halflife: float = 40.0) -> pd.DataFrame:
    """Per game and side: L_<comp>, E_<comp> (expected: the team's
    exponentially weighted earlier lineups this season, i.e. the dress-
    probability-weighted value sum) and D_<comp> = L - E."""
    s = lineup_sums().sort_values(["team", "season_end", "date", "gid"]).reset_index(drop=True)
    lam = 0.5 ** (1.0 / halflife) if np.isfinite(halflife) else 1.0
    Lc = [f"L_{c}" for c in COMPS]
    E = np.zeros((len(s), len(COMPS)))
    L = s[Lc].to_numpy()
    k = np.zeros(len(s), int)
    for _, idx in s.groupby(["team", "season_end"]).indices.items():
        num = np.zeros(len(COMPS))
        den = 0.0
        for j, i in enumerate(idx):
            E[i] = L[i] if den == 0 else num / den
            k[i] = j
            num = lam * num + L[i]
            den = lam * den + 1.0
    for j, c in enumerate(COMPS):
        s[f"E_{c}"] = E[:, j]
        s[f"D_{c}"] = L[:, j] - E[:, j]
    s["k_team_game"] = k
    s["xg_pg"] = s.season_end.map({V: league_xg_pg(V) for V in sorted(set(s.season_end))})
    return s


def wide(halflife: float = 40.0) -> pd.DataFrame:
    """One row per gid with D_<comp>_h / D_<comp>_a and the league xG scale."""
    t = lineup_table(halflife)
    cols = [f"D_{c}" for c in COMPS]
    h = t[t.side == "h"].set_index("gid")[cols + ["xg_pg"]]
    a = t[t.side == "a"].set_index("gid")[cols]
    w = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    return w.reset_index()


def offsets(w: pd.DataFrame, beta_x: float, beta_st: float = 0.0, beta_t: float = 0.0,
            toi_scale: float = 300.0) -> pd.DataFrame:
    """Log offsets on regulation goals (home goals lo_h, away goals lo_a).

    lo_h = beta_x  * (D_xgf_h + D_xga_a) / xg_pg
         + beta_st * (D_ppf_h + D_pka_a) / xg_pg
         + beta_t  * (D_toi_h - D_toi_a) / toi_scale     (and symmetrically lo_a)
    """
    x = w.xg_pg.to_numpy()
    lo_h = (beta_x * (w.D_v_xgf_h + w.D_v_xga_a) + beta_st * (w.D_v_ppf_h + w.D_v_pka_a)) / x \
        + beta_t * (w.D_toi_h - w.D_toi_a) / toi_scale
    lo_a = (beta_x * (w.D_v_xgf_a + w.D_v_xga_h) + beta_st * (w.D_v_ppf_a + w.D_v_pka_h)) / x \
        + beta_t * (w.D_toi_a - w.D_toi_h) / toi_scale
    return pd.DataFrame({"gid": w.gid.to_numpy(), "lo_h": lo_h.to_numpy(), "lo_a": lo_a.to_numpy()})


if __name__ == "__main__":
    import time
    t0 = time.time()
    t = lineup_table(40.0)
    print(f"{len(t)} team-games, {time.time() - t0:.1f}s")
    print(t.groupby("season_end").agg(n=("gid", "size"), n_sk=("n_sk", "mean"),
                                       L_xgf=("L_v_xgf", "mean"), sdD_xgf=("D_v_xgf", "std"),
                                       sdD_xga=("D_v_xga", "std"), sdD_toi=("D_toi", "std")).round(4))
