"""NeurHL-G targets and histories: player-game and team-game xG and box score.

Per season, from the event stream (walk-forward house xG per unblocked
attempt, shooter, on-ice slots) and the stint table (strength by time):

  pgx_{s}.parquet  one row per (game, skater):
      ixg_{ev,pp,sh,all}, isog_{ev,pp,sh,all}, iatt_all, g_{ev,pp,sh,all},
      oi_xgf_{ev,pp,sh,all}, oi_xga_{ev,pp,sh,all}, oi_sogf_all, oi_soga_all,
      pen_taken, pen_drawn
  tgx_{s}.parquet  one row per (game, side):
      xgf/xga, sogf/soga, gf/ga by strength and all, attf/atta,
      pp_toi, sh_toi (seconds), pp_opps, en_gf (goals into an empty net)

Strength is the SHOOTING team's view by skater counts: pp if it has more
skaters, sh if fewer, ev if equal (5v5, 4v4, 3v3). Empty-net events (stream
strength_key 7/8) count only in "all". Goalies are removed from on-ice slots by
position (player_games pos_group == 2), never by slot index.

PP opportunities are maximal runs of stints in which a side has more skaters
with both goalies in net (stint strength_key 1/2 home, 3/4 away), merging runs
separated by less than 2 seconds.

These are OUTCOME tables: a game's rows use only that game's events. Model
features must shift them (models/player_state.py). Regular season only.

Usage: ... python neurhl/data/build_xg_games.py [--seasons 2009-2026]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

BLOCK, GOAL, MISS, PEN, SOG = 1, 7, 9, 10, 14
SLOTS_H = [f"h_on{i}" for i in range(7)]
SLOTS_A = [f"a_on{i}" for i in range(7)]
STR = ("ev", "pp", "sh")


def goalie_ids(season: int) -> set:
    pg = pd.read_parquet(TENSORS / f"player_games_{season}.parquet",
                         columns=["player_id", "pos_group"])
    return set(pg.loc[pg.pos_group == 2, "player_id"].astype(int))


def strength(ev: pd.DataFrame) -> pd.Series:
    s = np.where(ev.n_for > ev.n_against, "pp",
                 np.where(ev.n_for < ev.n_against, "sh", "ev"))
    s = pd.Series(s, index=ev.index)
    s[ev.strength_key.isin([7, 8])] = "en"
    return s


def build_player(ev: pd.DataFrame, gids: set, season: int) -> pd.DataFrame:
    rs = ev[ev.game_type == 2].copy()
    rs["str"] = strength(rs)
    shots = rs[rs.event_type.isin([BLOCK, MISS, SOG, GOAL])].copy()
    shots["is_sog"] = shots.event_type.isin([SOG, GOAL]).astype(float)
    shots["is_goal"] = (shots.event_type == GOAL).astype(float)
    shots["xg0"] = np.where(shots.has_xg == 1, shots.xg, 0.0)

    # individual: shooter = p1
    ind = shots[shots.p1 > 0]
    rows = {}
    for st in STR + ("all",):
        m = ind if st == "all" else ind[ind.str == st]
        g = m.groupby(["game_id", "p1"])
        rows[f"ixg_{st}"] = g.xg0.sum()
        rows[f"isog_{st}"] = g.is_sog.sum()
        rows[f"g_{st}"] = g.is_goal.sum()
    rows["iatt_all"] = ind.groupby(["game_id", "p1"]).size().astype(float)
    indiv = pd.DataFrame(rows)
    indiv.index.names = ["game_id", "player_id"]

    # on-ice: melt slots; for = shooter's side
    base = shots[["game_id", "home_event", "str", "xg0", "is_sog"] + SLOTS_H + SLOTS_A]
    parts = []
    for side_cols, is_home in ((SLOTS_H, 1), (SLOTS_A, 0)):
        m = base.melt(id_vars=["game_id", "home_event", "str", "xg0", "is_sog"],
                      value_vars=side_cols, value_name="player_id")
        m = m[(m.player_id > 0)]
        m = m[~m.player_id.astype(int).isin(gids)]
        m["for"] = (m.home_event == is_home)
        m["is_home"] = is_home
        parts.append(m[["game_id", "player_id", "is_home", "for", "str", "xg0",
                        "is_sog"]])
    oi = pd.concat(parts, ignore_index=True)
    # strength from the PLAYER's side: invert pp/sh when the event is against
    oi["pstr"] = oi["str"]
    oi.loc[~oi["for"] & (oi["str"] == "pp"), "pstr"] = "sh"
    oi.loc[~oi["for"] & (oi["str"] == "sh"), "pstr"] = "pp"
    orow = {}
    for st in STR + ("all",):
        m = oi if st == "all" else oi[oi.pstr == st]
        for side, lab in ((True, "f"), (False, "a")):
            mm = m[m["for"] == side]
            orow[f"oi_xg{lab}_{st}"] = mm.groupby(["game_id", "player_id"]).xg0.sum()
    for side, lab in ((True, "f"), (False, "a")):
        mm = oi[oi["for"] == side]
        orow[f"oi_sog{lab}_all"] = mm.groupby(["game_id", "player_id"]).is_sog.sum()
    onice = pd.DataFrame(orow)

    pen = rs[rs.event_type == PEN]
    pens = pd.DataFrame({
        "pen_taken": pen[pen.p1 > 0].groupby(["game_id", "p1"]).size(),
        "pen_drawn": pen[pen.p2 > 0].groupby(["game_id", "p2"]).size()})
    pens.index.names = ["game_id", "player_id"]

    # spine: every dressed regular-season skater
    pg = pd.read_parquet(TENSORS / f"player_games_{season}.parquet",
                         columns=["game_id", "game_type", "player_id", "is_home",
                                  "pos_group"])
    pg = pg[(pg.game_type == 2) & (pg.pos_group != 2)].drop(columns="game_type")
    out = pg.set_index(["game_id", "player_id"])
    out = out.join(indiv).join(onice.drop(columns="is_home", errors="ignore")) \
        .join(pens)
    num = [c for c in out.columns if c not in ("is_home", "pos_group")]
    out[num] = out[num].fillna(0.0).astype("float32")
    return out.reset_index()


def pp_runs(st: pd.DataFrame, keys: list) -> pd.Series:
    """Count maximal runs of stints with strength_key in `keys`, per game."""
    s = st[st.strength_key.isin(keys)].copy()
    s["a0"] = (s.period - 1) * 1200 + s.t0
    s["a1"] = (s.period - 1) * 1200 + s.t1
    s = s.sort_values(["game_id", "a0"])
    prev_end = s.groupby("game_id").a1.shift(1)
    new_run = prev_end.isna() | (s.a0 - prev_end >= 2)
    return new_run.groupby(s.game_id).sum()


def build_team(ev: pd.DataFrame, season: int) -> pd.DataFrame:
    rs = ev[ev.game_type == 2].copy()
    rs["str"] = strength(rs)
    shots = rs[rs.event_type.isin([BLOCK, MISS, SOG, GOAL])].copy()
    shots["is_sog"] = shots.event_type.isin([SOG, GOAL]).astype(float)
    shots["is_goal"] = (shots.event_type == GOAL).astype(float)
    shots["xg0"] = np.where(shots.has_xg == 1, shots.xg, 0.0)
    gc = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet",
                         columns=["game_id", "game_type"])
    gids = gc.loc[gc.game_type == 2, "game_id"].to_numpy()
    rows = []
    for is_home in (1, 0):
        d = pd.DataFrame(index=pd.Index(gids, name="game_id"))
        f = shots[shots.home_event == is_home]
        a = shots[shots.home_event != is_home]
        for st in STR + ("all",):
            ff = f if st == "all" else f[f.str == st]
            # against: shooter's pp is my sh
            ast = {"ev": "ev", "pp": "sh", "sh": "pp"}.get(st)
            aa = a if st == "all" else a[a.str == ast]
            d[f"xgf_{st}"] = ff.groupby("game_id").xg0.sum()
            d[f"xga_{st}"] = aa.groupby("game_id").xg0.sum()
            d[f"gf_{st}"] = ff.groupby("game_id").is_goal.sum()
            d[f"ga_{st}"] = aa.groupby("game_id").is_goal.sum()
        d["sogf"] = f.groupby("game_id").is_sog.sum()
        d["soga"] = a.groupby("game_id").is_sog.sum()
        d["attf"] = f.groupby("game_id").size()
        d["atta"] = a.groupby("game_id").size()
        d["en_gf"] = f[f.str == "en"].groupby("game_id").is_goal.sum()
        d["is_home"] = is_home
        rows.append(d)
    tg = pd.concat(rows).fillna(0.0)

    st = pd.read_parquet(TENSORS / f"stints_{season}.parquet",
                         columns=["game_id", "game_type", "period", "t0", "t1",
                                  "dur_s", "strength_key"])
    st = st[st.game_type == 2]
    home_pp = st[st.strength_key.isin([1, 2])].groupby("game_id").dur_s.sum()
    away_pp = st[st.strength_key.isin([3, 4])].groupby("game_id").dur_s.sum()
    opp_h, opp_a = pp_runs(st, [1, 2]), pp_runs(st, [3, 4])
    tg = tg.reset_index()
    h = tg.is_home == 1
    tg.loc[h, "pp_toi"] = tg.loc[h, "game_id"].map(home_pp).fillna(0).values
    tg.loc[h, "sh_toi"] = tg.loc[h, "game_id"].map(away_pp).fillna(0).values
    tg.loc[h, "pp_opps"] = tg.loc[h, "game_id"].map(opp_h).fillna(0).values
    tg.loc[~h, "pp_toi"] = tg.loc[~h, "game_id"].map(away_pp).fillna(0).values
    tg.loc[~h, "sh_toi"] = tg.loc[~h, "game_id"].map(home_pp).fillna(0).values
    tg.loc[~h, "pp_opps"] = tg.loc[~h, "game_id"].map(opp_a).fillna(0).values
    num = [c for c in tg.columns if c not in ("game_id", "is_home")]
    tg[num] = tg[num].astype("float32")
    return tg


def build(season: int):
    ev = pd.read_parquet(TENSORS / f"stream_{season}.parquet")
    pgx = build_player(ev, goalie_ids(season), season)
    tgx = build_team(ev, season)
    pgx.to_parquet(TENSORS / f"pgx_{season}.parquet", index=False)
    tgx.to_parquet(TENSORS / f"tgx_{season}.parquet", index=False)
    return pgx, tgx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2009-2026")
    a, b = map(int, ap.parse_args().seasons.split("-"))
    for s in range(a, b + 1):
        pgx, tgx = build(s)
        ev_sum = pgx.groupby(["game_id", "is_home"]).ixg_all.sum()
        tg_sum = tgx.set_index(["game_id", "is_home"]).xgf_all
        gap = (ev_sum.reindex(tg_sum.index).fillna(0) - tg_sum).abs()
        print(f"{s}: {len(pgx):,} skater-games, {len(tgx)//2:,} games | "
              f"xgf/team-game {tgx.xgf_all.mean():.2f}  sog {tgx.sogf.mean():.1f}  "
              f"pp_opps {tgx.pp_opps.mean():.2f}  pp_min {tgx.pp_toi.mean()/60:.2f} | "
              f"sum(ixg)-team xgf max gap {gap.max():.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
