"""NeurHL-3 — D5: per player-game usage structure from the stint table.

What a total-TOI EWMA cannot see, measured directly from on-ice sets:
strength-split minutes (EV/PP/SH), the player's 5v5 line rank within his
position group, his top 5v5 linemates (for churn and linemate-quality
features), and his PP-unit rank. All within-game facts — consumers turn them
into strictly-pre-game features (EWMAs, deltas) at feature-build time.

strength_key enum (build_stints.strength_enum): 0=5v5, 1/2=home PP +1/+2,
3/4=away PP +1/+2, 5=4v4, 6=3v3, 7/8=empty-net, 9=other. EV buckets {0,5,6};
PP/SH are side-dependent; {7,8,9} land in `other_toi` so the four buckets
reconcile against player_games.toi_sec (printed, and bounded per season).

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_usage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

H_SLOTS = [f"h_s{i}" for i in range(1, 7)]
A_SLOTS = [f"a_s{i}" for i in range(1, 7)]
SEASONS = list(range(2008, 2027))
CFG = {"version": 2}
EV_KEYS = {0, 5, 6}
HOME_PP, AWAY_PP = {1, 2}, {3, 4}


def melt_side(st: pd.DataFrame, cols, is_home: int) -> pd.DataFrame:
    parts = []
    for c in cols:
        d = st[["row", "game_id", "dur_s", "strength_key", c]].rename(
            columns={c: "player_id"})
        parts.append(d[d.player_id > 0])
    out = pd.concat(parts, ignore_index=True)
    out["is_home"] = is_home
    return out


def build(season: int) -> pd.DataFrame:
    st = pd.read_parquet(
        TENSORS / f"stints_{season}.parquet",
        columns=["game_id", "game_type", "dur_s", "strength_key"] + H_SLOTS
        + A_SLOTS)
    st = st[st.game_type == 2].reset_index(drop=True)
    st["row"] = np.arange(len(st))
    long = pd.concat([melt_side(st, H_SLOTS, 1), melt_side(st, A_SLOTS, 0)],
                     ignore_index=True)

    k = long.strength_key
    ev = long.dur_s.where(k.isin(EV_KEYS), 0.0)
    pp = long.dur_s.where(((long.is_home == 1) & k.isin(HOME_PP))
                          | ((long.is_home == 0) & k.isin(AWAY_PP)), 0.0)
    sh = long.dur_s.where(((long.is_home == 1) & k.isin(AWAY_PP))
                          | ((long.is_home == 0) & k.isin(HOME_PP)), 0.0)
    five = long.dur_s.where(k == 0, 0.0)
    long = long.assign(ev=ev, pp=pp, sh=sh, five=five)
    u = long.groupby(["game_id", "is_home", "player_id"], as_index=False).agg(
        ev_toi=("ev", "sum"), pp_toi=("pp", "sum"), sh_toi=("sh", "sum"),
        toi5=("five", "sum"), toi_all=("dur_s", "sum"))
    u["other_toi"] = u.toi_all - u.ev_toi - u.pp_toi - u.sh_toi

    # position + team from player_games (pos never from slot index)
    pg = pd.read_parquet(TENSORS / f"player_games_{season}.parquet",
                         columns=["game_id", "player_id", "pos_group",
                                  "toi_sec", "game_type"])
    pg = pg[pg.game_type == 2]
    u = u.merge(pg[["game_id", "player_id", "pos_group", "toi_sec"]],
                on=["game_id", "player_id"], how="left")
    gc = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet",
                         columns=["game_id", "home_idx", "away_idx"])
    u = u.merge(gc, on="game_id", how="left")
    u["team"] = np.where(u.is_home == 1, u.home_idx, u.away_idx)

    sk = u[u.pos_group.isin([0, 1])].copy()
    sk["rank5"] = (sk.groupby(["game_id", "team", "pos_group"]).toi5
                   .rank(ascending=False, method="first"))
    sk["pp_rank"] = (sk.groupby(["game_id", "team"]).pp_toi
                     .rank(ascending=False, method="first")
                     .where(sk.pp_toi > 0))

    # top-2 same-position 5v5 linemates by shared stint time
    st5 = st[st.strength_key == 0]
    lm_parts = []
    for cols, side in ((H_SLOTS, 1), (A_SLOTS, 0)):
        ln = pd.concat([st5[["row", "game_id", "dur_s", c]]
                        .rename(columns={c: "player_id"}) for c in cols],
                       ignore_index=True)
        ln = ln[ln.player_id > 0].assign(is_home=side)
        lm_parts.append(ln)
    ln = pd.concat(lm_parts, ignore_index=True)
    pos_map = pg.drop_duplicates(["game_id", "player_id"]).set_index(
        ["game_id", "player_id"]).pos_group
    ln["pos_group"] = ln.set_index(["game_id", "player_id"]).index.map(pos_map)
    ln = ln[ln.pos_group.isin([0, 1])]
    pairs = ln.merge(ln, on=["row", "is_home", "pos_group"],
                     suffixes=("", "_b"))
    pairs = pairs[pairs.player_id != pairs.player_id_b]
    co = pairs.groupby(["game_id", "player_id", "player_id_b"],
                       as_index=False).dur_s.sum()
    # deterministic tie-break: equal shared-TOI pairs order by partner id,
    # stably — an unstable sort let input row order decide ties, which the
    # causality audit flagged as irreproducibility
    co = co.sort_values(["dur_s", "player_id_b"], ascending=[False, True],
                        kind="stable")
    co["k"] = co.groupby(["game_id", "player_id"]).cumcount()
    top = co[co.k < 2].pivot_table(index=["game_id", "player_id"],
                                   columns="k", values="player_id_b",
                                   aggfunc="first")
    top.columns = [f"l{int(c) + 1}" for c in top.columns]
    sk = sk.merge(top.reset_index(), on=["game_id", "player_id"], how="left")
    for c in ("l1", "l2"):
        if c not in sk:
            sk[c] = 0
        sk[c] = sk[c].fillna(0).astype("int64")

    out_cols = ["game_id", "player_id", "team", "is_home", "pos_group",
                "ev_toi", "pp_toi", "sh_toi", "other_toi", "toi5", "toi_all",
                "toi_sec", "rank5", "pp_rank", "l1", "l2"]
    n0 = len(sk)
    sk = sk[sk.team.notna() & sk.rank5.notna()].copy()
    if len(sk) < n0:
        print(f"    dropped {n0 - len(sk)} rows with unresolvable team/rank")
    sk["toi_sec"] = sk.toi_sec.fillna(0).astype("int64")
    sk["pp_rank"] = sk.pp_rank.fillna(0).astype("int64")
    sk["rank5"] = sk.rank5.astype("int64")
    for c in ("team", "is_home", "pos_group"):
        sk[c] = sk[c].astype("int64")
    return sk[out_cols]


def main():
    for s in SEASONS:
        src = [TENSORS / f"stints_{s}.parquet",
               TENSORS / f"player_games_{s}.parquet",
               TENSORS / f"games_ctx_{s}.parquet"]
        if not all(p.exists() for p in src):
            print(f"{s}: missing sources, skipping")
            continue
        out = TENSORS / f"usage_{s}.parquet"
        if MAN.is_fresh(out, src, CFG):
            print(f"{s}: fresh, skipping")
            continue
        d = build(s)
        # reconciliation: stint-derived total vs shift-derived player_games TOI
        m = d[d.toi_sec > 0]
        rel = (m.toi_all - m.toi_sec).abs() / m.toi_sec.clip(lower=1)
        agree10 = float((rel <= 0.10).mean())
        d.to_parquet(out, index=False)
        MAN.write_manifest(out, src, CFG,
                           {"n_rows": len(d), "toi_agree_10pct": agree10})
        print(f"{s}: {len(d):,} skater-games; stint-vs-shift TOI within 10% "
              f"for {agree10:.1%}; median PP share "
              f"{(d.pp_toi / d.toi_all.clip(lower=1)).median():.3f}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
