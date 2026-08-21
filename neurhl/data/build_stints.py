"""NeurHL-2 — stint extraction: the unit of observation for player effects.

A **stint** is a maximal interval within one period over which the on-ice set is
constant. Change points are the union of every shift start/end plus period
boundaries, so constancy holds by construction. This is the Macdonald (2011) /
Thomas et al. (2013) unit, and it is what makes a player's contribution separable
from his teammates' and opponents': every stint is a different combination of the
two, so the design matrix has rank far above the number of players.

Why this is the right substrate: raw on-ice rates — what v1's Layer-1 consumed —
are a TEAM property confounded by linemates and matchups. The measured
consequence was a player model that regressed every player to the league mean
(sd 0.006 against an EWMA baseline's 0.042). Stints are where that confound
becomes removable rather than merely acknowledged.

Reads the unified shift table (`build_shifts.py`, both data eras) and the event
shard. Writes `neurhl/data/tensors/stints_<season_end>.parquet`, one row per
stint, carrying both on-ice sets and the outcomes accrued inside it.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_stints.py [--seasons 2016]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

MIN_DUR = 1                 # seconds; the >=3s analysis filter is downstream
N_SK = 6                    # skater slots (6 covers a pulled goalie)
CFG = {"min_dur": MIN_DUR, "n_sk": N_SK, "version": 2}

META = ["game_id", "season_end", "game_type", "period", "stint_idx",
        "t0", "t1", "dur_s", "h_g", "a_g", "n_h", "n_a", "sit_code",
        "strength_key", "is_5v5", "home_lead", "zone_start", "otf",
        "slots_ok", "src"]
SKATERS = [f"h_s{i}" for i in range(1, N_SK + 1)] + \
          [f"a_s{i}" for i in range(1, N_SK + 1)]
OUT = ["cf_h", "cf_a", "sog_h", "sog_a", "g_h", "g_a", "fo_h", "fo_a",
       "pen_h", "pen_a", "hit_h", "hit_a", "blk_h", "blk_a"]
COLS = META + SKATERS + OUT

# zone_start: 0 = none / on-the-fly, 1 = home O-zone, 2 = neutral, 3 = home D
Z_NONE, Z_HOFF, Z_NEU, Z_HDEF = 0, 1, 2, 3

_EIDS: dict = {}


def _init(eids):
    global _EIDS
    _EIDS = eids


def strength_enum(n_h, n_a, hg, ag) -> int:
    """0=5v5, 1/2=home PP by 1/2+, 3/4=away PP by 1/2+, 5=4v4, 6=3v3,
    7=home net empty, 8=away net empty, 9=other."""
    if not hg:
        return 7
    if not ag:
        return 8
    if n_h == 5 and n_a == 5:
        return 0
    if n_h == 4 and n_a == 4:
        return 5
    if n_h == 3 and n_a == 3:
        return 6
    d = n_h - n_a
    if d == 1:
        return 1
    if d >= 2:
        return 2
    if d == -1:
        return 3
    if d <= -2:
        return 4
    return 9


def _game(args):
    """Stint rows for one game. Pure numpy: no pandas inside the hot loop."""
    (gid, season_end, game_type, sh, ev) = args
    s_pid, s_home, s_g, s_t0, s_t1 = sh
    e_t, e_type, e_home, e_sit, e_sh, e_sa, e_zone = ev

    max_per = 4 if game_type == 2 else int(e_t.max() // 1200 + 1) if len(e_t) else 4
    rows = []
    for per in range(1, max_per + 1):
        lo = (per - 1) * 1200
        hi = lo + 1200
        sel = (s_t1 > lo) & (s_t0 < hi)
        if not sel.any():
            continue
        t0 = np.clip(s_t0[sel], lo, hi).astype(np.int32)
        t1 = np.clip(s_t1[sel], lo, hi).astype(np.int32)
        pid, home, isg = s_pid[sel], s_home[sel], s_g[sel]
        end = int(t1.max())
        if end <= lo:
            continue

        pts = np.unique(np.concatenate([[lo], t0, t1, [end]]))
        pts = pts[(pts >= lo) & (pts <= end)]
        if len(pts) < 2:
            continue
        a, b = pts[:-1], pts[1:]
        keep = (b - a) >= MIN_DUR
        if not keep.any():
            continue
        a, b = a[keep], b[keep]
        mid = (a + b) / 2.0

        # (K, S) membership. K ~ 250, S ~ 800 -> 200k bools, trivially fast.
        act = (t0[None, :] <= mid[:, None]) & (mid[:, None] < t1[None, :])
        K = len(a)

        # --- events -> stint index, vectorised, with the TWO boundary
        # conventions validated in build_onice (98.9% vs situationCode).
        # A faceoff belongs to the stint STARTING at its timestamp: the new
        # players are already on for it. Every other event belongs to the stint
        # ENDING there -- those are the conditions that produced it.
        #
        # This is not a detail. A power-play goal at time t coincides with the
        # penalised player stepping back on at t, so crediting the interval
        # starting at t hands the goal to the post-goal 5v5 unit. Measured on
        # 2016 with the single-convention version: 90% of goals were labelled
        # 5v5 against a true share near 73%, i.e. PP goals were contaminating
        # the 5v5 RAPM design.
        emask = (e_t >= lo) & (e_t < max(end, lo + 1))
        et, eh = e_type[emask], e_home[emask]
        i_start = np.clip(np.searchsorted(pts, e_t[emask], "right") - 1, 0, K - 1)
        i_end = np.clip(np.searchsorted(pts, e_t[emask], "left") - 1, 0, K - 1)
        is_fo = np.isin(et, _EIDS["faceoff"])
        idx = np.where(is_fo, i_start, i_end)
        acc = {}
        for nm, ids in (("cf", _EIDS["attempt"]), ("sog", _EIDS["sog"]),
                        ("g", _EIDS["goal"]), ("fo", _EIDS["faceoff"]),
                        ("pen", _EIDS["penalty"]), ("hit", _EIDS["hit"]),
                        ("blk", _EIDS["block"])):
            m = np.isin(et, ids)
            acc[nm + "_h"] = np.bincount(idx[m & (eh == 1)], minlength=K)
            acc[nm + "_a"] = np.bincount(idx[m & (eh == 0)], minlength=K)

        # --- per-stint context carried from the first event inside it
        first = np.full(K, -1, np.int64)
        order = np.argsort(idx, kind="stable")
        if len(order):
            si = idx[order]
            firsts = np.concatenate([[True], si[1:] != si[:-1]])
            first[si[firsts]] = np.flatnonzero(emask)[order][firsts]

        # --- zone / on-the-fly from a faceoff at the stint boundary
        fo_stint = np.full(K, -1, np.int64)
        if is_fo.any():
            fi = np.flatnonzero(emask)[is_fo]
            ks = idx[is_fo]
            at_start = np.abs(e_t[fi] - a[ks]) <= 2
            fo_stint[ks[at_start]] = fi[at_start]

        for k in range(K):
            on = act[k]
            hs = np.sort(pid[on & home & ~isg])[:N_SK]
            as_ = np.sort(pid[on & ~home & ~isg])[:N_SK]
            hgv = pid[on & home & isg]
            agv = pid[on & ~home & isg]
            hg = int(hgv[0]) if len(hgv) else 0
            ag = int(agv[0]) if len(agv) else 0
            nh, na = len(hs), len(as_)

            f = first[k]
            sit = int(e_sit[f]) if f >= 0 else 0
            lead = int(e_sh[f]) - int(e_sa[f]) if f >= 0 else 0
            ok = True
            ss = str(sit)
            if len(ss) == 4 and ss.isdigit():
                ok = (int(ss[2]) == nh) and (int(ss[1]) == na)

            fs = fo_stint[k]
            if fs < 0:
                zone, otf = Z_NONE, True
            else:
                z, hv = int(e_zone[fs]), int(e_home[fs])
                if z == _EIDS["z_neu"]:
                    zone = Z_NEU
                elif z == _EIDS["z_off"]:
                    zone = Z_HOFF if hv == 1 else Z_HDEF
                elif z == _EIDS["z_def"]:
                    zone = Z_HDEF if hv == 1 else Z_HOFF
                else:
                    zone = Z_NONE
                otf = False

            row = [gid, season_end, game_type, per, k, int(a[k]), int(b[k]),
                   int(b[k] - a[k]), hg, ag, nh, na, sit,
                   strength_enum(nh, na, hg, ag),
                   bool(nh == 5 and na == 5 and hg and ag),
                   max(-3, min(3, lead)), zone, otf, ok, 0]
            row += [int(hs[i]) if i < nh else 0 for i in range(N_SK)]
            row += [int(as_[i]) if i < na else 0 for i in range(N_SK)]
            row += [int(acc[c][k]) for c in OUT]
            rows.append(row)
    return rows or None


def event_ids(maps: dict) -> dict:
    et, zn = maps["event_type"], maps["zone"]
    g = lambda d, *k: [d[c] for c in k if c in d]  # noqa: E731
    return {
        "attempt": g(et, "shot-on-goal", "goal", "missed-shot", "blocked-shot"),
        "sog": g(et, "shot-on-goal", "goal"),
        "goal": g(et, "goal"), "faceoff": g(et, "faceoff"),
        "penalty": g(et, "penalty"), "hit": g(et, "hit"),
        "block": g(et, "blocked-shot"),
        "z_off": zn.get("O", -1), "z_neu": zn.get("N", -2),
        "z_def": zn.get("D", -3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    maps = json.loads((TENSORS / "maps.json").read_text())
    eids = event_ids(maps)

    seasons = args.seasons or sorted(
        int(p.stem.split("_")[1]) for p in TENSORS.glob("shifts_2*.parquet")
        if p.stem.split("_")[1].isdigit())

    for se in seasons:
        sp = TENSORS / f"shifts_{se}.parquet"
        ep = TENSORS / f"events_{se}.parquet"
        out = TENSORS / f"stints_{se}.parquet"
        if not sp.exists() or not ep.exists():
            print(f"{se}: missing shifts or events, skipping")
            continue
        if not args.force and MAN.is_fresh(out, [sp, ep], CFG):
            print(f"{se}: fresh, skipping")
            continue

        sh = pd.read_parquet(sp)
        ev = pd.read_parquet(ep, columns=["game_id", "game_type", "t",
                                          "event_type", "home_event",
                                          "strength", "score_h", "score_a",
                                          "zone"])
        ev = ev[ev.t.notna()]
        jobs = []
        sh_g = dict(list(sh.groupby("game_id", sort=False)))
        for gid, e in ev.groupby("game_id", sort=True):
            s = sh_g.get(gid)
            if s is None or not len(s):
                continue
            jobs.append((
                int(gid), se, int(e.game_type.iloc[0]),
                (s.player_id.to_numpy(np.int64), s.is_home.to_numpy(bool),
                 s.is_goalie.to_numpy(bool), s.t0.to_numpy(np.int32),
                 s.t1.to_numpy(np.int32)),
                (e.t.to_numpy(np.int32), e.event_type.to_numpy(np.int16),
                 e.home_event.to_numpy(np.int8), e.strength.to_numpy(np.int32),
                 e.score_h.to_numpy(np.int16), e.score_a.to_numpy(np.int16),
                 e.zone.to_numpy(np.int16))))

        rows = []
        with Pool(args.workers, initializer=_init, initargs=(eids,)) as pool:
            for r in pool.imap_unordered(_game, jobs, chunksize=8):
                if r:
                    rows.extend(r)
        if not rows:
            print(f"{se}: no stints produced")
            continue
        df = pd.DataFrame(rows, columns=COLS)
        df["src"] = int(sh.src.iloc[0])
        df.to_parquet(out, index=False)
        MAN.write_manifest(out, [sp, ep], CFG,
                           {"n_stints": len(df),
                            "n_games": int(df.game_id.nunique())})
        rs = df[df.game_type == 2]
        print(f"{se}: {len(df):,} stints / {df.game_id.nunique()} games "
              f"({len(df)/max(df.game_id.nunique(),1):.0f}/g) | "
              f"5v5 {df.is_5v5.mean():.3f} | slots_ok {df.slots_ok.mean():.3f} | "
              f"med dur {df.dur_s.median():.0f}s | "
              f"5v5 TOI/g {rs[rs.is_5v5].groupby('game_id').dur_s.sum().mean():.0f}s")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
