"""NeurHL-2 — stint/shift integrity battery (PLAN_NeurHL2 verification 1).

Every downstream player quantity is computed on the stint table, so an error
here propagates silently into RAPM, the hazard model and the simulator. This
battery re-derives the checks mechanically per season and refuses to pass on a
season it cannot verify.

The strongest check available is agreement with the NHL's own `situationCode`,
which is recorded independently of the shift charts we reconstruct from. It
catches boundary-convention errors that look perfectly reasonable in aggregate:
the first version of `build_stints` attributed every event to the stint STARTING
at its timestamp, which handed power-play goals to the post-goal 5v5 unit. Total
goals, total TOI and stint counts were all unchanged; only situationCode
agreement moved (0.972 -> 0.999), and the 5v5 goal share (0.90 vs a true 0.65).

Checks per season:
  T1  team TOI per game reconciles to 6 on-ice * 3600s within tolerance
  T2  derived on-ice counts agree with situationCode (event-bearing stints)
  T3  goal strength split agrees with situationCode's own split
  T4  5v5 rates (CF/60, SOG/60, G/60) inside the observed historical band
  T5  every game in the event shard has stints

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/eval/validate_stints.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

# Bands are deliberately wide: they exist to catch construction errors, not to
# encode a prior about hockey. The 2008-2011 penalty-crackdown era genuinely
# had less 5v5 time and more special teams, and that must not fail the battery.
BAND = {
    # Stints partition the game clock, so their durations sum to game length,
    # NOT to player-TOI. Both are checked, separately:
    "clock_per_game": (3540, 3660),          # regulation = 3600s
    "toi_per_team_game": (20000, 21800),     # sum(dur * on-ice) ~ 6 * 3600
    "cf60_5v5": (48.0, 62.0),
    "sog60_5v5": (25.0, 34.0),
    "g60_5v5": (1.8, 2.8),
    "sit_agree": 0.97,                       # T2 floor
    "goal_share_tol": 0.03,                  # T3 |ours - official|
    "toi5v5_per_game": (2300, 3050),
}


# A waiver is only legitimate when the SOURCE, not the reconstruction, is shown
# to be wrong -- and the evidence has to live next to the waiver, or it becomes
# a way to make a failing check quiet.
KNOWN_SOURCE_DEFECTS = {
    2020: ("T2", "situationCode is corrupt in ~112 of 1,082 games (10.4%), "
                 "over-reporting AWAY-team shorthandedness. Evidence: (a) the "
                 "n_h cross-tab is perfectly diagonal while n_a is not, so only "
                 "the away digit is affected; (b) in those games the code "
                 "claims away<5 for 19.0% of stints while our shifts say 10.2% "
                 "-- and the good-game norm is 8.7% code / 8.8% ours; (c) those "
                 "games average 8.62 penalties vs 6.78, a 27% rise that "
                 "supports our 10.2% and cannot support 19.0%; (d) 2019 has "
                 "ZERO such games through the identical pipeline. Shift records, "
                 "player counts and file sizes are indistinguishable from clean "
                 "games, so the shift feed is intact."),
}


def season_report(se: int) -> dict:
    sp = TENSORS / f"stints_{se}.parquet"
    ep = TENSORS / f"events_{se}.parquet"
    if not sp.exists():
        return {"season": se, "status": "MISSING"}
    d = pd.read_parquet(sp)
    rs = d[d.game_type == 2]
    if not len(rs):
        return {"season": se, "status": "NO_REGULAR_SEASON"}
    f = rs[rs.is_5v5]
    ng = rs.game_id.nunique()
    r = {"season": se, "n_stints": int(len(d)), "n_games": int(ng),
         "src": int(d.src.iloc[0]), "fails": []}

    # T1a -- stints tile the game clock exactly (regulation games)
    clock = rs.groupby("game_id").dur_s.sum().median()
    r["clock_per_game"] = float(clock)
    lo, hi = BAND["clock_per_game"]
    if not (lo <= clock <= hi):
        r["fails"].append(f"T1a stint clock/game {clock:.0f} outside [{lo},{hi}]")

    # T1b -- player-TOI reconciliation: every second a player is on the ice is
    # counted exactly once across the stints he appears in.
    dur = rs.dur_s.to_numpy(np.float64)
    onice_h = rs.n_h.to_numpy() + (rs.h_g.to_numpy() > 0)
    onice_a = rs.n_a.to_numpy() + (rs.a_g.to_numpy() > 0)
    toi = float((dur * (onice_h + onice_a)).sum() / 2 / ng)
    r["toi_per_team_game"] = round(toi, 1)
    lo, hi = BAND["toi_per_team_game"]
    if not (lo <= toi <= hi):
        r["fails"].append(f"T1b player-TOI/team/game {toi:.0f} outside "
                          f"[{lo},{hi}] (nominal 21600)")

    # T2 -- on-ice counts vs situationCode, on stints that carry an event
    ev = rs[rs.sit_code > 0].copy()
    s = ev.sit_code.astype(int).astype(str)
    ok4 = s.str.len() == 4
    ev, s = ev[ok4], s[ok4]
    agree = ((s.str[2].astype(int) == ev.n_h) &
             (s.str[1].astype(int) == ev.n_a)).mean() if len(ev) else np.nan
    r["sit_agree"] = round(float(agree), 5)
    r["sit_checked"] = int(len(ev))
    if len(ev) and agree < BAND["sit_agree"]:
        r["fails"].append(f"T2 situationCode agreement {agree:.4f} "
                          f"< {BAND['sit_agree']}")

    # T3 -- goal strength split vs situationCode's own split
    gs = rs[(rs.g_h + rs.g_a) > 0]
    if len(gs):
        n = (gs.g_h + gs.g_a)
        ours = float(n[gs.is_5v5].sum() / n.sum())
        code = gs.sit_code.astype(int).astype(str)
        offi = float(n[code == "1551"].sum() / n.sum())
        r["goal_5v5_share_ours"] = round(ours, 4)
        r["goal_5v5_share_official"] = round(offi, 4)
        if offi > 0 and abs(ours - offi) > BAND["goal_share_tol"]:
            r["fails"].append(f"T3 5v5 goal share ours {ours:.3f} vs official "
                              f"{offi:.3f}")

    # T4 -- 5v5 rate realism
    tot = f.dur_s.sum()
    if tot > 0:
        for nm, key in (("cf", "cf60_5v5"), ("sog", "sog60_5v5"),
                        ("g", "g60_5v5")):
            v = 3600.0 * (f[f"{nm}_h"].sum() + f[f"{nm}_a"].sum()) / 2 / tot
            r[key] = round(float(v), 3)
            lo, hi = BAND[key]
            if not (lo <= v <= hi):
                r["fails"].append(f"T4 {key} {v:.2f} outside [{lo},{hi}]")
    t5 = f.groupby("game_id").dur_s.sum().mean()
    r["toi5v5_per_game"] = round(float(t5), 1)
    lo, hi = BAND["toi5v5_per_game"]
    if not (lo <= t5 <= hi):
        r["fails"].append(f"T4 5v5 TOI/game {t5:.0f} outside [{lo},{hi}]")

    # T5 -- coverage against the event shard
    if ep.exists():
        e = pd.read_parquet(ep, columns=["game_id", "game_type"])
        want = set(e[e.game_type == 2].game_id.unique())
        have = set(rs.game_id.unique())
        miss = want - have
        r["games_missing_stints"] = len(miss)
        if len(miss) > 0.02 * max(len(want), 1):
            r["fails"].append(f"T5 {len(miss)}/{len(want)} games have no stints")

    waived = [f for f in r["fails"] if any(f.startswith(c) for c in
                                           KNOWN_SOURCE_DEFECTS.get(se, ()))]
    if waived:
        r["waived"] = waived
        r["waiver_reason"] = KNOWN_SOURCE_DEFECTS[se][-1]
        r["fails"] = [f for f in r["fails"] if f not in waived]
    r["status"] = ("PASS" if not r["fails"]
                   else "FAIL") if not waived else "PASS*"
    return r


def main():
    seasons = sorted(int(p.stem.split("_")[1])
                     for p in TENSORS.glob("stints_2*.parquet")
                     if p.stem.split("_")[1].isdigit())
    reps = [season_report(s) for s in seasons]
    out = TENSORS / "stint_validation.json"
    out.write_text(json.dumps(reps, indent=1))

    hdr = (f"{'yr':>5} {'src':>3} {'stints':>9} {'gm':>5} {'clock':>6} "
           f"{'TOI/g':>7} {'sitOK':>7} {'g5v5':>6} {'off':>6} {'CF60':>6} "
           f"{'SG60':>6} {'G60':>5} {'5v5TOI':>7}  status")
    print(hdr)
    print("-" * len(hdr))
    for r in reps:
        if r.get("status") in ("MISSING", "NO_REGULAR_SEASON"):
            print(f"{r['season']:>5}  {r['status']}")
            continue
        print(f"{r['season']:>5} {r['src']:>3} {r['n_stints']:>9,} "
              f"{r['n_games']:>5} {r['clock_per_game']:>6.0f} "
              f"{r['toi_per_team_game']:>7.0f} "
              f"{r.get('sit_agree', float('nan')):>7.4f} "
              f"{r.get('goal_5v5_share_ours', float('nan')):>6.3f} "
              f"{r.get('goal_5v5_share_official', float('nan')):>6.3f} "
              f"{r.get('cf60_5v5', float('nan')):>6.2f} "
              f"{r.get('sog60_5v5', float('nan')):>6.2f} "
              f"{r.get('g60_5v5', float('nan')):>5.2f} "
              f"{r.get('toi5v5_per_game', float('nan')):>7.0f}  {r['status']}")
        for f in r["fails"]:
            print(f"        ! {f}")
    n_fail = sum(r.get("status") not in ("PASS", "PASS*") for r in reps)
    n_waived = sum(r.get("status") == "PASS*" for r in reps)
    print(f"\n{len(reps) - n_fail}/{len(reps)} seasons PASS "
          f"({n_waived} with a documented source-defect waiver) -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
