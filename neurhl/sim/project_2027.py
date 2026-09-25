"""NeurHL-2 — S6/S5: the 2026-27 projection (season_end 2027).

THE DELIVERABLE. Projects the upcoming season only; nothing here produces
2027-28 outputs.

Built from the components that survived validation, and deliberately NOT from the
S1 hazard query, which the preregistered fallback retired for game outcomes after
it failed to beat Elo (blend 0.67763 vs Elo 0.67668 on TUNE):

  * **real 2026-27 schedule and rosters** — 32 team files each, so this is the
    actual season, not a synthetic one;
  * **Elo** carried through 2026 and regressed to the mean, reported as a
    reference column (`elo_start`); in-season variation is carried by the
    per-season team-strength draw (TEAM_SIGMA), not by Elo updates;
  * **team attack/defence** EB-shrunk from 2024-2026;
  * **roster RAPM** from `rapm_prior_2027` (fit on 2024-2026 only) applied to the
    ANNOUNCED 2026-27 rosters — this is what carries trades and free agency,
    which team-level rates structurally cannot see;
  * **S1's score-effect curve** inside the Kolmogorov integration.

Cold starts are handled explicitly rather than by accident: a player with no
RAPM history (2026 draft class, European signings) gets the position's
REPLACEMENT coefficient from the prior table, not the league average. Treating an
unknown rookie as an average NHLer is the single easiest way to make a projection
quietly wrong.

Points follow the real NHL system (regulation win 2-0, overtime/shootout 2-1), so
standings are on the scale the deliverable is consumed at.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/sim/project_2027.py --sims 20000
"""
import argparse
import glob
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
from sim.game_model import (TEAM_SIGMA, fit_tie_calibration,  # noqa: E402
                            integrate, observed_tie_rate, outcome, run_elo,
                            score_effect_curve, team_strength)
from eval.backtest_season import rate_sensitivity  # noqa: E402
import windows as W  # noqa: E402

SEASON = 2027                       # season_end 2027 == the 2026-27 season
PREV = 2026
CONF = {
    "E": ["BOS", "BUF", "CAR", "CBJ", "DET", "FLA", "MTL", "NJD", "NYI", "NYR",
          "OTT", "PHI", "PIT", "TBL", "TOR", "WSH"],
    "W": ["ANA", "CGY", "CHI", "COL", "DAL", "EDM", "LAK", "MIN", "NSH", "SEA",
          "SJS", "STL", "UTA", "VAN", "VGK", "WPG"],
}


def load_schedule() -> pd.DataFrame:
    """The real 2026-27 schedule, deduplicated across the 32 team files."""
    rows = []
    for f in sorted(glob.glob(str(RAW / "nhl_sched_*_20262027.json"))):
        d = json.loads(Path(f).read_text())
        games = d.get("games", d if isinstance(d, list) else [])
        for g in games:
            if int(g.get("gameType", 2)) != 2:
                continue
            rows.append({"game_id": int(g["id"]),
                         "date": g.get("gameDate", ""),
                         "home": g["homeTeam"]["abbrev"],
                         "away": g["awayTeam"]["abbrev"]})
    s = pd.DataFrame(rows).drop_duplicates("game_id").sort_values(
        ["date", "game_id"]).reset_index(drop=True)
    return s


def load_rosters() -> dict:
    out = {}
    for f in sorted(glob.glob(str(RAW / "nhl_roster_*_20262027.json"))):
        ab = Path(f).name.split("_")[2]
        d = json.loads(Path(f).read_text())
        out[ab] = {
            "F": [p["id"] for p in d.get("forwards", [])],
            "D": [p["id"] for p in d.get("defensemen", [])],
            "G": [p["id"] for p in d.get("goalies", [])],
        }
    return out


def roster_ratings(rosters: dict) -> tuple:
    """Team -> RAPM net rating from the ANNOUNCED roster, plus cold-start count.

    TOI weights come from the player's 2026 usage; a player with no NHL history
    is given the position REPLACEMENT coefficient and a fourth-line workload,
    which is the honest prior for an unknown rookie.
    """
    pr = pd.read_parquet(TENSORS / f"rapm_prior_{SEASON}.parquet")
    real = pr[~pr.is_replacement.astype(bool)].set_index("player_id")
    repl = pr[pr.is_replacement.astype(bool)]
    repl_net = float((repl.cf_off - repl.cf_def).mean()) if len(repl) else 0.0
    net = (real.cf_off - real.cf_def).to_dict()

    pg = pd.read_parquet(TENSORS / f"player_games_{PREV}.parquet",
                         columns=["player_id", "toi_sec", "game_type"])
    toi = pg[pg.game_type == 2].groupby("player_id").toi_sec.sum().to_dict()

    out, cold = {}, {}
    for ab, r in rosters.items():
        vals, wts, nc = [], [], 0
        for pid in r["F"][:13] + r["D"][:7]:
            v = net.get(pid)
            w = toi.get(pid, 0.0)
            if v is None or w < 6000:          # <100 min of NHL last season
                v, w, nc = repl_net, 40000.0, nc + 1
            vals.append(v)
            wts.append(max(w, 1.0))
        out[ab] = float(np.average(vals, weights=wts)) if vals else 0.0
        cold[ab] = nc
    mu = np.mean(list(out.values()))
    sd = np.std(list(out.values())) or 1.0
    return {k: (v - mu) / sd for k, v in out.items()}, cold, repl_net


def abbrev_map() -> dict:
    m = json.loads((TENSORS / "maps.json").read_text())["team"]
    return {k: int(v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()
    assert W.PROJECT == SEASON, "PROJECT season mismatch"

    sched = load_schedule()
    rosters = load_rosters()
    rap, cold, repl_net = roster_ratings(rosters)
    ab2i = abbrev_map()
    print(f"2026-27 (season_end {SEASON}): {len(sched):,} regular-season games, "
          f"{sched.home.nunique()} teams, rosters for {len(rosters)}")
    print(f"cold-start skaters given REPLACEMENT rating: "
          f"{sum(cold.values())} across {len(cold)} teams "
          f"(max {max(cold.values())} on one roster)")

    # ---- prior-season strength, all from seasons < 2027
    train = list(range(2008, SEASON))
    att, dfn, lg, hm = team_strength(train)
    elo_hist = run_elo(train)
    last = {}
    for s in (PREV,):
        e = elo_hist[elo_hist.season_end == s]
        g = pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet",
                            columns=["game_id", "game_type", "home_idx",
                                     "away_idx", "home_g", "away_g"])
        g = g[g.game_type == 2].merge(e, on="game_id")
        for r in g.itertuples():
            last[int(r.home_idx)] = r.elo_h
            last[int(r.away_idx)] = r.elo_a
    elo0 = {ab: 1500.0 + 0.7 * (last.get(ab2i.get(ab, -1), 1500.0) - 1500.0)
            for ab in rosters}
    curve = score_effect_curve(Path(__file__).resolve().parents[1] /
                               "configs" / "event_sim_gates.json")

    # ---- per-game base rates (fixed for the season; per-season strength
    # noise is added inside the Monte Carlo below)
    base_h, base_a = {}, {}
    for r in sched.itertuples():
        ah = att.get(ab2i.get(r.home, -1), 1.0)
        dh = dfn.get(ab2i.get(r.home, -1), 1.0)
        aa = att.get(ab2i.get(r.away, -1), 1.0)
        da = dfn.get(ab2i.get(r.away, -1), 1.0)
        rh, ra = rap.get(r.home, 0.0), rap.get(r.away, 0.0)
        base_h[r.game_id] = lg * ah * da * hm ** 0.5 * np.exp(0.045 * (rh - ra))
        base_a[r.game_id] = lg * aa * dh / hm ** 0.5 * np.exp(0.045 * (ra - rh))

    # precompute outcome splits per game at the base rates
    print("integrating per-game score distributions ...")
    raw = {g: integrate(base_h[g], base_a[g], curve) for g in sched.game_id}
    uncal = [outcome(P)["p_tie"] for P in raw.values()]
    tie_c = fit_tie_calibration(uncal, [2024, 2025, 2026])
    print(f"tie calibration fitted on 2024-2026: model {np.mean(uncal):.4f} -> "
          f"observed {np.mean(uncal)*tie_c:.4f}  (factor {tie_c:.3f})")
    P_reg_h, P_reg_a, P_ot = {}, {}, {}
    for gid, P in raw.items():
        o = outcome(P, tie_calib=tie_c)
        P_reg_h[gid] = o["p_reg_home"]
        P_ot[gid] = o["p_tie"]
        P_reg_a[gid] = o["p_reg_away"]

    # ---- season Monte Carlo: per-season strength draw + game outcome draws
    rng = np.random.default_rng(args.seed)
    teams = sorted(rosters)
    idx = {t: i for i, t in enumerate(teams)}
    gid = sched.game_id.to_numpy()
    hi = np.array([idx[t] for t in sched.home])
    ai = np.array([idx[t] for t in sched.away])
    prh = np.array([P_reg_h[g] for g in gid])
    pot = np.array([P_ot[g] for g in gid])
    pts_all = np.zeros((args.sims, len(teams)), np.int16)
    wins_all = np.zeros((args.sims, len(teams)), np.int16)
    rw_all = np.zeros((args.sims, len(teams)), np.int16)

    # PARAMETER UNCERTAINTY -- one draw of each team's true strength per
    # simulated season, held fixed across that season. Without it the only
    # variance is game-outcome noise and the 80% intervals cover 0.639 of
    # outcomes instead of 0.80.
    csens = rate_sensitivity(curve)
    l0 = np.log(np.clip(prh, 1e-6, 1 - 1e-6) / (1 - np.clip(prh, 1e-6, 1 - 1e-6)))
    for s in range(args.sims):
        dt = rng.normal(0.0, TEAM_SIGMA, len(teams))
        prs = (1.0 / (1.0 + np.exp(-(l0 + csens * (dt[hi] - dt[ai]))))) * (1 - pot)
        u = rng.random(len(gid))
        v = rng.random(len(gid))
        home_reg = u < prs
        ot = (u >= prs) & (u < prs + pot)
        home_ot = ot & (v < 0.53)
        pts = np.zeros(len(teams), np.int32)
        np.add.at(pts, hi[home_reg], 2)
        np.add.at(pts, ai[~home_reg & ~ot], 2)
        np.add.at(pts, hi[home_ot], 2)
        np.add.at(pts, ai[ot & ~home_ot], 2)
        np.add.at(pts, ai[home_ot], 1)
        np.add.at(pts, hi[ot & ~home_ot], 1)
        pts_all[s] = pts
        w = np.zeros(len(teams), np.int32)
        np.add.at(w, hi[home_reg | home_ot], 1)
        np.add.at(w, ai[(~home_reg & ~ot) | (ot & ~home_ot)], 1)
        wins_all[s] = w
        rw = np.zeros(len(teams), np.int32)
        np.add.at(rw, hi[home_reg], 1)
        np.add.at(rw, ai[~home_reg & ~ot], 1)
        rw_all[s] = rw

    # ---- standings and playoff odds (top 8 per conference)
    #
    # Tie resolution: points, then regulation wins, then total wins (the
    # tiebreakers the sim can see), then a COIN FLIP from a dedicated
    # generator. An earlier build relied on Python's stable sort over an
    # alphabetically ordered team list, which handed EVERY tie to the
    # alphabetically earlier club -- a systematic identity bias worth up to
    # ~1.6 playoff points (WSH -1.58, WPG -0.99; Spearman(alphabet rank,
    # bias) = -0.91). The tiebreak generator is separate from the game rng,
    # so the simulated seasons above are bit-identical either way. A
    # fractional split of the cutoff tie group is recorded alongside as an
    # identity-blind reference; the acceptance battery asserts the two agree.
    conf_of = {t: ("E" if t in CONF["E"] else "W") for t in teams}
    conf_ids = {c: [idx[t] for t in teams if conf_of[t] == c]
                for c in ("E", "W")}
    playoff = np.zeros(len(teams))
    playoff_frac = np.zeros(len(teams))
    tie_rng = np.random.default_rng(args.seed + 7)
    for s in range(args.sims):
        for c in ("E", "W"):
            ids = conf_ids[c]
            key = {i: (-int(pts_all[s, i]), -int(rw_all[s, i]),
                       -int(wins_all[s, i])) for i in ids}
            u = tie_rng.random(len(ids))
            order = sorted(range(len(ids)), key=lambda j: key[ids[j]] + (u[j],))
            for j in order[:8]:
                playoff[ids[j]] += 1
            cut = key[ids[order[7]]]
            above = [i for i in ids if key[i] < cut]
            ties = [i for i in ids if key[i] == cut]
            for i in above:
                playoff_frac[i] += 1.0
            share = (8 - len(above)) / len(ties)
            for i in ties:
                playoff_frac[i] += share
    playoff /= args.sims
    playoff_frac /= args.sims

    chk = {
        "rule": "points, regulation wins, total wins, dedicated-rng coin flip",
        "max_abs_diff_pct": float(np.max(np.abs(playoff - playoff_frac)) * 100.0),
        "teams": {teams[i]: {"actual_pct": round(float(playoff[i] * 100), 3),
                             "fractional_pct": round(float(playoff_frac[i] * 100), 3)}
                  for i in range(len(teams))},
    }
    cfg = Path(__file__).resolve().parents[1] / "configs"
    (cfg / "playoff_tiebreak_check.json").write_text(json.dumps(chk, indent=1))
    print(f"tiebreak guard: max |actual - fractional| = "
          f"{chk['max_abs_diff_pct']:.3f} playoff points")

    # ---- per-game preseason probabilities, frozen for live scoring. NeurHL:
    # the engine's base-rate split (no per-season strength draw), with the
    # overtime coin at the sim's 0.53 home share. Reference: v1 Elo frozen at
    # its preseason ratings (0.7 carry, H = 35), the same static footing.
    H_ELO = json.loads((Path(__file__).resolve().parents[2] / "output" /
                        "params.json").read_text())["H"]
    gm = sched.copy()
    gm["p_home_reg"] = [P_reg_h[g] for g in gm.game_id]
    gm["p_ot"] = [P_ot[g] for g in gm.game_id]
    gm["p_away_reg"] = [P_reg_a[g] for g in gm.game_id]
    gm["p_home_win"] = gm.p_home_reg + 0.53 * gm.p_ot
    d_elo = (gm.home.map(elo0) + H_ELO - gm.away.map(elo0)).to_numpy()
    gm["p_home_win_elo"] = 1.0 / (1.0 + 10 ** (-d_elo / 400.0))
    gm.round(6).to_csv(Path(__file__).resolve().parents[1] / "output" /
                       "games_2027.csv", index=False)

    res = pd.DataFrame({
        "team": teams,
        "conf": [conf_of[t] for t in teams],
        "proj_points": pts_all.mean(0),
        "p10": np.percentile(pts_all, 10, axis=0),
        "p90": np.percentile(pts_all, 90, axis=0),
        "proj_wins": wins_all.mean(0),
        "playoff_pct": playoff * 100,
        "roster_rapm": [rap.get(t, 0.0) for t in teams],
        "elo_start": [elo0.get(t, 1500.0) for t in teams],
        "cold_starts": [cold.get(t, 0) for t in teams],
    }).sort_values(["conf", "proj_points"], ascending=[True, False])

    out = Path(__file__).resolve().parents[1] / "output" / "projection_2027.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n2026-27 PROJECTION ({args.sims:,} season simulations)\n")
    for c in ("E", "W"):
        print(f"  --- {'EASTERN' if c=='E' else 'WESTERN'} CONFERENCE ---")
        print(f"  {'team':<5}{'pts':>7}{'80% range':>13}{'wins':>7}"
              f"{'playoff%':>10}{'RAPM':>7}{'Elo':>7}{'cold':>6}")
        for r in res[res.conf == c].itertuples():
            print(f"  {r.team:<5}{r.proj_points:>7.1f}"
                  f"{f'{r.p10:.0f}-{r.p90:.0f}':>13}{r.proj_wins:>7.1f}"
                  f"{r.playoff_pct:>10.1f}{r.roster_rapm:>7.2f}"
                  f"{r.elo_start:>7.0f}{r.cold_starts:>6}")
    n_g = len(sched)
    tot = res.proj_points.sum()
    print(f"\nleague mean points {res.proj_points.mean():.1f}; total {tot:.0f} "
          f"over {n_g} games ({n_g//16} per team)")
    print(f"  identity check: 2*games + OT games = {2*n_g} + "
          f"{tot-2*n_g:.0f} -> implied OT share {(tot-2*n_g)/n_g:.1%} "
          f"(observed 2024-2026: {observed_tie_rate([2024,2025,2026]):.1%})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
