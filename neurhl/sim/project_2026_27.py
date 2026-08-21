"""NeurHL — the headline deliverable: 2026-27 season projection (report-only).

SCOPE (directive 2026-08-19): the UPCOMING season only — no 2027-28
outputs. This projection runs ALONGSIDE the v1/v4/HOWE live holdout and never
replaces it (PLAN_NeurHL S / P8).

Vantage 2027 (all data <= 2026): REAL announced 2026-27 rosters
(data/raw/nhl_rosters_20262027.csv — better than prior-season inference and
legitimate at this vantage), real schedule, rest/travel computed from the
schedule + arenas.csv, form states entering the season, embeddings_v2027
(rookies via career encoder; no-data players -> position mean, counted).

Outputs:
  neurhl/output/preds/games_2027_neurhl.csv      per-game NN probabilities
  neurhl/output/projections_2026_27_neurhl.csv   xPts/SD/percentiles/odds table
season_sim.rebuild_sim() then serves the market-tooling dict on demand.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, NOUT, PROJ, RAW, TENSORS  # noqa: E402
from eval.gates import apply_temperature, fit_temperature  # noqa: E402
from models.game_model import from_config  # noqa: E402
from sim.preseason_inputs import final_form, final_team_form  # noqa: E402
from sim.ratings_bridge import simulate  # noqa: E402
from train.train_game import era_table_row, load_frames  # noqa: E402

import backtest as B  # noqa: E402
import engine as E  # noqa: E402
from build_travel import arena_of, haversine_km  # noqa: E402  (src/)

T = 2027
SEEDS = [0, 1, 2, 3, 4]
N_SIMS = 10_000
POS_GROUP = {"C": 0, "L": 0, "R": 0, "D": 1, "G": 2}


def schedule_context() -> pd.DataFrame:
    sched = pd.read_csv(RAW / "nhl_schedule_20262027.csv")
    arenas = pd.read_csv(RAW / "arenas.csv")
    sched["dt"] = pd.to_datetime(sched.date)
    sched = sched.sort_values("dt").reset_index(drop=True)
    first = sched.dt.min()
    sched["days_in"] = (sched.dt - first).dt.days
    # per-team prior game date/location walk
    def arena(team):
        # arenas.csv currently ends at 2026; use the latest known arena
        for se in (T, T - 1):
            try:
                return arena_of(team, se, arenas)
            except KeyError:
                continue
        return (0.0, 0.0), 0

    last: dict = {}
    rows = []
    for g in sched.itertuples(index=False):
        site, site_tz = arena(g.home)
        ctx = {}
        for side, team in (("home", g.home), ("away", g.away)):
            prev = last.get(team)
            rest = 9 if prev is None else min((g.dt - prev["dt"]).days, 9)
            km = float(haversine_km(prev["loc"], site)) if prev else 0.0
            dtz = abs(prev["tz"] - site_tz) if prev else 0
            ctx[f"{side}_rest"] = rest
            ctx[f"{side}_km3d"] = km if rest <= 3 else 0.0
            ctx[f"{side}_dtz"] = dtz
            last[team] = {"dt": g.dt, "loc": site, "tz": site_tz}
        rows.append(ctx)
    return pd.concat([sched, pd.DataFrame(rows)], axis=1)


def main():
    torch.manual_seed(711)
    roster = pd.read_csv(RAW / "nhl_rosters_20262027.csv")
    roster["pos_group"] = roster.position.map(POS_GROUP).fillna(0).astype(int)
    seasons = list(range(2009, T))
    gc, pg = load_frames(seasons)
    cfg = json.loads((TENSORS.parents[1] / "configs" / "game_model.json")
                     .read_text())
    tform = final_team_form(gc, T - 1) if cfg.get("use_team_form") else None
    maps = json.loads((TENSORS / "maps.json").read_text())
    form = final_form(pg).set_index("player_id")
    prior = pg[pg.season_end == T - 1]
    toi = prior.groupby("player_id").toi_sec.sum()
    starts = prior.groupby("player_id").goalie_start.sum()
    npz = np.load(TENSORS / f"embeddings_v{T}.npz")
    row = {int(p): i for i, p in enumerate(npz["ids"])}
    emb_m = npz["emb"]
    d = emb_m.shape[1]
    bios = pd.read_parquet(TENSORS / "career_bios.parquet")
    pos_mean = np.zeros((3, d), np.float32)
    for pgi in range(3):
        sel = [row[p] for p in bios[bios.pos_group == pgi].player_id if p in row]
        if sel:
            pos_mean[pgi] = emb_m[sel].mean(0)

    lineups = {}
    n_cold = 0
    for team, grp in roster.groupby("team"):
        grp = grp.assign(toi=grp.playerId.map(toi).fillna(0),
                         starts=grp.playerId.map(starts).fillna(0))
        gk = grp[grp.pos_group == 2].sort_values("starts",
                                                 ascending=False).head(2)
        sk = grp[grp.pos_group < 2].sort_values("toi", ascending=False).head(18)
        players = []
        for r in pd.concat([gk, sk]).itertuples(index=False):
            pid = int(r.playerId)
            ridx = row.get(pid)
            e = emb_m[ridx] if ridx is not None else pos_mean[r.pos_group]
            n_cold += ridx is None
            f = form.loc[pid] if pid in form.index else None
            players.append((e, r.pos_group,
                            0.0 if f is None else float(f.ewma_toi),
                            0.0 if f is None else float(f.gp_todate)))
        lineups[team] = players
    print(f"lineups built for {len(lineups)} teams "
          f"({n_cold} cold-start players -> position mean)")

    sched = schedule_context()
    era_vec = era_table_row(T)
    N = len(sched)
    tens = {"emb": np.zeros((N, 2, 20, d), np.float32),
            "pctx": np.zeros((N, 2, 20, 6), np.float32),
            "pad": np.ones((N, 2, 20), bool),
            "ctx": np.zeros((N, 16), np.float32),
            "era": np.tile(era_vec, (N, 1)).astype(np.float32)}
    for i, g in enumerate(sched.itertuples(index=False)):
        for t, team in ((0, g.home), (1, g.away)):
            for j, (e, pgi, etoi, gpd) in enumerate(lineups.get(team, [])[:20]):
                tens["emb"][i, t, j] = e
                pos1h = np.zeros(3, np.float32)
                pos1h[pgi] = 1
                tens["pctx"][i, t, j] = [*pos1h, etoi / 20.0,
                                         np.log1p(gpd) / 5.0,
                                         float(pgi == 2 and j == 0)]
                tens["pad"][i, t, j] = False
        tens["ctx"][i, :9] = [min(g.home_rest, 7) / 7, min(g.away_rest, 7) / 7,
                              float(g.home_rest <= 1), float(g.away_rest <= 1),
                              g.home_km3d / 1000, g.away_km3d / 1000,
                              g.home_dtz / 3, g.away_dtz / 3, g.days_in / 200]
        if tform is not None:
            hg, ha, hp = tform.get(maps["team"].get(g.home, 0), (2.7, 2.7, 1.1))
            ag, aa, ap = tform.get(maps["team"].get(g.away, 0), (2.7, 2.7, 1.1))
            tens["ctx"][i, 9:15] = [hg / 3.0, ha / 3.0, hp / 2.0,
                                    ag / 3.0, aa / 3.0, ap / 2.0]
    tens = {k: torch.as_tensor(np.nan_to_num(v) if v.dtype != bool else v)
            for k, v in tens.items()}

    ps = []
    for s in SEEDS:
        model = from_config()
        model.load_state_dict(torch.load(CKPT / f"game_T{T}_s{s}.pt",
                                         map_location="cpu"))
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, N, 512):
                b = {k: v[i:i + 512] for k, v in tens.items()}
                outs.append(torch.softmax(model(b)["out4"], -1))
        ps.append(torch.cat(outs).numpy())
    p4 = np.mean(ps, axis=0)
    val = pd.read_csv(NOUT / "preds" / f"game_val_{T}.csv")
    tau = fit_temperature(val[["p_home_reg", "p_away_reg", "p_home_extra",
                               "p_away_extra"]].to_numpy(),
                          val.outcome4.to_numpy())
    p4 = apply_temperature(p4, tau)
    games = sched[["date", "home", "away"]].assign(
        p_home=p4[:, 0] + p4[:, 2], p_home_reg=p4[:, 0],
        p_home_extra=p4[:, 2])
    games.to_csv(NOUT / "preds" / "games_2027_neurhl.csv", index=False)

    v1 = json.loads((PROJ / "output" / "params.json").read_text())
    preds, _, _ = E.run_elo(B.g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    om = E.fit_outcome(preds, list(range(2006, T)))
    sim = simulate(games[["home", "away", "p_home"]], om, N_SIMS, 711, T,
                   playoffs=True)
    teams = list(sim["teams"])
    pts = sim["pts"]
    tab = pd.DataFrame({
        "team": teams,
        "xPts": pts.mean(0), "SD": pts.std(0),
        "P5": np.percentile(pts, 5, axis=0),
        "P50": np.percentile(pts, 50, axis=0),
        "P95": np.percentile(pts, 95, axis=0),
        "Playoff%": sim["made_po"].mean(0) * 100,
        "Division%": sim["won_div"].mean(0) * 100,
        "Conference%": sim["won_conf"].mean(0) * 100,
        "Cup%": sim["won_cup"].mean(0) * 100,
    }).sort_values("xPts", ascending=False)
    tab["FairOdds Cup (US)"] = tab["Cup%"].apply(
        lambda p: int(round(100 * (100 - p) / p)) if p > 0 else 99999)
    tab.to_csv(NOUT / "projections_2026_27_neurhl.csv", index=False)
    print(tab.head(10).round(1).to_string(index=False))
    print(f"tau={tau:.3f}; wrote projections_2026_27_neurhl.csv "
          f"(NeurHL, report-only alongside HOWE)")


if __name__ == "__main__":
    main()
