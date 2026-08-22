"""NeurHL-2 — S4-v2: game outcomes from validated components.

PLAN_NeurHL2 validation item 7 preregistered this path and named it the most
likely route to the best game-level number:

  "the engine's hazard rates can also be BLENDED into the hierarchical head as an
   additional feature, which is the most likely route to the best game-level
   number."

It is taken here because the direct S1 hazard query failed three separate ways
and the failures were all in the QUERY, not the model. On natural data the S1
ensemble reproduces home advantage correctly (expected +0.329 goals/game against
an observed +0.450, and +0.195 vs +0.268 at 5v5). But asking it for a hazard at a
fabricated state required inventing a context the game has not played yet, and
every attempt to fabricate one distorted the answer:

  * overwriting period/score onto arbitrary contexts produced "period 3" on a
    clock reading early first period -> lambda_away 3.09 vs lambda_home 2.48 with
    identical mirrored units;
  * six contexts per state cell gave sd 1.41 g/60 on the away rate alone;
  * slicing mid-game windows to positional slots 0..T-1 told the model it was
    watching an opening faceoff.

Each was fixed and the home/away split still would not come right, because
conditioning on a single (period, score, 5v5) cell is a selection the natural
sequence never makes.

So the game layer is built from the pieces that ARE validated:

  * **team attack/defence** from prior-season scoring rates, empirical-Bayes
    shrunk — the part a rate model does well;
  * **roster strength** from the walk-forward RAPM prior, TOI-weighted over the
    projected lineup, which carries player turnover that team-level rates cannot;
  * **score effects** from S1's own curve, the thing E2 verified at ratio 0.979
    and correlation 0.979 — applied inside the integration so leading teams
    suppress and trailing teams push;
  * **Kolmogorov forward integration** over the joint score distribution, as in
    S4-v1, so this is still an integration rather than a rollout.

Everything is walk-forward: to score season V, every input is computed from
seasons < V.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

PERIOD_LEN = 1200.0
MAX_GOALS = 12
SHRINK_GAMES = 30.0        # EB prior weight, in games


def team_game_rates(seasons) -> pd.DataFrame:
    """Per team-game goals for/against, regular season only."""
    rows = []
    for s in seasons:
        p = TENSORS / f"games_ctx_{s}.parquet"
        if not p.exists():
            continue
        g = pd.read_parquet(p, columns=["game_id", "game_type", "date",
                                        "home_idx", "away_idx", "home_g",
                                        "away_g"])
        g = g[g.game_type == 2]
        for side, opp, t in (("home", "away", "home_idx"),
                             ("away", "home", "away_idx")):
            rows.append(pd.DataFrame({
                "season_end": s, "date": g.date, "team": g[t],
                "gf": g[f"{side}_g"], "ga": g[f"{opp}_g"],
                "is_home": int(side == "home")}))
    return pd.concat(rows, ignore_index=True)


def team_strength(train_seasons) -> tuple:
    """EB-shrunk attack/defence multipliers per team, plus league rate and
    home advantage, all from `train_seasons` only."""
    t = team_game_rates(train_seasons)
    if not len(t):
        return {}, {}, 2.7, 1.0
    lg_gf = t.gf.mean()
    home_mult = (t[t.is_home == 1].gf.mean() / max(t[t.is_home == 0].gf.mean(), 1e-9))
    agg = t.groupby("team").agg(gf=("gf", "sum"), ga=("ga", "sum"),
                                n=("gf", "size"))
    w = agg.n / (agg.n + SHRINK_GAMES)
    att = (w * (agg.gf / agg.n) / lg_gf + (1 - w) * 1.0).to_dict()
    dfn = (w * (agg.ga / agg.n) / lg_gf + (1 - w) * 1.0).to_dict()
    return att, dfn, float(lg_gf), float(home_mult)


def roster_strength(season: int, prev: int) -> dict:
    """TOI-weighted RAPM net rating per team, from the walk-forward prior.

    Team-level scoring rates cannot see a roster that turned over; RAPM can.
    Both are walk-forward, so this stays a projection.
    """
    pr = TENSORS / f"rapm_prior_{season}.parquet"
    pg = TENSORS / f"player_games_{prev}.parquet"
    gc = TENSORS / f"games_ctx_{prev}.parquet"
    if not (pr.exists() and pg.exists() and gc.exists()):
        return {}
    r = pd.read_parquet(pr)
    r = r[~r.is_replacement.astype(bool)][["player_id", "cf_off", "cf_def"]]
    r["net"] = r.cf_off - r.cf_def
    p = pd.read_parquet(pg, columns=["game_id", "player_id", "is_home",
                                     "toi_sec", "pos_group", "game_type"])
    p = p[(p.game_type == 2) & (p.pos_group != 2)]
    g = pd.read_parquet(gc, columns=["game_id", "home_idx", "away_idx"])
    p = p.merge(g, on="game_id", how="left")
    p["team"] = np.where(p.is_home, p.home_idx, p.away_idx)
    a = p.groupby(["team", "player_id"], as_index=False).toi_sec.sum()
    a = a.merge(r[["player_id", "net"]], on="player_id", how="left").dropna()
    out = {}
    for team, d in a.groupby("team"):
        out[int(team)] = float(np.average(d.net, weights=d.toi_sec))
    mu = np.mean(list(out.values())) if out else 0.0
    sd = np.std(list(out.values())) or 1.0
    return {k: (v - mu) / sd for k, v in out.items()}


def run_elo(seasons, K=8.0, H=30.0, phi_s=0.7) -> pd.DataFrame:
    """Walk-forward Elo over the game sequence -- PRE-game rating for each game.

    Implemented here rather than imported from src/engine.py: NeurHL is required
    to leave src/ untouched, and an earlier accidental import of a same-named
    module executed it and truncated NOTES.md. Parameters match the house Elo
    (K=8, H=30, season carry-over phi=0.7) so the incumbent is nested rather than
    approximated -- PLAN_NeurHL2 records that nesting Elo cuts the paired SD 2.4x,
    which is ~5.7x effective sample size.

    Every rating used for a game is computed strictly from EARLIER games.
    """
    frames = []
    for s in seasons:
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            g = pd.read_parquet(p, columns=["game_id", "game_type", "date",
                                            "home_idx", "away_idx",
                                            "home_g", "away_g"])
            frames.append(g[g.game_type == 2].assign(season_end=s))
    if not frames:
        return pd.DataFrame()
    g = pd.concat(frames, ignore_index=True).sort_values(
        ["season_end", "date", "game_id"], kind="stable").reset_index(drop=True)

    R = {}
    cur_season = None
    out_h, out_a = np.zeros(len(g)), np.zeros(len(g))
    for i, r in enumerate(g.itertuples()):
        if r.season_end != cur_season:
            cur_season = r.season_end
            R = {k: 1500.0 + phi_s * (v - 1500.0) for k, v in R.items()}
        h, a = int(r.home_idx), int(r.away_idx)
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        out_h[i], out_a[i] = rh, ra
        e = 1.0 / (1.0 + 10 ** (-((rh + H) - ra) / 400.0))
        s_ = 1.0 if r.home_g > r.away_g else 0.0
        R[h] = rh + K * (s_ - e)
        R[a] = ra - K * (s_ - e)
    g["elo_h"], g["elo_a"] = out_h, out_a
    g["elo_logit"] = ((g.elo_h + H) - g.elo_a) * np.log(10) / 400.0
    return g[["game_id", "season_end", "elo_h", "elo_a", "elo_logit"]]


def score_effect_curve(gate_json: Path) -> dict:
    """Multiplier on a side's rate as a function of ITS OWN lead, taken from
    S1's measured E2 curve rather than assumed."""
    import json
    try:
        d = json.loads(gate_json.read_text())
        rows = d["E2"]["rows"]
        base = next(r["pred_share"] for r in rows if r["own_lead"] == 0)
        return {int(r["own_lead"]): r["pred_share"] / base for r in rows}
    except Exception:
        return {k: 1.0 for k in range(-4, 5)}


def integrate(lam_h, lam_a, curve, step=10.0, max_goals=MAX_GOALS,
              n_periods=3):
    """Forward Kolmogorov step over the joint (goals_home, goals_away) grid,
    with each side's rate modulated by its own current lead."""
    n = max_goals + 1
    P = np.zeros((n, n))
    P[0, 0] = 1.0
    gh = np.arange(n)[:, None] * np.ones((1, n))
    ga = np.ones((n, 1)) * np.arange(n)[None, :]
    lead_h = np.clip(gh - ga, -4, 4).astype(int)
    mh = np.vectorize(lambda d: curve.get(int(d), 1.0))(lead_h)
    ma = np.vectorize(lambda d: curve.get(int(-d), 1.0))(lead_h)

    T = n_periods * PERIOD_LEN
    t = 0.0
    while t < T - 1e-9:
        dt = min(step, T - t)
        ph = 1.0 - np.exp(-(lam_h / 3600.0) * mh * dt)
        pa = 1.0 - np.exp(-(lam_a / 3600.0) * ma * dt)
        nxt = P * (1 - ph) * (1 - pa)
        nxt[1:, :] += (P * ph * (1 - pa))[:-1, :]
        nxt[:, 1:] += (P * (1 - ph) * pa)[:, :-1]
        nxt[1:, 1:] += (P * ph * pa)[:-1, :-1]
        P = nxt / nxt.sum()
        t += dt
    return P


def outcome(P, ot_home_edge=0.53):
    n = P.shape[0]
    gh = np.arange(n)[:, None]
    ga = np.arange(n)[None, :]
    reg_h = float(P[gh > ga].sum())
    tie = float(P[gh == ga].sum())
    return {"p_home_win": reg_h + tie * ot_home_edge,
            "p_reg_home": reg_h, "p_tie": tie,
            "exp_gh": float((P * gh).sum()), "exp_ga": float((P * ga).sum())}
