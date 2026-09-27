"""NeurHL-G stat sheet: Monte Carlo box scores for one game (PLAN_NeurHL4 M).

Draws are consistent with the model's win probability by construction: each
draw first samples the outcome category from NeurHL-G's outcome4 probabilities
(home or away in regulation, home or away in overtime or shootout), then a
regulation score conditional on that category (rejection sampling from the
side goal means under a shared pace shock), then everything else conditional
on the score:

  team SOG      negative binomial around the model mean, floored at goals
  team xG       gamma around the model mean
  PP opps       Poisson(model mean)
  player SOG    multinomial split of team SOG by the model's SOG shares
  player goals  multinomial split of regulation goals by goal shares
  assists       per goal 0/1/2 assists (NHL-typical mix), split by assist shares
  player ixG    Dirichlet split of team xG by ixG shares
  ice time      model means (deterministic budgets)
  goalie        shots against, goals against (excluding the OT/SO winner),
                saves

Output: per player mean, 10th and 90th percentile of SOG, goals, assists,
points, ixG, plus P(goal >= 1) and P(point >= 1); per team the same for goals,
SOG, xG, PP opportunities; the five likeliest final scores.
"""
import numpy as np

ASSIST_MIX = np.array([0.06, 0.30, 0.64])     # P(0, 1, 2 assists) on a goal
SOG_DISP = 40.0                                # NB dispersion for team SOG
XG_SHAPE = 9.0
P_SO_GIVEN_TIE = 0.38                          # 2016-2026 share of ties ending in a shootout


def _conditional_scores(rng, lh, la, cat, sigma, n):
    """Regulation scores for n draws of one category (0 home, 1 away, 2 tie)."""
    out_h = np.empty(n, int)
    out_a = np.empty(n, int)
    filled = 0
    while filled < n:
        m = max(4 * (n - filled), 256)
        eps = rng.normal(-0.5 * sigma ** 2, sigma, m)
        h = rng.poisson(lh * np.exp(eps))
        a = rng.poisson(la * np.exp(eps))
        ok = (h > a) if cat == 0 else (a > h) if cat == 1 else (h == a)
        k = min(int(ok.sum()), n - filled)
        out_h[filled:filled + k] = h[ok][:k]
        out_a[filled:filled + k] = a[ok][:k]
        filled += k
    return out_h, out_a


def simulate(game: dict, n: int = 10_000, seed: int = 711) -> dict:
    """game: dict of one game's model outputs (numpy):
       o4 (4,), goals (2,), sogf (2,), xgf (2,), pp_opps (2,), sigma (float),
       mask (2,20), isog/g/a/ixg/toi_ev/toi_pp/toi_sh (2,20), player_id (2,20)."""
    rng = np.random.default_rng(seed)
    o4 = np.asarray(game["o4"], float)
    o4 = o4 / o4.sum()
    cat4 = rng.choice(4, size=n, p=o4)
    reg_cat = np.where(cat4 == 0, 0, np.where(cat4 == 1, 1, 2))
    gh = np.empty(n, int)
    ga = np.empty(n, int)
    for c in (0, 1, 2):
        idx = np.where(reg_cat == c)[0]
        if len(idx):
            gh[idx], ga[idx] = _conditional_scores(rng, game["goals"][0], game["goals"][1],
                                                   c, game.get("sigma", 0.12), len(idx))
    so = (cat4 >= 2) & (rng.random(n) < P_SO_GIVEN_TIE)
    fin_h = gh + (cat4 == 2)
    fin_a = ga + (cat4 == 3)
    reg_goals = np.stack([gh, ga], 1)
    ot_goal = np.stack([(cat4 == 2) & ~so, (cat4 == 3) & ~so], 1).astype(int)

    team = {}
    players = []
    m = game["mask"] > 0
    for s in (0, 1):
        g_s = reg_goals[:, s] + ot_goal[:, s]              # skater-credited goals
        mu = game["sogf"][s]
        p = SOG_DISP / (SOG_DISP + mu)
        sog = np.maximum(rng.negative_binomial(SOG_DISP, p, n), g_s)
        xg = rng.gamma(XG_SHAPE, game["xgf"][s] / XG_SHAPE, n)
        ppo = rng.poisson(game["pp_opps"][s], n)
        team[s] = {"goals": fin_h if s == 0 else fin_a, "sog": sog, "xg": xg, "pp_opps": ppo}
        ids = game["player_id"][s][m[s]]
        def shares(key):
            w = np.clip(game[key][s][m[s]], 1e-9, None)
            return w / w.sum()
        ps, pg, pa, px = shares("isog"), shares("g"), shares("a"), shares("ixg")
        sog_i = np.array([rng.multinomial(k, ps) for k in sog])
        # goals: at least the goal count assigned to shooters with shots
        g_i = np.array([rng.multinomial(k, pg) for k in g_s])
        sog_i = np.maximum(sog_i, g_i)
        n_ast = np.array([rng.choice(3, size=k, p=ASSIST_MIX).sum() if k else 0 for k in g_s])
        a_i = np.array([rng.multinomial(k, pa) for k in n_ast])
        xg_i = rng.dirichlet(np.clip(px * 60.0, 1e-3, None), n) * xg[:, None]
        pts = g_i + a_i
        for j, pid in enumerate(ids):
            q = lambda x: [float(x.mean()), float(np.percentile(x, 10)), float(np.percentile(x, 90))]
            players.append({
                "side": s, "player_id": int(pid),
                "toi_ev": float(game["toi_ev"][s][m[s]][j]),
                "toi_pp": float(game["toi_pp"][s][m[s]][j]),
                "toi_sh": float(game["toi_sh"][s][m[s]][j]),
                "sog": q(sog_i[:, j]), "goals": q(g_i[:, j]), "assists": q(a_i[:, j]),
                "points": q(pts[:, j]), "ixg": q(xg_i[:, j]),
                "p_goal": float((g_i[:, j] >= 1).mean()),
                "p_point": float((pts[:, j] >= 1).mean())})
    for s in (0, 1):
        opp = 1 - s
        ga_goalie = reg_goals[:, opp]
        team[s]["goalie_ga"] = ga_goalie
        team[s]["goalie_saves"] = np.maximum(team[opp]["sog"] - ga_goalie, 0)
    scores, counts = np.unique(np.stack([fin_h, fin_a], 1), axis=0, return_counts=True)
    top = np.argsort(-counts)[:5]
    summ = lambda x: {"mean": float(np.mean(x)), "p10": float(np.percentile(x, 10)),
                      "p90": float(np.percentile(x, 90))}
    return {
        "p_home_win": float(((fin_h > fin_a)).mean()),
        "o4": o4.tolist(),
        "team": {("home" if s == 0 else "away"): {k: summ(v) for k, v in team[s].items()}
                 for s in (0, 1)},
        "top_scores": [{"home": int(scores[i][0]), "away": int(scores[i][1]),
                        "p": float(counts[i] / n)} for i in top],
        "players": players,
    }
