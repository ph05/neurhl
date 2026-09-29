"""Rookie priors from pre-NHL records (PLAN_NeurHL_1_1 A15).

A skater with no NHL history enters the engine at his position's average
rates. This module learns, walk-forward, how scoring in another league
translates to a first NHL season:

  1. League factors (NHLe): for every move "league L in season t, NHL in
     season t+1 (>= 20 games)", f_L = sum(NHL goals) / sum(L goals per game x
     NHL games); separately for goals and assists, by position group.
     Only NHL seasons < V are used when predicting season V.
  2. A rookie's translated rate: the games-weighted average of f_L x his rate
     in each non-NHL league over the two seasons before his first NHL season
     (the later season counts twice).
  3. Shrinkage: NHL rate = c + beta x translated + gamma x (age - 21), fitted
     on earlier rookies with at least 20 games; players with little pre-NHL
     data are pulled toward the rookie average by their games.

evaluate() compares, for rookies of seasons V = 2014..2024 (never 2025 or
2026), the predicted goals and assists per game with the position average,
weighting by games played.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402

NCAA = {"NCAA", "WCHA", "H-East", "Hockey East", "CCHA", "ECAC", "Big Ten", "Big-10", "NCHC", "CHA",
        "Atlantic Hockey", "AHA", "Big10"}
ALIAS = {"Sweden": "SHL", "Elitserien": "SHL", "Finland": "Liiga", "SM-liiga": "Liiga", "Russia": "KHL",
         "Czechia": "Czech", "CzRep": "Czech", "Swiss": "NL", "NLA": "NL", "Germany": "DEL",
         "HockeyAllsvenskan": "Allsvenskan", "Sweden-2": "Allsvenskan", "Finland-2": "Mestis",
         "Swe-Jr.": "SwedenJr", "Sweden Jr.": "SwedenJr", "J20 Nationell": "SwedenJr",
         "Fin-Jr.": "FinlandJr", "FInland-Jr.": "FinlandJr", "Russia-2": "VHL", "USDP": "USNTDP",
         "USNTDP": "USNTDP", "NTDP": "USNTDP"}
MAIN = {"AHL", "OHL", "WHL", "QMJHL", "NCAA", "USHL", "KHL", "SHL", "Liiga", "NL", "DEL", "Czech",
        "Allsvenskan", "Mestis", "SwedenJr", "FinlandJr", "VHL", "MHL", "ECHL", "USNTDP", "BCHL", "AJHL",
        "EBEL", "ICEHL", "Slovakia"}
EXCLUDE_PREFIX = ("WC", "WJC", "WJ18", "OG", "Olympics", "4 Nations", "4-Nations", "5-Nations", "8-Nations",
                  "EHT", "Champions HL", "Hlinka", "U17", "U18", "Can-Cup", "Spengler", "AS-Game", "Exhib")
MIN_NHL_GP = 20


def norm_league(x: str) -> str:
    if x == "NHL":
        return "NHL"
    if any(x.startswith(p) for p in EXCLUDE_PREFIX):
        return "EVENT"
    x = "NCAA" if x in NCAA else ALIAS.get(x, x)
    return x if x in MAIN else "OTHER"


def load() -> pd.DataFrame:
    d = pd.read_parquet(TENSORS / "prenhl_seasons.parquet")
    d["lg"] = d.league.map(norm_league)
    d = d[(d.lg != "EVENT") & (d.pos_group < 2)]
    return d.groupby(["player_id", "season_end", "lg"], as_index=False).agg(
        gp=("gp", "sum"), g=("g", "sum"), a=("a", "sum"), age=("age", "first"),
        pos=("pos_group", "first"), pick=("draft_overall", "first"))


def factors(d: pd.DataFrame, before: int) -> dict:
    """League factors from moves into NHL seasons < before: {(lg, pos, stat): f}."""
    nhl = d[(d.lg == "NHL") & (d.gp >= MIN_NHL_GP)][["player_id", "season_end", "gp", "g", "a"]]
    nhl = nhl.assign(season_end=nhl.season_end - 1)          # align: other league at t, NHL at t+1
    m = d[(d.lg != "NHL") & (d.gp >= 10)].merge(nhl, on=["player_id", "season_end"], suffixes=("", "_n"))
    m = m[m.season_end + 1 < before]
    out = {}
    for (lg, pos), g_ in m.groupby(["lg", "pos"]):
        for st in ("g", "a"):
            den = (g_[st] / g_.gp * g_.gp_n).sum()
            if len(g_) >= 15 and den > 0:
                out[(lg, pos, st)] = float(g_[f"{st}_n"].sum() / den)
    # pooled fallbacks per position
    for pos in (0, 1):
        for st in ("g", "a"):
            g_ = m[m.pos == pos]
            den = (g_[st] / g_.gp * g_.gp_n).sum()
            out[("ALL", pos, st)] = float(g_[f"{st}_n"].sum() / den) if den > 0 else 0.2
    return out


def translated(d: pd.DataFrame, pid_season: pd.DataFrame, f: dict) -> pd.DataFrame:
    """For each (player_id, T): games-weighted translated g/gp and a/gp over the
    non-NHL leagues of seasons T-1 (weight 2) and T-2 (weight 1), and the
    pre-NHL games that informed it."""
    o = d[d.lg != "NHL"]
    rows = []
    idx = o.set_index("player_id")
    for pid, T in zip(pid_season.player_id, pid_season.season_end):
        if pid not in idx.index:
            rows.append((pid, T, np.nan, np.nan, 0.0))
            continue
        r = idx.loc[[pid]]
        r = r[(r.season_end >= T - 2) & (r.season_end <= T - 1)]
        if not len(r):
            rows.append((pid, T, np.nan, np.nan, 0.0))
            continue
        w = r.gp * np.where(r.season_end == T - 1, 2.0, 1.0)
        tg, ta = [], []
        for x in r.itertuples():
            fg = f.get((x.lg, x.pos, "g"), f[("ALL", x.pos, "g")] * (0.5 if x.lg == "OTHER" else 1.0))
            fa = f.get((x.lg, x.pos, "a"), f[("ALL", x.pos, "a")] * (0.5 if x.lg == "OTHER" else 1.0))
            tg.append(fg * x.g / x.gp)
            ta.append(fa * x.a / x.gp)
        rows.append((pid, T, float(np.average(tg, weights=w)), float(np.average(ta, weights=w)), float(r.gp.sum())))
    return pd.DataFrame(rows, columns=["player_id", "season_end", "tg", "ta", "pre_gp"])


def rookies(d: pd.DataFrame) -> pd.DataFrame:
    """First NHL season with >= MIN_NHL_GP games, for players with < 20 NHL games before it."""
    n = d[d.lg == "NHL"].sort_values(["player_id", "season_end"])
    n["prior_gp"] = n.groupby("player_id").gp.cumsum() - n.gp
    r = n[(n.gp >= MIN_NHL_GP) & (n.prior_gp < 20)].drop_duplicates("player_id")
    return r[["player_id", "season_end", "gp", "g", "a", "age", "pos", "pick"]]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Per position and stat: rate = c + beta x translated + gamma x (age - 21), weighted by NHL games;
    rookies with no translated rate get the fitted intercept at their age; pre-NHL games shrink."""
    out = test.copy()
    for pos in (0, 1):
        tr = train[(train.pos == pos) & train.tg.notna()]
        te = out.pos == pos
        for st, col in (("g", "tg"), ("a", "ta")):
            y = tr[st] / tr.gp
            X = np.c_[np.ones(len(tr)), tr[col], tr.age - 21]
            W = np.sqrt(tr.gp.to_numpy())
            coef = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
            base = float(np.average(train.loc[train.pos == pos, st] / train.loc[train.pos == pos, "gp"],
                                    weights=train.loc[train.pos == pos, "gp"]))
            x = out.loc[te, col].to_numpy()
            pred = coef[0] + coef[1] * np.nan_to_num(x, nan=0) + coef[2] * (out.loc[te, "age"] - 21)
            rel = out.loc[te, "pre_gp"] / (out.loc[te, "pre_gp"] + 40.0)        # little pre-NHL data -> average
            pred = np.where(np.isfinite(x), rel * pred + (1 - rel) * base, base)
            out.loc[te, f"pred_{st}"] = np.clip(pred, 0.0, None)
            out.loc[te, f"base_{st}"] = base
    return out


def evaluate(seasons=range(2014, 2025)):
    d = load()
    rk = rookies(d)
    res = []
    for V in seasons:
        f = factors(d, V)
        tr = rk[rk.season_end < V]
        te = rk[rk.season_end == V]
        tr = tr.merge(translated(d, tr[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        te = te.merge(translated(d, te[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        res.append(fit_predict(tr, te))
    r = pd.concat(res, ignore_index=True)
    for st in ("g", "a"):
        y = r[st] / r.gp
        w = r.gp
        e_m = np.average(np.abs(r[f"pred_{st}"] - y), weights=w)
        e_b = np.average(np.abs(r[f"base_{st}"] - y), weights=w)
        c = np.corrcoef(r[f"pred_{st}"], y)[0, 1]
        print(f"{st}/gp: MAE model {e_m:.4f} vs position average {e_b:.4f} ({100 * (e_m / e_b - 1):+.1f}%), "
              f"corr {c:.3f}, n {len(r)} rookies")
    pts_m = (r.pred_g + r.pred_a) * r.gp
    pts_b = (r.base_g + r.base_a) * r.gp
    act = r.g + r.a
    print(f"season points MAE: model {np.abs(pts_m - act).mean():.2f} vs position average {np.abs(pts_b - act).mean():.2f}")
    return r


if __name__ == "__main__":
    evaluate()
