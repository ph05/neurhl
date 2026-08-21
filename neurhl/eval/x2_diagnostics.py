"""NeurHL-2 — X2 sensitivity analyses, written BEFORE the full result was seen.

PLAN_NeurHL2 states that sensitivity analyses are "reported alongside, never as
substitutes". X2 itself is a head-to-head — team xG differential vs team Corsi
differential at predicting future goal share — and if it fails, the prereg says
L0 is deleted and the engine uses Corsi. That is binding and is NOT relaxed here.

These diagnostics exist to characterise HOW it failed or passed, not to rescue
it. The 2-vantage smoke test showed Corsi ahead (0.408 vs 0.239, n=60), so this
file is written now, with the full 18-vantage result still running, precisely so
the analysis cannot be tuned to the answer.

Four things worth knowing whichever way the head-to-head lands:

  1. **Incremental value.** Even where Corsi wins alone, does xG add anything on
     top of it? Partial correlation of xG with future goals, controlling for
     Corsi, and the R^2 of both together vs Corsi alone.
  2. **Sample-size confound.** Corsi counts ~2x the events xG does (every
     blocked shot included), so part of any Corsi win is lower sampling noise
     rather than better information. Comparing xG against FENWICK (unblocked
     attempts, the same event set xG is built on) isolates that.
  3. **Horizon.** Half-season targets are noisy. Next-SEASON goal share is a
     harder, cleaner target and is what a projection model actually cares about.
  4. **Stability.** Split-half repeatability of each differential — the property
     that makes Corsi useful despite being cruder.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/eval/x2_diagnostics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402


def load_team_games() -> pd.DataFrame:
    """One row per team-game: 5v5 xG, Fenwick, Corsi and goals, for/against."""
    rows = []
    for p in sorted(TENSORS.glob("xg_shots_2*.parquet")):
        v = int(p.stem.split("_")[-1])
        d = pd.read_parquet(p)
        f = d[(d.sk_h == 5) & (d.sk_a == 5) & (~d.en.astype(bool))]
        g = (f.assign(side=np.where(f.is_home == 1, "h", "a"))
             .groupby(["game_id", "side"])
             .agg(xg=("xg", "sum"), gl=("goal", "sum"),
                  ff=("goal", "size")).reset_index())
        w = g.pivot(index="game_id", columns="side",
                    values=["xg", "gl", "ff"]).fillna(0.0)
        w.columns = [f"{a}_{b}" for a, b in w.columns]
        w = w.reset_index()
        w["season_end"] = v

        sp = TENSORS / f"stints_{v}.parquet"
        st = pd.read_parquet(sp, columns=["game_id", "game_type", "is_5v5",
                                          "cf_h", "cf_a"])
        st = st[(st.game_type == 2) & st.is_5v5]
        cz = st.groupby("game_id").agg(cf_h=("cf_h", "sum"),
                                       cf_a=("cf_a", "sum")).reset_index()
        gc = pd.read_parquet(TENSORS / f"games_ctx_{v}.parquet",
                             columns=["game_id", "home_idx", "away_idx", "date"])
        w = w.merge(cz, on="game_id").merge(gc, on="game_id")
        rows.append(w)
    tg = pd.concat(rows, ignore_index=True)

    out = []
    for side, opp, tcol in (("h", "a", "home_idx"), ("a", "h", "away_idx")):
        out.append(pd.DataFrame({
            "season_end": tg.season_end, "date": tg.date, "team": tg[tcol],
            "xg_f": tg[f"xg_{side}"], "xg_a": tg[f"xg_{opp}"],
            "ff_f": tg[f"ff_{side}"], "ff_a": tg[f"ff_{opp}"],
            "cf_f": tg[f"cf_{side}"], "cf_a": tg[f"cf_{opp}"],
            "g_f": tg[f"gl_{side}"], "g_a": tg[f"gl_{opp}"]}))
    return pd.concat(out, ignore_index=True).sort_values(
        ["season_end", "team", "date"])


def shares(x: pd.DataFrame) -> pd.DataFrame:
    g = x.groupby(["season_end", "team"]).sum(numeric_only=True)
    return pd.DataFrame({
        "xg": g.xg_f / (g.xg_f + g.xg_a).replace(0, np.nan),
        "ff": g.ff_f / (g.ff_f + g.ff_a).replace(0, np.nan),
        "cf": g.cf_f / (g.cf_f + g.cf_a).replace(0, np.nan),
        "gf": g.g_f / (g.g_f + g.g_a).replace(0, np.nan)})


def partial_corr(y, x, z):
    """corr(y, x | z) — does x add anything once z is known?"""
    def resid(a, b):
        b1 = np.column_stack([np.ones(len(b)), b])
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(resid(y, z), resid(x, z))[0, 1])


def r2(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [np.asarray(c) for c in X])
    pred = X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    t = load_team_games()
    t["k"] = t.groupby(["season_end", "team"]).cumcount()
    t["n"] = t.groupby(["season_end", "team"]).k.transform("max") + 1
    h1, h2 = shares(t[t.k < t.n / 2]), shares(t[t.k >= t.n / 2])
    j = h1.join(h2, lsuffix="_1", rsuffix="_2").dropna()
    print(f"team-seasons: {len(j)}\n")

    # ---- 1. head-to-head (the gate itself) and Fenwick control
    print("1. predicting SECOND-half 5v5 GF% from first half")
    for nm in ("xg", "cf", "ff"):
        r = float(np.corrcoef(j[f"{nm}_1"], j.gf_2)[0, 1])
        print(f"     {nm:>3}%  r = {r:+.4f}")
    print("   (ff = Fenwick, the same unblocked event set xG is built on --")
    print("    isolates how much of any Corsi edge is just counting more events)")

    # ---- 2. incremental value of xG over Corsi
    y = j.gf_2.to_numpy()
    pc = partial_corr(y, j.xg_1.to_numpy(), j.cf_1.to_numpy()[:, None])
    pc_rev = partial_corr(y, j.cf_1.to_numpy(), j.xg_1.to_numpy()[:, None])
    print(f"\n2. incremental (partial correlation with future GF%)")
    print(f"     xG | Corsi     = {pc:+.4f}")
    print(f"     Corsi | xG     = {pc_rev:+.4f}")
    print(f"     R2 Corsi alone = {r2(y, [j.cf_1]):.4f}")
    print(f"     R2 xG alone    = {r2(y, [j.xg_1]):.4f}")
    print(f"     R2 both        = {r2(y, [j.cf_1, j.xg_1]):.4f}")

    # ---- 3. next-season horizon
    nxt = h1.join(h2, lsuffix="_1", rsuffix="_2")
    full = shares(t).reset_index()
    full["next"] = full.season_end + 1
    nx = full.merge(full[["season_end", "team", "gf"]]
                    .rename(columns={"season_end": "next", "gf": "gf_next"}),
                    on=["next", "team"], how="inner").dropna(
                        subset=["xg", "cf", "gf_next"])
    print(f"\n3. predicting NEXT-SEASON 5v5 GF% (n={len(nx)} team-season pairs)")
    for nm in ("xg", "cf", "ff", "gf"):
        r = float(np.corrcoef(nx[nm], nx.gf_next)[0, 1])
        print(f"     {nm:>3}%  r = {r:+.4f}")
    pcn = partial_corr(nx.gf_next.to_numpy(), nx.xg.to_numpy(),
                       nx.cf.to_numpy()[:, None])
    print(f"     xG | Corsi     = {pcn:+.4f}")

    # ---- 4. stability
    print("\n4. split-half stability of each differential (first vs second half)")
    for nm in ("xg", "cf", "ff", "gf"):
        r = float(np.corrcoef(j[f"{nm}_1"], j[f"{nm}_2"])[0, 1])
        print(f"     {nm:>3}%  r = {r:+.4f}")

    out = {
        "n_team_seasons": int(len(j)),
        "half_season": {nm: round(float(np.corrcoef(j[f"{nm}_1"], j.gf_2)[0, 1]), 4)
                        for nm in ("xg", "cf", "ff")},
        "incremental": {"xg_given_corsi": round(pc, 4),
                        "corsi_given_xg": round(pc_rev, 4),
                        "r2_corsi": round(r2(y, [j.cf_1]), 4),
                        "r2_xg": round(r2(y, [j.xg_1]), 4),
                        "r2_both": round(r2(y, [j.cf_1, j.xg_1]), 4)},
        "next_season": {"n": int(len(nx)),
                        **{nm: round(float(np.corrcoef(nx[nm], nx.gf_next)[0, 1]), 4)
                           for nm in ("xg", "cf", "ff", "gf")},
                        "xg_given_corsi": round(pcn, 4)},
        "stability": {nm: round(float(np.corrcoef(j[f"{nm}_1"], j[f"{nm}_2"])[0, 1]), 4)
                      for nm in ("xg", "cf", "ff", "gf")},
    }
    p = Path(__file__).resolve().parents[1] / "configs" / "x2_diagnostics.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
