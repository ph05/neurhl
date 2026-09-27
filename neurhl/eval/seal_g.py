"""NeurHL-G one-shot seal (PLAN_NeurHL4, SEAL and C). Runs once.

Refuses unless:
  * output/g_seal_result.json does not exist (never run before);
  * PLAN_NeurHL4.md carries a committed "## FREEZE" section that is on
    origin/main (checked after git fetch);
  * output/g_sstop.json records an S-STOP pass;
  * the frozen config hash matches the one in the FREEZE section;
  * every sealed input matches configs/sealed_inputs.sha256.

Then unseals, trains snapshot 2025 (seasons <= 2024) and snapshot 2026 (seasons
<= 2025) with the frozen configuration and seeds, fits the walk-forward stack
on out-of-sample predictions of earlier seasons, and scores the 2,624 games:
  S1  paired per-game log loss NeurHL-G - Elo, two-sided p < 0.05, same
      direction in both seasons (primary)
  S2  only if S1 passes: NeurHL-G - NeurHL-H, two-sided p < 0.05
  secondary (Holm): team SOG, xGF, xGA, goals vs a log5 team-history baseline;
      skater TOI, SOG, ixG, goals, assists vs shrunk own-history baselines
Writes output/g_seal_result.json and output/preds/g_seal_games.csv, then (as
the user directed: seal, then retrain for the live season) trains the live
bundle on seasons <= 2026 with the frozen configuration.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT, TENSORS  # noqa: E402
import windows as W  # noqa: E402

OUT = NOUT / "g_seal_result.json"


def git(*a):
    return subprocess.run(["git", "-C", str(PROJ), *a], capture_output=True, text=True)


def preconditions():
    assert not OUT.exists(), "the seal has already been spent (single use)"
    git("fetch", "-q", "origin")
    plan = (PROJ / "PLAN_NeurHL4.md").read_text()
    m = re.search(r"## FREEZE.*?config `([\w-]+)`.*?sha256 `([0-9a-f]{64})`", plan, re.S)
    assert m, "PLAN_NeurHL4.md has no FREEZE section naming config and sha256"
    cfg_name, cfg_sha = m.group(1), m.group(2)
    fz = git("log", "-1", "--format=%H", "-S", "## FREEZE", "--", "PLAN_NeurHL4.md").stdout.strip()
    assert fz, "FREEZE section is not committed"
    assert git("merge-base", "--is-ancestor", fz, "origin/main").returncode == 0, \
        "FREEZE commit is not on origin/main"
    ss = json.loads((NOUT / "g_sstop.json").read_text())
    assert ss.get("pass") is True, "S-STOP did not pass; the seal is not spent"
    cp = CONFIGS / "neurhl_g" / f"{cfg_name}.json"
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == cfg_sha, "frozen config changed"
    for line in (CONFIGS / "sealed_inputs.sha256").read_text().splitlines():
        h, f = line.split()
        assert hashlib.sha256((TENSORS / f).read_bytes()).hexdigest() == h, f"sealed input changed: {f}"
    return cfg_name, fz


def paired(a, b):
    from scipy import stats
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    z = d.mean() / se
    return {"diff": float(d.mean()), "se": float(se), "z": float(z),
            "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "p": float(2 * stats.norm.sf(abs(z)))}


def main():
    cfg_name, fz = preconditions()
    W.unseal(str(Path(__file__).resolve()))
    from eval.run_g import fit_stack, h_ref, logit, nll
    from train.train_live_g import oos_preds
    from train.train_neurhl_g import Data, predict, train_snapshot

    cfg = json.loads((CONFIGS / "neurhl_g" / f"{cfg_name}.json").read_text())
    seeds = list(range(cfg.get("seeds", 5)))
    train_cfg = {k: v for k, v in cfg.items() if k not in ("stack_h", "parent", "delta", "seeds")}
    D = Data("score")
    rows = []
    for T in W.SEALED:
        te = np.where(D.meta.season_end.to_numpy() == T)[0]
        outs = []
        for sd in seeds:
            m, P = train_snapshot(D, T, {**train_cfg, "seed": sd})
            outs.append(predict(m, P, te))
        o = {k: np.mean([x[k] for x in outs], 0) for k in outs[0]}
        hist_seasons = sorted(set(range(2011, T)) - W.NO_SCORE)
        hist = oos_preds(D, cfg, hist_seasons, seeds)
        cols = [hist.elo_logit, logit(hist.p_g)]
        gid_te = D.meta.game_id.iloc[te].values
        cols_t = [D.A["CTX"][te, 0], logit(o["p_home_win"])]
        if cfg.get("stack_h"):          # the candidate's declared stack form
            href = h_ref(hist_seasons + [T])
            cols.append(logit(hist.game_id.map(href).to_numpy()))
            cols_t.append(logit(pd.Series(gid_te).map(href).to_numpy()))
        X = np.column_stack(cols)
        ok = np.isfinite(X).all(1)
        mdl, use = fit_stack(X[ok], hist.y.to_numpy()[ok], hist.season.to_numpy()[ok])
        Xt = np.column_stack(cols_t)
        p = mdl.predict_proba(np.nan_to_num(Xt[:, use]))[:, 1]
        df = D.meta.iloc[te][["game_id", "season_end", "outcome4"]].copy()
        df["p_g"], df["p_g_raw"] = p, o["p_home_win"]
        df["p_elo"] = 1 / (1 + np.exp(-D.A["CTX"][te, 0]))
        tmt, skt = D.names["tm_tgt"], D.names["sk_tgt"]
        for k, yk in (("sogf", "sogf"), ("xgf", "xgf_all"), ("goals", "gf_reg")):
            for s, lab in ((0, "h"), (1, "a")):
                df[f"{k}_{lab}"] = o[k][:, s]
                df[f"y_{k}_{lab}"] = D.A["TMY"][te, s, tmt.index(yk)]
        df.attrs = {}
        rows.append((df, o, te))
    ev = pd.concat([r[0] for r in rows], ignore_index=True)
    ev["y"] = ev.outcome4.isin([0, 2]).astype(float)
    h = pd.read_csv(NOUT / "preds" / "hier_restatement_games.csv").set_index("game_id").p_neurhl_h
    ev["p_h"] = ev.game_id.map(h)

    lg, le, lh = nll(ev.p_g, ev.y), nll(ev.p_elo, ev.y), nll(ev.p_h, ev.y)
    s1 = paired(lg, le)
    per = ev.assign(d=lg - le).groupby("season_end").d.mean()
    s1["per_season"] = {str(k): float(v) for k, v in per.items()}
    s1["pass"] = bool(s1["diff"] < 0 and s1["p"] < 0.05 and (per < 0).all())
    ok_h = ev.p_h.notna()
    s2 = paired(lg[ok_h], lh[ok_h]) if s1["pass"] else {"skipped": "S1 did not pass"}
    if s1["pass"]:
        s2["pass"] = bool(s2["diff"] < 0 and s2["p"] < 0.05)

    # secondary family: G vs history baselines, Holm
    sec = {}
    tm_raw = D.A["TM"]
    names = D.names["tm_feat"]
    base = {"sogf": ("tm_sogf_pg_d95", "tm_soga_pg_d95"),
            "xgf": ("tm_xgf_all_pg_d95", "tm_xga_all_pg_d95"),
            "goals": ("tm_gf_pg_d95", "tm_ga_pg_d95")}
    te_all = np.concatenate([r[2] for r in rows])
    for k, (fcol, acol) in base.items():
        f = tm_raw[te_all][..., names.index(fcol)]
        a = tm_raw[te_all][..., names.index(acol)]
        lg_mean = np.nanmean(f)
        bh = f[:, 0] * a[:, 1] / lg_mean
        ba = f[:, 1] * a[:, 0] / lg_mean
        mu = np.r_[ev[f"{k}_h"], ev[f"{k}_a"]]
        bb = np.nan_to_num(np.r_[bh, ba], nan=lg_mean)
        y = np.r_[ev[f"y_{k}_h"], ev[f"y_{k}_a"]]
        okk = np.isfinite(y)
        dev = lambda m_: np.clip(m_, 1e-6, None) - y * np.log(np.clip(m_, 1e-6, None))
        sec[f"team_{k}"] = paired(dev(mu)[okk], dev(bb)[okk])
        if k == "xgf":   # xGA is each side's view of the opponent's xGF: same values
            sec["team_xga"] = dict(sec["team_xgf"], note="identical to team_xgf by construction")
    # skaters: model means vs shrunk own-history baselines
    bi = {c: i for i, c in enumerate(D.names["sk_base"])}
    skt = D.names["sk_tgt"]
    SKB, SKY, SKM = D.A["SKB"][te_all], D.A["SKY"][te_all], D.A["SKM"][te_all] > 0
    O = {k: np.concatenate([r[1][k] for r in rows]) for k in ("toi_ev", "toi_pp", "toi_sh",
                                                             "isog", "ixg", "g", "a")}
    toi_b = SKB[..., bi["b_toi_ev"]] + SKB[..., bi["b_toi_pp"]] + SKB[..., bi["b_toi_sh"]]
    basel = {"toi": toi_b,
             "sog": SKB[..., bi["b_isog60"]] * toi_b / 60,
             "ixg": SKB[..., bi["b_ixg60"]] * toi_b / 60,
             "goals": SKB[..., bi["b_g_per_sog"]] * SKB[..., bi["b_isog60"]] * toi_b / 60,
             "assists": SKB[..., bi["b_a60"]] * toi_b / 60}
    model = {"toi": O["toi_ev"] + O["toi_pp"] + O["toi_sh"], "sog": O["isog"],
             "ixg": O["ixg"], "goals": O["g"], "assists": O["a"]}
    ytgt = {"toi": SKY[..., skt.index("toi_ev")] + SKY[..., skt.index("toi_pp")]
            + SKY[..., skt.index("toi_sh")],
            "sog": SKY[..., skt.index("isog")], "ixg": SKY[..., skt.index("ixg_all")],
            "goals": SKY[..., skt.index("g")], "assists": SKY[..., skt.index("a")]}
    for k in model:
        y = ytgt[k][SKM]
        mu, bb = model[k][SKM], np.nan_to_num(basel[k][SKM], nan=np.nanmean(basel[k][SKM]))
        okk = np.isfinite(y)
        if k == "toi":
            sec["skater_toi"] = paired(np.abs(mu - y)[okk], np.abs(bb - y)[okk])
        else:
            dv = lambda m_: np.clip(m_, 1e-6, None) - y * np.log(np.clip(m_, 1e-6, None))
            sec[f"skater_{k}"] = paired(dv(mu)[okk], dv(bb)[okk])
    order = sorted(sec, key=lambda k: sec[k]["p"])
    for i, k in enumerate(order):
        sec[k]["p_holm"] = min(1.0, sec[k]["p"] * (len(order) - i))
        sec[k]["better"] = bool(sec[k]["diff"] < 0 and sec[k]["p_holm"] < 0.05)

    res = {"test": "NeurHL-G one-shot seal (PLAN_NeurHL4 SEAL)", "config": cfg_name,
           "freeze_commit": fz, "n": int(len(ev)),
           "ll": {"neurhl_g": float(lg.mean()), "elo": float(le.mean()),
                  "neurhl_h": float(lh[ok_h].mean())},
           "S1": s1, "S2": s2, "secondary": sec,
           "verdict": ("PASS" if s1["pass"] and s2.get("pass") else
                       "PARTIAL" if s1["pass"] else "NULL")}
    (NOUT / "preds").mkdir(exist_ok=True)
    ev.round(6).to_csv(NOUT / "preds" / "g_seal_games.csv", index=False)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "secondary"}, indent=1))
    print("secondary:", {k: round(v["diff"], 5) for k, v in sec.items()})

    from train.train_live_g import build
    build(cfg_name, 2026, "g2027_v2", purpose="score")
    p = CONFIGS / "live_models.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur["neurhl_g"] = "g2027_v2"
    cur["neurhl_g_v2_verdict"] = res["verdict"]
    p.write_text(json.dumps(cur, indent=1))


if __name__ == "__main__":
    main()
