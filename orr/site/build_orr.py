"""Build the ORR projection page: NeurHL's site layout (docs/index.html) filled
with this project's forecasts. ORR (Odds, Ratings & Rosters) is the public
name of the ORR model.

    python3 -m orr.site.build_orr   ->  docs/orr/index.html (GitHub Pages)

Reads only committed outputs: the 2026-27 freeze, the latest daily forecast,
the scorecard and the backtest files behind each accuracy row. HOWE and the
Elo reference come from the same files NeurHL's site uses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from orr import config as C

F = C.OUT / "freeze_2027"
BT = C.OUT / "backtest"
SITE = Path(__file__).resolve().parent
OUT = C.ROOT / "docs" / "orr" / "index.html"     # GitHub Pages: ph05.github.io/neurhl/orr/
HASHED = ["teams_2027.csv", "games_2027.csv", "skaters_2027.csv", "goalies_2027.csv"]


def _r(x, n=1):
    return None if x is None or pd.isna(x) else round(float(x), n)


def _rd(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def start_times() -> dict:
    out = {}
    for f in sorted((C.ROOT / "data" / "raw").glob("nhl_sched_*_20262027.json")):
        for g in json.loads(f.read_text()).get("games", []):
            if g.get("gameType") == 2:
                out[int(g["id"])] = g.get("startTimeUTC")
    return out


def tonight(pre: pd.DataFrame, elo: pd.Series) -> dict:
    days = sorted(d for d in (C.OUT / "live").iterdir() if d.is_dir()) if (C.OUT / "live").exists() else []
    if not days:
        return {}
    d = days[-1]
    g = pd.read_csv(d / f"games_{d.name}.csv")
    pl = pd.read_csv(d / f"players_{d.name}.csv")
    run = _rd(d / f"run_{d.name}.json") or {}
    st = start_times()
    from orr import gamemodel as GM
    sf = GM.schedule_features(pd.read_csv(F / "schedule_2027.csv", parse_dates=["date"]), C.TARGET_SEASON)
    sf = sf.set_index("game_id")
    games = []
    for r in g.itertuples():
        q = pl[pl.game_id == r.game_id]
        sog = q.groupby("team").exp_sog.sum()
        top = q.sort_values("p_goal", ascending=False).head(4)
        games.append({"id": int(r.game_id), "start": st.get(int(r.game_id)), "home": r.home, "away": r.away,
                      "kind": run.get("model", "daily"), "g": _r(r.p_home_win, 4), "pre": _r(pre.get(r.game_id), 4),
                      "rest": [_r(sf.rest_h.get(r.game_id), 0), _r(sf.rest_a.get(r.game_id), 0)],
                      "km": [_r(sf.km_h.get(r.game_id), 0), _r(sf.km_a.get(r.game_id), 0)],
                      "elo": _r(elo.get(r.game_id), 4), "ot": _r(r.p_ot, 4),
                      "score": [_r(r.exp_goals_home, 2), _r(r.exp_goals_away, 2)],
                      "sog": [_r(sog.get(r.home), 1), _r(sog.get(r.away), 1)],
                      "top": [[x.name, x.team, _r(x.exp_sog / max(x.p_dress, 1e-9), 1), _r(x.p_goal, 3), _r(x.p_point, 3),
                               _r(getattr(x, "p_sog3", None), 3)] for x in top.itertuples()]})
    games.sort(key=lambda x: x["start"] or "")
    through = run.get("results_through")
    return {"date": d.name, "through": through[:10] if through else None, "games": games}


def _ev(c, w, n, diff, se=None, ci=None, fmt=4):
    if ci is None and se is not None:
        ci = [diff - 1.96 * se, diff + 1.96 * se]
    p = "" if se in (None, 0) else ("<0.0001" if 2 * norm.sf(abs(diff / se)) < 1e-4
                                    else f"{2 * norm.sf(abs(diff / se)):.4f}")
    return {"c": c, "w": w, "n": n, "d": f"{diff:+.{fmt}f}",
            "ci": "" if ci is None else f"{ci[0]:+.{fmt}f} to {ci[1]:+.{fmt}f}", "p": p}


def evidence() -> list:
    rows = []
    ib = _rd(BT / "inseason_bt.json")
    if ib:
        a = [r for r in ib["modes"]["known_starters"] if r["season"] == "all"][0]
        for key, name in (("d_shipped_vs_neurhl_g", "NeurHL-G stack"), ("d_shipped_vs_neurhl_elo", "NeurHL's Elo")):
            x = a[key]
            rows.append(_ev(f"In-season ORR (goals and shots) vs {name}, log loss per game", "2018-19 to 2023-24 (excl. 2020-21)",
                            a["n"], x["diff"], x.get("se"), x["ci95"]))
        if "goals_only_live" in ib["modes"]:
            a = [r for r in ib["modes"]["goals_only_live"] if r["season"] == "all"][0]
            x = a["d_shipped_vs_neurhl_g"]
            rows.append(_ev("In-season ORR (goals only, as live until results carry shots) vs NeurHL-G stack",
                            "2018-19 to 2023-24 (excl. 2020-21)", a["n"], x["diff"], x.get("se"), x["ci95"]))
    xr = _rd(BT / "inseason_bt_x3r.json")
    if xr:
        d = xr["variants"]["goals_shots"]["pooled"]["d"]
        rows.append(_ev("ORR 1.1 (lineups) vs ORR 1.0, log loss per game, seasons no experiment used",
                        "2010-11 and 2020-21", xr["n"], d["diff"], d.get("se"), d["ci95"]))
    pu = _rd(BT / "player_update_bt.json")
    if pu:
        t = pu["test"]
        rows.append({"c": "In-season skater rates vs preseason rates, rest-of-season points MAE (1.2)",
                     "w": "2021-22 to 2022-23", "n": t["n"], "d": f"{t['d']['diff']:+.3f}",
                     "ci": f"{t['d']['ci95'][0]:+.3f} to {t['d']['ci95'][1]:+.3f}", "p": "<0.0001"})
    sb = _rd(BT / "standings_inseason_bt.json")
    if sb:
        t = sb["test"]
        rows.append({"c": "In-season standings vs preseason forecast, final points CRPS (1.4)", "w": "2021-22 to 2022-23",
                     "n": t["filter_kbest"]["n"], "d": f"{t['filter_kbest']['crps'] - t['preseason']['crps']:+.2f}",
                     "ci": "", "p": ""})
    ss = _rd(BT / "start_share_bt.json")
    if ss:
        t = ss["test"]
        rows.append({"c": "In-season goalie start shares vs preseason, rest-of-season starts MAE (1.4)",
                     "w": "2021-22 to 2022-23", "n": t["n"], "d": f"{t['diff']:+.2f}",
                     "ci": f"{t['ci95'][0]:+.2f} to {t['ci95'][1]:+.2f}", "p": ""})
    ri = _rd(BT / "ros_interval_bt.json")
    if ri:
        t = ri["test"]
        rows.append({"c": "Rest-of-season skater intervals, 80% interval score vs 1.6 (1.7)", "w": "2021-22 to 2022-23",
                     "n": t["n"], "d": f"{t['diff']:+.2f}", "ci": f"{t['ci95'][0]:+.2f} to {t['ci95'][1]:+.2f}",
                     "p": f"coverage {t['cover_best']:.2f} vs {t['cover_1_6']:.2f}"})
    gb = _rd(BT / "gamefile_bt.json")
    if gb:
        v = gb["variants"]["mkt+td (shipped)"]["by_season"]
        a = [r for r in v if r["season"] == "all"][0]
        x = a["d_gamefile_vs_elo_frozen"]
        rows.append(_ev("Preseason ORR game file vs preseason Elo, log loss per game", "2018-19 to 2023-24 (5 seasons)",
                        a["n"], x["diff"], x.get("se"), x["ci95"]))
    tb = _rd(BT / "teams_bt.json")
    if tb:
        b = tb["B_market_seasons"]
        ship = b["mkt+td@clean"]
        nj = b["neurhl_judge_2019_2024"]
        rows.append(_ev("Standings vs NeurHL engine layer, points MAE", "2018-19 to 2023-24 (judge window)", 158,
                        ship["judge_raw_mae"] - nj["engine_layer_mae"], fmt=2))
        rows.append(_ev("Standings vs NeurHL engine layer, points CRPS", "2018-19 to 2023-24 (judge window)", 158,
                        ship["judge_raw_crps"] - nj["engine_layer_crps"], fmt=2))
    pb = _rd(BT / "players_bt.json")
    if pb:
        a = pb["pooled_test"]["strict"]["A"]
        rows.append({"c": "Skaters vs NeurHL, points MAE, players with 40+ GP", "w": "2021-22 to 2025-26", "n": a["n"],
                     "d": f"{a['mae'] - 9.543:+.2f}", "ci": "", "p": ""})
    gl = _rd(BT / "goalies_bt.json")
    if gl:
        t = gl["pooled_test"]
        rows.append({"c": "Goalies vs league average, GSAx/60 MAE (1,000+ shots)", "w": "2021-22 to 2025-26",
                     "n": t["n"], "d": f"{t['hattrick']['gsax60_mae'] - t['naive']['gsax60_mae']:+.3f}", "ci": "", "p": ""})
    return rows


def main():
    howe = pd.read_csv(C.ROOT / "output" / "projections_2026_27_howe.csv").set_index("Abbr")
    t = pd.read_csv(F / "teams_2027.csv")
    sk = pd.read_csv(F / "skaters_2027.csv")
    gl = pd.read_csv(F / "goalies_2027.csv")
    g = pd.read_csv(F / "games_2027.csv")
    n13 = pd.read_csv(C.ROOT / "neurhl/output/neurhl_1_3/season/games_2027.csv").set_index("game_id")
    elo = n13.p_home_win_elo
    st = _rd(F / "state_2027.json")

    teams = [{"ab": r.team, "name": howe.Team.get(r.team, r.team), "div": howe.Division.get(r.team, r.div),
              "conf": r.conf, "pts": _r(r.points), "p10": int(round(r.points_p10)), "p90": int(round(r.points_p90)),
              "wins": _r(r.w), "po": _r(r.playoff_pct), "cup": _r(r.cup_pct), "howe": _r(howe.xPts.get(r.team))}
             for r in t.itertuples()]
    sf = sk.groupby("team").sog.sum()
    ppg = sk.groupby("team").ppg.sum()
    sa = gl.groupby("team").sa.sum()
    gag = gl.groupby("team").ga.sum()
    tx = []
    for r in t.itertuples():
        tx.append({"ab": r.team, "gp": int(r.gp), "w": _r(r.w), "l": _r(r.l), "otl": _r(r.otl), "rw": _r(r.rw),
                   "pts": _r(r.points), "p10": int(round(r.points_p10)), "p50": int(round(r.points_p50)),
                   "p90": int(round(r.points_p90)), "gf": _r(r.gf), "ga": _r(r.ga),
                   "sf": _r(sf.get(r.team)), "sa": _r(sa.get(r.team)), "ppg": _r(ppg.get(r.team)),
                   "sh": _r(100 * r.gf / sf.get(r.team), 2) if sf.get(r.team) else None,
                   "sv": _r(100 * (1 - gag.get(r.team) / sa.get(r.team)), 2) if sa.get(r.team) else None,
                   "po": _r(r.playoff_pct), "div_p": _r(r.division_pct), "pres": _r(r.presidents_pct),
                   "r2": _r(r.round2_pct), "cf": _r(r.conf_final_pct), "fin": _r(r.cup_final_pct), "cup": _r(r.cup_pct)})

    rp = C.OUT / "live" / "results_2027.csv"
    res = pd.read_csv(rp).set_index("game_id") if rp.exists() else pd.DataFrame()
    games = []
    for r in g.itertuples():
        row = [int(r.game_id), str(r.date), r.away, r.home, _r(r.p_home_win, 4), _r(elo.get(r.game_id), 4), _r(r.p_ot, 4)]
        if len(res) and r.game_id in res.index:
            x = res.loc[r.game_id]
            row.append([int(x.away_g), int(x.home_g), str(x.last_period)])
        games.append(row)

    s = sk[sk.gp >= 1]
    fo = s.fow + s.fol
    # ORR 1.3: season to date from the box scores, and rest of season from the updated rates
    td = None
    bp = C.OUT / "live" / f"boxes_{C.TARGET_SEASON}.csv"
    if bp.exists():
        from orr import player_update as PU
        b = pd.read_csv(bp)
        b = b[(b.pos != "G") & (b.toi > 0)]
        if len(b):
            tot_td = PU.totals(b).set_index("player_id")
            upd = PU.live_rates(s).set_index("player_id")
            team_gp = pd.read_csv(rp) if rp.exists() else None
            played = (pd.concat([team_gp.home, team_gp.away]).value_counts() if team_gp is not None
                      else pd.Series(dtype=float))
            left = 1 - s.team.map(played).fillna(0).to_numpy() / C.GAMES_PER_TEAM[C.TARGET_SEASON]
            ros = (upd.g_pg + upd.a_pg).reindex(s.player_id).to_numpy() * s.gp.to_numpy() * left
            td = {"gp": tot_td.n.reindex(s.player_id).fillna(0).to_numpy(),
                  "p": (tot_td.g + tot_td.a).reindex(s.player_id).fillna(0).to_numpy(), "ros": ros}
    skx = [[r.name, r.team, r.pos, _r(r.gp), _r(r.toi / max(r.gp, 1e-9)), _r(r.g), _r(r.a), _r(r.p),
            _r(r.p_p10), _r(r.p_p90), _r(r.sog), _r(r.ixg), _r(100 * r.g / r.sog) if r.sog > 0 else None,
            _r(r.pim), _r(r.hits), _r(r.blk), _r(100 * r.fow / f) if f > 50 else None]
           + ([_r(td["gp"][i], 0), _r(td["p"][i], 0), _r(td["ros"][i])] if td is not None else [])
           for i, (r, f) in enumerate(zip(s.itertuples(), fo))]
    # ORR 1.6: in-season goalie columns (box scores, updated talent and start share)
    gtd, tal0, tal1, share1 = None, None, None, None
    if bp.exists():
        bx = pd.read_csv(bp)
        bx = bx[(bx.pos == "G") & bx.shots_against.notna()]
        if len(bx):
            from orr import structural as ST
            from orr.inseason import update_start_shares_live
            gtd = bx.groupby("player_id").agg(gp=("game_id", "nunique"), sa=("shots_against", "sum"),
                                             ga=("goals_against", "sum"))
            day = pd.Timestamp.now("UTC").tz_convert(None).normalize() + pd.Timedelta(days=1)
            tal0, tal1 = ST.goalie_talent_2027(), ST.goalie_talent_live(day)
            share1 = update_start_shares_live(gl[["player_id", "team", "start_share"]].assign(p_present=1.0), day) \
                .set_index("player_id").start_share
    gx = []
    for r in gl[gl.starts >= 1].itertuples():
        row = [r.name, r.team, _r(r.starts), _r(r.wins), _r(r.sa), _r(r.ga), _r(r.sv_pct, 4),
               _r(r.ga / r.gp, 2) if r.gp > 0 else None, _r(r.gsax)]
        if gtd is not None:
            gq = gtd.reindex([r.player_id]).iloc[0]
            row += [_r(gq.gp, 0), _r(gq.sa, 0), _r(1 - gq.ga / gq.sa, 4) if gq.sa > 0 else None,
                    _r(100 * tal0.get(r.player_id, np.nan), 2), _r(100 * tal1.get(r.player_id, np.nan), 2),
                    _r(r.start_share, 3), _r(share1.get(r.player_id), 3)]
        gx.append(row)
    # ORR 1.6: remaining strength of schedule and the latest daily files
    sos, daily_files = {}, []
    lf = sorted((C.OUT / "live").glob("*/standings_*.csv"))
    if lf:
        ls = pd.read_csv(lf[-1])
        if "sos_remaining" in ls:
            sos = {r.team: _r(100 * r.sos_remaining, 1) for r in ls.itertuples()}
        daily_files = [str(f.relative_to(C.ROOT)) for f in sorted(lf[-1].parent.glob("*.csv"))]
    for row in tx:
        row["sos"] = sos.get(row["ab"])
    teams_live, live_date = {}, None
    if lf:
        live_date = lf[-1].parent.name
        for r in ls.itertuples():
            teams_live[r.team] = {"pts": _r(r.points), "p10": int(round(r.points_p10)), "p50": int(round(r.points_p50)),
                                  "p90": int(round(r.points_p90)), "w": _r(r.w), "l": _r(r.l), "otl": _r(r.otl),
                                  "rw": _r(r.rw), "gf": _r(r.gf), "ga": _r(r.ga), "po": _r(r.playoff_pct),
                                  "div_p": _r(r.division_pct), "pres": _r(r.presidents_pct), "r2": _r(r.round2_pct),
                                  "cf": _r(r.conf_final_pct), "fin": _r(r.cup_final_pct), "cup": _r(r.cup_pct)}
    daily_models = {}
    for f in sorted((C.OUT / "live").glob("*/run_*.json")):
        daily_models[f.parent.name] = (_rd(f) or {}).get("model")

    card = _rd(C.OUT / "scorecard_2027.json") or {}
    names = {"orr_preseason": "ORR preseason", "orr_inseason": "ORR daily",
             "neurhl_1.0": "NeurHL 1.0", "neurhl_1.1": "NeurHL 1.1", "neurhl_1.2": "NeurHL 1.2", "neurhl_1.3": "NeurHL 1.3"}
    live_rows = []
    for k, v in (card.get("games") or {}).items():
        if k not in names:
            continue
        ap = v.get("after_publication", v) if k != "orr_inseason" else v
        if ap and ap.get("n"):
            live_rows.append({"name": names[k], "n": ap["n"], "log_loss": ap["log_loss"]})
    from orr.inseason import DEFAULT_MODEL, MODELS
    # ORR 1.4: playoff odds by date (preseason file, then every committed daily run)
    hist_dates, hist = ["2026-09-29"], {r.team: [_r(r.playoff_pct)] for r in t.itertuples()}
    for f in sorted((C.OUT / "live").glob("*/standings_*.csv")):
        sd = pd.read_csv(f).set_index("team").playoff_pct
        hist_dates.append(f.parent.name)
        for tm in hist:
            hist[tm].append(_r(sd.get(tm)))
    odds_history = {"dates": hist_dates, "labels": ["Preseason"] + hist_dates[1:], "teams": hist}
    # ORR 1.5: biggest movers between the last two committed forecasts, with the games between them
    movers = []
    if len(hist_dates) >= 2:
        d0, d1 = hist_dates[-2], hist_dates[-1]
        rr = pd.read_csv(rp) if rp.exists() else pd.DataFrame(columns=["date", "home", "away", "home_g", "away_g", "last_period"])
        between = rr[(rr.date >= d0) & (rr.date < d1)]
        for tm, v in hist.items():
            if v[-1] is None or v[-2] is None:
                continue
            gs = []
            for x in between.itertuples():
                if tm in (x.home, x.away):
                    us, them = (x.home_g, x.away_g) if tm == x.home else (x.away_g, x.home_g)
                    opp = x.away if tm == x.home else "@" + x.home
                    res_ = "W" if us > them else ("OTL" if x.last_period != "REG" else "L")
                    gs.append(f"{res_} {us}-{them} {opp if tm == x.home else opp}")
            movers.append({"team": tm, "from": v[-2], "to": v[-1], "delta": round(v[-1] - v[-2], 1), "games": "; ".join(gs)})
        movers.sort(key=lambda m: -abs(m["delta"]))
        movers = {"from": "Preseason" if d0 == hist_dates[0] else d0, "to": d1, "rows": movers[:8]}
    data = {"meta": {"release": f"{MODELS[DEFAULT_MODEL]['version']} in-season (preseason file: ORR 1.0)", "cutoff": "2026-09-29 17:00 ET", "draws": 400, "sims": st.get("sims", 40000),
                     "tests": "7/7"},
            "live": {"as_of": card.get("through", "")[:10], "games_played": card.get("games_played", 0), "rows": live_rows,
                     "running": card.get("running", {}), "reliability": card.get("reliability", [])},
            "movers": movers,
            "teams": teams, "teams_x": tx, "tonight": tonight(g.set_index("game_id").p_home_win, elo),
            "games": games, "skaters_x": skx, "has_td": td is not None, "odds_history": odds_history, "goalies": gx, "evidence": evidence(),
            "has_gtd": gtd is not None, "daily_files": daily_files, "teams_live": teams_live,
            "live_date": live_date, "daily_models": daily_models,
            "sha256": {f"orr/output/freeze_2027/{f}": hashlib.sha256((F / f).read_bytes()).hexdigest() for f in HASHED}}
    blob = json.dumps(data, separators=(",", ":"), default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    html = (SITE / "orr_template.html").read_text().replace("__ORR_DATA__", blob.replace("</", "<\\/"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"-> {OUT} ({OUT.stat().st_size / 1024:.0f} KB): {len(teams)} teams, {len(games)} games, "
          f"{len(skx)} skaters, {len(gx)} goalies, {len(data['evidence'])} accuracy rows")


if __name__ == "__main__":
    main()
