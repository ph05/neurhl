"""Final production run: project 2026-27 (one-ahead) and 2027-28 (two-ahead, headline),
simulate 10,000 seasons each, write output/nhl_2027_28_projections.xlsx (+ headline CSV)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
N_SIMS = 10_000
LAST_SEASON = 2026  # 2025-26, last completed

FULL = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes", "CBJ": "Columbus Blue Jackets", "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers", "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils", "NSH": "Nashville Predators", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken", "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}


def fair_odds(p):
    if p <= 0:
        return np.nan, ""
    dec = 1.0 / p
    amer = f"+{round(100*(dec-1)):d}" if dec >= 2 else f"-{round(100/(dec-1)):d}"
    return round(dec, 2), amer


def summarize(sim, ratings_mean):
    teams = sim["teams"]
    pts = sim["pts"].astype(float)
    div_of = {t: dv for dv, ts_ in E.DIVISIONS_CURRENT.items() for t in ts_}
    conf_of = {t: c for c, dvs in E.CONFS.items() for dv in dvs
               for t in E.DIVISIONS_CURRENT[dv]}
    # presidents' trophy / bottom-3 via same tiebreak key as standings
    key = pts * 1e8 + sim["rw"] * 1e4 + sim["row"] + np.random.default_rng(1).random(pts.shape) * 0.5
    pres = np.zeros(len(teams))
    bot3 = np.zeros(len(teams))
    top_idx = key.argmax(axis=1)
    bot_idx = np.argsort(key, axis=1)[:, :3]
    for i in range(len(teams)):
        pres[i] = (top_idx == i).mean()
        bot3[i] = (bot_idx == i).any(axis=1).mean()
    rows = []
    for i, t in enumerate(teams):
        p = pts[:, i]
        playoff = sim["made_po"][:, i].mean()
        cup = sim["won_cup"][:, i].mean()
        po_dec, po_amer = fair_odds(playoff)
        cup_dec, cup_amer = fair_odds(cup)
        rows.append({
            "Team": FULL[t], "Abbr": t, "Conference": conf_of[t], "Division": div_of[t],
            "Rating": round(ratings_mean[t], 1),
            "xPts": p.mean(), "SD": p.std(ddof=1),
            "P5": np.percentile(p, 5), "P25": np.percentile(p, 25),
            "P50": np.percentile(p, 50), "P75": np.percentile(p, 75),
            "P95": np.percentile(p, 95),
            "Playoff%": playoff, "Division%": sim["won_div"][:, i].mean(),
            "Presidents%": pres[i], "Conference%": sim["won_conf"][:, i].mean(),
            "Cup%": cup, "Bottom3%": bot3[i],
            "P(>=100pts)": (p >= 100).mean(), "P(>=110pts)": (p >= 110).mean(),
            "FairOdds Playoff (dec)": po_dec, "FairOdds Cup (dec)": cup_dec,
            "FairOdds Cup (US)": cup_amer,
        })
    df = pd.DataFrame(rows).sort_values("xPts", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def league_aggregates(sim, season_label):
    pts = sim["pts"].astype(float)
    return {
        f"{season_label}: league mean points": round(float(pts.mean()), 2),
        f"{season_label}: SD of team points (mean across sims)":
            round(float(pts.std(axis=1, ddof=1).mean()), 2),
        f"{season_label}: expected # teams >=100 pts": round(float((pts >= 100).sum(1).mean()), 2),
        f"{season_label}: P(some team >=120 pts)": round(float((pts >= 120).any(1).mean()), 3),
        f"{season_label}: expected Presidents' Trophy points":
            round(float(pts.max(axis=1).mean()), 1),
        f"{season_label}: expected last-place points": round(float(pts.min(axis=1).mean()), 1),
    }


def main():
    params = json.loads((OUT / "params.json").read_text())
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=params["K"], H=params["H"], phi_s=params["phi_s"])
    beta = E.fit_xg_beta(end_r, ts, list(range(2008, LAST_SEASON + 1)))
    om = E.fit_outcome(preds, list(range(2006, LAST_SEASON + 1)))
    print(f"beta={beta:.0f}; outcome: p_ot@even={1/(1+np.exp(-om['b_pot'][0])):.3f}, "
          f"so_share={om['so_share']:.3f}")

    results = {}
    sims = {}
    for label, horizon, phi, sigma, seed in (
            ("2026_27", 1, params["phi1"], params["sigma1"], 101),
            ("2027_28", 2, params["phi2"], params["sigma2"], 202)):
        ratings = E.project_ratings(end_r, ts, beta, LAST_SEASON, w=params["w"], phi=phi)
        rng = np.random.default_rng(seed)
        sched = E.synthetic_schedule_84(E.DIVISIONS_CURRENT, rng)
        sim = E.simulate_season(ratings, sigma, sched, om, E.DIVISIONS_CURRENT,
                                N_SIMS, rng, playoffs=True)
        # invariants
        assert sim["made_po"].sum(1).std() == 0 and sim["made_po"].sum(1).mean() == 16
        assert sim["won_cup"].sum(1).mean() == 1.0
        assert abs(sim["pts"].mean() * 32 - 32 * 84 * 1.115) < 32 * 84 * 0.05
        results[label] = summarize(sim, ratings)
        sims[label] = sim
        print(f"{label}: simulated {N_SIMS} seasons; league mean {sim['pts'].mean():.1f} pts; "
              f"top team {results[label].iloc[0].Team} {results[label].iloc[0].xPts:.1f}")

    write_spreadsheet(results, sims, params, beta)


def write_spreadsheet(results, sims, params, beta):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    xlsx = OUT / "nhl_2027_28_projections.xlsx"
    bt = pd.read_csv(OUT / "backtest_seasons.csv")
    po = pd.read_csv(OUT / "backtest_playoff_calibration.csv")

    readme_lines = build_readme(params, beta, sims)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"NHL 2027-28 Projection Model": readme_lines}).to_excel(
            xw, sheet_name="README", index=False)
        results["2027_28"].to_excel(xw, sheet_name="Projections_2027_28", index=False)
        results["2026_27"].to_excel(xw, sheet_name="Projections_2026_27", index=False)
        dist_sheet(sims["2027_28"]).to_excel(xw, sheet_name="Sim_Distributions_2027_28",
                                             index=False)
        ratings_sheet(params, beta).to_excel(xw, sheet_name="Ratings", index=False)
        bt_out = backtest_sheet(bt, po, params)
        bt_out.to_excel(xw, sheet_name="Backtest", index=False)

        wb = xw.book
        pct_cols = {"Playoff%", "Division%", "Presidents%", "Conference%", "Cup%",
                    "Bottom3%", "P(>=100pts)", "P(>=110pts)"}
        num1_cols = {"xPts", "SD", "P5", "P25", "P50", "P75", "P95", "Rating"}
        hdr_fill = PatternFill("solid", fgColor="1F3864")
        hdr_font = Font(bold=True, color="FFFFFF")
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            headers = [c.value for c in ws[1]]
            for c in ws[1]:
                c.fill = hdr_fill
                c.font = hdr_font
                c.alignment = Alignment(vertical="center")
            for j, h in enumerate(headers, start=1):
                width = 14
                if h in ("Team", "NHL 2027-28 Projection Model", "metric"):
                    width = 100 if "README" in ws.title else 24
                ws.column_dimensions[get_column_letter(j)].width = width
                if h in pct_cols:
                    for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                        row[0].number_format = "0.0%"
                elif h in num1_cols:
                    for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                        row[0].number_format = "0.0"
            if ws.title == "README":
                for row in ws.iter_rows(min_row=2):
                    row[0].alignment = Alignment(wrap_text=False)
    results["2027_28"].to_csv(OUT / "projections_2027_28.csv", index=False)
    print(f"wrote {xlsx}")


def dist_sheet(sim):
    teams = sim["teams"]
    pts = sim["pts"].astype(float)
    rows = []
    for i, t in enumerate(teams):
        p = pts[:, i]
        row = {"Team": FULL[t]}
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            row[f"P{q}"] = round(float(np.percentile(p, q)), 1)
        row["P(<=75)"] = round(float((p <= 75).mean()), 3)
        row["P(>=100)"] = round(float((p >= 100).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("P50", ascending=False)


def ratings_sheet(params, beta):
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=params["K"], H=params["H"], phi_s=params["phi_s"])
    t26 = ts[ts.season_end == LAST_SEASON].set_index("team")
    rows = []
    for team, r_elo in sorted(end_r[LAST_SEASON].items(), key=lambda kv: -kv[1]):
        xg = t26.loc[team, "xg_pct_all"]
        r_xg = E.MEAN + beta * (xg - 0.5)
        blend = params["w"] * (r_elo - E.MEAN) + (1 - params["w"]) * (r_xg - E.MEAN)
        rows.append({
            "Team": FULL[team], "Abbr": team,
            "Elo end 2025-26": round(r_elo, 1),
            "xG% 2025-26 (all situations)": round(xg, 4),
            "xG-implied rating": round(r_xg, 1),
            "Blended deviation": round(blend, 1),
            "Projected mean 2026-27": round(E.MEAN + params["phi1"] * blend, 1),
            "Projected mean 2027-28": round(E.MEAN + params["phi2"] * blend, 1),
        })
    return pd.DataFrame(rows)


def backtest_sheet(bt, po, params):
    lines = []
    lines.append({"metric": "== Frozen parameters ==", "value": json.dumps(params)})
    for h in (1, 2):
        sub = bt[bt.horizon == h]
        lines.append({"metric": f"h{h} mean MAE/82 model (2018-2026)",
                      "value": round(sub.mae_model.mean(), 2)})
        lines.append({"metric": f"h{h} mean MAE/82 regressed-prior baseline",
                      "value": round(sub.mae_regressed_prior.mean(), 2)})
        lines.append({"metric": f"h{h} mean MAE/82 uniform baseline",
                      "value": round(sub.mae_uniform.mean(), 2)})
        pos = po[po.horizon == h]
        lines.append({"metric": f"h{h} playoff Brier (vs 0.25 constant)",
                      "value": round(((pos.p - pos.made) ** 2).mean(), 4)})
    head = pd.DataFrame(lines)
    per_season = bt.round(2)
    per_season.insert(0, "metric", "per-season detail")
    per_season = per_season.rename(columns={"value": "_"})
    merged = pd.concat([head, per_season], ignore_index=True)
    return merged


def build_readme(params, beta, sims):
    L = []
    a = L.append
    a("Aggregate expected returns of the 2027-28 NHL season - built 2026-08-12 from data through 2025-26.")
    a("")
    a("HEADLINE SHEET: Projections_2027_28. Each row aggregates 10,000 season simulations: expected points")
    a("(84-game schedule, new CBA), spread percentiles, and probabilities of playoffs/division/conference/Cup,")
    a("with fair (break-even, no-vig) odds implied by those probabilities.")
    a("")
    a("=== Method (discover -> build -> tune -> report) ===")
    a("1. DATA: all 27,166 NHL games 2005-06 through 2025-26 (results via Hockey-Reference season pages;")
    a("   attribution: Sports Reference LLC). Team-season expected goals 2007-08+ from MoneyPuck.com")
    a("   (free for non-commercial use with attribution). 84-game 2026-27+ schedule matrix per new CBA.")
    a("2. RATINGS: margin-of-victory Elo (K=6, home ice +35, ln(GD+1) multiplier with the FiveThirtyEight")
    a("   autocorrelation guard), run through playoffs; season carryover 0.70 toward the 1505 mean.")
    a("3. PROCESS BLEND: end-of-season rating = 0.6 x Elo + 0.4 x (all-situations xG share mapped to Elo,")
    a(f"   beta={beta:.0f} Elo per unit of xG share). EDA: xG% repeats year-over-year (r=0.61) better than")
    a("   points% (r=0.52); shooting/save % essentially don't repeat (r~0.006) - PDO is luck (Tulsky).")
    a("4. TWO-SEASON-AHEAD: 2027-28 strength = 1505 + 0.75 x blended deviation (shrinkage estimated")
    a("   directly on two-year gaps; quality has a persistent component, so phi2 >> phi1^2), plus")
    a("   sigma=40 Elo of drawn true-strength noise per simulation (covers two off-seasons of roster churn,")
    a("   aging, goaltending voodoo). 2026-27 uses phi1=0.85, sigma=30.")
    a("5. GAME MODEL: P(reach OT)~25% at even strength, decreasing with mismatch; regulation winner from")
    a("   full-slope logistic; OT/SO winner from a heavily compressed slope (better team wins only ~55%")
    a("   of OT games vs ~63% of regulation games); SO share ~34% of past-regulation games; W=2, OTL=1.")
    a("6. SIMULATION: 10,000 x 1,344-game seasons on a CBA-correct synthetic schedule (28 division /")
    a("   24 conference / 32 inter-conference), standings tiebreakers (RW then ROW), real playoff seeding")
    a("   (top-3 + 2 wild cards, fixed bracket), best-of-7 series with 2-2-1-1-1 home ice via a playoff-fit")
    a("   game model.")
    a("")
    a("=== Backtest (walk-forward, params frozen on <=2016-17, validated once on 2017-18..2025-26) ===")
    a("One-ahead: MAE 10.54 pts/82 vs 11.03 regressed-prior baseline, 13.24 uniform (best in 7/9 seasons).")
    a("Two-ahead (this task): MAE 11.57 vs 11.90 regressed, 13.24 uniform, 14.02 raw-prior.")
    a("Playoff-odds Brier 0.195/0.208 (h1/h2) vs 0.25 constant; binned calibration within noise.")
    a("Interpretation (Silver): two years out, the edge over 'regress everything heavily to the mean' is")
    a("real but thin; the value of the model is calibrated DISTRIBUTIONS, not point estimates.")
    a("Trajectory/momentum term tested on train: exactly null (gamma=0 optimal). Dropped.")
    a("")
    a("=== League aggregates (from simulations) ===")
    for k, v in {**league_aggregates(sims["2027_28"], "2027-28"),
                 **league_aggregates(sims["2026_27"], "2026-27")}.items():
        a(f"{k}: {v}")
    a("")
    a("=== Caveats (read before using) ===")
    a("- The model knows nothing after June 2026: 2026 and 2027 free agency, trades, drafts, injuries,")
    a("  goalie changes are all in the error term (sigma), not the point estimates.")
    a("- Real 2027-28 schedule is unpublished; a synthetic CBA-matrix schedule is used (order/rest ignored).")
    a("- 84-game era begins 2026-27; per-game modeling scaled up is assumed valid.")
    a("- Fair odds are break-even prices, not predictions of market prices; futures markets embed roster")
    a("  news this model cannot know. Differences are not automatically edges.")
    a("- 32 teams assumed (no expansion approved for 2027-28 as of Aug 2026).")
    a("- Playoff seeding uses points/RW keys; obscure tiebreakers (H2H) approximated by random draw.")
    a("")
    a("=== Teachers honored ===")
    a("Nate Silver (Elo, regression to the mean, calibration over confidence, uncertainty as information);")
    a("Eric Tulsky (shot/xG metrics out-predict goals in small samples; PDO luck; skepticism of stories);")
    a("Tom Tango (Marcel: the regressed-prior baseline every model must beat); Dom Luszczyszyn (full-")
    a("distribution team projections); Micah Blake McCurdy (score/venue adjustment); Evolving-Hockey /")
    a("MoneyPuck (public xG models). Errors are mine, not theirs.")
    return L


if __name__ == "__main__":
    main()
