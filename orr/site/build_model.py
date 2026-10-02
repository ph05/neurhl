"""The ORR model card (ORR 2.0): docs/orr/model.html on GitHub Pages.

    python3 -m orr.site.build_model

The page is built from the code's own records, so it cannot drift from what
runs. Those records are:
  * the MODELS table in orr/inseason.py, for each switch, the version that
    introduced it and whether the default model uses it;
  * the parameter files in orr/output/params/, which hold the
    pre-declared adoption decisions;
  * orr/output/evaluation_2027.json, the preregistered evaluation;
  * orr/output/reproduce/, the reproducibility checks.
"""
from __future__ import annotations

import html
import json

from orr import config as C
from orr.inseason import DEFAULT_MODEL, MODELS
from orr.site.build_changelog import PAGE

OUT = C.ROOT / "docs" / "orr" / "model.html"
REPO = "https://github.com/ph05/neurhl/blob/claude/gallant-maxwell-rbw7u6/"


def _p(name: str) -> dict:
    f = C.PARAMS / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


# switch -> (what it does, held-out evidence, evidence file, live-state rule)
SWITCHES = {
    "past_starters": ("Past games use the goalie who actually started.", "Part of X1.", "orr/RESULTS.md", None),
    "lineups": ("Team ratings are adjusted for who dresses and who starts in goal (X1).",
                "Log loss −0.00130 (CI −0.00255 to −0.00006) on 2019-24 gate games; −0.00232 (CI −0.00438 to −0.00019) on two unused seasons.",
                "orr/output/backtest/inseason_bt_x3r.json", None),
    "player_update": ("Skater goal, assist and shot rates are updated with his games so far (Gamma-Poisson, n0 = 40/40/20).",
                      "Rest-of-season points MAE 4.07 vs 4.48 (CI −0.49 to −0.34) on 2021-23.",
                      "orr/output/backtest/player_update_bt.json", None),
    "box_first": ("Past lineups and starters come from box scores first, then NeurHL's pregame files.",
                  "Matches the X1 backtest's inputs.", "orr/releases/v1.3.md", None),
    "goalie_update": ("Goalie talent is updated with this season's saves above average.",
                      "The backtests' own rule, applied live.", "orr/releases/v1.3.md", None),
    "player_calibration": ("Platt correction of per-game P(goal) and P(point).",
                           "Made held-out log loss worse for both; not adopted.",
                           "orr/output/backtest/player_prob_bt.json",
                           lambda: bool(_p("player_prob").get("adopted"))),
    "standings_drift": ("Rest-of-season rating drift in the standings simulation (k = 0.5).",
                        "Final-points CRPS 3.84 vs 4.99 for the preseason forecast on 2021-23.",
                        "orr/output/backtest/standings_inseason_bt.json", None),
    "start_share_update": ("Goalie start shares are updated with starts so far (Dirichlet, α = 10).",
                           "Rest-of-season starts MAE 5.29 vs 6.13 (CI −1.12 to −0.56) on 2021-23.",
                           "orr/output/backtest/start_share_bt.json", None),
    "absence": ("Long-term absences lower a team's rest-of-season strength.",
                "+0.0008 CRPS (CI ±0.01): no measurable effect; ships off.",
                "orr/output/backtest/standings_inseason_bt.json",
                lambda: bool(_p("standings_inseason").get("absence"))),
    "standings_sharp": ("Multiplier on in-season rating uncertainty in the standings simulation.",
                        "Tuned to 1.0, the existing setting: no change.",
                        "orr/output/backtest/standings_sharp_bt.json",
                        lambda: float(_p("standings_inseason").get("sd_mult", 1.0)) != 1.0),
    "sog_dist": ("Shots on goal follow a negative binomial (r = 16.6) rather than a Poisson.",
                 "Held-out log loss 0.48632 vs 0.48653.", "orr/output/backtest/sog_dist_bt.json",
                 lambda: bool(_p("sog_dist").get("adopted"))),
    "sos": ("Remaining strength of schedule in the daily standings.", "Descriptive.", "orr/releases/v1.6.md", None),
    "ros_file": ("Daily rest-of-season skater file with 80% intervals.", "Descriptive file.", "orr/releases/v1.6.md", None),
    "ros_interval": ("Rest-of-season intervals simulate games played, rate variance (v = 6) and injury spells.",
                     "Interval score 25.07 vs 25.53 (CI −0.73 to −0.18); coverage 0.80 on 2021-23.",
                     "orr/output/backtest/ros_interval2_bt.json",
                     lambda: bool(_p("ros_interval").get("adopted"))),
    "goalie_ros": ("Daily rest-of-season goalie file: starts and save % with intervals.", "Descriptive file.",
                   "orr/releases/v1.8.md", None),
    "forecast_diff": ("Each game's change since its previous forecast, and why.", "Descriptive.", "orr/releases/v1.8.md", None),
    "clinch": ("Clinch and elimination flags and playoff magic numbers.", "Exact by enumeration of results.",
               "orr/releases/v1.9.md", None),
}


def _vkey(v: str) -> list[int]:
    return [int(x) for x in v.split(".")]


def first_version(sw: str) -> str:
    for v in sorted(MODELS, key=_vkey):
        if MODELS[v].get(sw):
            return v
    return ""


def switch_rows() -> str:
    cfg = MODELS[DEFAULT_MODEL]
    rows = []
    for sw, (what, ev, f, rule) in SWITCHES.items():
        on = bool(cfg.get(sw))
        live = on and (rule() if rule else True)
        state = "live" if live else ("off (by its test)" if on else "off")
        rows.append(f"<tr><td><code>{sw}</code></td><td>{first_version(sw)}</td><td>{html.escape(what)}</td>"
                    f"<td>{html.escape(ev)} <a href=\"{REPO}{f}\">evidence</a></td><td><b>{state}</b></td></tr>")
    return ("<table><thead><tr><th>Switch</th><th>Since</th><th>What it does</th><th>Held-out evidence</th>"
            "<th>State</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def evaluation() -> str:
    f = C.OUT / "evaluation_2027.json"
    if not f.exists():
        return "<p>Not yet run.</p>"
    ev = json.loads(f.read_text())
    rows = []
    for k, v in ev.items():
        if not (k[:1] == "P" and k[1:2].isdigit()):
            continue
        d = v.get("diff")
        rows.append(f"<tr><td>{k.split('_')[0]}</td><td>{html.escape(k.split('_', 1)[1].replace('_', ' '))}</td>"
                    f"<td>{v.get('status', '')}</td><td>{v.get('n', '')}</td><td>{'' if d is None else f'{d:+.4f}'}</td>"
                    f"<td>{html.escape(str(v.get('verdict', v.get('reason', ''))))}</td></tr>")
    return (f"<p>Run {html.escape(ev.get('evaluated_utc', '')[:16])} UTC on {ev.get('games_played', 0)} games. "
            "Verdicts are read only after the last regular-season game.</p><table><thead><tr><th></th><th>Comparison</th>"
            "<th>Status</th><th>N</th><th>ORR − NeurHL</th><th>Verdict / reason</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def reproducibility() -> str:
    rows = []
    for f in sorted((C.OUT / "reproduce").glob("reproduce_*.json"), reverse=True):
        r = json.loads(f.read_text())
        nl = sum(x["matched"] for x in r.get("lineups", []))
        rows.append(f"<tr><td>{r['date']}</td><td>ORR {html.escape(r.get('model', ''))}</td><td><code>{r['code']}</code></td>"
                    f"<td>{'yes' if r['results']['matched'] else 'no'}</td><td>{nl}/{len(r.get('lineups', []))}</td>"
                    f"<td><b>{html.escape(r['verdict'])}</b></td></tr>")
    if not rows:
        return "<p>No checks yet.</p>"
    return ("<table><thead><tr><th>Forecast</th><th>Model</th><th>Code</th><th>Results file matched</th>"
            "<th>Lineup files matched</th><th>Re-run</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def body() -> str:
    m = MODELS[DEFAULT_MODEL]["version"]
    return f"""
<h1>ORR model card</h1>
<p>The current model is <b>{m}</b>. ORR (Odds, Ratings &amp; Rosters) forecasts the 2026-27 NHL season: game probabilities, standings and playoff odds, and skater and goalie projections. It is a public, reproducible alternative to NeurHL. This page is generated from the code's own configuration, parameter files and evaluation records.</p>

<h2>What it is for</h2>
<ul>
<li>Pregame win probabilities, expected goals and lineup-aware player lines for each game day.</li>
<li>Season standings, playoff and Cup odds, clinch flags and magic numbers, and rest-of-season player and goalie intervals.</li>
<li>Comparison with NeurHL, on terms fixed in advance.</li>
</ul>
<p>It is not for betting. It does not model trades, coaching changes or injuries that are not yet visible in lineups. Its playoff-round probabilities have never been tested against held-out playoffs.</p>

<h2>Layers</h2>
<ol>
<li><b>Preseason file (ORR 1.0), frozen.</b> This layer is a market-anchored team rating with a roster-based prior. It gives one scoring model for every game (home ice, rest, travel and the starting goalie) and simulates the season. Player projections conserve team ice time. Inputs were dated before 2026-09-29 17:00 ET. The file is in <code>orr/output/freeze_2027/</code>.</li>
<li><b>Daily in-season update.</b> A Kalman-style filter updates each team's offence and defence from results and shots on goal. Ratings are adjusted for dressed lineups and starting goalies (X1). Goalie talent and start shares are updated with the season so far.</li>
<li><b>Players.</b> Each skater's per-game rates are posterior means given his games so far. Player lines give P(goal), P(point) and P(k+ shots), conditional on dressing and weighted by the probability of dressing.</li>
<li><b>Season simulation.</b> 20,000 seasons are simulated with completed games fixed. It includes rest-of-season rating drift and the full tiebreakers and playoff format, and produces standings, odds, clinch flags and remaining strength of schedule.</li>
<li><b>Rest-of-season files.</b> Skater and goalie intervals come from a Monte Carlo over games played, rate uncertainty and injury spells.</li>
</ol>

<h2>Switches in {m}</h2>
<p>Each feature can be switched off, so every earlier version stays reproducible with <code>python3 -m orr.inseason --model 1.x</code>. Tuning used seasons up to 2016-17 only. Each test ran once, on 2021-22 and 2022-23, under an adoption rule declared beforehand. A feature that failed its test ships switched off rather than being deleted.</p>
{switch_rows()}

<h2>Preregistered evaluation against NeurHL</h2>
<p>The protocol is <a href="{REPO}orr/EVALUATION_2027.md">orr/EVALUATION_2027.md</a> and the code is <a href="{REPO}orr/evaluate_2027.py">orr/evaluate_2027.py</a>. They were fixed on 2026-10-02 with the SHA-256 of every forecast file. There are four primary comparisons, each with a paired-bootstrap 95% CI. A comparison is decided only when its CI excludes zero.</p>
{evaluation()}

<h2>Reproducibility</h2>
<p><code>python3 -m orr.reproduce --date D</code> re-runs a published daily forecast. It checks out the recorded code commit in a temporary worktree and rebuilds the exact inputs: the results file and every lineup file, each matched by SHA-256. It then re-runs the day with the recorded seed and compares every output file.</p>
{reproducibility()}

<h2>Known limitations</h2>
<ul>
<li>Shots on goal enter the live update only where the NHL API is reachable, as in the daily GitHub Action. Without it the filter runs on goals alone, the variant that did not pass its test.</li>
<li>The preseason file was published at 2026-10-01 00:37 UTC, after seven of the first eight games had started, so it is scored only on later games.</li>
<li>Rest-of-season player intervals reach 80% coverage on held-out seasons, but goalie intervals have no held-out test.</li>
<li>Playoff-round probabilities have never been tested against held-out playoffs.</li>
</ul>
"""


def main():
    page = PAGE.replace("<title>ORR Changelog</title>", "<title>ORR Model Card</title>").replace("__BODY__", body())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
