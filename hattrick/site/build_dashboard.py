"""Build the HatTrick 2026-27 dashboard (one self-contained HTML page).

    python3 -m hattrick.site.build_dashboard   ->  hattrick/output/dashboard.html

Data: the preseason freeze (hattrick/output/freeze_2027/), NeurHL's 1.1 and
1.3 releases for comparison, the 2026-08-17 market lines, and the scorecard.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hattrick import config as C

F = C.OUT / "freeze_2027"
OUT = C.OUT / "dashboard.html"


def _r(x, n=1):
    return None if pd.isna(x) else round(float(x), n)


def data() -> dict:
    t = pd.read_csv(F / "teams_2027.csv")
    n13 = pd.read_csv(C.ROOT / "neurhl/output/neurhl_1_3/season/teams_2027.csv").set_index("team")
    m = pd.read_csv(C.CACHE / "snapshot/data/market/nhl_totals_ou_2027.csv").set_index("team")
    teams = [{"team": r.team, "conf": r.conf, "div": r.div, "pts": _r(r.points), "p10": _r(r.points_p10, 0),
              "p90": _r(r.points_p90, 0), "po": _r(r.playoff_pct), "divw": _r(r.division_pct),
              "cup": _r(r.cup_pct), "pres": _r(r.presidents_pct), "gf": _r(r.gf, 0), "ga": _r(r.ga, 0),
              "news": _r(r.news_rel82 * 84 / 82), "n13": _r(n13.points.get(r.team)),
              "n13po": _r(n13.playoff_pct.get(r.team)), "line": _r(m.line.get(r.team))}
             for r in t.itertuples()]
    sk = pd.read_csv(F / "skaters_2027.csv")
    ns = pd.read_csv(C.ROOT / "neurhl/output/neurhl_1_3/season/skaters_2027.csv").set_index("player_id")
    sk = sk.sort_values("p", ascending=False).head(400)
    skaters = [{"name": r.name, "team": r.team, "pos": r.pos, "gp": _r(r.gp, 0), "g": _r(r.g), "a": _r(r.a),
                "p": _r(r.p), "lo": _r(r.p_p10, 0), "hi": _r(r.p_p90, 0),
                "n13": _r(ns.points.get(r.player_id))} for r in sk.itertuples()]
    gl = pd.read_csv(F / "goalies_2027.csv")
    ng = pd.read_csv(C.ROOT / "neurhl/output/neurhl_1_3/season/goalies_2027.csv").set_index("player_id")
    gl = gl[gl.starts >= 15].sort_values("starts", ascending=False)
    goalies = [{"name": r.name, "team": r.team, "starts": _r(r.starts), "sv": _r(r.sv_pct, 3),
                "gsax": _r(r.gsax), "n13": _r(ng.starts.get(r.player_id))} for r in gl.itertuples()]
    g = pd.read_csv(F / "games_2027.csv")
    n13g = pd.read_csv(C.ROOT / "neurhl/output/neurhl_1_3/season/games_2027.csv").set_index("game_id")
    res = {}
    rp = C.OUT / "live" / "results_2027.csv"
    src = rp if rp.exists() else C.ROOT / "neurhl/output/live/results_2027.csv"
    if src.exists():
        rr = pd.read_csv(src)
        res = {int(r.game_id): f"{r.away} {int(r.away_g)}–{int(r.home_g)} {r.home}"
               + ("" if r.last_period == "REG" else f" ({r.last_period})") for r in rr.itertuples()}
    live = {}
    lf = sorted((C.OUT / "live").glob("*/games_*.csv"))
    if lf:
        from hattrick.score import _deadline
        ld = pd.concat([pd.read_csv(f) for f in lf])
        ld = ld[pd.to_datetime(ld.created_utc).dt.tz_convert(None) < pd.to_datetime(ld.date).map(_deadline)]
        live = ld.sort_values("created_utc").drop_duplicates("game_id").set_index("game_id").p_home_win.to_dict()
    games = [{"id": int(r.game_id), "d": str(r.date), "h": r.home, "a": r.away, "p": _r(r.p_home_win, 3),
              "live": _r(live.get(r.game_id), 3),
              "ot": _r(r.p_ot, 3), "n13": _r(n13g.p_home_win.get(r.game_id), 3),
              "res": res.get(int(r.game_id))} for r in g.itertuples()]
    card = json.loads((C.OUT / "scorecard_2027.json").read_text()) if (C.OUT / "scorecard_2027.json").exists() else {}
    state = json.loads((F / "state_2027.json").read_text())
    return {"teams": teams, "skaters": skaters, "goalies": goalies, "games": games,
            "card": card, "blend": state.get("blend", {}), "created": state.get("created_utc"),
            "cutoff": state.get("cutoff_utc")}


SUMMARY = [
    ["Standings", "NeurHL's judge window 2019-24 (raw points MAE / CRPS)", "9.23 / 6.70", "9.42 / 6.80", "win",
     "Market + team history, convex weights, leave-one-season-out; the market alone is 9.32 / 6.71; Elo 9.47 / 6.87"],
    ["Standings", "NeurHL's backtest seasons 2012, 2014-17 (per-82 MAE)", "8.96", "9.99", "win",
     "Team-history view; NeurHL's v1 Elo + xG 9.17; HatTrick loses 2012"],
    ["Games", "Shipped preseason game-file pipeline, 2019-24 (6,289 games)", "0.6674", "—", "win",
     "No historical NeurHL preseason file; beats a frozen Elo (0.6718) by 0.0044, CI excludes 0"],
    ["Games", "NeurHL's 11,052 games, in-season log loss", "0.6635", "0.6645", "tie",
     "HatTrick with starters where known, no lineups; NeurHL-H with actual lineups and starters"],
    ["Games", "NeurHL-G gate games 2019-24", "0.6603", "0.6601", "tie", "Difference not significant"],
    ["Skaters", "NeurHL's protocol, ≥40 GP, held-out 2022-26 (points MAE)", "9.41", "9.54", "win",
     "2022-24 (HatTrick: first-10-games rosters; NeurHL: actual season team) 9.49 vs 9.74; 2025-26 9.30 vs 9.24"],
    ["Skaters", "Per-82 restatement, same players", "9.22", "9.08", "loss", "NeurHL's A/B blend is 0.14 better; 0.33 better on 2025-26"],
    ["Skaters", "Regression toward the mean (slope on last season)", "0.83", "0.96", "win", "Below 1 is what a projection should do"],
    ["Goalies", "GSAx/60 MAE, held-out 2022-26, ≥1,000 shots", "0.237", "—", "tie",
     "League average 0.241, last season 0.335; save talent is barely predictable"],
]


PAGE = r"""<title>HatTrick 2026–27</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a rink-board scoreboard header, then stacked panels; tables scroll inside their own frames. */
:root {
  --ice: #f4f7f9; --board: #ffffff; --ink: #0d1b26; --muted: #5b6b78; --rule: #d9e1e7;
  --blue: #1d4fb5; --red: #c8102e; --win: #1f7a4d; --tie: #8a6d1c; --loss: #b3261e;
  --band: #c9d7f2; --bar: #1d4fb5;
  --display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --body: "Barlow", "Helvetica Neue", Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ice: #0b1218; --board: #111b23; --ink: #e5edf3; --muted: #93a3b1; --rule: #22303b;
  --blue: #7aa2ff; --red: #ff5d6c; --win: #4cc38a; --tie: #e2b84a; --loss: #ff7a70;
  --band: #24395f; --bar: #7aa2ff; color-scheme: dark; } }
:root[data-theme="dark"] {
  --ice: #0b1218; --board: #111b23; --ink: #e5edf3; --muted: #93a3b1; --rule: #22303b;
  --blue: #7aa2ff; --red: #ff5d6c; --win: #4cc38a; --tie: #e2b84a; --loss: #ff7a70;
  --band: #24395f; --bar: #7aa2ff; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--ice); color: var(--ink); font: 15px/1.5 var(--body); }
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 16px; padding-block: 24px 64px; display: grid; gap: 28px; }
header.board { background: var(--board); border: 1px solid var(--rule); border-top: 6px solid var(--red);
  padding: 20px 20px 16px; display: grid; gap: 10px; }
header.board h1 { font: 700 clamp(34px, 6vw, 56px)/0.95 var(--display); letter-spacing: 0.01em; margin: 0; text-wrap: balance; }
header.board h1 span { color: var(--blue); }
.sub { color: var(--muted); max-width: 70ch; margin: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font: 500 12px/1 var(--mono); padding: 6px 8px; border: 1px solid var(--rule); background: var(--ice); }
section { display: grid; gap: 12px; min-width: 0; }
h2 { font: 600 26px/1.1 var(--display); margin: 0; letter-spacing: 0.02em; text-transform: uppercase; }
h2 small { font: 500 13px/1 var(--body); color: var(--muted); text-transform: none; letter-spacing: 0; margin-left: 8px; }
.note { color: var(--muted); font-size: 13px; max-width: 75ch; margin: 0; }
.frame { background: var(--board); border: 1px solid var(--rule); overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { font: 600 12px/1.2 var(--body); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); background: var(--board); position: sticky; top: 0; }
th button { all: unset; cursor: pointer; }
th button:focus-visible, .seg button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
td.note { white-space: normal; min-width: 220px; }
td.num, th.num { text-align: right; font-family: var(--mono); font-size: 13px; }
td.team { font: 600 15px/1 var(--display); letter-spacing: 0.04em; }
tr:hover td { background: color-mix(in srgb, var(--blue) 6%, transparent); }
.verdict { font: 600 11px/1 var(--mono); padding: 4px 6px; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid currentColor; }
.v-win { color: var(--win); } .v-tie { color: var(--tie); } .v-loss { color: var(--loss); }
.rangecell { min-width: 170px; }
.range { position: relative; height: 14px; }
.range .b { position: absolute; top: 4px; height: 6px; background: var(--band); }
.range .m { position: absolute; top: 0; width: 2px; height: 14px; background: var(--bar); }
.range .n { position: absolute; top: 2px; width: 8px; height: 8px; border: 2px solid var(--red); border-radius: 50%; transform: translateX(-4px); }
.pos { color: var(--win); } .neg { color: var(--loss); }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.seg { display: inline-flex; flex-wrap: wrap; max-width: 100%; border: 1px solid var(--rule); background: var(--board); }
.seg button { font: 500 13px/1 var(--body); color: var(--ink); background: none; border: 0; padding: 8px 10px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--board); }
input[type=search], select { font: 14px var(--body); color: var(--ink); background: var(--board); border: 1px solid var(--rule); padding: 7px 9px; min-width: 0; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
.legend i { display: inline-block; width: 18px; height: 6px; background: var(--band); vertical-align: middle; margin-right: 6px; }
.legend b { display: inline-block; width: 8px; height: 8px; border: 2px solid var(--red); border-radius: 50%; vertical-align: middle; margin-right: 6px; }
.prob { display: inline-block; width: 80px; height: 8px; background: var(--band); position: relative; vertical-align: middle; }
.prob span { position: absolute; left: 0; top: 0; bottom: 0; background: var(--bar); }
footer { color: var(--muted); font-size: 12px; }
@media (prefers-reduced-motion: no-preference) { tr td { transition: background .12s; } }
</style>

<div class="wrap">
<header class="board">
  <h1>HatTrick <span>2026–27</span></h1>
  <p class="sub">NHL projections from information dated before the first puck drop (2026-09-29, 5:00 pm ET), built after the season began and scored against every NeurHL release only on games after publication. Standings anchor to the sportsbook line, re-price the news it had not seen with the player model, and blend in team history. Games come from one scoring model, players from regressed per-60 rates with conserved ice time.</p>
  <div class="chips" id="chips"></div>
</header>

<section aria-labelledby="h-sum">
  <h2 id="h-sum">Head to head<small>backtests on NeurHL's own data and protocols</small></h2>
  <div class="frame"><table id="sum"></table></div>
  <p class="note">Lower is better for every number. Games use log loss, standings use points per 82 games, skaters use season points. Full detail is in hattrick/RESULTS.md.</p>
</section>

<section aria-labelledby="h-st">
  <h2 id="h-st">Standings<small>40,000 simulated seasons</small></h2>
  <div class="controls">
    <div class="seg" role="group" aria-label="Filter teams" id="stfilter"></div>
  </div>
  <div class="legend"><span><i></i>HatTrick 80% range</span><span><b></b>NeurHL 1.3</span></div>
  <div class="frame"><table id="st"></table></div>
  <p class="note">News is points HatTrick added or removed for events after the 2026-08-17 line: trades, signings, camp injuries, suspensions, and the Hellebuyck standoff.</p>
</section>

<section aria-labelledby="h-sk">
  <h2 id="h-sk">Skaters<small>84 games, 80% intervals</small></h2>
  <div class="controls"><label for="sksearch" class="note">Find</label><input id="sksearch" type="search" placeholder="Player or team"></div>
  <div class="frame"><table id="sk"></table></div>
</section>

<section aria-labelledby="h-gl">
  <h2 id="h-gl">Goalies<small>no start cap; injury risk simulated</small></h2>
  <div class="frame"><table id="gl"></table></div>
</section>

<section aria-labelledby="h-gm">
  <h2 id="h-gm">Games<small>preseason file and daily in-season forecasts</small></h2>
  <div class="controls"><label for="gmdate" class="note">Date</label><select id="gmdate"></select></div>
  <div class="frame"><table id="gm"></table></div>
</section>

<section aria-labelledby="h-sc">
  <h2 id="h-sc">Scorecard so far</h2>
  <div class="frame"><table id="sc"></table></div>
  <p class="note" id="scnote"></p>
</section>
<footer id="foot"></footer>
</div>

<script>
const D = __DATA__;
const SUMMARY = __SUMMARY__;
const $ = s => document.querySelector(s);
const fmt = (x, d = 1) => x == null ? "—" : Number(x).toFixed(d);
const sign = x => x == null ? "—" : (x > 0 ? "+" : "") + Number(x).toFixed(1);

$("#chips").innerHTML = [
  `cutoff ${D.cutoff.replace("+00:00", " UTC")}`,
  `blend ${Object.entries(D.blend.weights || {}).map(([k, v]) => k.replace("_rel82", "") + " " + v.toFixed(2)).join(" · ")}`,
  `1,344 games · 32 teams · ${D.skaters.length}+ skaters`,
].map(t => `<span class="chip">${t}</span>`).join("");

$("#sum").innerHTML = `<thead><tr><th>Layer</th><th>Test</th><th class="num">HatTrick</th><th class="num">NeurHL</th><th>Result</th><th>Note</th></tr></thead><tbody>` +
  SUMMARY.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="num">${r[2]}</td><td class="num">${r[3]}</td><td><span class="verdict v-${r[4]}">${r[4]}</span></td><td class="note">${r[5]}</td></tr>`).join("") + "</tbody>";

function sortable(id, cols, rows, render, initial) {
  const el = $(id); let key = initial[0], dir = initial[1];
  function draw() {
    const rs = [...rows()].sort((a, b) => { const x = a[key], y = b[key];
      if (x == null) return 1; if (y == null) return -1;
      return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir; });
    el.innerHTML = "<thead><tr>" + cols.map(c => `<th class="${c.num ? "num" : ""}" aria-sort="${c.k === key ? (dir > 0 ? "ascending" : "descending") : "none"}">` +
      (c.k ? `<button data-k="${c.k}">${c.t}</button>` : c.t) + "</th>").join("") + "</tr></thead><tbody>" + rs.map(render).join("") + "</tbody>";
    el.querySelectorAll("th button").forEach(b => b.onclick = () => { const k = b.dataset.k; dir = k === key ? -dir : -1; key = k; draw(); });
  }
  draw(); return draw;
}

const lo = 55, hi = 135, pct = v => ((v - lo) / (hi - lo) * 100).toFixed(2) + "%";
let stView = "All";
const stRows = () => D.teams.filter(t => stView === "All" || t.conf === stView || t.div === stView);
const drawSt = sortable("#st", [
  { t: "Team", k: "team" }, { t: "Division", k: "div" }, { t: "Pts", k: "pts", num: 1 }, { t: "Range" },
  { t: "NeurHL 1.3", k: "n13", num: 1 }, { t: "Line", k: "line", num: 1 }, { t: "News", k: "news", num: 1 },
  { t: "Playoffs %", k: "po", num: 1 }, { t: "Division %", k: "divw", num: 1 }, { t: "Cup %", k: "cup", num: 1 },
  { t: "GF", k: "gf", num: 1 }, { t: "GA", k: "ga", num: 1 }], stRows,
  t => `<tr><td class="team">${t.team}</td><td>${t.div}</td><td class="num">${fmt(t.pts)}</td>
    <td class="rangecell"><div class="range" title="80%: ${t.p10}–${t.p90}"><span class="b" style="left:${pct(t.p10)};width:calc(${pct(t.p90)} - ${pct(t.p10)})"></span><span class="m" style="left:${pct(t.pts)}"></span>${t.n13 != null ? `<span class="n" style="left:${pct(t.n13)}"></span>` : ""}</div></td>
    <td class="num">${fmt(t.n13)}</td><td class="num">${fmt(t.line)}</td><td class="num ${t.news > 0.5 ? "pos" : t.news < -0.5 ? "neg" : ""}">${sign(t.news)}</td>
    <td class="num">${fmt(t.po)}</td><td class="num">${fmt(t.divw)}</td><td class="num">${fmt(t.cup)}</td><td class="num">${fmt(t.gf, 0)}</td><td class="num">${fmt(t.ga, 0)}</td></tr>`, ["pts", -1]);
const views = ["All", "E", "W", "Atlantic", "Metropolitan", "Central", "Pacific"];
$("#stfilter").innerHTML = views.map(v => `<button type="button" aria-pressed="${v === "All"}" data-v="${v}">${v === "E" ? "East" : v === "W" ? "West" : v}</button>`).join("");
$("#stfilter").querySelectorAll("button").forEach(b => b.onclick = () => {
  stView = b.dataset.v; $("#stfilter").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); drawSt(); });

let q = "";
const drawSk = sortable("#sk", [
  { t: "Player", k: "name" }, { t: "Team", k: "team" }, { t: "Pos", k: "pos" }, { t: "GP", k: "gp", num: 1 },
  { t: "G", k: "g", num: 1 }, { t: "A", k: "a", num: 1 }, { t: "Pts", k: "p", num: 1 }, { t: "80% range", k: "hi", num: 1 },
  { t: "NeurHL 1.3", k: "n13", num: 1 }],
  () => D.skaters.filter(s => !q || (s.name + " " + s.team).toLowerCase().includes(q)).slice(0, 150),
  s => `<tr><td>${s.name}</td><td class="team">${s.team}</td><td>${s.pos}</td><td class="num">${fmt(s.gp, 0)}</td><td class="num">${fmt(s.g)}</td>
    <td class="num">${fmt(s.a)}</td><td class="num"><b>${fmt(s.p)}</b></td><td class="num">${s.lo}–${s.hi}</td><td class="num">${fmt(s.n13)}</td></tr>`, ["p", -1]);
$("#sksearch").addEventListener("input", e => { q = e.target.value.trim().toLowerCase(); drawSk(); });

sortable("#gl", [{ t: "Goalie", k: "name" }, { t: "Team", k: "team" }, { t: "Starts", k: "starts", num: 1 },
  { t: "SV%", k: "sv", num: 1 }, { t: "GSAx", k: "gsax", num: 1 }, { t: "NeurHL 1.3 starts", k: "n13", num: 1 }], () => D.goalies,
  g => `<tr><td>${g.name}</td><td class="team">${g.team}</td><td class="num">${fmt(g.starts)}</td><td class="num">${fmt(g.sv, 3).replace(/^0/, "")}</td>
    <td class="num ${g.gsax > 0 ? "pos" : "neg"}">${sign(g.gsax)}</td><td class="num">${fmt(g.n13)}</td></tr>`, ["starts", -1]);

const dates = [...new Set(D.games.map(g => g.d))];
$("#gmdate").innerHTML = dates.map(d => `<option>${d}</option>`).join("");
function drawGames() {
  const d = $("#gmdate").value;
  const rows = D.games.filter(g => g.d === d);
  $("#gm").innerHTML = `<thead><tr><th>Matchup</th><th class="num">HatTrick P(home)</th><th></th><th class="num">HatTrick in-season</th><th class="num">NeurHL 1.3</th><th class="num">P(past regulation)</th><th>Result</th></tr></thead><tbody>` +
    rows.map(g => `<tr><td class="team">${g.a} @ ${g.h}</td><td class="num">${fmt(g.p * 100, 1)}%</td><td><span class="prob"><span style="width:${(g.p * 100).toFixed(1)}%"></span></span></td>
    <td class="num">${g.live == null ? "—" : fmt(g.live * 100, 1) + "%"}</td><td class="num">${g.n13 == null ? "—" : fmt(g.n13 * 100, 1) + "%"}</td><td class="num">${fmt(g.ot * 100, 1)}%</td><td>${g.res || ""}</td></tr>`).join("") + "</tbody>";
}
$("#gmdate").onchange = drawGames; drawGames();

const games = (D.card.games || {});
const names = { hattrick_preseason: "HatTrick preseason (built after puck drop)", hattrick_inseason: "HatTrick in-season", "neurhl_1.0": "NeurHL 1.0", "neurhl_1.1": "NeurHL 1.1",
  "neurhl_1.2": "NeurHL 1.2 (after puck drop)", "neurhl_1.3": "NeurHL 1.3 (after puck drop)", neurhl_0925_freeze: "NeurHL 09-25 freeze",
  neurhl_G_pregame: "NeurHL-G pregame", neurhl_H_pregame: "NeurHL-H pregame", elo_pregame: "Elo pregame" };
$("#sc").innerHTML = `<thead><tr><th>Forecast</th><th class="num">Games</th><th class="num">Log loss</th><th class="num">Brier</th><th class="num">Accuracy</th><th class="num">After publication</th></tr></thead><tbody>` +
  Object.entries(games).sort((a, b) => a[1].log_loss - b[1].log_loss).map(([k, v]) => {
    const ap = v.after_publication;
    const apc = ap == null ? "—" : ap.n ? `${ap.n} games · ${fmt(ap.log_loss, 3)}` : "0 games";
    return `<tr><td>${names[k] || k}</td><td class="num">${v.n}</td><td class="num">${fmt(v.log_loss, 3)}</td><td class="num">${fmt(v.brier, 3)}</td><td class="num">${fmt(v.accuracy * 100, 0)}%</td><td class="num">${apc}</td></tr>`;
  }).join("") + "</tbody>";
$("#scnote").textContent = `Through ${D.card.through || "—"}: ${D.card.games_played || 0} games. "After publication" counts only games that started after the file was published, the fair comparison. At this sample size the ranking is noise; a season of 1,344 games separates models by about 0.005.`;
$("#foot").textContent = `Preseason file created ${D.created} from inputs dated before ${D.cutoff}. Built from hattrick/output/freeze_2027 and NeurHL's published files.`;
</script>
"""


def main():
    d = data()
    html = PAGE.replace("__DATA__", json.dumps(d, separators=(",", ":"))).replace(
        "__SUMMARY__", json.dumps(SUMMARY, ensure_ascii=False))
    OUT.write_text(html)
    print(f"-> {OUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
