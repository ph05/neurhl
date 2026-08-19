"""Build v5 derived tables from the PLAN_V5 data acquisition (D1-D6).

Outputs (committed):
  data/processed/team_seasons_v5.csv  one row per (season_end, team): official
      aggregates (faceoffs, penalties, special teams, SAT%), MoneyPuck situational
      metrics (score-adjusted Corsi, flurry-adjusted xG%, danger shares), and
      shot-level rush xG metrics.
  data/processed/hr_league.csv        hockey-reference standings (SRS/SOS) + team
      stats per season.
  data/processed/edge_team.csv        NHL EDGE team tracking aggregates (2022+).
  data/processed/pbp_team_seasons.csv PBP-derived per-team-season aggregates
      (built by build_pbp.py, merged here if present).
  output/v5_crosschecks.json          cross-source agreement record (PLAN_V5
      integrity bar: mismatches > 2% investigated before use).

Team identity = franchise codes as everywhere else (ATL->WPG, ARI/PHX->UTA).
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import NAME2FRAN
from players import MP_FRAN

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"
OUT = PROJ / "output"


# ------------------------------------------------------------- NHL stats-rest
def nhl_reports() -> pd.DataFrame:
    rows = {}
    for end in range(2006, 2027):
        recs = {}
        for rep in ("summary", "penalties", "realtime", "faceoffpercentages"):
            p = RAW / "nhl_reports" / f"{rep}_{end}.json"
            if not p.exists():
                continue
            recs[rep] = {d["teamFullName"]: d for d in json.loads(p.read_text())}
        if "summary" not in recs:
            continue
        for name, s in recs["summary"].items():
            team = NAME2FRAN.get(name) or NAME2FRAN.get(
                name.replace("é", "e"))          # "Montréal Canadiens"
            assert team is not None, f"unmapped NHL team name: {name}"
            pen = recs.get("penalties", {}).get(name, {})
            rt = recs.get("realtime", {}).get(name, {})
            rows[(end, team)] = {
                "season_end": end, "team": team,
                "fo_pct": s.get("faceoffWinPct"),
                "pp_pct": s.get("powerPlayPct"), "pk_pct": s.get("penaltyKillPct"),
                "pen_drawn60": pen.get("penaltiesDrawnPer60"),
                "pen_taken60": pen.get("penaltiesTakenPer60"),
                "pen_net60": pen.get("netPenaltiesPer60"),
                "sat_pct": rt.get("satPct"),
                "hits60": rt.get("hitsPer60"),
                "giveaways60": rt.get("giveawaysPer60"),
                "takeaways60": rt.get("takeawaysPer60"),
            }
    return pd.DataFrame(rows.values())


# ------------------------------------------------------------- MoneyPuck teams
def mp_situational() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("mp_teams_*.csv")):
        end = int(f.stem.split("_")[-1]) + 1
        mp = pd.read_csv(f)
        mp = mp.rename(columns={"penalitiesFor": "penaltiesFor",       # MP typo
                                "penalitiesAgainst": "penaltiesAgainst"})
        mp["team"] = mp.team.replace(MP_FRAN)
        s5 = mp[mp.situation == "5on5"].set_index("team")
        sall = mp[mp.situation == "all"].set_index("team")
        for t in s5.index:
            r5, ra = s5.loc[t], sall.loc[t]
            corsi_f = r5.scoreAdjustedShotsAttemptsFor
            corsi_a = r5.scoreAdjustedShotsAttemptsAgainst
            fxf = r5.flurryScoreVenueAdjustedxGoalsFor
            fxa = r5.flurryScoreVenueAdjustedxGoalsAgainst
            hdf, hda = r5.highDangerxGoalsFor, r5.highDangerxGoalsAgainst
            rows.append({
                "season_end": end, "team": t,
                "corsi_sa_pct": corsi_f / (corsi_f + corsi_a),
                "flurry_xg5_pct": fxf / (fxf + fxa),
                "hd_share_f": hdf / r5.xGoalsFor,
                "hd_xg_pct": hdf / (hdf + hda),
                "fo_pct_mp": ra.faceOffsWonFor
                             / (ra.faceOffsWonFor + ra.faceOffsWonAgainst),
                "pen_for_mp": ra.penaltiesFor, "pen_against_mp": ra.penaltiesAgainst,
                "gp_mp": ra.games_played,
                "xgf_all_mp": ra.xGoalsFor, "xga_all_mp": ra.xGoalsAgainst,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- MoneyPuck shots
SHOT_COLS = ["season", "teamCode", "homeTeamCode", "awayTeamCode", "isHomeTeam",
             "isPlayoffGame", "homeSkatersOnIce", "awaySkatersOnIce", "shotRush",
             "xGoal", "goal", "event"]


def mp_rush() -> pd.DataFrame:
    rows = {}
    for f in sorted((RAW / "mp_shots").glob("shots_*.zip")):
        end = int(f.stem.split("_")[-1]) + 1
        with zipfile.ZipFile(f) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                df = pd.read_csv(io.BytesIO(fh.read()), usecols=lambda c: c in SHOT_COLS)
        df = df[df.isPlayoffGame == 0]
        is5v5 = (df.homeSkatersOnIce == 5) & (df.awaySkatersOnIce == 5)
        df = df[is5v5].copy()
        df["shooter"] = np.where(df.isHomeTeam == 1, df.homeTeamCode, df.awayTeamCode)
        df["defender"] = np.where(df.isHomeTeam == 1, df.awayTeamCode, df.homeTeamCode)
        for side, key in (("shooter", "f"), ("defender", "a")):
            g = df.groupby(side)
            agg = pd.DataFrame({
                f"xg5_{key}_shots": g.xGoal.sum(),
                f"rush_xg_{key}": g.apply(
                    lambda d: d.loc[d.shotRush == 1, "xGoal"].sum(),
                    include_groups=False),
            })
            for team, r in agg.iterrows():
                t = MP_FRAN.get(team, team)
                rows.setdefault((end, t), {"season_end": end, "team": t}).update(
                    r.to_dict())
    df = pd.DataFrame(rows.values())
    df["rush_xg_pct"] = df.rush_xg_f / (df.rush_xg_f + df.rush_xg_a)
    df["rush_share_f"] = df.rush_xg_f / df.xg5_f_shots
    return df


# ------------------------------------------------------------- hockey-reference
def hr_league() -> pd.DataFrame:
    rows = []
    for y in range(2006, 2027):
        p = RAW / "hr_html" / f"NHL_{y}.html"
        if not p.exists():
            continue
        html = p.read_text().replace("<!--", "").replace("-->", "")
        tables = pd.read_html(io.StringIO(html))
        srs = {}
        for t in tables:
            cols = [str(c[-1] if isinstance(c, tuple) else c) for c in t.columns]
            if "SRS" in cols and "SOS" in cols:
                t = t.copy()
                t.columns = cols
                namecol = cols[0]
                for _, r in t.iterrows():
                    nm = str(r[namecol]).replace("*", "").strip()
                    if nm in NAME2FRAN:
                        srs[NAME2FRAN[nm]] = (float(r["SRS"]), float(r["SOS"]))
        for team, (v, s) in srs.items():
            rows.append({"season_end": y, "team": team, "srs": v, "sos": s})
    return pd.DataFrame(rows).drop_duplicates(["season_end", "team"])


# ------------------------------------------------------------- NHL EDGE
def edge_team() -> pd.DataFrame:
    rows = []
    for p in sorted((RAW / "edge").glob("team-detail_*.json")):
        _, abbr, end = p.stem.split("_")
        d = json.loads(p.read_text())
        if not d:
            continue
        t = MP_FRAN.get(abbr, abbr)
        row = {"season_end": int(end), "team": t}
        ss = d.get("skatingSpeed", {})
        row["bursts_20"] = _val(ss.get("burstsOver20"), "value")
        row["bursts_22"] = _val(ss.get("burstsOver22"), "value")
        row["speed_max"] = _val(ss.get("speedMax"), "imperial")
        row["dist_total_mi"] = _val(d.get("distanceSkated", {}).get("total"),
                                    "imperial")
        shs = d.get("shotSpeed", {})
        row["shot_att_over90"] = _val(shs.get("shotAttemptsOver90"), "value")
        row["shot_speed_max"] = _val(shs.get("topShotSpeed"), "imperial")
        zp = RAW / "edge" / f"team-zone-time-details_{abbr}_{end}.json"
        if zp.exists():
            z = json.loads(zp.read_text())
            for zt in z.get("zoneTimeDetails", []):
                if zt.get("strengthCode") in ("all", "ev", "es"):
                    sfx = "" if zt["strengthCode"] in ("ev", "es") else "_all"
                    row[f"oz_pct{sfx}"] = zt.get("offensiveZonePctg")
                    row[f"dz_pct{sfx}"] = zt.get("defensiveZonePctg")
            sd = z.get("shotDifferential", {})
            row["sat_diff_pg"] = sd.get("shotAttemptDifferential")
        rows.append(row)
    return pd.DataFrame(rows)


def _val(x, key):
    if isinstance(x, dict) and isinstance(x.get(key), (int, float)):
        return x[key]
    return np.nan


# ------------------------------------------------------------- assemble + checks
def main():
    nhl = nhl_reports()
    mp = mp_situational()
    rush = mp_rush()
    hr = hr_league()

    v5 = nhl.merge(mp, on=["season_end", "team"], how="outer") \
            .merge(rush[["season_end", "team", "rush_xg_pct", "rush_share_f",
                         "xg5_f_shots"]], on=["season_end", "team"], how="left") \
            .merge(hr, on=["season_end", "team"], how="left") \
            .sort_values(["season_end", "team"])

    pbp_p = PROC / "pbp_team_seasons.csv"
    if pbp_p.exists():
        pbp = pd.read_csv(pbp_p)
        v5 = v5.merge(pbp, on=["season_end", "team"], how="left")

    checks = {}
    both = v5.dropna(subset=["fo_pct", "fo_pct_mp"])
    checks["fo_nhl_vs_mp"] = {
        "n": len(both), "corr": float(both.fo_pct.corr(both.fo_pct_mp)),
        "mean_abs_diff": float((both.fo_pct - both.fo_pct_mp).abs().mean())}
    # MP penalty semantics resolved against official drawn/taken:
    pen = v5.dropna(subset=["pen_net60", "pen_for_mp", "pen_against_mp", "gp_mp"])
    mp_net_a = (pen.pen_against_mp - pen.pen_for_mp) / pen.gp_mp
    mp_net_b = -mp_net_a
    ca, cb = pen.pen_net60.corr(mp_net_a), pen.pen_net60.corr(mp_net_b)
    checks["pen_semantics"] = {
        "corr_drawnFor": float(ca), "corr_takenFor": float(cb),
        "resolution": "penaltiesFor = drawn by team" if ca > cb
                      else "penaltiesFor = taken by team"}
    ts = pd.read_csv(PROC / "team_seasons.csv")
    m = v5.merge(ts[["season_end", "team", "xgf_all"]], on=["season_end", "team"])
    m = m.dropna(subset=["xgf_all_mp", "xgf_all"])
    checks["mp_teams_vs_team_seasons_xg"] = {
        "n": len(m),
        "max_rel_diff": float(((m.xgf_all_mp - m.xgf_all).abs()
                               / m.xgf_all).max())}
    if "fo_pct_pbp" in v5.columns:
        b = v5.dropna(subset=["fo_pct", "fo_pct_pbp"])
        checks["fo_nhl_vs_pbp"] = {
            "n": len(b), "corr": float(b.fo_pct.corr(b.fo_pct_pbp)),
            "mean_abs_diff": float((b.fo_pct - b.fo_pct_pbp).abs().mean())}
    if "pen_net_g_pbp" in v5.columns:
        b = v5.dropna(subset=["pen_net60", "pen_net_g_pbp"])
        checks["pen_nhl_vs_pbp_corr"] = {
            "n": len(b), "corr": float(b.pen_net60.corr(b.pen_net_g_pbp))}
    fr_dir = RAW / "fastrhockey"
    if fr_dir.exists():
        frames = []
        for f in sorted(fr_dir.glob("team_box_*.parquet")):
            try:
                frames.append(pd.read_parquet(
                    f, columns=["season", "tri_code", "face_off_win_percentage"]))
            except Exception as e:      # noqa: BLE001 — schema drift across years
                print(f"fastrhockey skip {f.name}: {e!r}")
        if frames:
            fr = pd.concat(frames, ignore_index=True)
            fr["face_off_win_percentage"] = pd.to_numeric(
                fr.face_off_win_percentage, errors="coerce")   # strings in some years
            fr["team"] = fr.tri_code.replace(MP_FRAN)
            fr["season_end"] = fr.season.astype(int) % 10000
            fo_fr = fr.groupby(["season_end", "team"], as_index=False) \
                .face_off_win_percentage.mean() \
                .rename(columns={"face_off_win_percentage": "fo_pct_fr"})
            b = v5.merge(fo_fr, on=["season_end", "team"]) \
                .dropna(subset=["fo_pct", "fo_pct_fr"])
            if len(b):
                checks["fo_nhl_vs_fastrhockey"] = {
                    "n": len(b),
                    "corr": float(b.fo_pct.corr(b.fo_pct_fr / 100.0)),
                    "mean_abs_diff": float(
                        (b.fo_pct - b.fo_pct_fr / 100.0).abs().mean())}
    if "sat5_pct_own" in v5.columns:
        b = v5.dropna(subset=["sat_pct", "sat5_pct_own", "sat5_pct_opp"])
        r_own = float(b.sat_pct.corr(b.sat5_pct_own))
        r_opp = float(b.sat_pct.corr(b.sat5_pct_opp))
        checks["sat_attribution"] = {
            "n": len(b), "corr_blk_owner_shoots": r_own,
            "corr_blk_owner_blocks": r_opp,
            "chosen": "sat5_pct_own" if r_own >= r_opp else "sat5_pct_opp"}

    v5.round(6).to_csv(PROC / "team_seasons_v5.csv", index=False)
    hr.round(4).to_csv(PROC / "hr_league.csv", index=False)
    ed = edge_team()
    if len(ed):
        ed.round(4).to_csv(PROC / "edge_team.csv", index=False)
    OUT.mkdir(exist_ok=True)
    (OUT / "v5_crosschecks.json").write_text(json.dumps(checks, indent=2))
    print(f"team_seasons_v5: {len(v5)} rows, {v5.season_end.min()}-{v5.season_end.max()}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
