"""NeurHL LIVE: parse DailyFaceoff snapshots (PLAN_NeurHL4 LIVE lineup source).

Reads the raw pages written daily by neurhl/data/fetch/snapshot_lineups.py:

  data/raw/lineup_snapshots/<date>/df_lines_<team-slug>.html.gz   (32 teams)
  data/raw/lineup_snapshots/<date>/df_injuries.html.gz
  data/raw/lineup_snapshots/<date>/df_goalies.html.gz

Every page embeds <script id="__NEXT_DATA__" type="application/json">; all
parsing goes through that JSON, never the rendered HTML.

parse_lines(dir)     one row per (player, slot) on the team line-combination
                     pages: EV groups f1-f4 / d1-d3(4) / g, pp1-pp2, pk1-pk2 and
                     ir (category "oi"). Slot is DailyFaceoff's positionIdentifier
                     (lw/c/rw/ld/rd for EV lines, g1/g2 goalies, sk1-sk5 PP/PK,
                     ir1..irN); the line number is in `group`. updated_at is the
                     page-level combinations.updatedAt (UTC) -- when DailyFaceoff
                     last edited that team's lines, NOT the snapshot time.
parse_injuries(dir)  the injury NEWS feed (first page, ~20 latest items). It is
                     a feed, not a complete injury list; the `ir` group of the
                     lines pages is the per-team list. Rows carry DailyFaceoff's
                     playerFiveVFiveId, which is the NHL player id (nhl_id).
parse_goalies(dir)   the starting-goalies page. pageProps.data is an EMPTY list
                     on days without regular-season games (all of preseason).
                     Non-empty schema (checked against the archived page
                     /starting-goalies/2026-04-09): one dict per game with
                     date, dateGmt, home/away{TeamSlug,TeamName,GoalieName,
                     NewsStrengthName (Confirmed/Likely/Unconfirmed/None)}.
                     Any schema drift logs a warning and returns an empty frame.

check_lines(df)      per-team schema check: exactly 1 g1, >=12 EV forwards,
                     >=6 EV defense.

CLI: python neurhl/live/parse_dailyfaceoff.py [--date YYYY-MM-DD] [--csv OUTDIR]
"""
import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW  # noqa: E402

log = logging.getLogger("neurhl.live.dailyfaceoff")
SNAP = RAW / "lineup_snapshots"

# DailyFaceoff team slug -> NHL API abbreviation (explicit, all 32 + legacy Utah slugs).
# DailyFaceoff's sortedTeams short names differ from NHL for 8 teams (LA, MON, NAS,
# NJ, SJ, TB, VEG, WAS), so never key on them; use this table.
SLUG_TO_NHL = {
    "anaheim-ducks": "ANA", "boston-bruins": "BOS", "buffalo-sabres": "BUF",
    "calgary-flames": "CGY", "carolina-hurricanes": "CAR", "chicago-blackhawks": "CHI",
    "colorado-avalanche": "COL", "columbus-blue-jackets": "CBJ", "dallas-stars": "DAL",
    "detroit-red-wings": "DET", "edmonton-oilers": "EDM", "florida-panthers": "FLA",
    "los-angeles-kings": "LAK", "minnesota-wild": "MIN", "montreal-canadiens": "MTL",
    "nashville-predators": "NSH", "new-jersey-devils": "NJD", "new-york-islanders": "NYI",
    "new-york-rangers": "NYR", "ottawa-senators": "OTT", "philadelphia-flyers": "PHI",
    "pittsburgh-penguins": "PIT", "san-jose-sharks": "SJS", "seattle-kraken": "SEA",
    "st-louis-blues": "STL", "tampa-bay-lightning": "TBL", "toronto-maple-leafs": "TOR",
    "utah-mammoth": "UTA", "vancouver-canucks": "VAN", "vegas-golden-knights": "VGK",
    "washington-capitals": "WSH", "winnipeg-jets": "WPG",
    # legacy Utah slugs (snapshot files are always saved as df_lines_utah-mammoth)
    "utah-hockey-club": "UTA", "utah-hc": "UTA",
}
NHL_TEAMS = sorted(set(SLUG_TO_NHL.values()))
assert len(NHL_TEAMS) == 32

EV_FWD = {"lw", "c", "rw"}
EV_DEF = {"ld", "rd"}
LINE_COLS = ["team_abbrev", "name", "jersey", "group", "slot", "category",
             "injury_status", "gtd", "updated_at", "position_name", "df_player_id",
             "player_slug", "source_name", "team_slug"]
INJ_COLS = ["team_abbrev", "name", "nhl_id", "df_player_id", "position", "date",
            "category", "strength", "details", "source_name", "source_url",
            "created_at", "updated_at"]
GOALIE_COLS = ["game_date", "start_utc", "side", "team_abbrev", "opp_abbrev",
               "goalie", "df_goalie_id", "status", "news_created_at", "details"]

_NEXT_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>', re.S)


def next_data(path: Path) -> dict:
    """The page's __NEXT_DATA__ JSON."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        html = f.read()
    m = _NEXT_RE.search(html)
    if not m:
        raise ValueError(f"{path.name}: no __NEXT_DATA__ script")
    return json.loads(m.group(1))


def snapshot_dir(date: str | None = None) -> Path:
    """Directory for a date, or the latest snapshot when date is None."""
    if date:
        return SNAP / date
    days = sorted(p for p in SNAP.iterdir() if p.is_dir() and re.match(r"\d{4}-\d\d-\d\d$", p.name))
    return days[-1]


# ---------------------------------------------------------------- lines
def parse_lines(snapshot_dir: Path) -> pd.DataFrame:
    snapshot_dir = Path(snapshot_dir)
    rows = []
    files = sorted(snapshot_dir.glob("df_lines_*.html.gz"))
    for f in files:
        slug = f.name[len("df_lines_"):-len(".html.gz")]
        if slug not in SLUG_TO_NHL:
            log.warning("%s: unknown team slug %r, skipped", f.name, slug)
            continue
        team = SLUG_TO_NHL[slug]
        try:
            comb = next_data(f)["props"]["pageProps"]["combinations"]
            players = comb["players"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            log.warning("%s: unparseable (%s), skipped", f.name, e)
            continue
        page_abbrev = comb.get("teamAbbreviation")
        if page_abbrev and page_abbrev != team:
            log.warning("%s: page teamAbbreviation %s != mapped %s", f.name, page_abbrev, team)
        for p in players:
            rows.append({
                "team_abbrev": team,
                "name": (p.get("name") or "").strip(),
                "jersey": p.get("jerseyNumber"),
                "group": p.get("groupIdentifier"),
                "slot": p.get("positionIdentifier"),
                "category": p.get("categoryIdentifier"),
                "injury_status": p.get("injuryStatus"),
                "gtd": bool(p.get("gameTimeDecision")),
                "updated_at": comb.get("updatedAt"),
                "position_name": p.get("positionName"),
                "df_player_id": p.get("playerId"),
                "player_slug": p.get("playerSlug"),
                "source_name": comb.get("sourceName"),
                "team_slug": slug,
            })
    df = pd.DataFrame(rows, columns=LINE_COLS)
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["df_player_id"] = pd.to_numeric(df["df_player_id"], errors="coerce").astype("Int64")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    missing = set(NHL_TEAMS) - set(df["team_abbrev"])
    if missing:
        log.warning("%s: no lines for %s", snapshot_dir.name, sorted(missing))
    return df


def ev_role(df: pd.DataFrame) -> pd.Series:
    """F / D / G for EV line slots, '' otherwise."""
    ev = df["category"].eq("ev")
    role = pd.Series("", index=df.index)
    role[ev & df["slot"].isin(EV_FWD)] = "F"
    role[ev & df["slot"].isin(EV_DEF)] = "D"
    role[ev & df["slot"].isin({"g1", "g2"})] = "G"
    return role


def check_lines(df: pd.DataFrame) -> list[str]:
    """Schema check per team; returns human-readable violations (empty = pass)."""
    out = []
    role = ev_role(df)
    for team in NHL_TEAMS:
        m = df["team_abbrev"].eq(team)
        if not m.any():
            out.append(f"{team}: no lines parsed")
            continue
        g1 = int((m & df["slot"].eq("g1") & df["category"].eq("ev")).sum())
        nf = int((m & role.eq("F")).sum())
        nd = int((m & role.eq("D")).sum())
        if g1 != 1:
            out.append(f"{team}: {g1} starting-goalie (g1) slots, expected 1")
        if nf < 12:
            out.append(f"{team}: {nf} EV forwards, expected >= 12")
        if nd < 6:
            out.append(f"{team}: {nd} EV defense, expected >= 6")
        dup = df[m & role.ne("")].duplicated(subset=["name"])
        if dup.any():
            out.append(f"{team}: player(s) in two EV slots: "
                       f"{sorted(df[m & role.ne('')][dup]['name'])}")
    return out


# ---------------------------------------------------------------- injuries
def parse_injuries(snapshot_dir: Path) -> pd.DataFrame:
    f = Path(snapshot_dir) / "df_injuries.html.gz"
    empty = pd.DataFrame(columns=INJ_COLS)
    if not f.exists():
        log.warning("%s: missing", f)
        return empty
    try:
        items = next_data(f)["props"]["pageProps"]["data"]["data"]
        assert isinstance(items, list), "pageProps.data.data is not a list"
    except (KeyError, TypeError, ValueError, AssertionError) as e:
        log.warning("%s: unexpected schema (%s); returning empty frame", f.name, e)
        return empty
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        slug = it.get("teamSlug")
        rows.append({
            "team_abbrev": SLUG_TO_NHL.get(slug, it.get("teamAbbreviation")),
            "name": it.get("playerName"),
            "nhl_id": it.get("playerFiveVFiveId"),
            "df_player_id": it.get("playerId"),
            "position": it.get("playerPosition"),
            "date": it.get("date"),
            "category": it.get("newsCategoryName"),
            "strength": it.get("newsStrengthName"),
            "details": (it.get("details") or "").strip(),
            "source_name": it.get("sourceName"),
            "source_url": it.get("sourceUrl"),
            "created_at": it.get("createdAt"),
            "updated_at": it.get("updatedAt"),
        })
    df = pd.DataFrame(rows, columns=INJ_COLS)
    for c in ("nhl_id", "df_player_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("created_at", "updated_at"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------- goalies
def _goalie_game_rows(e: dict) -> list[dict]:
    """Two rows (home, away) from one game entry; raises AssertionError on drift."""
    assert isinstance(e, dict), f"entry is {type(e).__name__}, not dict"
    need = ["homeTeamSlug", "awayTeamSlug", "homeGoalieName", "awayGoalieName"]
    miss = [k for k in need if k not in e]
    assert not miss, f"entry missing keys {miss}; keys seen: {sorted(e)[:25]}"
    assert e.get("date") or e.get("dateGmt"), "entry has neither date nor dateGmt"
    rows = []
    teams = {}
    for side in ("home", "away"):
        slug = e[f"{side}TeamSlug"]
        assert slug in SLUG_TO_NHL, f"unknown {side} team slug {slug!r}"
        teams[side] = SLUG_TO_NHL[slug]
    for side, opp in (("home", "away"), ("away", "home")):
        status = e.get(f"{side}NewsStrengthName")
        if status is None and f"{side}NewsStrengthName" not in e:
            status = e.get(f"{side}GoalieStatus")        # tolerate a rename
        rows.append({
            "game_date": e.get("date"),
            "start_utc": e.get("dateGmt"),
            "side": side,
            "team_abbrev": teams[side],
            "opp_abbrev": teams[opp],
            "goalie": e.get(f"{side}GoalieName"),
            "df_goalie_id": e.get(f"{side}GoalieId"),
            "status": status if status else "Unknown",
            "news_created_at": e.get(f"{side}NewsCreatedAt"),
            "details": (e.get(f"{side}NewsDetails") or "").strip(),
        })
    return rows


def parse_goalies(snapshot_dir: Path) -> pd.DataFrame:
    """Starting goalies; empty frame on no-game days or any schema drift."""
    f = Path(snapshot_dir) / "df_goalies.html.gz"
    empty = pd.DataFrame(columns=GOALIE_COLS)
    if not f.exists():
        log.info("%s: missing (goalie page not snapshotted that day)", f)
        return empty
    try:
        pp = next_data(f)["props"]["pageProps"]
        data = pp.get("data")
    except (KeyError, TypeError, ValueError) as e:
        log.warning("%s: unreadable (%s); returning empty frame", f.name, e)
        return empty
    if data is None or (isinstance(data, list) and not data):
        log.info("%s: no games listed for %s (normal outside the regular season)",
                 f.name, pp.get("date"))
        return empty
    if not isinstance(data, list):
        log.warning("%s: pageProps.data is %s, expected list; returning empty frame",
                    f.name, type(data).__name__)
        return empty
    rows = []
    try:
        for e in data:
            rows += _goalie_game_rows(e)
    except AssertionError as err:
        log.warning("%s: starting-goalie schema check failed (%s); returning empty frame. "
                    "Inspect the page and update parse_goalies.", f.name, err)
        return empty
    df = pd.DataFrame(rows, columns=GOALIE_COLS)
    df["df_goalie_id"] = pd.to_numeric(df["df_goalie_id"], errors="coerce").astype("Int64")
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    bad = df[~df["status"].isin({"Confirmed", "Likely", "Unconfirmed", "Unknown"})]
    if len(bad):
        log.warning("%s: unfamiliar goalie status values %s", f.name, sorted(bad["status"].unique()))
    if pp.get("date") and (df["game_date"] != pp["date"]).any():
        log.warning("%s: some games dated differently from page date %s", f.name, pp["date"])
    return df


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Parse a DailyFaceoff snapshot day.")
    ap.add_argument("--date", help="YYYY-MM-DD (default: latest snapshot)")
    ap.add_argument("--dir", help="explicit snapshot directory (overrides --date)")
    ap.add_argument("--csv", help="write lines/injuries/goalies CSVs to this directory")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = Path(a.dir) if a.dir else snapshot_dir(a.date)
    lines, inj, gk = parse_lines(d), parse_injuries(d), parse_goalies(d)

    role = ev_role(lines)
    t = pd.DataFrame({
        "F": lines[role.eq("F")].groupby("team_abbrev").size(),
        "D": lines[role.eq("D")].groupby("team_abbrev").size(),
        "G": lines[role.eq("G")].groupby("team_abbrev").size(),
        "PP": lines[lines["category"].eq("pp")].groupby("team_abbrev").size(),
        "PK": lines[lines["category"].eq("pk")].groupby("team_abbrev").size(),
        "IR": lines[lines["group"].eq("ir")].groupby("team_abbrev").size(),
        "gtd": lines[lines["gtd"]].drop_duplicates(["team_abbrev", "name"]).groupby("team_abbrev").size(),
    }).reindex(NHL_TEAMS).fillna(0).astype(int)
    g1 = lines[lines["slot"].eq("g1")].set_index("team_abbrev")["name"]
    t["g1"] = g1.reindex(t.index).fillna("-")
    upd = lines.groupby("team_abbrev")["updated_at"].first()
    t["updated"] = upd.reindex(t.index).dt.strftime("%m-%d %H:%M").fillna("-")
    t["source"] = lines.groupby("team_abbrev")["source_name"].first().reindex(t.index)
    print(f"snapshot {d.name}: {len(lines)} line rows, {lines['team_abbrev'].nunique()} teams")
    print(t.to_string())
    print(f"\ninjury feed: {len(inj)} items "
          f"({inj['name'].nunique()} players, {inj['team_abbrev'].nunique()} teams, "
          f"{inj['nhl_id'].isna().sum()} without NHL id)")
    print(f"starting goalies: {len(gk) // 2} games" + ("" if len(gk) else " (none listed)"))
    if len(gk):
        print(gk[["game_date", "side", "team_abbrev", "goalie", "status"]].to_string(index=False))
    viol = check_lines(lines)
    print(f"\nschema check: {'PASS' if not viol else f'{len(viol)} violation(s)'}")
    for v in viol:
        print("  " + v)
    if a.csv:
        out = Path(a.csv)
        out.mkdir(parents=True, exist_ok=True)
        lines.to_csv(out / f"df_lines_{d.name}.csv", index=False)
        inj.to_csv(out / f"df_injuries_{d.name}.csv", index=False)
        gk.to_csv(out / f"df_goalies_{d.name}.csv", index=False)


if __name__ == "__main__":
    main()
