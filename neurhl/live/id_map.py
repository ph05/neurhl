"""NeurHL LIVE: map DailyFaceoff players to NHL player ids.

Primary reference: the latest dated roster snapshot
data/raw/rosters/<date>/rosters.csv (neurhl/live/fetch_rosters.py). For each
DailyFaceoff row (team_abbrev, name, jersey), in order:

  alias    data/manual/df_alias.csv (df_name, team, player_id); a blank team
           matches any team. Consulted first so a manual entry can also
           override a wrong automatic match.
  jersey   unique roster player with the same (team, sweater) whose name is
           consistent (same normalized last name, or same first name and
           overlapping last-name tokens). A jersey hit with an inconsistent
           name is NOT accepted (camp numbers change) and falls through.
  name     unique roster player on the same team with the same normalized
           name (lowercase, accents and punctuation stripped, Jr./Sr./II/III
           dropped, hyphens as spaces).
  name_lg  unique normalized-name match on another team's current roster
           (player moved; visible as roster_team != team_abbrev).
  fuzzy    unique same-team player with the same normalized last name and a
           compatible first name (prefix, e.g. Alex/Alexander, or same initial
           when the last name is unique on the team).

Fallback references (read-only), for players DailyFaceoff lists who are NOT on
the current NHL roster endpoint. The endpoint drops injured players and
players assigned to the minors (e.g. Bedard on IR, Milic sent down), which
DailyFaceoff still shows in its lines/IR groups:

  august   the frozen 1.0 roster files data/raw/nhl_roster_{TEAM}_20262027.json
  lookup   data/raw/mp_lookup.csv (NHL ids with last known team)

Both match by normalized name, same team first, else a league-unique name.
roster_team is the fallback source's team, and on_roster says whether the
player is on the primary (current) roster.

Returns (mapped frame, unmatched frame). Unmatched rows need a df_alias.csv
entry.

CLI: python neurhl/live/id_map.py [--date SNAPSHOT_DATE] [--rosters ROSTER_DATE] [--out CSV]
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJ, RAW  # noqa: E402
# siblings: relative when imported as neurhl.live.id_map; plain when run as a script.
# (Never `import live.*`: common.py puts src/ first on sys.path and src/live.py shadows it.)
try:
    from .fetch_rosters import latest_rosters, load_august
    from .parse_dailyfaceoff import parse_lines, snapshot_dir, ev_role
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_rosters import latest_rosters, load_august  # noqa: E402
    from parse_dailyfaceoff import parse_lines, snapshot_dir, ev_role  # noqa: E402

ALIAS_CSV = PROJ / "data" / "manual" / "df_alias.csv"
MP_LOOKUP = RAW / "mp_lookup.csv"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def norm(name) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z ]", "", s)            # drops . ' , and anything else
    toks = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(toks)


def _split(n: str) -> tuple[str, str]:
    toks = n.split()
    return (toks[0], " ".join(toks[1:])) if len(toks) > 1 else ("", n)


def _consistent(df_norm: str, r_first: str, r_last: str) -> bool:
    f, l = _split(df_norm)
    if l == r_last:
        return True
    return bool(f == r_first and set(l.split()) & set(r_last.split()))


def load_aliases(path: Path = ALIAS_CSV) -> dict[tuple[str, str], int]:
    if not path.exists():
        return {}
    a = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {(norm(r.df_name), r.team.strip().upper()): int(r.player_id)
            for r in a.itertuples() if r.df_name and r.player_id}


def load_fallback() -> list[tuple[str, pd.DataFrame]]:
    """[(method, frame[player_id, team, full])] in priority order."""
    aug = load_august()
    aug = pd.DataFrame({"player_id": aug["player_id"], "team": aug["team"],
                        "full": aug["first"] + " " + aug["last"]})
    out = [("august", aug)]
    if MP_LOOKUP.exists():
        mp = pd.read_csv(MP_LOOKUP, usecols=["playerId", "name", "team"])
        out.append(("lookup", pd.DataFrame({"player_id": mp["playerId"].astype(int),
                                            "team": mp["team"], "full": mp["name"]})))
    return out


def _prep(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    if "full" not in r:
        r["full"] = r["first"] + " " + r["last"]
    r["n_full"] = r["full"].map(norm)
    return r


def map_ids(df: pd.DataFrame, rosters: pd.DataFrame, aliases: dict | None = None,
            fallback: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add player_id, match_method, roster_team, roster_name, on_roster to a DailyFaceoff frame."""
    aliases = load_aliases() if aliases is None else aliases
    fallback = load_fallback() if fallback is None else fallback
    r = _prep(rosters)
    r["n_first"] = r["first"].map(norm)
    r["n_last"] = r["last"].map(norm)
    fb = [(m, _prep(f)) for m, f in fallback]
    on_roster = set(int(x) for x in r["player_id"])
    by_id = r.drop_duplicates("player_id").set_index("player_id")

    def hit(row, method):
        return (int(row.player_id), method, row.team, row.full)

    def resolve(team, name, jersey):
        n = norm(name)
        pid = aliases.get((n, team), aliases.get((n, "")))
        if pid is not None:
            if pid in by_id.index:
                b = by_id.loc[pid]
                return pid, "alias", b["team"], b["full"]
            return pid, "alias", None, None
        tr = r[r["team"] == team]
        if pd.notna(jersey):
            j = tr[tr["sweater"] == jersey]
            if len(j) == 1 and _consistent(n, j.iloc[0]["n_first"], j.iloc[0]["n_last"]):
                return hit(j.iloc[0], "jersey")
        m = tr[tr["n_full"] == n]
        if len(m) == 1:
            return hit(m.iloc[0], "name")
        lg = r[r["n_full"] == n].drop_duplicates("player_id")
        if len(lg) == 1:
            return hit(lg.iloc[0], "name_lg")
        f, l = _split(n)
        same_last = tr[tr["n_last"] == l]
        comp = same_last[same_last["n_first"].map(
            lambda rf: bool(f) and (rf.startswith(f) or f.startswith(rf)))]
        if len(comp) == 1:
            return hit(comp.iloc[0], "fuzzy")
        if len(same_last) == 1 and f and same_last.iloc[0]["n_first"][:1] == f[:1]:
            return hit(same_last.iloc[0], "fuzzy")
        for method, ref in fb:
            m = ref[ref["n_full"] == n].drop_duplicates("player_id")
            mt = m[m["team"] == team]
            if len(mt) == 1:
                return hit(mt.iloc[0], method)
            if len(m) == 1:
                return hit(m.iloc[0], method)
        return None, None, None, None

    keys = df[["team_abbrev", "name", "jersey"]].drop_duplicates()
    res = {(t, nm, jy if pd.notna(jy) else None): resolve(t, nm, jy)
           for t, nm, jy in keys.itertuples(index=False)}
    out = df.copy()
    vals = [res[(t, nm, jy if pd.notna(jy) else None)]
            for t, nm, jy in out[["team_abbrev", "name", "jersey"]].itertuples(index=False)]
    out["player_id"] = pd.array([v[0] for v in vals], dtype="Int64")
    out["match_method"] = [v[1] for v in vals]
    out["roster_team"] = [v[2] for v in vals]
    out["roster_name"] = [v[3] for v in vals]
    out["on_roster"] = [v[0] is not None and int(v[0]) in on_roster for v in vals]
    unmatched = (out[out["player_id"].isna()]
                 .drop_duplicates(["team_abbrev", "name"])
                 [["team_abbrev", "name", "jersey", "group", "slot"]]
                 .reset_index(drop=True))
    return out, unmatched


def main():
    ap = argparse.ArgumentParser(description="Map DailyFaceoff lines to NHL player ids.")
    ap.add_argument("--date", help="DailyFaceoff snapshot date (default: latest)")
    ap.add_argument("--rosters", help="roster snapshot date to use (default: latest)")
    ap.add_argument("--out", help="write the mapped frame to this CSV")
    a = ap.parse_args()
    d = snapshot_dir(a.date)
    rday, rosters = latest_rosters(a.rosters)
    lines = parse_lines(d)
    mapped, unmatched = map_ids(lines, rosters)

    role = ev_role(mapped)
    core = mapped[role.ne("")].drop_duplicates(["team_abbrev", "name"])
    allp = mapped.drop_duplicates(["team_abbrev", "name"])
    pct = lambda s: f"{s.sum()}/{len(s)} ({100 * s.mean():.2f}%)"  # noqa: E731
    print(f"DailyFaceoff {d.name} vs rosters {rday}")
    print(f"EV F/D/G players matched:          {pct(core['player_id'].notna())}")
    print(f"  ...matched on the current roster: {pct(core['on_roster'])}")
    print(f"all players (incl. IR, PP/PK-only): {pct(allp['player_id'].notna())}")
    print("methods (unique players):", allp["match_method"].value_counts(dropna=False).to_dict())
    moved = allp[allp["player_id"].notna() & allp["roster_team"].notna()
                 & (allp["roster_team"] != allp["team_abbrev"])]
    if len(moved):
        print(f"\nmatched to a different team ({len(moved)}):")
        print(moved[["team_abbrev", "name", "roster_team", "roster_name", "match_method"]]
              .to_string(index=False))
    off = allp[allp["player_id"].notna() & ~allp["on_roster"]]
    if len(off):
        print(f"\nlisted by DailyFaceoff but NOT on the current NHL roster ({len(off)}):")
        print(off[["team_abbrev", "name", "group", "slot", "player_id", "match_method"]]
              .to_string(index=False))
    fz = allp[allp["match_method"].eq("fuzzy")]
    if len(fz):
        print(f"\nfuzzy matches ({len(fz)}), please eyeball (add to df_alias.csv once checked):")
        print(fz[["team_abbrev", "name", "roster_name", "player_id"]].to_string(index=False))
    if len(unmatched):
        print(f"\nunmatched ({len(unmatched)}):")
        print(unmatched.to_string(index=False))
    if a.out:
        mapped.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
