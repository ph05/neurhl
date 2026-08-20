"""NeurHL — HTM report parsers (A1 backfill, seasons 2008-2012 + shift gap-fill).

Parses the official HTM game reports into the same structures as the JSON-era
corpus:
  parse_pl(html)      -> per-event dicts (code, period, t, strength, teams,
                         actor/secondary jersey+name, zone, distance, shot type,
                         on-ice [jersey,pos,name] per side)
  parse_toi(html)     -> per-shift dicts (jersey, name, period, start, end)
  NameResolver        -> (team, season, FULL NAME) -> playerId, built from
                         data/raw/nhl_player_reports (official) with
                         player_landing as fallback

Era notes (verified on 2008/2010 files): PL on-ice cells carry
<font title="Position - FULL NAME">jersey</font>; descriptions carry dotted
team codes (L.A, N.J, S.J, T.B); TH/TV shift times are "elapsed / remaining".
No x/y coordinates exist in HTM — coordinates for 2008-2011 shots come from
MoneyPuck raw fields (P3), keyed by (game, time, shooter).
"""
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW  # noqa: E402

DOTTED = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
CODE_MAP = {"FAC": "faceoff", "SHOT": "shot-on-goal", "MISS": "missed-shot",
            "BLOCK": "blocked-shot", "GOAL": "goal", "HIT": "hit",
            "GIVE": "giveaway", "TAKE": "takeaway", "PENL": "penalty",
            "STOP": "stoppage", "PSTR": "period-start", "PEND": "period-end",
            "GEND": "game-end", "SOC": "shootout-complete",
            "DELPEN": "delayed-penalty", "CHL": "stoppage", "GOFF": "stoppage",
            "EISTR": "stoppage", "EIEND": "stoppage", "EGT": "stoppage",
            "EGPID": "stoppage"}
ZONE_MAP = {"Off": "O", "Def": "D", "Neu": "N"}
POS_MAP = {"Center": 0, "Left Wing": 0, "Right Wing": 0, "Defense": 1,
           "Goalie": 2}

_TAG = re.compile(r"<[^>]+>")
_FONT = re.compile(r'<font[^>]*title="([^"]+)"[^>]*>\s*(\d+)\s*</font>')
_TEAMJ = re.compile(
    r"([A-Z]\.?[A-Z]\.?[A-Z]?)\s+(?:ONGOAL - |GIVEAWAY - |TAKEAWAY - )?"
    r"#?(\d+)\s+([A-Z' .-]+?)(?:\(\d+\))?(?=,|$|\s(?:HIT|BLOCKED|vs|Drawn)|\s[A-Z][a-z])",
    re.M)
_ZONE = re.compile(r"(Off|Def|Neu)\. Zone")
_DIST = re.compile(r"(\d+) ft\.")
_MMSS = re.compile(r"(\d+):(\d\d)")


def strip(s: str) -> str:
    return _TAG.sub("", s).replace("&nbsp;", " ").strip()


def mmss(s: str) -> int:
    m = _MMSS.search(s)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def norm_team(code: str) -> str:
    return DOTTED.get(code, code.replace(".", ""))


def norm_name(s: str) -> str:
    """'GETZLAF, RYAN' or 'Ryan Getzlaf' -> 'RYAN GETZLAF' (accent-stripped)."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper().strip()
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return re.sub(r"\s+", " ", s)


def split_cells(chunk: str) -> list[str]:
    """Top-level <td> cells of a row chunk (nested tables kept inside cells)."""
    out, depth, start = [], 0, None
    for m in re.finditer(r"<(/?)(td|table|tr)[^>]*>", chunk):
        closing, tag = m.group(1) == "/", m.group(2)
        if tag == "table":
            depth += -1 if closing else 1
        elif tag == "td" and depth == 0:
            if not closing and start is None:
                start = m.end()
            elif closing and start is not None:
                out.append(chunk[start:m.start()])
                start = None
        elif tag == "tr" and closing and depth == 0 and out:
            break                              # end of this row
    return out


def onice_abbrevs(html: str) -> tuple[str, str]:
    """(away, home) team codes from the 'XXX On Ice' column headers."""
    hits = re.findall(r">\s*([A-Z]\.?[A-Z]\.?[A-Z]?)\s+On Ice\s*<", html)
    if len(hits) >= 2:
        return norm_team(hits[0]), norm_team(hits[1])
    return "", ""


def _fonts(cell: str) -> list[tuple]:
    out = []
    for title, jersey in _FONT.findall(cell):
        pos, _, name = title.partition(" - ")
        out.append((int(jersey), POS_MAP.get(pos.strip(), 0), norm_name(name)))
    return out


def parse_pl(html: str) -> list[dict]:
    chunks = re.split(r'<tr[^>]*class="evenColor"[^>]*>', html)[1:]
    events = []
    for ch in chunks:
        cells = split_cells(ch)
        if len(cells) < 6:
            continue
        num, per_s, strength, times, code, desc = [strip(c) for c in cells[:6]]
        if not num.isdigit():
            continue
        per = 4 if per_s in ("OT", "4") else (5 if per_s == "5" else
                                              int(per_s) if per_s.isdigit() else 0)
        t = (per - 1) * 1200 + mmss(times) if per else 0
        away_on = _fonts(cells[6]) if len(cells) > 6 else []
        home_on = _fonts(cells[7]) if len(cells) > 7 else []
        pairs = [(norm_team(tm), int(j), norm_name(nm))
                 for tm, j, nm in _TEAMJ.findall(desc)]
        zone = ZONE_MAP.get((_ZONE.search(desc) or [None, ""])[1], "")
        dist = int(d.group(1)) if (d := _DIST.search(desc)) else -1
        shot_type = ""
        for st in ("Wrist", "Slap", "Snap", "Backhand", "Tip-In", "Deflected",
                   "Wrap-around"):
            if f", {st}," in desc or desc.endswith(st):
                shot_type = st.lower()
                break
        events.append({"idx": int(num), "period": per, "t": t,
                       "strength": strength, "code": code,
                       "type": CODE_MAP.get(code, "?"), "desc": desc,
                       "actors": pairs, "zone": zone, "dist": dist,
                       "shot_type": shot_type,
                       "away_on": away_on, "home_on": home_on})
    return events


def parse_toi(html: str) -> list[dict]:
    out = []
    blocks = re.split(r'class="playerHeading[^"]*"[^>]*>', html)[1:]
    for blk in blocks:
        head = strip(blk.split("</td>")[0])
        m = re.match(r"(\d+)\s+(.+)", head)
        if not m:
            continue
        jersey, name = int(m.group(1)), norm_name(m.group(2))
        for row in re.findall(r'<tr class="[^"]*Color">(.*?)</tr>', blk, re.S):
            cells = [strip(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            if len(cells) < 5 or not cells[0].isdigit():
                continue
            per = 4 if cells[1] == "OT" else (int(cells[1]) if cells[1].isdigit() else 0)
            if not per or per >= 5:
                continue
            start = (per - 1) * 1200 + mmss(cells[2].split("/")[0])
            end = (per - 1) * 1200 + mmss(cells[3].split("/")[0])
            if end > start:
                out.append({"jersey": jersey, "name": name, "period": per,
                            "t0": start, "t1": end})
    return out


class NameResolver:
    """(team, FULL NAME) -> playerId for one season, official reports first."""

    def __init__(self, season_end: int):
        self.direct: dict = {}
        self.by_name: dict = {}
        self.by_last: dict = {}                 # (team, LAST) -> {pid}
        rep = RAW / "nhl_player_reports"
        for pos, fld in (("skater", "skaterFullName"), ("goalie", "goalieFullName")):
            for tag in ("", "_po"):
                p = rep / f"{pos}_summary_{season_end}{tag}.json.gz"
                if not p.exists():
                    continue
                with gzip.open(p, "rt") as f:
                    for r in json.load(f):
                        nm = norm_name(r[fld])
                        pid = r["playerId"]
                        last = nm.rsplit(" ", 1)[-1] if " " in nm else nm
                        for tm in str(r.get("teamAbbrevs") or "").split(","):
                            if tm:
                                self.direct[(tm.strip(), nm)] = pid
                                self.by_last.setdefault((tm.strip(), last),
                                                        set()).add(pid)
                        self.by_name.setdefault(nm, set()).add(pid)

    def jersey_map(self, events: list, toi_away: list, toi_home: list,
                   away_ab: str, home_ab: str) -> dict:
        """(team_abbrev, jersey) -> playerId for one game.

        TH/TV rosters are complete (every dressed player); PL on-ice fonts add
        anyone missing. Names are full, so resolution is near-exact.
        """
        pairs: dict = {}
        for toi, ab in ((toi_away, away_ab), (toi_home, home_ab)):
            for s in toi:
                pairs.setdefault((ab, s["jersey"]), s["name"])
        for e in events:
            for side, ab in (("away_on", away_ab), ("home_on", home_ab)):
                for jersey, _, name in e[side]:
                    pairs.setdefault((ab, jersey), name)
        out = {}
        for (ab, jersey), name in pairs.items():
            pid = self.resolve(ab, name)
            if pid:
                out[(ab, jersey)] = pid
        return out

    def resolve(self, team: str, name: str) -> int:
        pid = self.direct.get((team, name))
        if pid:
            return pid
        cands = self.by_name.get(name)
        if cands and len(cands) == 1:
            return next(iter(cands))
        # last-name fallback within team: covers first-name spelling variants
        # (NIKOLAI/NIKOLAY, ZACHERY/ZACK) and bare-last-name description forms
        last = name.rsplit(" ", 1)[-1]
        hits = self.by_last.get((team, last), set())
        if len(hits) == 1:
            return next(iter(hits))
        if " " not in name:                      # bare last name, any team
            all_hits = {p for nm, ps in self.by_name.items()
                        if nm.endswith(" " + name) for p in ps}
            if len(all_hits) == 1:
                return all_hits.pop()
        return 0
