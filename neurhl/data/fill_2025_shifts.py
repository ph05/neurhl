"""NeurHL — synthesize the 57 missing 2024-25 shift charts from HTM TOI reports.

The shiftcharts API returned empty for the final ~2 weeks of 2024-25 (EDA-03);
the HTM TH/TV reports for those games were fetched by A1. This converter parses
them (build_htm.parse_toi), resolves players via NameResolver(2025), and writes
standard shift-chart JSON (typeCode 517 records) into data/raw/shifts/2025/ so
every downstream consumer (build_onice, tensorizers) sees a complete season.
Files are marked with "source": "htm_toi" inside each record. Idempotent.
"""
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW  # noqa: E402
from data.build_htm import NameResolver, parse_toi  # noqa: E402


def mmss_str(t: int, period: int) -> str:
    s = t - (period - 1) * 1200
    return f"{s // 60:02d}:{s % 60:02d}"


def main():
    pbp = {int(p.name.split(".")[0]) for p in (RAW / "pbp" / "2025").glob("*.json.gz")}
    shf = {int(p.name.split(".")[0]) for p in (RAW / "shifts" / "2025").glob("*.json.gz")}
    missing = sorted(pbp - shf)
    if not missing:
        print("no missing 2025 shift files")
        return
    res = NameResolver(2025)
    made, failed = 0, []
    for gid in missing:
        code = f"{gid % 1_000_000:06d}"
        with gzip.open(RAW / "pbp" / "2025" / f"{gid}.json.gz", "rt") as f:
            g = json.load(f)
        sides = {"TH": (g["homeTeam"]["id"], g["homeTeam"]["abbrev"]),
                 "TV": (g["awayTeam"]["id"], g["awayTeam"]["abbrev"])}
        recs = []
        for rpt, (team_id, ab) in sides.items():
            p = RAW / "htm_reports" / "2025" / f"{rpt}{code}.htm.gz"
            if not p.exists():
                continue
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                shifts = parse_toi(f.read())
            for s in shifts:
                pid = res.resolve(ab, s["name"])
                if not pid:
                    continue
                recs.append({"typeCode": 517, "playerId": pid,
                             "teamId": team_id, "period": s["period"],
                             "startTime": mmss_str(s["t0"], s["period"]),
                             "endTime": mmss_str(s["t1"], s["period"]),
                             "gameId": gid, "source": "htm_toi"})
        if len(recs) > 400:
            dest = RAW / "shifts" / "2025" / f"{gid}.json.gz"
            tmp = dest.with_suffix(".part")
            with gzip.open(tmp, "wt") as f:
                json.dump(recs, f, separators=(",", ":"))
            tmp.rename(dest)
            made += 1
        else:
            failed.append((gid, len(recs)))
    print(f"synthesized {made}/{len(missing)} shift files"
          + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
