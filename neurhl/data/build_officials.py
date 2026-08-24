"""NeurHL-3 — D3: parse HTM RO reports into officials + scratches tensors.

Fills the G4 pre-2012 hole from the newly fetched RO files: per game the
official SCRATCHES (jersey/pos/name per side), HEAD COACHES, and OFFICIALS
(2 referees + 2 linesmen, column-major in the RO layout). Names are stored
raw; the referee features key on names directly, and scratch->player_id
resolution happens at the consumer against season name maps.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_officials.py
"""
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

HTM = RAW / "htm_reports"
SEASONS = [2008, 2009, 2010, 2011, 2012]
CFG = {"version": 1}
_TAG = re.compile(r"<[^>]+>")
_OFF = re.compile(r"#(\d+)\s+(.+)")


def tokens(html: str) -> list:
    t = _TAG.sub("|", html).replace("&nbsp;", " ")
    return [x.strip() for x in t.split("|") if x.strip()]


def parse_ro(html: str):
    tk = tokens(html)

    def idx(label, start=0):
        try:
            return tk.index(label, start)
        except ValueError:
            return -1
    out_scr, coaches, refs, lines = [], [], [], []
    i = idx("Scratches")
    j = idx("Head Coaches")
    if i >= 0 and j > i:
        side = -1
        k = i
        while k < j:
            if tk[k] == "#" and k + 1 < j and tk[k + 1] == "Pos":
                side += 1
                k += 3
                continue
            if (side >= 0 and tk[k].isdigit() and k + 2 < j
                    and len(tk[k + 1]) <= 2):
                out_scr.append((min(side, 1), int(tk[k]), tk[k + 1],
                                tk[k + 2]))
                k += 3
                continue
            k += 1
    o = idx("Officials")
    if j >= 0 and o > j:
        coaches = [x for x in tk[j + 1:o]
                   if x.isupper() and len(x) > 4 and not x.isdigit()]
    s = idx("Standby", o if o >= 0 else 0)
    if o >= 0:
        end = s if s > o else min(o + 40, len(tk))
        offs = [m.group(2).strip() for x in tk[o:end]
                if (m := _OFF.match(x))]
        half = max(len(offs) // 2, 1)
        refs, lines = offs[:half], offs[half:]
    return out_scr, coaches, refs, lines


def build(season: int):
    d = HTM / str(season)
    files = sorted(d.glob("RO*.htm.gz"))
    off_rows, scr_rows = [], []
    for f in files:
        code = int(f.name[2:8])
        game_id = int(f"{season - 1}{code:06d}")
        html = gzip.open(f, "rt", errors="ignore").read()
        scr, coaches, refs, lines = parse_ro(html)
        off_rows.append((game_id,
                         refs[0] if len(refs) > 0 else "",
                         refs[1] if len(refs) > 1 else "",
                         lines[0] if len(lines) > 0 else "",
                         lines[1] if len(lines) > 1 else "",
                         coaches[0] if len(coaches) > 0 else "",
                         coaches[1] if len(coaches) > 1 else ""))
        for side, num, pos, name in scr:
            scr_rows.append((game_id, side, num, pos, name))
    off = pd.DataFrame(off_rows, columns=["game_id", "ref1", "ref2",
                                          "lines1", "lines2", "coach_v",
                                          "coach_h"])
    scr = pd.DataFrame(scr_rows, columns=["game_id", "is_home", "jersey",
                                          "pos", "name"])
    return off, scr


def main():
    for s in SEASONS:
        d = HTM / str(s)
        if not d.exists() or not list(d.glob("RO*.htm.gz")):
            print(f"{s}: no RO files, skipping")
            continue
        out_o = TENSORS / f"officials_{s}.parquet"
        out_s = TENSORS / f"scratches_{s}.parquet"
        probe = sorted(d.glob("RO*.htm.gz"))[:1]
        if MAN.is_fresh(out_o, probe, CFG):
            print(f"{s}: fresh, skipping")
            continue
        off, scr = build(s)
        off.to_parquet(out_o, index=False)
        scr.to_parquet(out_s, index=False)
        MAN.write_manifest(out_o, probe, CFG, {"n_games": len(off)})
        both_refs = float(((off.ref1 != "") & (off.ref2 != "")).mean())
        print(f"{s}: {len(off):,} games, both refs {both_refs:.1%}, "
              f"{len(scr):,} scratch rows "
              f"({len(scr) / max(len(off), 1):.2f}/game), coaches "
              f"{float((off.coach_v != '').mean()):.1%}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
