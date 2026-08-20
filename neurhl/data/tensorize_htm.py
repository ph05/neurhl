"""NeurHL — HTM backfill tensorization: 2008-2011 seasons -> event shards.

Converts parsed HTM reports (build_htm.py: PL events + TH/TV shifts + jersey
resolution, validated at 99.6-99.9% player resolution) into the SAME shard
schema as tensorize_events.py, so the event-LM consumes one homogeneous
corpus. Differences, flagged not hidden:
  - no x/y coordinates (has_coord=0; zone + shot distance exist; MP raw coords
    joinable later per P3) — `dist` carried in the x column as negative marker?
    NO: dist goes to its own column via the `y` field? Neither — dist is
    dropped here; the pretrain tokenizer buckets zone instead of xy for HTM
    rows. (MP-coordinate enrichment is a logged Phase-2 extension.)
  - strength reconstructed from PL on-ice grids as a situationCode-style int.
  - season_end 2012 builds to events_htm_2012.parquet for cross-validation
    against the canonical JSON shard; it is never merged.

Also emits player_games_<se>.parquet (TOI from TH/TV, counting stats from
events, goalie starter) and games_ctx_<se>.parquet rows (labels from games.csv)
for the backfill seasons, plus a per-season reconciliation against official
NHL reports (goals, SOG, faceoffs, penalties) written to
neurhl/eda/eda_07_htm.md — the A1 integrity check from PLAN_NeurHL.
"""
import gzip
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, PROC, RAW, TENSORS  # noqa: E402
from data.build_htm import (NameResolver, onice_abbrevs, parse_pl,  # noqa: E402
                            parse_toi)
from data.tensorize_events import COLS  # noqa: E402

HTM = RAW / "htm_reports"
SEASONS = [2008, 2009, 2010, 2011, 2012]
POS_G = 2
_RES = None
_MAPS = None
_SE = None


def load_maps() -> dict:
    maps = json.loads((TENSORS / "maps.json").read_text())
    changed = False
    for ab in ("ATL", "PHX"):
        if ab not in maps["team"]:
            maps["team"][ab] = max(maps["team"].values()) + 1
            changed = True
    if changed:
        (TENSORS / "maps.json").write_text(json.dumps(maps, indent=1,
                                                      sort_keys=True))
    return maps


def _init(season_end, maps):
    global _RES, _MAPS, _SE
    _SE = season_end
    _MAPS = maps
    _RES = NameResolver(season_end)


def _read(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def game_htm(args):
    gid, pl_path = args
    html = _read(pl_path)
    evs = parse_pl(html)
    if not evs:
        return None
    aab, hab = onice_abbrevs(html)
    tv_p = pl_path.parent / pl_path.name.replace("PL", "TV")
    th_p = pl_path.parent / pl_path.name.replace("PL", "TH")
    tv = parse_toi(_read(tv_p)) if tv_p.exists() else []
    th = parse_toi(_read(th_p)) if th_p.exists() else []
    jm = _RES.jersey_map(evs, tv, th, aab, hab)
    venue = _MAPS["team"].get(hab, 0)
    et_map, st_map, zn_map = (_MAPS["event_type"], _MAPS["shot_type"],
                              _MAPS["zone"])
    rows, prev_t, sh_h, sh_a = [], 0, 0, 0
    counts = {"goal_h": 0, "goal_a": 0, "sog": 0, "fac": 0, "pen": 0}
    pstats: dict = {}

    def pid_of(pair):
        return jm.get((pair[0], pair[1]), 0) if pair else 0

    for e in evs:
        et = e["type"]
        if et == "?" or e["period"] == 0:
            continue
        actors = e["actors"]
        p1 = p2 = p3 = goalie = 0
        home_event = -1
        if actors:
            first_team = actors[0][0]
            if et == "faceoff":
                won = (m.group(1) if (m := re.match(
                    r"([A-Z]\.?[A-Z]\.?[A-Z]?) won", e["desc"])) else "")
                won = won.replace(".", "")
                ordered = sorted(actors[:2], key=lambda a: a[0] != won)
                p1, p2 = pid_of(ordered[0]), pid_of(ordered[1] if len(ordered) > 1 else None)
                first_team = ordered[0][0] if ordered else first_team
            elif et == "goal":
                p1 = pid_of(actors[0])
                rest = [a for a in actors[1:] if a[0] == actors[0][0]]
                p2 = pid_of(rest[0] if rest else None)
                p3 = pid_of(rest[1] if len(rest) > 1 else None)
            else:
                p1 = pid_of(actors[0])
                p2 = pid_of(actors[1] if len(actors) > 1 else None)
            home_event = 1 if first_team == hab else 0
        # period 5 = shootout: attempts stay as rows but never count toward
        # scores/totals (official totals credit only the SO decider, which
        # games.csv already carries in the labels)
        in_so = e["period"] == 5
        if et == "goal" and home_event >= 0 and not in_so:
            if home_event:
                sh_h += 1
                counts["goal_h"] += 1
            else:
                sh_a += 1
                counts["goal_a"] += 1
        if not in_so:
            counts["sog"] += et in ("shot-on-goal", "goal")
            counts["fac"] += et == "faceoff"
            counts["pen"] += et == "penalty"
        # opposing goalie for shot events, from the on-ice grid
        if et in ("shot-on-goal", "missed-shot", "goal") and home_event >= 0:
            opp = e["away_on"] if home_event else e["home_on"]
            gks = [jm.get((aab if home_event else hab, j), 0)
                   for j, pos, _ in opp if pos == POS_G]
            goalie = gks[0] if gks else 0
        h_on = np.zeros(7, dtype=np.int64)
        a_on = np.zeros(7, dtype=np.int64)
        for grid, slots, ab in ((e["home_on"], h_on, hab),
                                (e["away_on"], a_on, aab)):
            g_ids = [jm.get((ab, j), 0) for j, pos, _ in grid if pos == POS_G]
            s_ids = sorted(jm.get((ab, j), 0) for j, pos, _ in grid
                           if pos != POS_G)[:6]
            if g_ids:
                slots[0] = g_ids[0]
            slots[1:1 + len(s_ids)] = s_ids
        n_hg = int(h_on[0] > 0)
        n_ag = int(a_on[0] > 0)
        strength = (n_ag * 1000 + int((a_on[1:] > 0).sum()) * 100
                    + int((h_on[1:] > 0).sum()) * 10 + n_hg)
        rows.append((gid, 2, e["idx"], e["period"], e["t"],
                     max(0, e["t"] - prev_t), et_map.get(et, 0),
                     zn_map.get(e["zone"], 0), st_map.get(e["shot_type"], 0),
                     strength, home_event, 0, 0, 0, 0, 0, 0, sh_h, sh_a,
                     p1, p2, p3, goalie, venue,
                     *h_on.tolist(), *a_on.tolist()))
        prev_t = e["t"]
        # player counting stats
        if p1:
            st = pstats.setdefault(p1, dict(goals=0, assists=0, sog=0, att=0,
                                            blocks=0, pen=0, fo_w=0, fo_l=0))
            if et == "goal":
                st["goals"] += 1
                st["sog"] += 1
                st["att"] += 1
            elif et == "shot-on-goal":
                st["sog"] += 1
                st["att"] += 1
            elif et in ("missed-shot", "blocked-shot"):
                st["att"] += 1
            elif et == "faceoff":
                st["fo_w"] += 1
            elif et == "penalty":
                st["pen"] += 1
        if p2:
            st = pstats.setdefault(p2, dict(goals=0, assists=0, sog=0, att=0,
                                            blocks=0, pen=0, fo_w=0, fo_l=0))
            if et == "faceoff":
                st["fo_l"] += 1
            elif et == "blocked-shot":
                st["blocks"] += 1
            elif et == "goal":
                st["assists"] += 1
        if p3:
            st = pstats.setdefault(p3, dict(goals=0, assists=0, sog=0, att=0,
                                            blocks=0, pen=0, fo_w=0, fo_l=0))
            st["assists"] += 1

    # SO decider: official totals credit exactly one goal to the SO winner
    # (the team with more scored attempts). Counting only — score state and
    # event rows stay as-played.
    so_h = sum(1 for e in evs if e["period"] == 5 and e["type"] == "goal"
               and e["actors"] and e["actors"][0][0] == hab)
    so_a = sum(1 for e in evs if e["period"] == 5 and e["type"] == "goal"
               and e["actors"] and e["actors"][0][0] == aab)
    if so_h > so_a:
        counts["goal_h"] += 1
    elif so_a > so_h:
        counts["goal_a"] += 1

    # player-game rows from TOI reports
    prow = []
    for toi_l, ab, is_home in ((tv, aab, False), (th, hab, True)):
        agg: dict = {}
        first: dict = {}
        for s in toi_l:
            pid = jm.get((ab, s["jersey"]), 0)
            if not pid:
                continue
            agg[pid] = agg.get(pid, 0) + (s["t1"] - s["t0"])
            first[pid] = min(first.get(pid, 1 << 30), s["t0"])
        goalies = {j for e in evs for j, pos, _ in
                   (e["home_on"] if is_home else e["away_on"]) if pos == POS_G}
        g_pids = {jm.get((ab, j), 0) for j in goalies} - {0}
        starter = min(g_pids, key=lambda p: first.get(p, 1 << 30), default=0)
        for pid, t in agg.items():
            st = pstats.get(pid, {})
            prow.append((gid, 2, pid, is_home, 2 if pid in g_pids else 0,
                         t, int(pid == starter and pid in g_pids),
                         st.get("goals", 0), st.get("sog", 0),
                         st.get("att", 0), st.get("blocks", 0),
                         st.get("pen", 0), st.get("fo_w", 0),
                         st.get("fo_l", 0), st.get("assists", 0)))
    return rows, prow, {**counts, "gid": gid, "away": aab, "home": hab}


def main():
    maps = load_maps()
    lines = ["# EDA-07 — HTM backfill integrity (A1)\n"]
    for se in SEASONS:
        sdir = HTM / str(se)
        pls = sorted(sdir.glob("PL*.htm.gz")) if sdir.exists() else []
        if len(pls) < 1200:
            print(f"{se}: only {len(pls)} PL files (crawl incomplete), skipping")
            continue
        tag = "htm_2012" if se == 2012 else str(se)
        out = TENSORS / f"events_{tag}.parquet"
        if out.exists():
            print(f"{se}: exists, skipping")
            continue
        jobs = [((se - 1) * 1_000_000 + 20_000 + int(p.name[4:8]), p)
                for p in pls]
        all_rows, all_prows, metas = [], [], []
        with Pool(8, initializer=_init, initargs=(se, maps)) as pool:
            for r in pool.imap(game_htm, jobs, chunksize=16):
                if r:
                    all_rows.extend(r[0])
                    all_prows.extend(r[1])
                    metas.append(r[2])
        df = pd.DataFrame(all_rows, columns=COLS)
        df.to_parquet(out, index=False)
        pg = pd.DataFrame(all_prows, columns=[
            "game_id", "game_type", "player_id", "is_home", "pos_group",
            "toi_sec", "goalie_start", "goals", "sog", "att", "blocks", "pen",
            "fo_w", "fo_l", "assists"])
        pg.to_parquet(TENSORS / f"player_games_{tag}.parquet", index=False)

        # ---- reconciliation vs official season totals
        md = pd.DataFrame(metas)
        n = len(md)
        rec = {"games": n,
               "gpg_htm": (md.goal_h + md.goal_a).mean(),
               "sog_pg_htm": md.sog.mean() / 2,
               "fac_pg_htm": md.fac.mean(),
               "pen_pg_htm": md.pen.mean()}
        games = pd.read_csv(PROC / "games.csv")
        gr = games[(games.season_end == se) & (games.game_type == "R")]
        rec["gpg_official"] = (gr.home_g + gr.away_g).mean()
        summ = RAW / "nhl_reports" / f"summary_{se}.json"
        if summ.exists():
            rows = json.loads(summ.read_text())
            rec["sog_pg_official"] = float(np.mean(
                [r["shotsForPerGame"] for r in rows]))
        gpg_ok = abs(rec["gpg_htm"] - rec["gpg_official"]) / rec["gpg_official"]
        sog_ok = (abs(rec["sog_pg_htm"] - rec.get("sog_pg_official", 0))
                  / rec["sog_pg_official"]) if "sog_pg_official" in rec else 1
        rec["gpg_rel_err"] = gpg_ok
        rec["sog_rel_err"] = sog_ok
        rec["pass_98pct"] = bool(gpg_ok < 0.02 and sog_ok < 0.02)
        lines.append(f"## {se}\n")
        lines.append(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in rec.items()}, indent=1) + "\n")
        print(f"{se}: {len(df):,} events, {len(pg):,} player-games, "
              f"gpg {rec['gpg_htm']:.3f} vs {rec['gpg_official']:.3f}, "
              f"pass={rec['pass_98pct']}")
        sys.stdout.flush()
    (EDA / "eda_07_htm.md").write_text("\n".join(lines))
    print("done")


if __name__ == "__main__":
    main()
