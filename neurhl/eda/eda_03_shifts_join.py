"""NeurHL EDA-03 — shifts→PBP on-ice join validation (PLAN_NeurHL Phase 0).

Prototypes the exact on-ice reconstruction build_onice.py will use, on a random
per-season sample, and validates it three ways:
  1. internal: on-ice skater/goalie counts vs the event's situationCode
     (two boundary conventions — shift intervals are half-open on one side and
     events at faceoffs sit exactly on shift boundaries);
  2. external: exact on-ice set agreement vs fastRhockey's pre-reconstructed
     home_on_*/away_on_* columns on the 2012-2023 overlap (P4: check, never input);
  3. inventory: shift files missing relative to the PBP corpus, enumerated.
Writes eda/eda_03_shifts_join.md.
"""
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, RAW  # noqa: E402

N_SAMPLE = 30           # games per season for the situationCode check
N_FR = 5                # games per season for the fastRhockey set check
SEED = 711
SKIP_EVENTS = {"period-start", "period-end", "game-end", "stoppage",
               "shootout-complete", "delayed-penalty"}


def mmss(s: str) -> int:
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


def load_game(gid_path: Path) -> dict:
    with gzip.open(gid_path, "rt") as f:
        return json.load(f)


def shift_intervals(shift_path: Path, goalies: set):
    with gzip.open(shift_path, "rt") as f:
        recs = json.load(f)
    if isinstance(recs, dict):
        recs = recs.get("data", [])
    out = []
    for r in recs:
        if r.get("typeCode") != 517 or not r.get("startTime") or not r.get("endTime"):
            continue
        per = r["period"]
        if per >= 5:
            continue
        t0 = (per - 1) * 1200 + mmss(r["startTime"])
        t1 = (per - 1) * 1200 + mmss(r["endTime"])
        if t1 <= t0:
            continue
        out.append((r["teamId"], r["playerId"], r["playerId"] in goalies, t0, t1))
    return out


def onice_counts(iv: np.ndarray, is_goalie: np.ndarray, team_is_home: np.ndarray,
                 t: int, conv: str):
    if conv == "A":
        on = (iv[:, 0] <= t) & (t < iv[:, 1])
    else:
        on = (iv[:, 0] < t) & (t <= iv[:, 1])
    h_sk = int((on & team_is_home & ~is_goalie).sum())
    a_sk = int((on & ~team_is_home & ~is_goalie).sum())
    h_g = int((on & team_is_home & is_goalie).sum())
    a_g = int((on & ~team_is_home & is_goalie).sum())
    return a_g, a_sk, h_sk, h_g


def check_game(pbp_path: Path, shift_path: Path):
    g = load_game(pbp_path)
    home_id = g["homeTeam"]["id"]
    goalies = {r["playerId"] for r in g.get("rosterSpots", [])
               if r.get("positionCode") == "G"}
    shifts = shift_intervals(shift_path, goalies)
    if not shifts:
        return None
    team = np.array([s[0] for s in shifts])
    is_g = np.array([s[2] for s in shifts])
    iv = np.array([(s[3], s[4]) for s in shifts], dtype=float)
    is_home = team == home_id
    res = {"A": [0, 0], "B": [0, 0], "either": [0, 0]}
    for p in g.get("plays", []):
        et = p.get("typeDescKey", "")
        sc = p.get("situationCode")
        per = p.get("periodDescriptor", {}).get("number", 0)
        if et in SKIP_EVENTS or sc is None or not str(sc).isdigit() or len(str(sc)) != 4 \
                or per >= 5:
            continue
        t = (per - 1) * 1200 + mmss(p["timeInPeriod"])
        sc = str(sc)
        want = (int(sc[0]), int(sc[1]), int(sc[2]), int(sc[3]))
        got_a = onice_counts(iv, is_g, is_home, t, "A")
        got_b = onice_counts(iv, is_g, is_home, t, "B")
        for name, hit in (("A", got_a == want), ("B", got_b == want),
                          ("either", got_a == want or got_b == want)):
            res[name][0] += hit
            res[name][1] += 1
    return res


def main():
    rng = random.Random(SEED)
    lines = ["# EDA-03 — shifts→PBP on-ice join validation\n"]

    # ---- inventory of gaps
    lines.append("## Shift-file inventory vs PBP corpus\n")
    inv = []
    for sdir in sorted((RAW / "pbp").iterdir()):
        se = sdir.name
        pbp_ids = {p.name for p in sdir.glob("*.json.gz")}
        shift_dir = RAW / "shifts" / se
        shift_ids = {p.name for p in shift_dir.glob("*.json.gz")} if shift_dir.exists() else set()
        inv.append((se, len(pbp_ids), len(shift_ids), len(pbp_ids - shift_ids)))
    inv_df = pd.DataFrame(inv, columns=["season_end", "pbp_games", "shift_files",
                                        "missing_shifts"])
    lines.append(inv_df.to_markdown(index=False) + "\n")

    # ---- situationCode agreement on a sample
    rows = []
    for sdir in sorted((RAW / "pbp").iterdir()):
        se = sdir.name
        cands = [p for p in sdir.glob("*.json.gz")
                 if (RAW / "shifts" / se / p.name).exists()]
        if not cands:
            continue
        sample = rng.sample(cands, min(N_SAMPLE, len(cands)))
        agg = {"A": [0, 0], "B": [0, 0], "either": [0, 0]}
        for p in sample:
            r = check_game(p, RAW / "shifts" / se / p.name)
            if r is None:
                continue
            for k in agg:
                agg[k][0] += r[k][0]
                agg[k][1] += r[k][1]
        rows.append((int(se), len(sample),
                     agg["A"][0] / max(agg["A"][1], 1),
                     agg["B"][0] / max(agg["B"][1], 1),
                     agg["either"][0] / max(agg["either"][1], 1),
                     agg["A"][1]))
    ag = pd.DataFrame(rows, columns=["season_end", "games", "conv_A", "conv_B",
                                     "either", "events"])
    lines.append("## situationCode agreement (sampled games; A: start<=t<end, "
                 "B: start<t<=end)\n")
    lines.append(ag.round(4).to_markdown(index=False) + "\n")
    lines.append(f"\nOverall either-convention agreement: "
                 f"**{(ag.either * ag.events).sum() / ag.events.sum():.4f}** "
                 f"(events sampled: {int(ag.events.sum()):,}). The tensorizer will "
                 f"use the better single convention per event type (faceoffs sit on "
                 f"boundaries) and fall back to situationCode as truth for counts.\n")

    # ---- fastRhockey skater-count check (P4 — sampled overlap)
    # EDA finding: fR parquet schemas drift per file (home_on_7/away_on_7 appear
    # in some files only, on-ice slots hold NAMES as strings, not ids), so the
    # robust cross-check is the home_skaters/away_skaters count columns, which
    # exist in every file. Season is derived from the file's own game ids.
    fr_dir = RAW / "fastrhockey"
    fr_rows = []
    for f in sorted(fr_dir.glob("play_by_play_*.parquet")):
        try:
            fr = pd.read_parquet(f, columns=["game_id", "period", "game_seconds",
                                             "home_skaters", "away_skaters",
                                             "event_type"])
        except Exception as e:
            fr_rows.append((f.name, 0, float("nan"), f"load failed: {type(e).__name__}"))
            continue
        gids = fr.game_id.dropna().astype(int)
        reg = gids[(gids // 10000) % 100 == 2]           # gameType 2 only
        if reg.empty:
            fr_rows.append((f.name, 0, float("nan"), "no regular-season ids"))
            continue
        se = str(int(str(reg.iloc[0])[:4]) + 1)          # season_end from gid
        have = [g for g in reg.unique()
                if (RAW / "pbp" / se / f"{g}.json.gz").exists()
                and (RAW / "shifts" / se / f"{g}.json.gz").exists()]
        take = rng.sample(list(have), min(N_FR, len(have))) if have else []
        n_ev, n_match = 0, 0
        for gid in take:
            g = load_game(RAW / "pbp" / se / f"{gid}.json.gz")
            goalies = {r["playerId"] for r in g.get("rosterSpots", [])
                       if r.get("positionCode") == "G"}
            shifts = shift_intervals(RAW / "shifts" / se / f"{gid}.json.gz", goalies)
            if not shifts:
                continue
            team = np.array([s[0] for s in shifts])
            is_g = np.array([s[2] for s in shifts])
            iv = np.array([(s[3], s[4]) for s in shifts], dtype=float)
            is_home = team == g["homeTeam"]["id"]
            sub = fr[(fr.game_id == gid) & (fr.period <= 4)].dropna(
                subset=["game_seconds", "home_skaters", "away_skaters"])
            sub = sub[~sub.event_type.isin(["PERIOD_START", "PERIOD_END",
                                            "GAME_END", "STOP"])]
            if len(sub) > 120:
                sub = sub.sample(120, random_state=SEED)
            for _, r in sub.iterrows():
                t = float(r.game_seconds)
                want = (int(r.home_skaters), int(r.away_skaters))
                hit = False
                for conv in ("A", "B"):
                    a_g, a_sk, h_sk, h_g = onice_counts(iv, is_g, is_home, t, conv)
                    if (h_sk, a_sk) == want or (h_sk + h_g, a_sk + a_g) == want:
                        hit = True
                        break
                n_ev += 1
                n_match += hit
        fr_rows.append((f.name, len(take), n_match / max(n_ev, 1), f"{n_ev} events, se={se}"))
    frd = pd.DataFrame(fr_rows, columns=["file", "games", "count_agreement", "note"])
    lines.append("## fastRhockey skater-count agreement (P4 check — sampled)\n")
    lines.append(frd.round(4).to_markdown(index=False) + "\n")
    lines.append("\nNote: fR on-ice slots store player NAMES (strings) with per-file "
                 "schema drift, so the durable cross-check is skater counts (either "
                 "boundary convention, goalie-inclusive or not). The situationCode "
                 "check above remains the authoritative internal validation.\n")

    lines.append("HTM backfill (A1) validation is appended in Phase 1 once the crawl "
                 "completes: TH/TV-derived intervals for 2008-2011 + the 57-game "
                 "2024-25 gap run through the identical checks.\n")
    (EDA / "eda_03_shifts_join.md").write_text("\n".join(lines))
    print("wrote eda_03_shifts_join.md")


if __name__ == "__main__":
    main()
