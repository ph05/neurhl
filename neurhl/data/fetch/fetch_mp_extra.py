"""NeurHL-2 — fetch the MoneyPuck datasets we did NOT already have.

Already on disk before this: season summaries (skaters/goalies/teams), the
player lookup, and the per-shot corpus. Missing, and fetched here:

  * **game-by-game** skaters / goalies / lines -- per-player-per-GAME rows with
    situation splits (5v5 / 5v4 / 4v5 / other). We can now derive these from our
    own stint table, so their value is as an INDEPENDENT CROSS-CHECK, in the same
    way the HTM/JSON shift overlap validated the TH/TV parser at 0.32s MAE. A
    derived quantity nobody can check is a derived quantity nobody should trust.
  * **lines season summary** -- forward-line and defence-pair aggregates.
  * **data dictionaries** -- column definitions for both corpora.

P3 applies at USE time, not fetch time: these files contain MoneyPuck's fitted
xGoals columns, which are excluded from any feature by `parse_mp_shots.admissible`
and must stay excluded here too. Fetching them is fine; feeding them a model is
not.

Run: uv run --no-project --python 3.12 --with requests \
     python neurhl/data/fetch/fetch_mp_extra.py
"""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW  # noqa: E402

OUT = RAW / "mp_extra"
MP = "https://moneypuck.com/moneypuck"
PT = "https://peter-tanner.com/moneypuck/downloads"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

TARGETS = [
    (f"{PT}/MoneyPuckDataDictionaryForPlayers.csv", "dict_players.csv"),
    (f"{PT}/MoneyPuck_Shot_Data_Dictionary.csv", "dict_shots.csv"),
    (f"{MP}/playerData/playerBios/allPlayersLookup.csv", "player_bios.csv"),
    (f"{MP}/playerData/careers/gameByGame/all_teams.csv", "all_teams_gbg.csv"),
    (f"{PT}/seasonPlayersSummary/skaters/2008_to_2024.zip", "gbg_skaters.zip"),
    (f"{PT}/seasonPlayersSummary/goalies/2008_to_2024.zip", "gbg_goalies.zip"),
    (f"{PT}/seasonPlayersSummary/lines/2008_to_2024.zip", "gbg_lines.zip"),
]
# lines season summaries, one per season
TARGETS += [(f"{MP}/playerData/seasonSummary/{y}/regular/lines.csv",
             f"lines_{y + 1}.csv") for y in range(2008, 2025)]


def get(url: str, dest: Path, tries: int = 3) -> tuple:
    if dest.exists() and dest.stat().st_size > 0:
        return "skip", dest.stat().st_size
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=180)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return "ok", len(r.content)
            last = f"HTTP {r.status_code}"
        except Exception as e:                       # noqa: BLE001
            last = type(e).__name__
        time.sleep(2 * (k + 1))
    return f"FAIL ({last})", 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tot = 0
    for url, name in TARGETS:
        status, n = get(url, OUT / name)
        tot += n
        flag = "" if status in ("ok", "skip") else "  <-- "
        print(f"  {name:<24} {status:<16} {n/1e6:8.2f} MB{flag}")
        sys.stdout.flush()
    print(f"\n{tot/1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
