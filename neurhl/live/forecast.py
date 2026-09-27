"""NeurHL LIVE: forecast entry point (PLAN_NeurHL4 LIVE). PLACEHOLDER.

Called by launchd:
  com.neurhl.morning  (11:00 ET, via neurhl/live/run_morning.sh)   --mode morning
  com.neurhl.pregame  (every 10 min, 11:00-23:50 ET)                --mode pregame

The real implementation (lineups via neurhl/live/lineup_resolver.py, NeurHL-G /
NeurHL-H / Elo forecasts, neurhl/output/live/ files, neurhl/live/publish.sh) is
still to be written. Until then this only logs that it ran.

CLI: python neurhl/live/forecast.py --mode {morning,pregame,nightly}
"""
import argparse
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["morning", "pregame", "nightly"], required=True)
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    print(f"[forecast {now:%Y-%m-%dT%H:%M:%SZ}] mode={a.mode}: placeholder, no forecast produced")


if __name__ == "__main__":
    main()
