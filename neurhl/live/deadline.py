"""NeurHL LIVE: did a forecast commit precede its game's start? (PLAN_NeurHL4 LIVE, Q)

A forecast counts only if its commit is on origin/main before the game's
scheduled start (startTimeUTC). This module checks, through the GitHub API
(`gh`, already authenticated):

  commit_time(sha)    gh api repos/<repo>/commits/<sha> --jq .commit.committer.date
  on_main(sha)        compare <sha>...main is "ahead" or "identical" (sha is in main)
  stamp_time(sha)     created_at of the earliest stamp.yml Actions run for that
                      head sha (server-side time; push happened no later than this)
  check(start, sha)   dict with all of the above and `ok`

`ok` requires committer date < start and the commit being on main. The
committer date is written by the machine that made the commit, so the stamp
run is the server-side evidence; with require_stamp=True a missing or late
stamp run also fails the check. Note a stamp run exists only for pushes whose
head commit touched neurhl/output/live/**; for a multi-commit push the run is
keyed to the push's head sha, so the stamp is looked up for the given sha only.

CLI: python neurhl/live/deadline.py --sha SHA (--start ISO | --game GID)
                                    [--repo ph05/neurhl] [--require-stamp]
Exit 0 = precedes, 1 = does not, 2 = could not check.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

REPO = "ph05/neurhl"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _iso(s: str) -> datetime:
    t = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _gh(path: str, jq: str | None = None) -> str:
    cmd = ["gh", "api", path] + (["--jq", jq] if jq else [])
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"gh api {path}: {(p.stderr or p.stdout).strip()}")
    return p.stdout.strip()


def commit_time(sha: str, repo: str = REPO) -> datetime:
    return _iso(_gh(f"repos/{repo}/commits/{sha}", ".commit.committer.date"))


def on_main(sha: str, repo: str = REPO) -> bool:
    return _gh(f"repos/{repo}/compare/{sha}...main", ".status") in ("ahead", "identical")


def stamp_time(sha: str, repo: str = REPO, workflow: str = "stamp.yml") -> datetime | None:
    out = _gh(f"repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}&per_page=100",
              "[.workflow_runs[].created_at]")
    times = [_iso(t) for t in json.loads(out or "[]")]
    return min(times) if times else None


def game_start(game_id: int) -> datetime:
    req = urllib.request.Request(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/landing", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return _iso(json.load(r)["startTimeUTC"])


def check(start_utc, sha: str, repo: str = REPO, require_stamp: bool = False) -> dict:
    start = start_utc if isinstance(start_utc, datetime) else _iso(str(start_utc))
    ct = commit_time(sha, repo)
    res = {"sha": sha, "start_utc": start.isoformat(), "committer_utc": ct.isoformat(),
           "commit_precedes": ct < start, "on_main": on_main(sha, repo),
           "stamp_utc": None, "stamp_precedes": None}
    try:
        st = stamp_time(sha, repo)
    except RuntimeError as e:        # workflow not on the server yet, etc.
        st, res["stamp_error"] = None, str(e)
    if st is not None:
        res["stamp_utc"], res["stamp_precedes"] = st.isoformat(), st < start
    res["ok"] = bool(res["commit_precedes"] and res["on_main"]
                     and (not require_stamp or res["stamp_precedes"]))
    return res


def precedes(start_utc, sha: str, repo: str = REPO) -> bool:
    """True iff the commit's committer date is strictly before start_utc."""
    start = start_utc if isinstance(start_utc, datetime) else _iso(str(start_utc))
    return commit_time(sha, repo) < start


def main():
    ap = argparse.ArgumentParser(description="Check that a forecast commit precedes a game start.")
    ap.add_argument("--sha", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--start", help="game start, UTC ISO (e.g. 2026-09-29T21:00:00Z)")
    g.add_argument("--game", type=int, help="NHL game id; start taken from the NHL API")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--require-stamp", action="store_true",
                    help="also require a stamp.yml run for the sha created before the start")
    a = ap.parse_args()
    try:
        start = game_start(a.game) if a.game else _iso(a.start)
        res = check(start, a.sha, a.repo, a.require_stamp)
    except Exception as e:  # noqa: BLE001
        print(f"could not check: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
