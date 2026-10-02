"""ORR configuration: paths, the information cutoff, and league constants.

Every number that shapes a forecast lives here or in a fitted-parameter JSON
under orr/output/params/, never inline in model code.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repository root
PKG = Path(__file__).resolve().parent               # orr/
OUT = PKG / "output"
PARAMS = OUT / "params"
CACHE = PKG / "cache"                               # gitignored working cache

# ---------------------------------------------------------------------------
# Information cutoff for the 2026-27 preseason freeze.
# 2026-09-29 17:00 America/New_York (EDT, UTC-4) = 21:00 UTC, the first puck
# drop of the season. Every input the freeze reads from this repository is read
# from the tree of CUTOFF_COMMIT, the last commit authored before the cutoff,
# so nothing committed later (results, NeurHL 1.2/1.3, later injury research)
# can reach the preseason numbers. External files fetched afterwards
# (fastRhockey box scores) contain only seasons that ended in 2024 or earlier.
# ---------------------------------------------------------------------------
CUTOFF_UTC = datetime(2026, 9, 29, 21, 0, 0, tzinfo=timezone.utc)
CUTOFF_COMMIT = "95806068c62596c4596475a5b9cece9809556683"   # 2026-09-29T16:53:49-04:00

TARGET_SEASON = 2027            # season_end convention: 2026-27 -> 2027
GAMES_PER_TEAM = {2027: 84}
DEFAULT_GAMES = 82

# Seasons that are unusual (shortened, division-only, bubble). They stay in
# training data with weights but are never used to score a model.
BROKEN_SEASONS = {2013, 2020, 2021}

# Franchise continuity: historical codes -> current code.
FRANCHISE = {"ATL": "WPG", "PHX": "UTA", "ARI": "UTA",
             "L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}

TEAMS_2027 = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
              "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI",
              "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR",
              "UTA", "VAN", "VGK", "WPG", "WSH"]

# 2026-27 alignment (unchanged from 2025-26).
DIVISIONS = {
    "Atlantic": ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    "Central": ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"],
    "Pacific": ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"],
}
CONFERENCES = {"E": ["Atlantic", "Metropolitan"], "W": ["Central", "Pacific"]}
DIV_OF = {t: d for d, ts in DIVISIONS.items() for t in ts}
CONF_OF = {t: c for c, ds in CONFERENCES.items() for d in ds for t in DIVISIONS[d]}

SEED = 20260929
