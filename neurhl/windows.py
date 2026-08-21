"""NeurHL-2 — evaluation windows and structurally-broken seasons.

One module owns every "which seasons may I look at?" decision, so window
discipline is enforced mechanically instead of remembered.

Three windows, not two. The v1 build had only TUNE (which it then spent on ~10
configurations) and CONFIRM (sacred), which left nowhere legitimate to make
architecture decisions — so architecture choices leaked into the spent window and
its p-values became uninterpretable. DEV fixes that at zero cost: seasons
2009-2011 were previously training-only, so using them for architecture and
calibration choices costs nothing and protects both other windows.

Structurally broken seasons TRAIN but never SCORE (amendment A4). The criterion
is the season's SCHEDULE, fixed by history, never by any observed result:
  2013 = 2012-13 lockout: 48 games/team, CONFERENCE-ONLY, no preseason
  2021 = 2020-21 COVID:   56 games/team, DIVISION-ONLY, no preseason, no crowds
2020 (2019-20) is NOT excluded: it merely stopped early; the games played had
ordinary structure.
"""

# ---------------------------------------------------------------- windows
TRAIN_FROM = 2008                       # corpus start (HTM backfill)
DEV = [2009, 2010, 2011]                # architecture / calibration decisions
TUNE = [2012, 2013, 2014, 2015, 2016, 2017]   # SPENT by v1 — development only
CONFIRM = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
PROJECT = 2027                          # the 2026-27 season being projected

# ------------------------------------------------- structurally broken seasons
NO_SCORE = {2013, 2021}

SEASON_NOTES = {
    2013: "2012-13 lockout: 48 games/team, conference-only, no preseason",
    2021: "2020-21 COVID: 56 games/team, division-only, no preseason, no crowds",
    2020: "2019-20 stopped early but structurally normal — SCORED",
}


def scored(seasons) -> list:
    """Seasons from `seasons` that may be used to SCORE a model."""
    return [s for s in seasons if s not in NO_SCORE]


DEV_SCORED = scored(DEV)                      # [2009, 2010, 2011]
TUNE_SCORED = scored(TUNE)                    # [2012, 2014, 2015, 2016, 2017]
CONFIRM_SCORED = scored(CONFIRM)              # 8 seasons, ~10,184 games


def window_of(season: int) -> str:
    if season in DEV:
        return "dev"
    if season in TUNE:
        return "tune"
    if season in CONFIRM:
        return "confirm"
    if season == PROJECT:
        return "project"
    return "train_only"


def assert_scorable(seasons, window: str) -> None:
    """Guard: refuse to score a broken season, or to cross window boundaries."""
    allowed = {"dev": DEV, "tune": TUNE, "confirm": CONFIRM}[window]
    for s in seasons:
        if s in NO_SCORE:
            raise ValueError(
                f"season {s} is structurally broken and must never be scored "
                f"({SEASON_NOTES[s]}). It may still be TRAINED on.")
        if s not in allowed:
            raise ValueError(
                f"season {s} is not in the '{window}' window "
                f"(it is '{window_of(s)}') — window boundaries are not "
                f"negotiable after the fact.")


def train_seasons_for(predict_season: int, start: int = TRAIN_FROM) -> list:
    """Every season usable to TRAIN a model that predicts `predict_season`.

    Strictly < predict_season (P1 vantage rule). Broken seasons ARE included:
    they are real hockey and the model should learn from them.
    """
    return [s for s in range(start, predict_season)]
