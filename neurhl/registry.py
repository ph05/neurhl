"""NeurHL-2 — feature registry: one declaration per feature, machine-checked.

Every feature declares its coverage group, vantage rule, source and transform in
ONE place. Availability masks and leakage checks are then generated FROM the
registry rather than reimplemented per model — which is the only way this stays
correct once there are several models reading overlapping tables.

COVERAGE GROUPS (the ladder). Sources cover different eras; the 2026-27 season
being projected has ALL of them, so a source that exists only from 2022 is fully
present at deployment. The asymmetry is between training/validation and
deployment — never a reason to discard signal.

VANTAGE RULES (P1 anti-leakage):
  PRIOR    computed only from season_end <= V-1        (preseason-safe)
  INGAME   may use season-V games strictly BEFORE this game's date
  STATIC   time-invariant (birth date, draft slot, handedness)
  OUTCOME  a label — never an input
"""
from dataclasses import dataclass, field

# ------------------------------------------------------------ coverage groups
GROUPS = {
    "G0": dict(name="core events", first=2008, last=2026,
               desc="event types, actors, on-ice sets, strength, score state"),
    "G1": dict(name="coordinates", first=2008, last=2026,
               desc="x/y. NHL JSON carries none before 2012 (events_2011."
                    "has_coord == 0.000); mp_shots supplies 100% coverage back "
                    "to 2008, so the hole is CLOSED. Use |x|-derived distance/"
                    "angle, never raw x/y: the two feeds disagree on rink-end "
                    "convention for 15.8% of GAMES (whole-game flips), which "
                    "distance is invariant to (corr 0.99972, 98.6% within 1ft)"),
    "G2": dict(name="shot distance", first=2008, last=2026,
               desc="dist_ft from HTM PL + derived from JSON coords"),
    "G3": dict(name="shifts / exact on-ice", first=2008, last=2026,
               desc="2010+ native shift JSON; 2008-2009 via HTM TH/TV"),
    "G4": dict(name="scratches / referees / coaches", first=2012, last=2026,
               desc="api-web right-rail"),
    "G5": dict(name="NHL EDGE tracking", first=2022, last=2026,
               desc="shot speed, skating bursts, 17-area profile, zone time"),
    "G6": dict(name="declared lineups / injuries", first=2027, last=2027,
               desc="daily DailyFaceoff snapshots; forward-looking only"),
    "G7": dict(name="careers / bios / draft", first=1999, last=2026,
               desc="all-league career rows, bios, draft records"),
}

VANTAGE_RULES = {"PRIOR", "INGAME", "STATIC", "OUTCOME"}


@dataclass(frozen=True)
class Feature:
    name: str
    group: str                  # key of GROUPS
    vantage: str                # one of VANTAGE_RULES
    source: str                 # table / file it is derived from
    desc: str = ""
    sparse_ok: bool = True      # may be absent -> learned "absent" embedding
    tags: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: unknown coverage group {self.group}")
        if self.vantage not in VANTAGE_RULES:
            raise ValueError(f"{self.name}: unknown vantage rule {self.vantage}")

    def available(self, season: int) -> bool:
        """Usable when predicting `season` — which depends on the vantage rule.

        A PRIOR feature predicting season V reads season V-1, so EDGE
        (2022-2026) IS available for the 2027 projection even though 2027 is
        outside its own coverage. Conflating "data exists for season V" with
        "usable when predicting V" is what makes coverage reasoning go wrong.
        """
        if self.vantage == "STATIC":
            return True
        g = GROUPS[self.group]
        need = season - 1 if self.vantage == "PRIOR" else season
        return g["first"] <= need <= g["last"]


class Registry:
    def __init__(self):
        self._f: dict = {}

    def add(self, *features: Feature) -> "Registry":
        for f in features:
            if f.name in self._f:
                raise ValueError(f"duplicate feature: {f.name}")
            self._f[f.name] = f
        return self

    def __getitem__(self, name) -> Feature:
        return self._f[name]

    def __len__(self):
        return len(self._f)

    def names(self, group=None, vantage=None) -> list:
        return sorted(n for n, f in self._f.items()
                      if (group is None or f.group == group)
                      and (vantage is None or f.vantage == vantage))

    def inputs(self) -> list:
        """Everything that may legitimately be fed to a model."""
        return sorted(n for n, f in self._f.items() if f.vantage != "OUTCOME")

    def availability_mask(self, season: int) -> dict:
        """{group: bool} when PREDICTING `season` — drives the pattern-aware
        missingness embeddings and the group-dropout schedule.

        A group counts as available if any registered feature in it is usable
        for that season under its own vantage rule.
        """
        out = {}
        for g in GROUPS:
            feats = [self._f[n] for n in self.names(group=g)
                     if self._f[n].vantage != "OUTCOME"]
            out[g] = any(f.available(season) for f in feats) if feats else False
        return out

    def assert_no_outcome_inputs(self, feature_names) -> None:
        """Hard guard: a label must never be handed to a model as an input."""
        bad = [n for n in feature_names
               if n in self._f and self._f[n].vantage == "OUTCOME"]
        if bad:
            raise ValueError(f"OUTCOME features used as inputs: {bad}")

    def assert_available(self, feature_names, season: int) -> None:
        """Guard against a silent look-ahead: a feature marked present outside
        its true coverage window would otherwise be invisible."""
        bad = [n for n in feature_names
               if n in self._f and not self._f[n].available(season)]
        if bad:
            raise ValueError(
                f"features used outside their coverage window in {season}: "
                f"{bad}")


# ------------------------------------------------------------ the registry
REG = Registry().add(
    # --- G0 core, always available
    Feature("elo_logit", "G0", "INGAME", "src/engine.run_elo",
            "incumbent pre-game Elo expectation; ALWAYS nested (2.4x paired-SD "
            "reduction => ~5.7x effective sample size)"),
    Feature("shot_form_diff", "G0", "INGAME", "stints",
            "rolling shot-attempt share differential; complementary to Elo"),
    Feature("score_state", "G0", "INGAME", "events",
            "score differential clipped; drives score effects"),
    Feature("strength_key", "G0", "INGAME", "events", "5v5/PP/PK/EN/3v3"),
    Feature("pen_drawn_rate", "G0", "PRIOR", "events.p1_p2",
            "player penalty-drawn per 60 — repeatable skill, converts to PP"),
    Feature("pen_taken_rate", "G0", "PRIOR", "events.p1_p2",
            "player penalty-taken per 60"),
    Feature("faceoff_skill", "G0", "PRIOR", "events.faceoff",
            "taker win rate, opponent-adjusted; possession starts"),
    Feature("rebound_gen", "G0", "PRIOR", "events",
            "rebound attempts generated per 60 (9.1% of attempts are rebounds)"),
    Feature("rush_gen", "G0", "PRIOR", "events",
            "rush attempts generated per 60 (6.5% of attempts)"),
    Feature("sched_strength_todate", "G0", "INGAME", "games_ctx",
            "opponent quality faced so far — deconfounds rolling form"),

    # --- G1/G2 geometry
    Feature("shot_xy", "G1", "INGAME", "events", "normalised coordinates"),
    Feature("shot_dist", "G2", "INGAME", "events+htm",
            "distance in feet; HTM PL carries it pre-2012"),
    Feature("rink_dist_offset", "G2", "INGAME", "mp_shots",
            "A11: expanding per-venue recorded-distance bias, strictly "
            "pre-game — house-built, never MoneyPuck's arenaAdjusted"),
    Feature("rink_disagree", "G1", "INGAME", "events+mp_shots",
            "A11: trailing per-rink-season NHL-vs-MP location disagreement — "
            "shot-location uncertainty; 2012+ (NaN-masked before)"),

    # --- G3 shifts / deployment / fatigue
    Feature("toi_share_proj", "G3", "PRIOR", "player_games",
            "projected TOI share — the L2 aggregation weight (never actual)"),
    Feature("rest_days", "G3", "INGAME", "games_ctx", "days since last game"),
    Feature("b2b", "G3", "INGAME", "games_ctx", "back-to-back flag"),
    Feature("travel_km_3d", "G3", "INGAME", "travel_games", "km over 3 days"),
    Feature("travel_km_7d", "G3", "INGAME", "travel_games", "km over 7 days"),
    Feature("tz_delta_signed", "G3", "INGAME", "travel_games+arenas",
            "SIGNED timezone shift — eastward and westward impair differently; "
            "unsigned dtz discards half the effect"),
    Feature("games_last_7d", "G3", "INGAME", "games_ctx", "schedule density"),
    Feature("toi_last_7d", "G3", "INGAME", "player_games", "cumulative load"),
    Feature("rapm_off", "G3", "PRIOR", "rapm", "stint-ridge offensive impact"),
    Feature("rapm_def", "G3", "PRIOR", "rapm", "stint-ridge defensive impact"),
    Feature("rapm_se", "G3", "PRIOR", "rapm",
            "posterior SD — sampled in S5 so a 20-game rookie is not treated "
            "as confidently as a 500-game veteran"),
    Feature("goalie_gsax", "G3", "PRIOR", "rapm+events",
            "EB-shrunk goals saved above expected per 60"),
    Feature("goalie_starts_7d", "G3", "INGAME", "player_games",
            "goalie workload; backup usually starts a back-to-back"),

    # --- G3 deployment structure (NeurHL-3, from stints/usage + mp_extra)
    Feature("pp_toi_share", "G3", "INGAME", "usage",
            "power-play TOI share — opportunity a total-TOI EWMA cannot see"),
    Feature("line_rank", "G3", "INGAME", "usage",
            "5v5 line rank within (team, pos_group); its short-vs-long delta "
            "is the role-change detector"),
    Feature("linemate_quality", "G3", "INGAME", "usage",
            "TOI-weighted modal-linemate EWMA production"),
    Feature("linemate_churn", "G3", "INGAME", "usage",
            "Jaccard distance of the modal 5v5 linemate set vs last game"),
    Feature("vacated_toi", "G3", "INGAME", "absences",
            "EWMA-TOI of same-position absent regulars — minutes opened up"),

    # --- G4 lineup availability
    Feature("scratches", "G4", "INGAME", "right_rail",
            "who is OUT — precisely Elo's blind spot"),
    Feature("coach_change", "G4", "INGAME", "coaches.csv",
            "bench boss changed since last game — deployment reset risk"),
    Feature("replacement_delta", "G4", "INGAME", "right_rail+rapm",
            "quality drop from absent player to his replacement — the quantity "
            "that actually moves a game, not mere absence"),
    Feature("referee_pen_rate", "G4", "PRIOR", "right_rail",
            "crew penalty tendency; conditions the penalty process"),

    # --- G5 EDGE (sparse; zero-init residual path)
    Feature("edge_shot_speed", "G5", "PRIOR", "edge_players",
            "top shot speed — orthogonal to location, invisible to geometric xG"),
    Feature("edge_skating", "G5", "PRIOR", "edge_players",
            "max speed and bursts >20/22 mph"),
    Feature("edge_shot_profile", "G5", "PRIOR", "edge_players",
            "17-area shot distribution — shot-selection fingerprint"),
    Feature("edge_zone_time", "G5", "PRIOR", "edge_players",
            "O/N/D zone-time share (all + EV)"),

    # --- G6 forward-looking
    Feature("declared_lines", "G6", "INGAME", "lineup_snapshots",
            "declared combinations; 2026-27 deployment prior"),

    # --- G7 static
    Feature("age", "G7", "STATIC", "career_bios", "with empirically fitted curve"),
    Feature("draft_slot", "G7", "STATIC", "career_bios", "cold-start prior"),
    Feature("pos_group", "G7", "STATIC", "career_bios",
            "F/D/G — MUST come from rosterSpots/bios, never from on-ice slot "
            "index (slots are sorted by playerId, not position)"),

    # --- outcomes (labels only)
    Feature("home_win", "G0", "OUTCOME", "games_ctx", "primary label"),
    Feature("goals_h", "G0", "OUTCOME", "games_ctx", ""),
    Feature("goals_a", "G0", "OUTCOME", "games_ctx", ""),
)
