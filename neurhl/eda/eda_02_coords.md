# EDA-02 — shot coordinates & arena scorer bias

Shot-family events (regular season): **2,060,098**; share with x/y: **1.0000**.

**FINDING: `homeTeamDefendingSide` is populated only from season_end 2020** — it is NOT era-stable. Attacking direction is inferred per (game, side, period) from the median x of offensive-zone shots and validated against defendingSide where it exists.

Direction inference: coverage **0.9997** of shots; agreement with defendingSide (2020+): **0.9997**.

## Attacking-direction normalization sanity

Offensive-zone (zoneCode=O) shots with normalized x>25: **0.9981** overall.

|   season_end |   oz_x_gt_25 |
|-------------:|-------------:|
|         2012 |       0.9984 |
|         2013 |       0.9984 |
|         2014 |       0.9983 |
|         2015 |       0.9984 |
|         2016 |       0.9983 |
|         2017 |       0.9986 |
|         2018 |       0.998  |
|         2019 |       0.9982 |
|         2020 |       0.9969 |
|         2021 |       0.9973 |
|         2022 |       0.9972 |
|         2023 |       0.9973 |
|         2024 |       0.9983 |
|         2025 |       0.9987 |
|         2026 |       0.9983 |

## League mean SOG distance by season (recording drift)

|   season_end |   mean_dist |   mean_absy |     n |
|-------------:|------------:|------------:|------:|
|         2012 |       34.54 |       15.36 | 74171 |
|         2013 |       34.1  |       15.24 | 42511 |
|         2014 |       34.68 |       15.43 | 74978 |
|         2015 |       34.65 |       15.53 | 74541 |
|         2016 |       34.61 |       15.53 | 73769 |
|         2017 |       34.49 |       15.51 | 74941 |
|         2018 |       36.21 |       15.43 | 81816 |
|         2019 |       36.34 |       15.33 | 80433 |
|         2020 |       36.6  |       15.29 | 68497 |
|         2021 |       35.97 |       15.06 | 52408 |
|         2022 |       35.23 |       14.81 | 83610 |
|         2023 |       34.97 |       14.93 | 82537 |
|         2024 |       35.73 |       16.1  | 79981 |
|         2025 |       36.5  |       16.22 | 74640 |
|         2026 |       35.24 |       15.36 | 73718 |

## Venue-season shot-distance bias (venue mean − league mean, ft)

Worst 15 venue-seasons:

|   season_end | venue_team   |   dev |    n |
|-------------:|:-------------|------:|-----:|
|         2012 | NYR          | -6.67 | 2316 |
|         2013 | NYR          | -5.08 | 1452 |
|         2014 | PHI          |  4.82 | 2572 |
|         2012 | BOS          |  4.63 | 2620 |
|         2015 | NYI          | -4.61 | 2701 |
|         2020 | CHI          | -4.56 | 2290 |
|         2014 | NYR          | -4.51 | 2621 |
|         2014 | NYI          | -4.48 | 2607 |
|         2015 | BOS          |  4.14 | 2567 |
|         2021 | CHI          | -4.06 | 1848 |
|         2014 | BOS          |  3.96 | 2507 |
|         2016 | NYR          | -3.95 | 2323 |
|         2016 | BOS          |  3.88 | 2593 |
|         2013 | NYI          | -3.79 | 1402 |
|         2017 | COL          |  3.69 | 2477 |

Persistent venue bias (mean dev across seasons):

| venue_team   |   mean_dev_ft |
|:-------------|--------------:|
| NYR          |         -2.31 |
| CHI          |         -2.07 |
| NYI          |         -2.07 |
| ANA          |         -1.89 |
| STL          |         -1.13 |
| EDM          |         -1.11 |
| PIT          |         -0.98 |
| NJD          |         -0.62 |
| NSH          |         -0.51 |
| CGY          |         -0.44 |
| DET          |         -0.33 |
| SJS          |         -0.27 |
| CAR          |         -0.23 |
| DAL          |         -0.16 |
| CBJ          |         -0.05 |
| VAN          |         -0.05 |
| VGK          |          0.05 |
| TBL          |          0.16 |
| WPG          |          0.21 |
| LAK          |          0.28 |
| ARI          |          0.4  |
| TOR          |          0.53 |
| FLA          |          0.58 |
| WSH          |          0.71 |
| MTL          |          0.75 |
| OTT          |          1.07 |
| BUF          |          1.3  |
| COL          |          1.38 |
| PHX          |          1.45 |
| UTA          |          1.52 |
| MIN          |          1.69 |
| SEA          |          1.71 |
| PHI          |          2.03 |
| BOS          |          2.1  |

Prototype train-window (≤2017) arena offsets written to cache (`arena_offsets_train_prototype.csv`); production offsets will be per-vantage expanding versions of the same estimator. Range: -4.37 to 3.66 ft.
