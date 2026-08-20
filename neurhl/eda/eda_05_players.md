# EDA-05 — player vocabulary & cold-start coverage

## Vocabulary by season (dressed skaters+goalies in PBP rosterSpots)

|      |   dressed_players |   rookies |   vocab |
|-----:|------------------:|----------:|--------:|
| 2012 |              1006 |      1006 |    1006 |
| 2013 |               940 |       152 |    1158 |
| 2014 |              1000 |       180 |    1338 |
| 2015 |               996 |       150 |    1488 |
| 2016 |              1012 |       173 |    1661 |
| 2017 |              1002 |       165 |    1826 |
| 2018 |              1005 |       136 |    1962 |
| 2019 |              1028 |       159 |    2121 |
| 2020 |               991 |       131 |    2252 |
| 2021 |              1037 |       132 |    2384 |
| 2022 |              1150 |       186 |    2570 |
| 2023 |              1079 |       126 |    2696 |
| 2024 |              1044 |       128 |    2824 |
| 2025 |              1046 |       123 |    2947 |
| 2026 |              1063 |       133 |    3080 |

Total distinct players 2012–2026: **3,080** (embedding table size at the 2026 vantage).

## Event-mention long tail (pretraining signal per player)

|      |   event_mentions |
|-----:|-----------------:|
| 0.1  |               16 |
| 0.25 |              133 |
| 0.5  |              956 |
| 0.75 |             4140 |
| 0.9  |             8889 |
| 0.99 |            23983 |

Players with <100 career event mentions: **0.221** of vocab (680 players) — these lean on the career encoder / position mean.

## Career-encoder coverage (player_landing files = drafted players)

|   season_end |   dressed_with_career_file |   rookies_with_career_file |
|-------------:|---------------------------:|---------------------------:|
|         2012 |                      0.299 |                      0.299 |
|         2013 |                      0.353 |                      0.612 |
|         2014 |                      0.428 |                      0.639 |
|         2015 |                      0.506 |                      0.773 |
|         2016 |                      0.582 |                      0.74  |
|         2017 |                      0.638 |                      0.776 |
|         2018 |                      0.676 |                      0.662 |
|         2019 |                      0.704 |                      0.66  |
|         2020 |                      0.754 |                      0.786 |
|         2021 |                      0.756 |                      0.697 |
|         2022 |                      0.783 |                      0.78  |
|         2023 |                      0.8   |                      0.722 |
|         2024 |                      0.82  |                      0.75  |
|         2025 |                      0.83  |                      0.772 |
|         2026 |                      0.833 |                      0.789 |


Overall: 0.656 of dressed player-seasons and 0.585 of rookie debuts have a career file. The remainder (undrafted college/European FAs) is the cold-start null population (position-mean embedding, counted at report time).

Draft records on disk: 4,311 picks (2006–2025, incl. never-made-NHL — no survivorship bias).
