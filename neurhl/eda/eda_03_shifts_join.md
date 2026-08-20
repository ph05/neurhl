# EDA-03 — shifts→PBP on-ice join validation

## Shift-file inventory vs PBP corpus

|   season_end |   pbp_games |   shift_files |   missing_shifts |
|-------------:|------------:|--------------:|-----------------:|
|         2012 |        1230 |          1230 |                0 |
|         2013 |         720 |           720 |                0 |
|         2014 |        1230 |          1230 |                0 |
|         2015 |        1230 |          1230 |                0 |
|         2016 |        1230 |          1230 |                0 |
|         2017 |        1230 |          1230 |                0 |
|         2018 |        1271 |          1271 |                0 |
|         2019 |        1271 |          1271 |                0 |
|         2020 |        1082 |          1082 |                0 |
|         2021 |         868 |           868 |                0 |
|         2022 |        1312 |          1312 |                0 |
|         2023 |        1312 |          1312 |                0 |
|         2024 |        1312 |          1312 |                0 |
|         2025 |        1312 |          1255 |               57 |
|         2026 |        1312 |          1312 |                0 |

## situationCode agreement (sampled games; A: start<=t<end, B: start<t<=end)

|   season_end |   games |   conv_A |   conv_B |   either |   events |
|-------------:|--------:|---------:|---------:|---------:|---------:|
|         2012 |      30 |   0.9604 |   0.9573 |   0.9955 |     7352 |
|         2013 |      30 |   0.9502 |   0.947  |   0.9852 |     7644 |
|         2014 |      30 |   0.963  |   0.9617 |   0.9986 |     7698 |
|         2015 |      30 |   0.9658 |   0.962  |   0.9968 |     7603 |
|         2016 |      30 |   0.9604 |   0.9578 |   0.9967 |     7846 |
|         2017 |      30 |   0.95   |   0.9471 |   0.9849 |     7561 |
|         2018 |      30 |   0.9601 |   0.9584 |   0.9966 |     7951 |
|         2019 |      30 |   0.9602 |   0.9577 |   0.9974 |     8033 |
|         2020 |      30 |   0.9554 |   0.9528 |   0.9896 |     7819 |
|         2021 |      30 |   0.8995 |   0.8921 |   0.9354 |     7332 |
|         2022 |      30 |   0.9592 |   0.9539 |   0.9919 |     7555 |
|         2023 |      30 |   0.9537 |   0.948  |   0.9918 |     7577 |
|         2024 |      30 |   0.9526 |   0.9497 |   0.9802 |     7873 |
|         2025 |      30 |   0.9617 |   0.9584 |   0.9956 |     7709 |
|         2026 |      30 |   0.96   |   0.9598 |   0.9949 |     7876 |


Overall either-convention agreement: **0.9889** (events sampled: 115,429). The tensorizer will use the better single convention per event type (faceoffs sit on boundaries) and fall back to situationCode as truth for counts.

## fastRhockey skater-count agreement (P4 check — sampled)

| file                      |   games |   count_agreement | note                |
|:--------------------------|--------:|------------------:|:--------------------|
| play_by_play_2011.parquet |       0 |            0      | 0 events, se=2011   |
| play_by_play_2012.parquet |       5 |            0.9117 | 600 events, se=2012 |
| play_by_play_2013.parquet |       5 |            0.97   | 600 events, se=2013 |
| play_by_play_2014.parquet |       5 |            0.945  | 600 events, se=2014 |
| play_by_play_2015.parquet |       5 |            0.9333 | 600 events, se=2015 |
| play_by_play_2016.parquet |       5 |            0.8917 | 600 events, se=2016 |
| play_by_play_2017.parquet |       5 |            0.9783 | 600 events, se=2017 |
| play_by_play_2018.parquet |       5 |            0.9817 | 600 events, se=2018 |
| play_by_play_2019.parquet |       5 |            0.9783 | 600 events, se=2019 |
| play_by_play_2020.parquet |       5 |            0.9717 | 600 events, se=2020 |
| play_by_play_2021.parquet |       5 |            0.975  | 600 events, se=2021 |
| play_by_play_2022.parquet |       5 |            0.9667 | 600 events, se=2022 |
| play_by_play_2023.parquet |       5 |            0.945  | 600 events, se=2023 |
| play_by_play_2024.parquet |       5 |            0.9817 | 600 events, se=2024 |


Note: fR on-ice slots store player NAMES (strings) with per-file schema drift, so the durable cross-check is skater counts (either boundary convention, goalie-inclusive or not). The situationCode check above remains the authoritative internal validation.

HTM backfill (A1) validation is appended in Phase 1 once the crawl completes: TH/TV-derived intervals for 2008-2011 + the 57-game 2024-25 gap run through the identical checks.
