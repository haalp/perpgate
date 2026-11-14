# Protocol

Written before any model is trained. Changes go at the bottom, dated, with the reason. Nothing above the
change log is edited after results exist.

## Question

Can a model pick the better trades out of the rule-based candidates in `data/candidates.parquet`, after
costs, on data it has not seen?

## Splits (by entry time, UTC)

| split | period | use |
|---|---|---|
| train | 2020-02 to 2024-12 | everything: exploring, choosing rules and features, fitting |
| validation | 2025-01 to 2025-12 | choosing between models and thresholds; opened only for that |
| test | 2026-01 onward | opened once, at the end, for the final table |

A trade belongs to the split of its entry. Training rows whose exit falls in the validation period are dropped,
so no training label depends on validation prices. The same rule applies between validation and test.

Rules, thresholds, features and the pair list are chosen on train only. `scripts/report.py` shows train by
default for that reason.

## Candidates compared

1. Take every candidate (the floor).
2. Fixed rule chosen on train (for example "setup B only").
3. GBM on the feature columns.
4. Decision model, zero-shot, on typed questions.
5. Decision model, fine-tuned on train.
6. GBM and decision model combined.

## What counts as working

Measured on the test split, per trade, in R after costs:

- H1: the model's kept trades have a higher mean `r_net` than candidate 2, and the 95% interval of the
  difference, bootstrapped by calendar week, excludes zero.
- H2: the kept trades have a mean `r_net` above zero, same interval.
- H3: the model keeps at least 10% of candidates. A filter that keeps almost nothing is not a result.
- H4: a decision model counts as adding something only if it beats the GBM under H1.

Weeks, not trades, are resampled because candidates from different pairs in the same week move together.

The keep/skip threshold is fixed on validation and not moved on test.

## Costs

Taker fee 0.05% and slippage 0.02% per side. Funding is not modelled in v1; it is added before the test split
is opened.

## Not allowed

- Tuning anything on validation or test, including the stop bounds, level list and leverage mix in `liq_map.py`.
- Reporting a subset (a pair, a year, a side) picked after seeing test results.
- Choosing pairs because they trended later. Trend strength enters only as a feature measured at entry.

## Change log

- 2026-11-14. Deviation, before this file existed: the first full run printed one table of hit rate and mean
  `r_net` by setup, level type and side over the whole period, 2026 included. No per-year or per-pair test
  figures were seen. The observations taken from it: all setup A groups sit near the coin-flip hit rate and
  lose after costs; setup B is the only positive group; shorts beat longs in every row.
- 2026-11-14. Stock, ETF, metal and oil perps removed from the universe (decided from the ticker list, not
  from results). Momentum rank features added. Opens moved to Istanbul time on 2026-11-12.
