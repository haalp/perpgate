# perpgate

A gate for perpetual-futures trade setups: rules propose a trade, a model says take it or skip it.
The question this repo is built to answer is whether a GBM, a decision model, or both can pick the
better trades out of a rule-based signal stream on Binance USDT-M futures, after costs, on data the
models have not seen.

Research code. Nothing here places orders, and nothing here is investment advice.

## Pipeline

| step | script | output |
|---|---|---|
| 1 | `scripts/fetch_universe.py` | 15m / 4h / 1d klines, `data/universe.csv` (monthly top 50 by previous month's USD volume, delisted pairs included), `data/integrity.csv` |
| 1b | `scripts/fetch_metrics.py` | 5-minute open interest and long/short ratios from the Binance archive |
| 2 | `scripts/build_candidates.py` | `data/candidates.parquet`: one row per candidate trade with features, `label_2r` and `r_net` |
| - | `scripts/liq_map.py` | estimated liquidation map from open interest (library, not wired into step 2 yet) |
| 3 | not written yet | GBM baseline, decision model, comparison under a pre-registered protocol |

```bash
pip install -r requirements.txt
python scripts/fetch_universe.py
python scripts/build_candidates.py
python scripts/fetch_metrics.py      # optional for now; slow
```

## Rules in step 2

- Trend: 4h close vs 4h EMA50 and 1d close vs 1d EMA20 must agree; no trade against them.
- Setup A: a 1h close breaks a level, then a 15m bar retests it and closes back beyond it. Levels are the
  1d / 4h / 1h range boundary, the prior 48h extreme, and the yearly / monthly / weekly / daily open (UTC).
- Setup B: a 6h impulse of at least 2.5 ATR, a pullback of at most 50%, then a 15m close beyond the impulse extreme.
- Label: `label_2r` is 1 when +2R is touched before the stop; a bar that touches both counts as the stop.
- `r_net`: half off at 2R, stop to entry, the rest at 3R / entry / 7-day timeout, minus fees and slippage.

Every threshold is a named constant at the top of the script.

## Status

The scripts pass look-ahead and labelling tests on synthetic data. They have not been run on the real
dataset yet, so there are no results. Known gaps: funding is not in `r_net`, the volume unit of the
kline source is assumed to be base-asset quantity, and the liquidation map is unvalidated.

Data: [rogerdehe/klines-binance](https://huggingface.co/datasets/rogerdehe/klines-binance) (MIT) and
[data.binance.vision](https://data.binance.vision).
