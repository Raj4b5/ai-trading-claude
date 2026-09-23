# Quiet second alert + break of the previous high

A test harness for one question about a repeat-alert strategy that buys a stock on its second
screener alert:

> When a stock goes *quiet* into its second alert, what is the return if we skip the
> alert-close buy, wait for the stock to **break the previous high**, and enter there?

Everything except the entry is held at the baseline rules, so any change in return comes from
the quiet filter and the entry timing alone.

## Definitions

**Second alert.** The symbol's true second alert session, counted across the whole alert history.
A third or later alert never qualifies. Alerts stamped on a holiday roll to the next session.

**Quiet.** ATR(14) on the alert day divided by ATR(14) sixty sessions earlier is below 1, meaning
volatility contracted into the alert. Alerts with fewer than 74 bars of history cannot be scored,
so the quiet filter leaves them out.

**Break of the previous high.** "The previous high" can be read several ways, so the harness tests
each:

| code | reference high | must break within |
|---|---|---|
| `alert` | the high of the second-alert session. This is the main reading: the next sessions trade above the alert candle | 5 / 10 / 20 sessions |
| `hh20` | the highest high of the 20 sessions ending on the alert day | 10 sessions |
| `hh60` | the highest high of the 60-session base ending on the alert day | 20 sessions |
| `pause` | the running post-alert high, broken only after at least 3 sessions without a new high (the stock goes quiet *after* the alert, then breaks out) | 20 sessions |

Fill models:

- `stop`: a buy-stop at the level, filled at `max(open, level)`, so a gap-up fills at the open.
- `close`: the session must close above the level, and the buy is at that close. A once-a-day
  job near the close can run this without a resting order.

If the stock doesn't break inside the window, there is no trade. The alert isn't queued.

## What stays fixed (baseline rules)

| item | value |
|---|---|
| liquidity | 20-session mean volume, including the alert day with at least 10 bars, > 100,000, measured at the alert |
| exit | close ≤ stop, where stop = max(stop, highest close since entry − k × ATR14). The stop only ratchets up and is checked on closes, never on the entry day. Positions also close at 63 sessions |
| k | `--k-atr`, default 5.5 (run 4 as well) |
| book | 5 slots. Size is min(equity ÷ 5, cash). One entry per symbol, ever. Exits run before entries. A skipped alert is gone |
| costs | 0.30% round trip |
| ties | same-day candidates are taken in random order, with `--seeds` runs (default 200) giving the median and p10–p90 CAGR |
| eras | full window, then before and after `--split` (default 2024-07-01) |

Eligibility (second alert plus the volume floor) is the same for every variant. `A0` is the
baseline, so every other row reads directly as "what this entry change does".

## Variants run

| id | variant |
|---|---|
| A0 | baseline: buy the close of the second-alert session |
| A1 | quiet only: buy the alert close |
| B0 | every second alert, break of the alert-day high within 10 sessions (isolates the entry change) |
| **B1** | **quiet + break of the alert-day high within 10 sessions, buy-stop fill (the question as asked)** |
| B2 | quiet + close above the alert-day high within 10 sessions |
| B3 / B4 | B1 with a 5- or 20-session window |
| B5 | quiet + break of the 20-session high |
| B6 | quiet + break of the 60-session base high within 20 sessions |
| C0 / C1 | (all / quiet) at least 3 quiet sessions after the alert, then a break of the post-alert high |

## Running it

```bash
pip install pandas numpy                 # openpyxl too, for .xlsx alert files
python3 test_backtest.py                 # 11 hand-checkable tests
python3 backtest.py --demo               # synthetic data, machinery check only

python3 backtest.py \
  --alerts alerts.csv \                  # every alert row: symbol,date (not only 2nd alerts)
  --prices prices/ \                     # <SYMBOL>.csv files, or one long csv/parquet with a symbol column
  --start 2022-04-01 --end 2026-08-31 --k-atr 5.5 --seeds 200 --out results_k55
python3 backtest.py ... --k-atr 4 --out results_k4
```

- `--alerts` also accepts a workbook directly: `--alerts alerts.xlsx --alerts-sheet "<sheet name>"`.
  Column names are matched loosely, so `Symbol`/`nsecode` and `Date`/`Triggered at` both work.
- `--second-alerts` treats each alert row as a symbol's second alert, for when you already have
  that list.
- `--count-from YYYY-MM-DD` starts the alert count at a later date.
- Prices must be adjusted for splits and bonuses, with columns `date,open,high,low,close,volume`.

**Reconcile before trusting a result.** `--check-known trades.csv --prices prices/` replays trades
your own engine reported and compares exit date, price and reason, with one row per trade and
columns `symbol,entry_date,entry_px,exit_date,exit_px,exit_reason,k_atr`. Every row should say
`ok` before the variant table is trusted.

Outputs, in `--out`:

- `summary.md`: per-variant trades, mean/median return per trade, win rate, the median lag from
  alert to entry, the entry premium over the alert close, and CAGR (median, p10–p90) with max
  drawdown for the full window and both eras. It also has a paired table against A0 on the same
  symbols.
- `summary.json`: the same numbers plus the funnel (eligible, not quiet, unscored, no breakout).
- `candidate_trades.csv`: every candidate trade in every variant, with entry, exit, reason and
  return.
