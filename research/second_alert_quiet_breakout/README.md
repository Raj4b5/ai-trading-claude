# Quiet second alert + break of the previous high

A test harness for one question about a repeat-alert strategy that buys a stock on its second
screener alert:

> When a stock goes *quiet* into its second alert, what is the return if we skip the
> alert-close buy, wait for the stock to **break the previous high**, and enter there?

It also tests a follow-up question:

> After the quiet base, buy **50%** on the break of the previous high, then the other **50%**
> if the stock comes back to the **previous support**. What does that return?

Everything except the entry is held at the baseline rules, so any change in return comes from
the quiet filter and the entry timing and size alone.

## Definitions

**Second alert.** The symbol's true second alert session, counted across the whole alert history.
A third or later alert never qualifies. Alerts stamped on a holiday roll to the next session.
Alerts dated before a stock's price history starts still count, one per day, but can't be
traded. They aren't folded into the first bar.

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
| S0–S5 | scale-in: see the next section |

## Scale-in: 50% on the break, 50% back at support

The first part (`--first-frac`, default 0.5) is bought exactly as in B1 or B5: a quiet second
alert, then a buy-stop on the break of the previous high. The rest waits in a buy-limit at the
"previous support". If any session's low reaches that level while the trade is open, the
limit fills at `min(open, level)`, so a gap down fills at the open. "Previous support" can be
read three ways, so the harness tests each:

| code | support level | reads as |
|---|---|---|
| `retest` | the broken high itself | old resistance turns into support, and the break is retested |
| `alert_low` | the low of the second-alert session | the alert candle's support |
| `ll20` | the lowest low of the 20 sessions ending on the alert day | the floor of the quiet base |

| id | first part bought on | rest bought at |
|---|---|---|
| S0 | break of the alert-day high (as B1) | never. This is the control, so S1–S3 minus S0 is what the add is worth |
| **S1** | break of the alert-day high | a retest of that high |
| **S2** | break of the alert-day high | the alert-day low |
| **S3** | break of the alert-day high | the 20-session base low |
| S4 | break of the 20-session high (as B5) | a retest of that high |
| S5 | break of the 20-session high | the 20-session base low |

How the add is modelled:

- **The exit runs from the first entry**, unchanged. The add doesn't move the trailing stop or
  restart the 63-session cap, and both parts leave together.
- **The add can fill on a trail-exit day.** The limit fills intraday and the stop is checked on
  the close. It can't fill on the entry day, on the day the cap closes the trade, or after the exit.
- **The unbought part is held back as cash from the entry.** On the day you buy the first half
  you can't know whether the pullback will come, so that cash isn't lent to other alerts. It is
  spent at the add or released at the exit. Idle cash earns nothing, and that cost is included.
- **Per-trade returns are per slot.** A trade that never adds earns only on the part it bought,
  because the other part sat in cash. That makes S rows comparable with A0 and B1, which use a
  whole slot. The scale-in table also shows the return on money actually bought.

Extra outputs for the S rows:

- **What the second part did:** how often it filled and how many sessions after the entry, the
  add price against the first fill, and the slot return with and without the fill. It also
  shows the second part's own return and win rate, because buying the pullback only helps if
  the pullback trades recover.
- **Each S row against the same entry bought in one go** (S1–S3 against B1, S4–S5 against B5),
  with the same symbols and the CAGR and drawdown side by side.
- **A0 winners of +30% or more held at part size:** big runners that never came back to support,
  so only half of the position ever rode them.

## Running it

```bash
pip install pandas numpy                 # openpyxl too, for .xlsx alert files
python3 test_backtest.py                 # 18 hand-checkable tests
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
- `--first-frac 0.33` changes the scale-in split, for example a third on the break and two thirds
  at support.
- Prices must be adjusted for splits and bonuses, with columns `date,open,high,low,close,volume`.

### Prices from Zerodha Kite

`fetch_kite.py` downloads daily candles from Kite Connect into the folder layout `--prices`
reads. It needs a Kite Connect app with historical-data access, network access to
`api.kite.trade`, and two environment variables set in the environment's settings, never in
code or chat: `KITE_API_KEY` and `KITE_ACCESS_TOKEN`. Kite expires access tokens every morning.

```bash
pip install pandas numpy openpyxl requests
python3 test_fetch_kite.py               # offline, against a fake Kite server
python3 fetch_kite.py --alerts data/alerts.xlsx --alerts-sheet "<sheet name>" --out data/prices
python3 backtest.py --alerts data/alerts.xlsx --alerts-sheet "<sheet name>" --prices data/prices \
  --out results_k55
python3 backtest.py --alerts data/alerts.xlsx --alerts-sheet "<sheet name>" --prices data/prices \
  --k-atr 4 --out results_k4
```

- It fetches every stock with at least two alert days, starting 200 days before the first
  alert, so the ATR and the 60-session quiet lookback have history.
- It stays under Kite's 3 requests a second and retries rate limits.
- It stops at once if the token has expired or the app lacks historical access.
- Files already downloaded are kept, so a rerun resumes.
- `data/prices_report.csv` lists every stock: the instrument used (EQ series first, then BE/BZ
  and SME), the bars fetched, and whether the stock is missing from Kite today. A delisted or
  renamed stock can't be fetched, which leaves out mostly the worst trades, so check how many
  are missing. The report also lists overnight gaps big enough to be an unadjusted split or
  bonus. A few are normal; many mean the candles aren't adjusted.

This repository is public, so `data/`, `prices*/`, spreadsheets and alert CSVs are gitignored.
Keep alert lists and broker data out of commits.

**Reconcile before trusting a result.** `--check-known trades.csv --prices prices/` replays trades
your own engine reported and compares exit date, price and reason, with one row per trade and
columns `symbol,entry_date,entry_px,exit_date,exit_px,exit_reason,k_atr`. Every row should say
`ok` before the variant table is trusted.

Outputs, in `--out`:

- `summary.md`: per-variant trades, mean/median return per trade, win rate, the median lag from
  alert to entry, the entry premium over the alert close, and CAGR (median, p10–p90) with max
  drawdown for the full window and both eras. It also has a paired table against A0 on the same
  symbols, plus the two scale-in tables.
- `summary.json`: the same numbers plus the funnel (eligible, not quiet, unscored, no breakout,
  second part filled).
- `candidate_trades.csv`: every candidate trade in every variant, with entry, exit, reason and
  return. S rows also carry `first_frac`, `add_date`, `add_px`, `add_level`, `add_lag` and
  `slot_ret`. `ret` is the return on the money actually bought.
