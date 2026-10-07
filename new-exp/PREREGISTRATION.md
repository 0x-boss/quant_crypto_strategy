# new-exp - pre-registration (written BEFORE any strategy result was looked at)

Scope: this folder is self-contained. Nothing from the older BTC/ETH TrendCore work in the repo is used.

## 1. Target (user brief)
* Pool: the tickers in `data/binance_stocks.json` (Binance "stocks available", 7 966 rows; we use `os == true`, type STOCK/ETF).
* Swing and/or intraday trades, **spot, long-only, no leverage** (gross exposure <= 100 % of equity, cash otherwise).
* Cost: **0.2 % round trip** (0.1 % per side) charged on every traded notional. Extra slippage is shown only as a stress test.
* Goal: **median calendar-month net return > 5 %** and **annualised Sharpe > 1.5**.
* No selection bias in the pool: the universe is re-selected by a rule that only uses data available at the time.

## 2. Rules that keep the research honest
1. **Point-in-time universe.** On the first trading day of each month the tradable set is the top-N names by trailing
   63-day median dollar volume (`close x volume`), measured on data up to the *previous* close, min price $5, min 252 days of
   history. `market cap` and `opt` flags in the JSON are *current* attributes and are never used for selection. Sector labels
   (`ssi`) are static classifications and may be used for grouping.
2. **Execution = next bar.** A signal built from data through the close of day *t* trades at the **open of t+1** (daily
   engine) or the open of the next bar (intraday engine). Exits at stops use the stop price, or the open if it gaps through.
3. **Leveraged / inverse ETFs are excluded** from the main pool (they embed leverage, which the brief forbids).
4. **Parameters come from priors, not from tuning.** Each hypothesis below has one *primary* parameterisation fixed now,
   taken from the published literature / conventional practice. Neighbouring settings are run only as a plateau check.
5. **Time split.** DEV = 2012-01 -> 2021-12 (decisions allowed). HOLDOUT = 2022-01 -> 2026-10 (bear 2022, rebound,
   AI-led bull). A finalist touches the holdout **once**. A second, independent check is the real Binance stock-token price
   history (2026-06 ->), which also measures how far tokens trade from the underlying.
   Disclosure: the author model has general knowledge of market history up to mid-2026, so the holdout is not "blind" in the
   strict sense; this is why rule 4 (priors, not tuning) matters.
6. **Trial registry.** Every backtest configuration that is run is appended to `results/trials.csv`. The Deflated Sharpe
   Ratio (Bailey & Lopez de Prado) is computed with the full trial count, not only with the survivors.
7. **Survivorship.** The pool is today's list, so delisted names are absent. This flatters long-only dip-buying most. We
   report a delisting-haircut stress (random terminal losses injected) next to every headline number.
8. A strategy "passes" only if it meets the target on DEV **and** HOLDOUT separately, survives 2x costs, a 1-bar extra
   execution lag, and has a parameter plateau. Otherwise it is reported as failed. Negative results are results.

## 3. Hypothesis families (primary parameters fixed now)
**A. Daily cross-sectional ranks** - buy top-k (k=10 primary), equal weight, staggered holds. Signals: 12-1 momentum, 6-1
momentum, 3-month momentum, 52-week-high proximity, 1-day / 5-day / 21-day reversal, ATR-scaled 1-day reversal, volume-surge
direction, overnight-gap reversal, residual (beta-neutral) momentum, MAX-effect reversal, low volatility (baseline only).
Holding period H in {1, 5, 21} days.

**B. Event / pattern swing trades with explicit exits** - RSI(2)<10 pullback inside an uptrend (close>SMA200; exit close>SMA5
or 5 days); panic gap-down reversal (gap <= -3 ATR with volume spike; exit at the next close); gap-up with volume
continuation (PEAD proxy, 10-day hold); Donchian-20 breakout with ATR trailing stop; new-52-week-high breakout with volume;
dip-in-leaders (top-decile 6-month momentum, down >=2 ATR over 3 days).

**C. Intraday (needs intraday history)** - opening-range breakout on stocks "in play" (relative opening volume >= 2);
gap-and-go; intraday reversal of a >=x% drawdown by late morning; overnight hold of stocks closing strong on high volume;
last-hour momentum.

**D. Walk-forward ML ranker** - one fixed small-capacity model (gradient boosting, shallow, strong regularisation) on
price/volume features, label = next-5-day open-to-open return, expanding window refit once a year, top-k long. Hyper-parameters
fixed a priori; no tuning on the holdout.

**E. Overlays / portfolio** - SPY>SMA200 regime gate, volatility scale-down (never up: no leverage), sleeve blending.

## 4. Why 5 % / month is hard here (so we know what to look for)
Sharpe 1.5 with a 5 % median month implies ~11 % monthly (~40 % annual) volatility and a hit-rate of roughly 70 % of months.
Costs of 0.2 % per round trip mean a strategy turning over capital 6x per month pays 1.2 % per month just in fees, so a
per-trade net edge of ~0.8 % at 6 turns/month (or 2.5 % at 2 turns/month) is the arithmetic of the target. Edges of that
size exist only in high-volatility names; the research therefore concentrates on liquid-but-volatile names and on
capital-efficient (high-turnover but low-cost-ratio) designs.
