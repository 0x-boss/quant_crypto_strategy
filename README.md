# TrendCore - a volatility-targeted, regime-gated crypto trend strategy

Long-only BTC + ETH trend following with ex-ante volatility targeting, tested on **real daily data
(CoinMetrics, 2010 - 2026-05-23)** with fees, slippage and financing charged, and validated with a
battery of anti-overfitting tests.

> **Read this first - honest scorecard against the challenge targets**
>
> | Target | 2016-2026 (full) | 2017-2026 | 2018-2026 | 2022-2026 |
> |---|---|---|---|---|
> | CAGR >= 50 % | **60.5 %** | **56.6 %** | 35.3 % | 19.1 % |
> | Sharpe > 1.5 | **1.73** | **1.66** | 1.21 | 0.78 |
> | Max drawdown (minimal) | **-31.1 %** (BTC: -83.8 %) | -31.1 % | -31.1 % | -31.1 % |
>
> The three targets are met on the full 10-year history and on 2017+, with a drawdown roughly one third of
> BTC's. They are **not** met on windows that exclude the 2016-17 bubble (2018+: Sharpe 1.2, CAGR 35 %) or in the
> post-2022 regime (Sharpe 0.8). I could not find - despite ~180 backtests of alternatives (see
> [`research/RESEARCH_LOG.md`](research/RESEARCH_LOG.md)) - any robust, honestly-testable edge that lifts the
> recent regime to Sharpe 1.5 with the data available (daily closes, no funding/order-book data).
> I would rather report that than present a curve-fit number. Expect forward Sharpe nearer 1.0-1.3 than 1.7.

![equity](results/equity_drawdown.png)
![years](results/yearly_monthly.png)

## Results (net of costs, financing; no cash yield)

Config: vol target 35 %, vol-scalar cap 1.5, gross <= 2x NAV, 10 bp one-way (BTC/ETH), 10 %/yr financing on
borrowed notional, signal at 00:00-UTC close -> traded same close (stress-tested with 1-2 day lag below).

| | CAGR | Sharpe | Sortino | MaxDD | Vol | Calmar |
|---|---|---|---|---|---|---|
| **TrendCore 2016-2026** | 60.5 % | 1.73 | 2.15 | -31.1 % | 30.0 % | 1.94 |
| TrendCore 2017-2026 | 56.6 % | 1.66 | 2.05 | -31.1 % | 29.6 % | 1.82 |
| TrendCore 2018-2026 | 35.3 % | 1.21 | 1.46 | -31.1 % | 28.3 % | 1.13 |
| TrendCore 2020-2026 | 42.0 % | 1.32 | 1.64 | -31.1 % | 30.0 % | 1.35 |
| TrendCore 2022-2026 | 19.1 % | 0.78 | 0.89 | -31.1 % | 27.2 % | 0.61 |
| BTC buy & hold 2016-2026 | 64.6 % | 1.08 | 1.45 | -83.8 % | 66.9 % | 0.77 |
| ETH buy & hold 2016-2026 | 109.9 % | 1.24 | 1.86 | -94.0 % | 97.5 % | 1.17 |
| 50/50 BTC/ETH 2016-2026 | 98.8 % | 1.30 | 1.78 | -88.0 % | 74.6 % | 1.12 |

**What the strategy is and is not.** It is a *risk-reduction* machine: it earns BTC-like CAGR at ~45 % of BTC's
volatility and ~37 % of its drawdown, which is why Sharpe/Calmar are far higher. It does **not** beat
buy-and-hold ETH or BTC on raw return in this sample (crypto went up ~100-2000x); a buy-and-hold investor
would have had to sit through -84 % / -94 % drawdowns.

Calendar years: 2016 +104 % | 2017 +432 % | 2018 -10 % | 2019 +50 % | 2020 +175 % | 2021 +59 % | 2022 -10 % |
2023 +51 % | 2024 +49 % | 2025 +9 % | 2026 YTD -3 %.  Rolling 1-year windows: 86 % positive, median +51 %,
10th percentile -5.8 %, worst -25.9 %, median rolling Sharpe 1.58.

### Risk dial (same strategy, different volatility target) - [`results/risk_dial.csv`](results/risk_dial.csv)

| vol target | CAGR 2016+ | Sharpe | MaxDD | CAGR 2018+ | Sharpe 2018+ |
|---|---|---|---|---|---|
| 20 % | 33.4 % | 1.60 | -22.5 % | 19.6 % | 1.09 |
| 25 % | 42.0 % | 1.64 | -26.7 % | 23.8 % | 1.10 |
| 30 % | 50.8 % | 1.67 | -28.2 % | 28.5 % | 1.13 |
| **35 % (default)** | 60.5 % | 1.73 | -31.1 % | 35.3 % | 1.21 |
| 50 % | 75.7 % | 1.70 | -37.1 % | 42.8 % | 1.18 |

Sharpe is ~flat across the dial: the target is a pure risk-appetite choice. Return and drawdown scale together.

## Strategy (all choices canonical / a-priori; nothing optimised on returns)

1. **Universe** - the two deepest-liquidity assets, BTC and ETH.
2. **Trend score in [0,1]** per asset = average of three indicator families: momentum votes (14/30/60/90/180 d),
   price-above-SMA votes (20/50/100/200 d), and Carver EWMAC (8/32, 16/64, 32/128, 64/256, vol-normalised).
3. **Regime gate** - hold an asset only while price > its SMA(g); bagged over g in {100,150,200}.
4. **Sizing** - inverse-volatility (EWMA span in {20,30,60}, bagged) -> equal risk per asset.
5. **Portfolio** - realised-vol targeting at 35 % (EWMA 30 d), vol-scalar capped at 1.5, **gross exposure hard-capped at 2x NAV**.
6. **Execution** - daily, long-only, no shorting, 5 %-of-NAV no-trade band (turnover ~14x/yr, costs ~1.4 %/yr).
   Cash earns 0 (conservative; with a 4 % stablecoin/T-bill yield CAGR/Sharpe would be 64.6 % / 1.81 full-sample, 39.0 % / 1.31 from 2018).

Code: [`quant/strategy.py`](quant/strategy.py) (final spec) | [`quant/strategies.py`](quant/strategies.py) (building blocks)
| [`quant/engine.py`](quant/engine.py) (backtester) | `live_signal.py` (tomorrow's weights from a CSV of closes).

## Why I believe it is not overfit (and where I do not)

All of this is reproducible with `python validate.py` -> [`results/validation.json`](results/validation.json).

| Test | Result |
|---|---|
| **Causality** (`tests/test_no_lookahead.py`) | Weights at T are bit-identical with/without future data; a 3x price shock after T leaves earlier P&L unchanged. |
| **Placebo** - circularly shift positions (keeps exposure level, destroys timing), 400 draws | Placebo Sharpe mean 0.82, 95th pct 1.26, **max 1.54** vs actual **1.72** (p < 0.003). Timing adds ~0.9 Sharpe on top of the market drift that mere exposure earns. |
| **Neighbouring parameters** (16 single-change variants: gate length, vol span, signal family, vol target, leverage cap, band, BTC-only) | Sharpe 1.46 - 1.85, median 1.70. A plateau, not a spike. (Single families alone: momentum 1.85, SMA 1.78, EWMAC 1.57 - I kept the *average*, not the best.) |
| **Walk-forward selection** - choose among 18 configs on trailing 2y Sharpe, hold 6 months (2019+) | Selected 1.35 vs fixed config 1.35; whole grid spans 1.18-1.40. Selection adds nothing - there is nothing to over-select. |
| **Pre-sample test** - identical BTC-only rule on **2011-2015**, never examined during development | Sharpe 1.8-2.0 at 10-60 bp costs, DD -35..-40 % vs -84 % buy&hold; also positive (Sharpe 0.2-0.4) in the 2014-15 bear/sideways phase where B&H was ~0. |
| **Cost / lag / financing stress** | 4x costs: 54.1 % / 1.59; +2 days execution lag: 54.5 % / 1.56 (DD -42.5 %); 2x costs + 1d lag + 20 % financing: 55.4 % / 1.59 (DD -40.1 %); **no leverage at all (cap 1.0): 49.1 % / 1.71 / DD -26.6 %**. |
| **Block bootstrap** (20-day blocks) | Sharpe 90 % CI [1.15, 2.31], P(Sharpe>1)=98 %, P(>1.5)=76 %; CAGR CI [36 %, 95 %]; MaxDD CI [-49 %, -25 %]. |
| **Deflated Sharpe** (Bailey-Lopez de Prado) with 150 trials | 1.00 for Sharpe-dispersion 0.25, 0.90 for a very pessimistic 0.5. |
| **Cross-asset generalisation** - same rule on 64 other liquid assets it was never designed on | Sharpe beat buy&hold in only **50 %** (median 0.33 vs 0.30) - *no Sharpe edge on alts* - but max drawdown was shallower in **100 %** (median -44 % vs -97 %) and median CAGR +5 % vs -26 %. The rule is a robust *risk* control everywhere; the Sharpe uplift is specific to BTC/ETH. |

**Where I do not claim robustness (please weigh these):**

* Design decisions (BTC+ETH only, the 200-day gate, no alts/shorts/ML) were made *after* looking at results on this
  same history; the ~180-trial ledger is in `research/RESEARCH_LOG.md`. The 2011-15 pre-sample and the cross-asset
  test are the nearest thing to true out-of-sample.
* Performance is **regime dependent**: strategy Sharpe is roughly 1.6-1.7x BTC's own Sharpe in every window
  (1.08 -> 1.73 full, 0.48 -> 0.78 since 2022). If BTC's forward drift is weak, so is the strategy's absolute Sharpe.
  The full-sample figures are carried by 2016-17 (+982 % over two years) and 2020; **from 2018 the bootstrap Sharpe CI is
  [0.55, 1.78]** and from 2022 [-0.05, 1.66].
* Drawdowns are slow "chop" drawdowns (mid-2023, mid-2024, 2018-19), lasting up to ~2 years of flat equity, not crashes.

## What I tried that did **not** work (honest negative results)

Cross-sectional momentum/reversal/low-vol factors (negative since 2022 / killed by costs), diversified alt trend
books (Sharpe ~0.2 since 2022, DD -50..-75 %), long/short trend (shorts lose), ETH/BTC relative value, breadth gating,
on-chain & exchange-flow features (weak, and retro-labelled data = hindsight risk), LightGBM walk-forward (OOS IC 0.01-0.03,
net Sharpe negative), 1-5 day fast momentum/lead-lag/day-of-week effects, tokenised-gold sleeve (hindsight-driven),
drawdown overlay (cuts CAGR as much as DD), efficiency-ratio filter (unstable). Details in the ledger.

## Caveats for real trading

* Data are CoinMetrics daily reference rates (00:00 UTC), not exchange fills. 10 bp one-way for BTC/ETH is realistic for
  retail-to-mid size on liquid venues; the stress tests cover 4x costs and a 2-day delay.
* Leverage (<= 1.5x vol-scalar, <= 2x gross) assumes perps/margin with isolated margin and ample buffer; daily-close
  backtests cannot see intraday liquidations. Funding is modelled as a flat 10 %/yr on borrowed notional.
* Two correlated assets (0.8): idiosyncratic risk of BTC/ETH regimes, exchange/custody and stablecoin risk are not modelled.
* **Not investment advice.** Past performance of a backtest does not predict future returns.

## What would most likely lift recent-regime Sharpe (needs data I could not access)

The environment's network policy blocked every market-data host, so I used the CoinMetrics community CSVs from GitHub.
That data has no funding rates, OHLC or order books. The highest-value additions would be a **funding-rate / basis carry
sleeve** (a different, low-correlation return source that earns in the flat periods when trend is out of the market) and
intraday data to refine entries. Neither could be backtested honestly here, so neither is claimed.

## Data and license

Price/volume data: **Coin Metrics, Inc. community data** (https://github.com/coinmetrics/data, commit `f1a36af`, 2026-05-24),
licensed **CC BY-NC 4.0** (attribution, non-commercial). The parquet panels in `data/` are derived from it (subset of
columns/assets) and carry the same terms - research use only. For live/commercial trading use your own licensed price feed
(`live_signal.py` only needs daily closes). `PriceUSD` for day D is the 00:00-UTC close ending day D.
Own code: MIT-style, use at your own risk.

## Reproduce

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/coinmetrics/data /path/to/coinmetrics-data   # optional: panels are committed in data/
python scripts/build_dataset.py /path/to/coinmetrics-data/csv                       # optional rebuild (built from commit f1a36af)
python run_backtest.py          # headline metrics + charts -> results/
python validate.py              # robustness suite (~5 min)  -> results/validation.json
python tests/test_no_lookahead.py
python live_signal.py prices.csv   # tomorrow's target weights from a date,btc,eth CSV
```

Layout: `quant/` library | `scripts/build_dataset.py` | `data/` compact parquet panels | `research/` every experiment +
ledger | `results/` outputs | `tests/` causality tests.
