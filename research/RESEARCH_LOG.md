# Research log (trial ledger)

Everything below was run on the same real data (CoinMetrics daily `PriceUSD`, 2010-2026-05-23).  Metrics
are net of costs.  "Dev" = 2016/17-2021, "Hold" = 2022-2026 unless stated.  The ledger exists so the
number of trials behind the final design is on the record (see *Deflated Sharpe* in the README) and so
the dead ends are visible, not just the winner.  ~180 backtests were run in total.

| # | Script | Idea | Outcome |
|---|--------|------|---------|
| 0 | `00_baselines.py` | BTC/ETH buy&hold, BTC SMA / TSMOM rules | BTC B&H Sharpe 1.41 dev / **0.48** hold; simple BTC trend 1.6-1.9 dev / 0.7-0.9 hold. Regime dependence is the central fact of this data. |
| 1 | `01_universe_factors.py` | Cross-sectional momentum / reversal / low-vol, dollar-neutral, top-20 liquid | Momentum Sharpe **negative in hold** (-0.6..-1.3); reversal destroyed by costs. **Rejected.** |
| 2 | `02_trend_portfolio.py` | Diversified long-only trend on top-10/20 liquid alts, vol-targeted | dev 1.7-2.0, **hold 0.0-0.3**, DD -50..-75 %. Alts bleed vs BTC after 2021. **Rejected.** |
| 3 | `03_longshort_btc_eth.py` | Long/short trend on BTC, ETH | Shorts *reduce* Sharpe in both periods (hold 0.5 vs 0.85 long-only). **Rejected.** |
| 4 | `04_diversifiers.py` | ETH/BTC relative trend; breadth gating | ETH/BTC rel. trend hold Sharpe -0.4; breadth gate no help. **Rejected.** |
| 5 | `05_s1_anatomy.py` | Where does the BTC+ETH trend book lose? | 2018 & 2022 whipsaw losses in bear-market rallies; exposure ~0.3. |
| 6 | `06_onchain_ic.py` | 8 on-chain/volume features (stablecoin growth, exchange flows, MVRV, ...) | IC 0.0-0.15, same sign but not significant out of sample; exchange-flow data is **retroactively relabelled** (hindsight risk). **Rejected.** |
| 7 | `07_reversal_gross.py` | Gross vs net of short-term reversal | Gross is *negative* (continuation); breakeven cost 3-14 bp: untradable. **Rejected.** |
| 8 | `08_ewmac.py` | Canonical Carver EWMAC (8/32..64/256) | Same as binary ensemble (1.5 full / 0.55 hold). Kept as one of 3 signal families. |
| 9 | `09_autocorr.py` | 1-5 day autocorrelation / fast TSMOM | AC ~ -0.05 at lag 1; fast TSMOM hold Sharpe < 0.2. **Rejected.** |
| 10 | `10_gold_sleeve.py` | Tokenised-gold (PAXG/XAUT) trend as uncorrelated sleeve | Helps only because gold boomed 2024-25, hurts 2020-21; full-sample Sharpe 1.33 -> 1.35. **Rejected as hindsight.** |
| 11 | `11_ml_walkforward.py` | LightGBM, 18 features, pooled liquid universe, expanding-window walk-forward | OOS rank-IC 0.01-0.03; dollar-neutral gross Sharpe 0.67 -> **-1.9 net**. **Rejected.** |
| 12 | `12_regime_conditional.py` | Forward returns by trend regime x 3-day shock | No cell with |t| > 2 and consistent sign. **Rejected.** |
| 13 | `13_risk_layers.py` | Vol estimator, portfolio vol target, drawdown overlay | Overlay lowers CAGR as much as DD (Calmar flat) - **dropped**. |
| 14 | `14_regime_gate.py` | Price > SMA(100/150/200) gate | Consistent: DD -24 -> -19 %, Sharpe 1.58 -> 1.60-1.68; plateau across lengths. **Adopted (200, bagged 100/150/200).** |
| 15 | `15_universe_breadth.py` | BTC / BTC+ETH / top5 / top10 / top20 liquid universes | BTC+ETH best (1.54); broader universes worse (DD -41..-49 %). **Adopted BTC+ETH.** |
| 16 | `16_alt_satellite.py` | Strictly filtered alt satellite (0/15/30 % budget) | No gain at any budget. **Rejected.** |
| 17 | `17_conviction.py` | Exponent on trend score f^p (p=0.5..3) | Flat (1.49-1.63). Kept linear. |
| 18 | `18_risk_frontier.py` | Vol target 0.30-0.45 x drawdown overlay | Sharpe ~1.5 everywhere; the target is a risk dial. |
| 19 | `19_drawdown_anatomy.py` | Worst drawdowns | Chop drawdowns (mid-2023, mid-2024), not crash exposure. |
| 20 | `20_leverage_caps.py` | Leverage cap 1.0-3.0, fast vol estimators | Sharpe 1.5-1.6 throughout; cap 1.0 (no leverage) still 46 % CAGR / 1.60. Adopted cap 1.5. |
| 21 | `21_bagged.py` | Parameter-bagging over gate x vol span | ~Same as single config (1.61 vs 1.59): chosen as the more defensible one. |
| 22 | `22_efficiency_ratio.py` | Kaufman efficiency-ratio trend-quality filter | Hold-block Sharpe 0.60-1.18 depending on window = noise. **Rejected.** |
| 23 | `23_band.py` | 0-10 % no-trade band | Turnover 17x -> 13x/yr at no cost. **Adopted 5 %.** |
| 24 | `24_leadlag.py` | Alt-basket -> BTC/ETH next-day lead-lag | Sign flips across regimes. **Rejected.** |
| 25 | `25_presample.py` | Same rule, BTC-only, **2011-2015 (never examined)** | Sharpe 1.8-2.0 at 10-60 bp costs, DD -35..-40 % vs -84 % B&H. Supports generalisation. |
| 26 | `26_risk_dial_table.py` | Vol target 15-50 % with the final strategy | Sharpe flat 1.6-1.73; return and DD scale together (table in README). |
| 27 | `27_whipsaw_control.py` | EWMA-smoothed score (span 3/7/14), Schmitt-trigger hysteresis (3 settings) | Smoothing lowers Sharpe (1.66-1.69 vs 1.73); hysteresis +/-0.02 = noise. **Rejected, baseline unchanged.** |
| - | (inline) | Day-of-week / month effects | abs(t) < 2.5, unstable. **Rejected.** |

## What this says

* The only edge that survived was **bull-regime drift capture with volatility control** (trend + gate +
  vol targeting) on the two deepest-liquidity assets.  No stable alpha was found in cross-sectional
  factors, short-horizon effects, on-chain data, ML, or shorting.
* Strategy Sharpe is ~1.6-1.7x BTC's Sharpe in *every* regime (1.0 -> 1.7 full-sample, 0.5 -> 0.8 since
  2022), i.e. the value-add is real and proportional, but absolute Sharpe depends on the market's drift.
