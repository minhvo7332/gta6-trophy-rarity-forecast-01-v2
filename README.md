# Logit-space CQR quantile fan v2: dual-clock walk-forward blend + velocity signal

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minhvo7332/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minhvo7332/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

make.fun bounty `gta6-trophy-rarity-forecast-01` ($250, closes 21/11 18:30 +07).

Forecasts each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00Z:
five quantiles (5/25/50/75/95%) per trophy, from `node main.ts` (Node 22,
stdlib only, offline, deterministic).

## Method

- Per trophy, work in logit space. All fits use prefix-sum tables, so every
  walk-forward fit is O(1) and the whole run stays fast on ~300 reads.
- Three forward signals, each fitted on two clocks (`ln(1+h)` saturating,
  plain hours growing):
  - logit-rate trend (slope damped 0.75);
  - earners/players decomposition (separate lines for log earners and
    log players tracked, slopes damped 0.75 / 0.5, recombined into a rate);
  - **velocity (new in v2)**: the earn rate as a function of the players
    clock — logit rate regressed on log players tracked, read at the
    forecast players count.
- A 24-hour-ahead walk-forward backtest on the case's own history scores
  every signal on both clocks (a 1-step backtest wrongly rewards
  persistence); inverse-error weights blend the clocks, then the signals,
  then the blend with persistence. Bias correction clamped to ±0.5 logit.
- CQR-style calibration: the spread comes from the 90th percentile of the
  blended backtest error over calm rows only, floored at a random-walk
  `0.2*sqrt(days)`. Backtest rows straddling the first-refresh regime break
  are left out of the bias and spread — the break is a one-off, and the
  floor prices the remaining jump risk.
- Flat history (launch night, no Exophase refresh yet): refresh-shock prior.
  The median is the exact rate scaling from the rules (players 638 → 3,868,
  median earners growth 1.5x), and the spread carries the earners-growth
  uncertainty (log sd 0.5) plus refresh-timing noise.
- A trophy at 100% stays at 100%. The final case is never in any fit: every
  fit uses only the history inside the case file.

## Local probe (NOT a server grade)

Practice cases p00/p01, answers reconstructed from the 04:30 refresh
(players 638 → 3,868; earners growth factor g):

| g | this entry | house baseline |
|---|---:|---:|
| 1.0 | 0.201 | 1.688 |
| 1.5 | 0.113 | 1.289 |
| 2.0 | 0.169 | 1.006 |

Median across g: 0.169 (r2-v1: 0.157). The practice histories are flat, so
the probe only exercises the flat-history prior; the improvements below
show on non-flat histories instead.

Synthetic final-like cases (120 reads, flat night + 04:30 refresh jump +
growth, known truth), mean pinball over 3 seeds per family:

| family | this entry | r2-v1 | house baseline |
|---|---:|---:|---:|
| universal collapse | 0.078 | 0.225 | 0.243 |
| smooth drift | 0.104 | 0.236 | 0.261 |
| step-wise refreshes | 0.128 | 0.253 | 0.280 |
| step-wise + noise (stress) | 0.179 | 0.278 | 0.293 |
| **mean (12 cases)** | **0.122** | **0.248** | **0.270** |

3 runs byte-identical, ~0.2–0.5 s/run, 55/55 rows valid. (Numbers are
local probes, not a server grade.)
