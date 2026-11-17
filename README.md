# Forecast GTA VI's launch-day Twitch viewers hour by hour

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun's `gta6-twitch-ccu-forecast-01` bounty ($250 USDC, 20 paid
places, 1st $50). The program reads one case and forecasts, for each of
hours 2–24 of GTA VI's release day (UTC), five quantiles of that hour's
Twitch concurrent viewers. Deadline for entries: 2026-11-19 01:00 UTC
(09:00 WITA).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library
only, no network, 8 s / 1 CPU / 1024 MB per case). It reads every `*.in` in
the input directory and writes one `<id>.out` per case:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Method

A launch day's shape is driven by *when* each region unlocks. For a
rolling-midnight launch, hour 1 (00:00–01:00 UTC) only has the regions
already past midnight local; the US unlocks at 05:00/08:00 UTC and the
audience peaks in the US/EU evening. For a simultaneous global launch at a
later hour, hour 1 is still pre-launch and the later hours explode. So the
log-ratio `r_h = ln(viewers_h / viewers_1)` depends strongly on (a) how big
hour 1 already is and (b) whether the launch rolls at midnight local.

Per hour `h`, the program fits over the case's comparable launches that have
a (nearly) full release day and a sizeable hour 1 (≥ 10,000 viewers):

```
r_h ~ a_h + b_h * ln(viewers_1) + c_h * rolling
```

by ordinary least squares (tiny ridge for stability), then forecasts

```
q_tau(h) = viewers_1 * exp(mu_h + z_tau * sd_h)
```

where `mu_h` is the fitted value at this case's hour-1 print and rolling
flag, `z_tau` the standard normal quantile, and `sd_h` the residual spread
of that hour's fit, scaled by 0.85. The fit is redone from the case's own
comparables at every run, so the final case (42 comparables, including the
four practice launches) is used in full.

Fallbacks keep every case valid: if the fit fails for an hour, the median
ratio is used; if there are too few comparables at all, persistence (every
hour stays at hour 1, spread widening like a random walk) is used.

## Validation (local, against the practice answers)

Practice loss (mean pinball on the log scale, lower is better):

| model | p00 Wolverine | p01 RDR2 | p02 GoW Ragnarök | p03 Spider-Man 2 | mean |
|---|---|---|---|---|---|
| house baseline (persistence) | 0.744 | 0.287 | 0.215 | 0.303 | **0.387** |
| this entry | 0.164 | 0.115 | 0.120 | 0.212 | **0.153** |

Leave-one-out over the 27 big comparable launches: 0.234 vs 0.321 for the
baseline. Typical miss on the practice: ~26% (baseline ~83%); 90%
coverage: 0.91 (baseline 0.78). Practice answers were read once from
SullyGnome's public chart data for the four practice launches (the same
source the kit's NOTICE documents), only to score the practice runs.

The program is deterministic (two runs write byte-identical outputs) and
runs the four practice cases in ~0.3 s.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wffmhs5a&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
