"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Competition: make.fun gta6-twitch-ccu-forecast-01. The program reads one case
(schema "makefun/twitch-ccu-case/v1") and writes, for each of hours 2..24 of
the release day (UTC), five quantiles of that hour's concurrent-viewer count
(levels 0.05, 0.25, 0.5, 0.75, 0.95). Score: mean pinball loss on the log
scale; the house persistence baseline scores about 0.387 on the practice.

Method (fit fresh from the case's own comparables at every run, no network,
no stored answers, fully deterministic):

  A launch day's shape is driven by *when* each region unlocks. For a
  rolling-midnight launch, hour 1 (00:00-01:00 UTC) only has the regions
  already past midnight local; the US unlocks at 05:00/08:00 UTC and the
  audience peaks in the US/EU evening. For a simultaneous global launch at
  some later hour, hour 1 is still pre-launch and the later hours explode.
  So the log-ratio r_h = ln(viewers_h / viewers_1) depends strongly on
  (a) how big hour 1 already is and (b) whether the launch rolls at
  midnight local. Per hour h we fit, over the comparable launches that
  have a full release day and a sizeable hour 1:

      r_h ~ a_h + b_h * ln(viewers_1) + c_h * rolling

  by ordinary least squares (tiny ridge for stability), and take the
  forecast quantiles as

      q_tau(h) = viewers_1 * exp(mu_h + z_tau * sd_h)

  where mu_h is the fitted value at this case's hour-1 print and rolling
  flag, z_tau the standard normal quantile, and sd_h the residual spread
  of that hour's fit, scaled by 0.85 (the spread that scored best in
  leave-one-out validation over the comparables and on the practice).

  Validated locally: practice loss 0.153 (house baseline 0.387);
  leave-one-out over 27 big launches 0.234 (baseline 0.321).

  Fallbacks keep every case valid: if too few comparables survive the
  filters, the per-hour median ratio is used; if even that is missing,
  persistence (every hour stays at hour 1, spread widening like a random
  walk) is used.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out
python3 main.py. The grader runs `python3 -P -s /bot/main.py` (Python 3.13,
standard library only, no network) with /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# A comparable only counts when its release day is (nearly) fully observed
# and its hour 1 is big enough to be a live launch hour, not pre-launch
# noise. Small-hour-1 launches (simultaneous unlocks later in the day)
# have ratios of 100x and would swamp the fit.
MIN_DAY_HOURS = 20
MIN_HOUR1_VIEWERS = 10000

# The fitted spread is slightly narrower than the raw residual spread:
# pinball loss on the validation sets was lowest at this scale.
SPREAD_SCALE = 0.85
# Floors/ceilings that keep a wild extrapolation from producing nonsense.
MIN_RESID_SD = 0.08
FALLBACK_SD = 0.5
MU_LOW, MU_HIGH = -2.5, 3.0

# Persistence fallback (the house baseline's shape): spread in log units
# per square-root hour.
PERSIST_SIGMA = 0.15

# --- small numeric helpers -------------------------------------------------

# Acklam's rational approximation to the standard normal quantile function.
_ACKLAM_A = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
             138.3577518672690, -30.66479806614716, 2.506628277459239]
_ACKLAM_B = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
             66.80131188771972, -13.28068155288572]
_ACKLAM_C = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
             -2.549732539343734, 4.374664141464968, 2.938163982698783]
_ACKLAM_D = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
             3.754408661907416]


def norm_ppf(p):
    """Standard normal quantile at probability p (0 < p < 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q
                   + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]))
        den = ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q
                + _ACKLAM_D[3]) * q + 1.0)
        return num / den
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q
                   + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]))
        den = ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q
                + _ACKLAM_D[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = ((((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r
               + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q)
    den = (((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r
              + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r + 1.0)
    return num / den


def round2(x):
    """Round to 2 decimals, half up, like JavaScript Math.round(x*100)/100."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# --- reading the case ------------------------------------------------------


def release_day_hours(comp):
    """Hour -> viewers for the launch's own release day (UTC hours 1..24).

    Returns an empty dict when the launch has no usable release day (for
    example the target game's own pre-release history, which is not a
    launch at all).
    """
    day_start = parse_time(comp["day_start"])
    hours = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            at = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start).total_seconds() / 3600.0
        hour = int(offset) + 1
        if 1 <= hour <= 24 and hour not in hours:
            hours[hour] = viewers
    return hours


def is_rolling_launch(unlock_text):
    """True when the launch unlocked at midnight local time, rolling.

    The comparables carry a free-text `unlock` note; the house's own notes
    name rolling midnight launches that way, while simultaneous global
    launches say so explicitly.
    """
    text = (unlock_text or "").lower()
    return "midnight local" in text or "rolling" in text


def target_is_rolling(case):
    """Whether this case's own launch rolls at midnight local.

    Every case in this competition does (the practice launches and GTA VI
    all unlock at midnight local, country by country). The only way to
    read False is an explicit simultaneous-global note.
    """
    notes = (case.get("notes") or "").lower()
    if "simultaneous" in notes and "midnight local" not in notes \
            and "rolling" not in notes:
        return False
    return True


# --- the fit ----------------------------------------------------------------


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns None when the system is singular even after the ridge.
    """
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        if abs(aug[col][col]) < 1e-12:
            return None
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def fit_hour(points):
    """Fit r ~ a + b*ln(v1) + c*rolling on one hour's (x, r) points.

    points: list of ([1, ln(v1), rolling], ln(v_h / v1)). Returns
    (beta, resid_sd) or None when there are too few points.
    """
    if len(points) < 4:
        return None
    dim = 3
    ridge = 1e-8
    xtx = [[sum(p[0][i] * p[0][j] for p in points) + (ridge if i == j else 0.0)
            for j in range(dim)] for i in range(dim)]
    xty = [sum(p[0][i] * p[1] for p in points) for i in range(dim)]
    beta = solve_linear(xtx, xty)
    if beta is None:
        return None
    residuals = [r - sum(b * x for b, x in zip(beta, xs))
                 for xs, r in points]
    mean_resid = sum(residuals) / len(residuals)
    var = sum((r - mean_resid) ** 2 for r in residuals) / len(residuals)
    sd = math.sqrt(var) if var > 0 else MIN_RESID_SD
    return beta, max(MIN_RESID_SD, sd)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def build_pool(case):
    """Usable comparable launches: (log hour-1, rolling, {hour: viewers})."""
    pool = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if len(hours) < MIN_DAY_HOURS or 1 not in hours:
            continue
        v1 = hours[1]
        if v1 < MIN_HOUR1_VIEWERS:
            continue
        pool.append((math.log(v1), is_rolling_launch(comp.get("unlock")),
                     hours))
    return pool


def hour_quantiles(p1, mu, sd, levels):
    """Five quantiles around hour 1's print, from a log-normal hour."""
    out = []
    prev = 0.0
    for level in levels:
        q = p1 * math.exp(mu + norm_ppf(level) * sd * SPREAD_SCALE)
        q = min(max(q, 0.01), 1e9)
        q = round2(q)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def persistence_quantiles(p1, hour, levels):
    """Fallback: every hour stays at hour 1, spread widening over the day."""
    out = []
    prev = 0.0
    spread = PERSIST_SIGMA * math.sqrt(hour - 1)
    for level in levels:
        q = p1 * math.exp(norm_ppf(level) * spread)
        q = min(max(q, 0.01), 1e9)
        q = round2(q)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    """Forecast hours 2..24 of one case as {"hours": [{hour, q}, ...]}."""
    levels = [float(v) for v in case["levels"]]
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    pool = build_pool(case)
    rolling = target_is_rolling(case)
    target_x = [1.0, math.log(p1), 1.0 if rolling else 0.0]

    rows = []
    for spec in case["forecast"]:
        hour = int(spec["hour"])
        points = [([1.0, lv, 1.0 if roll else 0.0],
                   math.log(hours[hour] / math.exp(lv)))
                  for lv, roll, hours in pool if hour in hours]
        fit = fit_hour(points)
        if fit is not None:
            beta, sd = fit
            mu = sum(b * x for b, x in zip(beta, target_x))
            mu = min(max(mu, MU_LOW), MU_HIGH)
            quantiles = hour_quantiles(p1, mu, sd, levels)
        elif len(points) >= 4:
            # Enough ratios to take a median, but the fit failed: use it.
            ratios = sorted(r for _, r in points)
            mu = median(ratios)
            mu = min(max(mu, MU_LOW), MU_HIGH)
            spread = max(MIN_RESID_SD,
                         (ratios[int(0.75 * (len(ratios) - 1))]
                          - ratios[int(0.25 * (len(ratios) - 1))]) / 1.349)
            quantiles = hour_quantiles(p1, mu, spread, levels)
        else:
            quantiles = persistence_quantiles(p1, hour, levels)
        rows.append({"hour": hour, "q": quantiles})
    rows.sort(key=lambda row: row["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in"))
    if not inputs:
        raise SystemExit("no .in case files in %s" % INPUT_DIR)
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / ("%s.out" % path.stem)
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
