"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Twitch viewers for Grand Theft Auto VI on its release day,
hours 2..24, as five quantiles an hour, from the hour-1 print plus the
median launch-day shape of comparable past big launches.

Model (per case):
  - p1 is the case's hour-1 print (at least 1).
  - Keep a comparable only if its unlock text mentions "midnight local"
    or "rolling", it is not early-access-contaminated and not Tier weak,
    it is not the GTA VI category's own history, it has all 24 release-day
    hourly prints, its hour-1 print is at least 1000, and its peak over
    hour 1 is under 10.
  - For each hour h = 2..24, across the kept comparables:
        m_h = median(log(viewers_h / viewers_1))
        s_h = pstdev of those log ratios
  - Scale adjustment (bigger hour-1 launches rise less), phased in over
    the morning: the slope is 0 at hour 2 and ramps linearly to -0.25 by
    hour 8, staying -0.25 after that, because in hours 2-4 the shape is
    the timezone trickle and the dip is the same at any launch size,
    while the scale effect belongs to the day's peak and plateau:
        m_h_adj = clamp(m_h + slope(h) * (log(p1) - med_log_p1), -0.7, 1.2)
    where med_log_p1 is the median log hour-1 print of the kept set.
    Leave-one-out on the 10 kept comparables: ramped 0.1880 vs 0.1985
    with the full slope from hour 2, the gain all in hours 2-8.
  - Spread: s_h_adj = clamp(s_h, 0.30, 0.65); if the case's hour-1 print
    is more than 1 log unit from the kept median, widen by 0.05 per extra
    log unit, capped at +0.15.
  - Quantiles: q_tau = p1 * exp(m_h_adj + z_tau * s_h_adj), rounded to
    2 decimals half up (whole numbers as ints), z = (-1.645, -0.674, 0,
    0.674, 1.645) for levels (0.05, 0.25, 0.5, 0.75, 0.95).
  - Fallback (fewer than 3 kept comparables): the house persistence
    model, q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)).

Deterministic: no randomness, no clock, no network; input files are read
in name order and every output byte is a pure function of the input.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only. Outside the grader's
container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence fallback spread (house baseline): log units per sqrt hour.
SIGMA = 0.15

# Scale adjustment: how much the median log ratio moves per log unit the
# case's hour-1 print sits from the kept comparables' median hour-1 print.
# The slope phases in over the morning (see slope_at): 0 at hour 2,
# full strength from hour 8, since the scale effect belongs to the
# day's peak, not the early timezone-trickle dip.
SCALE_SLOPE = -0.25
SCALE_RAMP_HOURS = 6
# Bounds on the adjusted median log ratio.
MEDIAN_MIN = -0.7
MEDIAN_MAX = 1.2
# Bounds on the log spread, and the extra widening for far-from-median
# hour-1 prints.
SPREAD_MIN = 0.30
SPREAD_MAX = 0.65
SPREAD_EXTRA_PER_LOG = 0.05
SPREAD_EXTRA_CAP = 0.15
SPREAD_EXTRA_THRESHOLD = 1.0
# Minimum kept comparables for the shape model; below this, fall back.
MIN_KEPT = 3
# Comparable filters.
MIN_HOUR1 = 1000
MAX_PEAK_RATIO = 10


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def slope_at(h: int) -> float:
    """The scale-adjustment slope for hour h: 0 at hour 2, ramping
    linearly to SCALE_SLOPE by hour 2 + SCALE_RAMP_HOURS, flat after."""
    return SCALE_SLOPE * min(1.0, max(0.0, (h - 2) / SCALE_RAMP_HOURS))


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_hours(comp: dict) -> list[float] | None:
    """The comparable's 24 release-day hourly viewer counts, or None.

    Returns None when any of hours 1..24 (day_start + (h - 1) hours) is
    missing, or any count is not a positive number.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_time: dict[datetime, float] = {}
    for row in comp.get("hourly") or []:
        try:
            by_time[parse_time(row["from"])] = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
    hours: list[float] = []
    for h in range(1, 25):
        v = by_time.get(day_start + timedelta(hours=h - 1))
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            return None
        hours.append(float(v))
    return hours


def kept_hours(comp: dict) -> list[float] | None:
    """A comparable's 24 hourly counts if it passes every filter, else None."""
    name = (comp.get("name") or "").lower()
    if "grand theft auto vi" in name:
        return None
    unlock = (comp.get("unlock") or "").lower()
    if "midnight local" not in unlock and "rolling" not in unlock:
        return None
    if "early-access-contaminated" in unlock:
        return None
    if "tier: weak" in unlock:
        return None
    hours = release_day_hours(comp)
    if hours is None:
        return None
    if hours[0] < MIN_HOUR1:
        return None
    if max(hours) / hours[0] >= MAX_PEAK_RATIO:
        return None
    return hours


def hour_quantiles(p1: float, h: int, m_adj: float, s_adj: float) -> list[float | int]:
    """The five quantiles of hour h around the adjusted median ratio."""
    out: list[float | int] = []
    prev: float | int = 0
    for z in Z:
        q = round2(p1 * math.exp(m_adj + z * s_adj))
        q = min(max(q, 1), 1_000_000_000)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def persistence_quantiles(p1: float, h: int) -> list[float | int]:
    """House baseline fallback: every hour stays at the hour-1 print."""
    out: list[float | int] = []
    prev: float | int = 0
    for z in Z:
        q = round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1)))
        q = min(max(q, 1), 1_000_000_000)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    kept = [h for h in (kept_hours(c) for c in case.get("comparables") or []) if h is not None]

    hours_out = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        h = row["hour"]
        if len(kept) < MIN_KEPT:
            q = persistence_quantiles(p1, h)
        else:
            med_log_p1 = statistics.median([math.log(c[0]) for c in kept])
            diff = log_p1 - med_log_p1
            ratios = [math.log(c[h - 1] / c[0]) for c in kept]
            m_h = statistics.median(ratios)
            s_h = statistics.pstdev(ratios) if len(ratios) > 1 else SIGMA
            m_adj = clamp(m_h + slope_at(h) * diff, MEDIAN_MIN, MEDIAN_MAX)
            s_adj = clamp(s_h, SPREAD_MIN, SPREAD_MAX)
            if abs(diff) > SPREAD_EXTRA_THRESHOLD:
                s_adj += min(SPREAD_EXTRA_CAP, SPREAD_EXTRA_PER_LOG * (abs(diff) - SPREAD_EXTRA_THRESHOLD))
            q = hour_quantiles(p1, h, m_adj, s_adj)
        hours_out.append({"hour": h, "q": q})
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
