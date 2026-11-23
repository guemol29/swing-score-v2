"""Detectors for Swing Score V2. Pure functions, independent from Streamlit/Yahoo Finance."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


BASE_LOOKBACK = 20
BASE_MAX_RANGE = 0.18
BASE_MAX_DRIFT = 0.05
BASE_MAX_SLOPE_PER_BAR = 0.0015
BREAKOUT_BUFFER = 0.005
RETEST_UPPER = 0.03
RETEST_CLOSE_FLOOR = 0.98
SUPPORT_INVALIDATION = 0.97
BREAKOUT_WAIT_SESSIONS = 20
MAX_PATTERN_LOOKBACK = 60


def _date_or_none(index: pd.Index, position: int | None) -> str | None:
    if position is None or position < 0 or position >= len(index):
        return None
    try:
        return pd.Timestamp(index[position]).strftime("%Y-%m-%d")
    except Exception:
        return str(index[position])


def _base_metrics(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict[str, Any] | None:
    if len(highs) < BASE_LOOKBACK or len(lows) < BASE_LOOKBACK or len(closes) < BASE_LOOKBACK:
        return None
    h = highs[-BASE_LOOKBACK:]
    l = lows[-BASE_LOOKBACK:]
    c = closes[-BASE_LOOKBACK:]
    if not (np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()):
        return None
    resistance = float(np.max(h))
    support = float(np.min(l))
    if support <= 0:
        return None
    width = resistance / support - 1.0
    drift = abs(float(c[-1] / c[0] - 1.0)) if c[0] else np.inf
    x = np.arange(len(c), dtype=float)
    slope = float(np.polyfit(x, c, 1)[0] / max(1e-9, float(np.mean(c))))
    valid = (
        width <= BASE_MAX_RANGE
        and drift <= BASE_MAX_DRIFT
        and abs(slope) <= BASE_MAX_SLOPE_PER_BAR
    )
    return {
        "valid": bool(valid),
        "resistance": resistance,
        "base_support": support,
        "range_pct": float(width),
        "drift_pct": float(drift),
        "slope_per_bar": slope,
    }


def detect_breakout_throwback(df: pd.DataFrame) -> dict[str, Any]:
    """Detect a compact base, a closing breakout, a support-holding retest and a new-high close.

    A valid throwback requires a daily low that reaches the retest zone while the
    close holds above 98% of resistance. A daily close below 97% of resistance
    invalidates the pattern. Entry confirmation is a later daily close above the
    high of the breakout candle. All percentages are explicit heuristic thresholds.
    """
    empty = {
        "base_detected": False,
        "breakout_detected": False,
        "throwback_confirmed": False,
        "entry_confirmed": False,
        "invalidated": False,
        "pattern_status": "SENSE PATRÓ",
        "pattern_reason": "No s'ha identificat una base vàlida.",
        "resistance": None,
        "base_support": None,
        "support_floor": None,
        "breakout_high": None,
        "base_start": None,
        "base_end": None,
        "breakout_date": None,
        "throwback_date": None,
        "range_pct": None,
        "drift_pct": None,
    }
    required = {"High", "Low", "Close"}
    if df is None or df.empty or not required.issubset(df.columns):
        empty["pattern_reason"] = "Falten dades High, Low o Close."
        return empty

    data = df.dropna(subset=["High", "Low", "Close"]).tail(120).copy()
    if len(data) < BASE_LOOKBACK + 1:
        empty["pattern_reason"] = "Dades insuficients per avaluar una base de 20 sessions."
        return empty

    highs = pd.to_numeric(data["High"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(data["Low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(data["Close"], errors="coerce").to_numpy(dtype=float)
    n = len(data)
    index = data.index

    # Search most-recent first; only a completed daily close can confirm a breakout.
    first_i = max(BASE_LOOKBACK, n - MAX_PATTERN_LOOKBACK)
    for breakout_i in range(n - 1, first_i - 1, -1):
        start_i = breakout_i - BASE_LOOKBACK
        metrics = _base_metrics(
            highs[start_i:breakout_i],
            lows[start_i:breakout_i],
            closes[start_i:breakout_i],
        )
        if not metrics or not metrics["valid"]:
            continue

        resistance = metrics["resistance"]
        breakout_level = resistance * (1.0 + BREAKOUT_BUFFER)
        prior_close = closes[breakout_i - 1]
        breakout_close = closes[breakout_i]
        if not (np.isfinite(prior_close) and np.isfinite(breakout_close)):
            continue
        if prior_close > breakout_level or breakout_close <= breakout_level:
            continue

        breakout_high = float(highs[breakout_i])
        support_floor = resistance * SUPPORT_INVALIDATION
        retest_ceiling = resistance * (1.0 + RETEST_UPPER)
        retest_floor = resistance * RETEST_CLOSE_FLOOR
        retest_i = None
        invalidated = False
        end_i = min(n, breakout_i + BREAKOUT_WAIT_SESSIONS + 1)

        for j in range(breakout_i + 1, end_i):
            if closes[j] < support_floor:
                invalidated = True
                break
            if lows[j] <= retest_ceiling and closes[j] >= retest_floor:
                retest_i = j
                break

        if invalidated:
            empty.update({
                "base_detected": True,
                "breakout_detected": True,
                "invalidated": True,
                "pattern_status": "PATRÓ INVALIDAT",
                "pattern_reason": f"Tancament diari per sota del suport d'invalidació ({support_floor:.2f}).",
                "resistance": resistance,
                "base_support": metrics["base_support"],
                "support_floor": support_floor,
                "breakout_high": breakout_high,
                "base_start": _date_or_none(index, start_i),
                "base_end": _date_or_none(index, breakout_i - 1),
                "breakout_date": _date_or_none(index, breakout_i),
                "range_pct": metrics["range_pct"],
                "drift_pct": metrics["drift_pct"],
            })
            return empty

        if retest_i is None:
            expired = n - 1 - breakout_i > BREAKOUT_WAIT_SESSIONS
            empty.update({
                "base_detected": True,
                "breakout_detected": True,
                "pattern_status": "BREAKOUT CADUCAT" if expired else "ESPERANT THROWBACK",
                "pattern_reason": (
                    "No hi ha hagut retest dins de 20 sessions."
                    if expired else "El breakout és vàlid, però encara no hi ha retest del nivell."
                ),
                "resistance": resistance,
                "base_support": metrics["base_support"],
                "support_floor": support_floor,
                "breakout_high": breakout_high,
                "base_start": _date_or_none(index, start_i),
                "base_end": _date_or_none(index, breakout_i - 1),
                "breakout_date": _date_or_none(index, breakout_i),
                "range_pct": metrics["range_pct"],
                "drift_pct": metrics["drift_pct"],
            })
            return empty

        # A retest is not confirmed if price subsequently closes below support.
        later_below = np.where(closes[retest_i + 1:] < support_floor)[0]
        if len(later_below):
            failure_i = retest_i + 1 + int(later_below[0])
            empty.update({
                "base_detected": True,
                "breakout_detected": True,
                "invalidated": True,
                "pattern_status": "PATRÓ INVALIDAT",
                "pattern_reason": f"El preu ha perdut el suport després del throwback ({support_floor:.2f}).",
                "resistance": resistance,
                "base_support": metrics["base_support"],
                "support_floor": support_floor,
                "breakout_high": breakout_high,
                "base_start": _date_or_none(index, start_i),
                "base_end": _date_or_none(index, breakout_i - 1),
                "breakout_date": _date_or_none(index, breakout_i),
                "throwback_date": _date_or_none(index, retest_i),
                "range_pct": metrics["range_pct"],
                "drift_pct": metrics["drift_pct"],
                "failure_date": _date_or_none(index, failure_i),
            })
            return empty

        confirmed = bool(n - 1 > retest_i and closes[-1] > breakout_high)
        empty.update({
            "base_detected": True,
            "breakout_detected": True,
            "throwback_confirmed": True,
            "entry_confirmed": confirmed,
            "pattern_status": "ENTRADA CONFIRMADA" if confirmed else "ESPERANT NOU MÀXIM",
            "pattern_reason": (
                "El tancament actual supera el màxim de la vela de breakout."
                if confirmed else "El retest ha mantingut el suport; falta tancar per sobre del màxim del breakout."
            ),
            "resistance": resistance,
            "base_support": metrics["base_support"],
            "support_floor": support_floor,
            "breakout_high": breakout_high,
            "base_start": _date_or_none(index, start_i),
            "base_end": _date_or_none(index, breakout_i - 1),
            "breakout_date": _date_or_none(index, breakout_i),
            "throwback_date": _date_or_none(index, retest_i),
            "range_pct": metrics["range_pct"],
            "drift_pct": metrics["drift_pct"],
        })
        return empty

    # No active breakout found: report a current, still-forming base if present.
    metrics = _base_metrics(highs[-BASE_LOOKBACK:], lows[-BASE_LOOKBACK:], closes[-BASE_LOOKBACK:])
    if metrics and metrics["valid"]:
        empty.update({
            "base_detected": True,
            "pattern_status": "BASE EN FORMACIÓ",
            "pattern_reason": "Consolidació lateral detectada; encara no hi ha tancament per sobre de la resistència.",
            "resistance": metrics["resistance"],
            "base_support": metrics["base_support"],
            "support_floor": metrics["resistance"] * SUPPORT_INVALIDATION,
            "base_start": _date_or_none(index, n - BASE_LOOKBACK),
            "base_end": _date_or_none(index, n - 1),
            "range_pct": metrics["range_pct"],
            "drift_pct": metrics["drift_pct"],
        })
    return empty


def build_pattern_chart(df: pd.DataFrame, pattern: dict[str, Any], max_rows: int = 120) -> pd.DataFrame:
    """Build numeric chart series for Streamlit with price and pattern reference levels."""
    if df is None or df.empty or "Close" not in df:
        return pd.DataFrame()
    full_close = pd.to_numeric(df["Close"], errors="coerce")
    source = df.tail(max_rows).copy()
    close = full_close.reindex(source.index)
    chart = pd.DataFrame(index=source.index)
    chart["Preu"] = close
    # Calculate the 200-session average on the full downloaded history,
    # then display only the latest chart window.
    chart["MM200 diària"] = full_close.rolling(200, min_periods=200).mean().reindex(source.index)

    levels = {
        "Resistència base": pattern.get("resistance"),
        "Suport invalidació": pattern.get("support_floor"),
        "Màxim vela breakout": pattern.get("breakout_high"),
    }
    for name, level in levels.items():
        if level is not None and np.isfinite(float(level)):
            chart[name] = float(level)
    return chart.dropna(how="all")
