

import math

import yfinance as yf

import pandas as pd

import numpy as np

from typing import Optional



EPS = 1e-9



# --- CONSTANTS ---

RISER_MAX_DIST_TO_HIGH   = 0.10

RISER_MAX_VARIABILITY    = 0.05

DOWN_MAX_VARIABILITY     = 0.05

TOP_MAX_DIST_TO_HIGH     = 0.03

TOP_MAX_VARIABILITY      = 0.05

MIN_PERIODS              = 20

REGRESSION_WINDOW        = 20

ERA_SIZE                 = 5



# --- STAGE 2 FILTER ---

MIN_RISER_SCORE          = 0.45

MA_WEEKS                 = 30

MA_SLOPE_LOOKBACK        = 4

RS_WEEKS                 = 26     # relative-strength window (weeks)

RS_BENCHMARK             = "SPY"

MIN_RS                   = 0.0    # stock return - benchmark return must exceed this

VOL_BASE_WEEKS           = 10     # baseline weeks for volume average

VOL_BREAKOUT_MULT        = 1.5    # last-week volume vs baseline

REQUIRE_VOLUME           = False  # True = volume surge is mandatory for Stage 2

SOFTMAX_TEMP             = 0.1    # low T -> sharper probabilities (scores are in 0-1)





# --- HELPERS ---

def _clamp(x, lo=0.0, hi=1.0):

    return max(lo, min(hi, x))



def _softmax(scores: dict, temperature=SOFTMAX_TEMP) -> dict:

    exp_vals = {k: math.exp(v / temperature) for k, v in scores.items()}

    total = sum(exp_vals.values()) or 1.0

    return {k: v / total for k, v in exp_vals.items()}



def _normalize(scores: dict) -> dict:

    max_val = max(scores.values()) if scores else 1.0

    if max_val < EPS:

        return {k: 0.0 for k in scores}

    return {k: v / max_val for k, v in scores.items()}



def _linear_regression_slope(series: np.ndarray) -> float:

    n = len(series)

    if n < 2:

        return 0.0

    x = np.arange(n, dtype=float)

    x -= x.mean()

    y = series - series.mean()

    denom = (x * x).sum()

    if abs(denom) < EPS:

        return 0.0

    slope = (x * y).sum() / denom

    return float(slope / max(EPS, series.mean()))



def _weekly(ticker: str) -> Optional[pd.DataFrame]:

    # Weekly data used for MA30w / RS / volume.

    # IMPORTANT: exclude the current, still-open week.

    df = yf.download(ticker, period="2y", interval="1wk",

                     progress=False, auto_adjust=True)

    if df is None or df.empty:

        return None

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)



    # Remove the current incomplete weekly candle.

    # Yahoo's last weekly row can represent the current week while the

    # market is open; Stage 2 must use only completed weeks for its

    # trend calculations.

    if len(df) > 1:

        df = df.iloc[:-1].copy()



    return df





# --- BENCHMARK (downloaded once) ---

_BENCH_CACHE: dict = {}



def get_benchmark() -> Optional[pd.Series]:

    if RS_BENCHMARK not in _BENCH_CACHE:

        df = _weekly(RS_BENCHMARK)

        _BENCH_CACHE[RS_BENCHMARK] = df["Close"].astype(float) if df is not None else None

    return _BENCH_CACHE[RS_BENCHMARK]





# --- 30-WEEK MA + RS + VOLUME ---

def fetch_ma30w(ticker: str) -> Optional[dict]:

    try:

        df = _weekly(ticker)

        if df is None or len(df) < MA_WEEKS + MA_SLOPE_LOOKBACK:

            print(f"[{ticker}] Not enough weekly data for {MA_WEEKS}w MA")

            return None



        close = df["Close"].astype(float)

        ma    = close.rolling(MA_WEEKS).mean().values



        # MA30w and its slope come ONLY from completed weeks.

        ma_now = ma[-1]

        ma_prev = ma[-1 - MA_SLOPE_LOOKBACK]



        # Compare the real/latest daily price against the stable MA30w.

        daily = yf.download(ticker, period="5d", interval="1d",

                            progress=False, auto_adjust=True)

        if daily is None or daily.empty:

            return None

        if isinstance(daily.columns, pd.MultiIndex):

            daily.columns = daily.columns.get_level_values(0)

        price = float(daily["Close"].iloc[-1])



        # Relative strength vs benchmark (aligned by date)

        rs = None

        bench = get_benchmark()

        if bench is not None and len(close) > RS_WEEKS:

            b = bench.reindex(close.index, method="ffill")

            if not b.isna().iloc[-RS_WEEKS - 1:].any():

                s_ret = close.iloc[-1] / close.iloc[-RS_WEEKS - 1] - 1.0

                b_ret = b.iloc[-1]     / b.iloc[-RS_WEEKS - 1]     - 1.0

                rs = float(s_ret - b_ret)



        # Volume surge: last week vs average of previous N weeks

        vol_ratio = None

        if "Volume" in df.columns and len(df) > VOL_BASE_WEEKS + 1:

            vol = df["Volume"].astype(float)

            base = vol.iloc[-VOL_BASE_WEEKS - 1:-1].mean()

            vol_ratio = float(vol.iloc[-1] / base) if base > 0 else None



        return {

            "ma30w":          round(float(ma_now), 4),

            "price_vs_ma30w": round(float(price / ma_now - 1.0), 4),

            "ma30w_slope":    round(float(ma_now / ma_prev - 1.0), 4),

            "above_ma30w":    bool(price > ma_now),

            "ma30w_rising":   bool(ma_now > ma_prev),

            "rs":             round(rs, 4) if rs is not None else None,

            "vol_ratio":      round(vol_ratio, 2) if vol_ratio is not None else None,

        }

    except Exception as e:

        print(f"[{ticker}] MA30w error: {e}")

        return None





# --- FEATURE EXTRACTION ---

_DAILY_BENCH_CACHE: dict = {}

def _daily_benchmark_close(period: str, interval: str) -> Optional[pd.Series]:
    """Cached benchmark closes aligned to the requested interval."""
    key = (RS_BENCHMARK, period, interval)
    if key not in _DAILY_BENCH_CACHE:
        try:
            bdf = yf.download(RS_BENCHMARK, period=period, interval=interval, progress=False, auto_adjust=True)
            if bdf is None or bdf.empty:
                _DAILY_BENCH_CACHE[key] = None
            else:
                if isinstance(bdf.columns, pd.MultiIndex):
                    bdf.columns = bdf.columns.get_level_values(0)
                _DAILY_BENCH_CACHE[key] = bdf["Close"].astype(float)
        except Exception:
            _DAILY_BENCH_CACHE[key] = None
    return _DAILY_BENCH_CACHE[key]


def fetch_features(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[dict]:

    try:

        df = yf.download(ticker, period=period, interval=interval,

                         progress=False, auto_adjust=True)

        if df is None or len(df) < MIN_PERIODS:

            print(f"[{ticker}] Insufficient data ({len(df) if df is not None else 0} candles)")

            return None



        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        closes = df["Close"].values.flatten().astype(float)

        highs  = df["High"].values.flatten().astype(float)

        lows   = df["Low"].values.flatten().astype(float)

        n      = len(closes)



        current_price = closes[-1]

        period_high   = highs.max()

        dist_to_high  = (period_high - current_price) / max(EPS, period_high)



        recent = closes[-REGRESSION_WINDOW:]

        slope  = _linear_regression_slope(recent)



        # Eras aligned to the END (most recent candle always included);

        # the oldest partial block is the one dropped.

        start = n % ERA_SIZE

        eras = [closes[i:i + ERA_SIZE] for i in range(start, n - ERA_SIZE + 1, ERA_SIZE)]

        era_returns = [(e[-1] - e[0]) / max(EPS, e[0]) for e in eras if len(e) == ERA_SIZE]



        up_eras    = sum(1 for r in era_returns if r > 0)

        down_eras  = sum(1 for r in era_returns if r < 0)

        total_eras = len(era_returns)



        story_score_up    = (up_eras / max(1, total_eras)) * 6.0

        story_score_down  = (down_eras / max(1, total_eras)) * 6.0

        story_variability = float(np.std(era_returns)) if era_returns else 0.0



        recent_returns = np.diff(recent) / recent[:-1]

        up_ratio   = float(np.mean(recent_returns > 0))

        down_ratio = float(np.mean(recent_returns < 0))



        rolling_max  = np.maximum.accumulate(closes)

        drawdowns    = (rolling_max - closes) / np.maximum(EPS, rolling_max)

        max_drawdown = float(drawdowns.max())



        recent_low = lows[-REGRESSION_WINDOW:].min()

        recovery   = (current_price - recent_low) / max(EPS, recent_low) * 6.0



        segments_up, streak = 0, 0

        for r in recent_returns:

            if r > 0:

                streak += 1

                segments_up = max(segments_up, streak)

            else:

                streak = 0



        # New swing-opportunity inputs, independent of regime classification.
        close_s = pd.Series(closes, index=df.index)
        sma20 = float(close_s.rolling(20).mean().iloc[-1]) if n >= 20 else np.nan
        sma50 = float(close_s.rolling(50).mean().iloc[-1]) if n >= 50 else np.nan
        ret = {days: (float(current_price / closes[-days-1] - 1.0) if n > days else None)
               for days in (5, 10, 20, 60)}

        rs20 = None
        bench = _daily_benchmark_close(period, interval)
        if bench is not None and len(bench) > 20:
            aligned = pd.concat([close_s.rename("ticker"), bench.rename("bench")], axis=1).dropna()
            if len(aligned) > 20:
                rs20 = float((aligned["ticker"].iloc[-1] / aligned["ticker"].iloc[-21] - 1.0) -
                             (aligned["bench"].iloc[-1] / aligned["bench"].iloc[-21] - 1.0))

        # Volume confirmation: mean volume of the latest 5 sessions versus
        # the preceding 20 sessions (less sensitive to one exceptional day).
        vol_ratio_5v20 = None
        if "Volume" in df.columns and n >= 26:
            volume_values = df["Volume"].values.flatten().astype(float)
            recent_vol = float(np.nanmean(volume_values[-5:]))
            baseline_vol = float(np.nanmean(volume_values[-25:-5]))
            if baseline_vol > 0 and np.isfinite(recent_vol):
                vol_ratio_5v20 = float(recent_vol / baseline_vol)

        atr_pct = None
        atr_pct_median60 = None
        if n >= 75:
            prev_close = np.r_[closes[0], closes[:-1]]
            true_range = np.maximum(highs - lows,
                                    np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
            atr14 = pd.Series(true_range, index=df.index).rolling(14).mean()
            atr_pct_series = atr14 / close_s.replace(0, np.nan)
            atr_pct = float(atr_pct_series.iloc[-1])
            atr_pct_median60 = float(atr_pct_series.iloc[-60:].median())
        elif n >= 15:
            prev_close = np.r_[closes[0], closes[:-1]]
            true_range = np.maximum(highs - lows,
                                    np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
            atr_pct = float(np.nanmean(true_range[-14:])) / max(EPS, current_price)
        recent_high20 = float(np.max(highs[-20:]))
        drawdown20 = float((recent_high20 - current_price) / max(EPS, recent_high20))

        return {

            "ticker":                       ticker,

            "current_price":                round(current_price, 4),

            "dist_to_high":                 round(dist_to_high, 4),

            "eras_recent_regression_slope": round(slope, 6),

            "swing_recovery":               round(min(recovery, 6.0), 4),

            "swing_drawdown_n":             round(_clamp(1.0 - max_drawdown), 4),

            "swing_segments_up_count":      float(segments_up),

            "riser_story_score":            round(story_score_up, 4),

            "riser_up_ratio":               round(up_ratio, 4),

            "riser_story_variability":      round(story_variability, 6),

            "down_story_score":             round(story_score_down, 4),

            "down_recent_down_ratio":       round(down_ratio, 4),

            "down_story_variability":       round(story_variability, 6),

            "top_story_score":              round(story_score_up, 4),

            "top_up_ratio":                 round(up_ratio, 4),

            "top_story_variability":        round(story_variability, 6),
            "swing_sma20":                  float(sma20) if np.isfinite(sma20) else None,
            "swing_sma50":                  float(sma50) if np.isfinite(sma50) else None,
            "swing_return_5d":              ret[5],
            "swing_return_10d":             ret[10],
            "swing_return_20d":             ret[20],
            "swing_return_60d":             ret[60],
            "swing_rs20":                   rs20,
            "swing_vol_ratio5v20":          vol_ratio_5v20,
            "swing_atr_pct":                atr_pct,
            "swing_atr_pct_median60":        atr_pct_median60,
            "swing_drawdown20":             drawdown20,
            "swing_last_day_up":            bool(n >= 2 and closes[-1] > closes[-2]),
            "swing_return_5d_positive":     bool(ret[5] is not None and ret[5] > 0),

        }



    except Exception as e:

        print(f"[{ticker}] Error: {e}")

        return None





# --- REGIME DETECTION ---

def detect_regime(f: dict) -> dict:

    slope = f["eras_recent_regression_slope"]

    dth   = f["dist_to_high"]

    up_r  = f["riser_up_ratio"]

    dn_r  = f["down_recent_down_ratio"]

    return {

        "swing_valid": 0.3 < up_r < 0.7,

        "riser_valid": slope > 0.002 and up_r > 0.55,

        "down_valid":  slope < -0.002 and dn_r > 0.55,

        "top_valid":   dth < TOP_MAX_DIST_TO_HIGH and up_r > 0.5,

    }





# --- SWING OPPORTUNITY SCORING ---

def _score_swing_opportunity(f: dict) -> tuple[float, dict]:
    """0-1 swing setup score; heuristic score, not a probability of profit."""
    price = float(f["current_price"])
    sma20, sma50 = f.get("swing_sma20"), f.get("swing_sma50")

    # 25% Trend/structure. Keep the existing transparent SMA checks.
    trend_parts = []
    if sma20 is not None and sma20 > 0:
        trend_parts.append(1.0 if price > sma20 else 0.0)
    if sma50 is not None and sma50 > 0:
        trend_parts.append(1.0 if price > sma50 else 0.0)
    if sma20 is not None and sma50 is not None and sma20 > 0 and sma50 > 0:
        trend_parts.append(1.0 if sma20 > sma50 else 0.0)
    trend = float(np.mean(trend_parts)) if trend_parts else 0.5

    # 20% Momentum. Use different saturation bands by horizon: 5d +/-8%,
    # 10d +/-12%, 20d +/-18%, 60d +/-30%. This avoids treating +10% over
    # five sessions as equivalent in strength to +10% over sixty sessions.
    momentum_bands = [(5, 0.15, 0.08), (10, 0.25, 0.12),
                      (20, 0.35, 0.18), (60, 0.25, 0.30)]
    momentum_values = [(_clamp(0.5 + float(f[f"swing_return_{days}d"]) / (2.0 * band)), weight)
                       for days, weight, band in momentum_bands
                       if f.get(f"swing_return_{days}d") is not None]
    momentum = (sum(value * weight for value, weight in momentum_values) /
                sum(weight for _, weight in momentum_values) if momentum_values else 0.5)

    # 20% Relative strength vs SPY over 20 sessions. 0% excess return = 50;
    # +/-10 percentage points or more = 0/100. Stage 2 RS remains a separate
    # 26-completed-week measure and must not be read as the same indicator.
    rs20 = f.get("swing_rs20")
    relative_strength = _clamp(0.5 + float(rs20) / 0.20) if rs20 is not None else 0.5

    # 20% Volume: latest 5-session average vs preceding 20-session average.
    # Reward sustained activity only when the 5-session price return is positive.
    vol_ratio = f.get("swing_vol_ratio5v20")
    if vol_ratio is None:
        volume = 0.5  # neutral when unavailable; do not invent a volume signal
    else:
        volume_activity = _clamp((float(vol_ratio) - 0.8) / 0.7)
        volume = volume_activity if f.get("swing_return_5d_positive") else volume_activity * 0.35

    # 15% Quality/risk. Penalize excessive distance from SMA20 and unusually
    # high volatility relative to the ticker's own 60-session ATR baseline.
    # A pullback from the recent high is reported but is not automatically bad.
    if sma20 is not None and sma20 > 0:
        distance = price / sma20 - 1.0
        if distance >= 0:
            proximity = 1.0 - _clamp(distance / 0.12)  # full penalty at +12% above SMA20
        else:
            proximity = 1.0 - _clamp(abs(distance) / 0.15)  # full penalty at 15% below SMA20
    else:
        proximity = 0.5

    atr_pct = f.get("swing_atr_pct")
    atr_median = f.get("swing_atr_pct_median60")
    if atr_pct is not None and atr_median is not None and atr_median > 0:
        atr_ratio = float(atr_pct) / float(atr_median)
        atr_quality = 1.0 - _clamp((atr_ratio - 1.25) / 0.75)
    else:
        atr_quality = 0.5
    quality_risk = 0.60 * proximity + 0.40 * atr_quality

    components = {"trend_structure": round(trend, 4), "momentum": round(momentum, 4),
                  "relative_strength": round(relative_strength, 4), "volume": round(volume, 4),
                  "quality_risk": round(quality_risk, 4)}
    score = (0.25 * trend + 0.20 * momentum + 0.20 * relative_strength +
             0.20 * volume + 0.15 * quality_risk)
    return round(_clamp(score), 4), components



def score_from_features(f: dict) -> dict:
    regime = detect_regime(f)
    swing_opportunity_score, swing_components = _score_swing_opportunity(f)

    # Legacy regime score: retained separately to preserve RISER/DOWN/TOP logic.
    swing_score_raw = 0.0
    if regime["swing_valid"]:
        swing_score_raw = (
            0.55 * _clamp(f["swing_recovery"] / 6.0) +
            0.30 * _clamp(f["swing_drawdown_n"]) +
            0.15 * _clamp(f["swing_segments_up_count"] / 3.0)
        )



    riser_score_raw = 0.0

    if regime["riser_valid"]:

        riser_headroom = _clamp((RISER_MAX_DIST_TO_HIGH - f["dist_to_high"]) /

                                 max(EPS, RISER_MAX_DIST_TO_HIGH))

        riser_clean    = 1.0 - _clamp(f["riser_story_variability"] /

                                       max(EPS, RISER_MAX_VARIABILITY))

        riser_score_raw = (

            0.40 * _clamp(f["riser_story_score"] / 6.0) +

            0.20 * _clamp(f["riser_up_ratio"]) +

            0.20 * _clamp(max(0.0, f["eras_recent_regression_slope"]) / 0.01) +

            0.10 * riser_headroom +

            0.10 * riser_clean

        )



    down_score_raw = 0.0

    if regime["down_valid"]:

        down_tail = 1.0 - _clamp(f["down_story_variability"] /

                                   max(EPS, DOWN_MAX_VARIABILITY))

        down_score_raw = (

            0.45 * _clamp(f["down_story_score"] / 6.0) +

            0.20 * _clamp(f["down_recent_down_ratio"]) +

            0.20 * _clamp(abs(min(0.0, f["eras_recent_regression_slope"])) / 0.01) +

            0.15 * down_tail

        )



    top_score_raw = 0.0

    if regime["top_valid"]:

        top_proximity = _clamp(1.0 - f["dist_to_high"] /

                                max(EPS, TOP_MAX_DIST_TO_HIGH))

        top_flatness  = 1.0 - _clamp(f["top_story_variability"] /

                                      max(EPS, TOP_MAX_VARIABILITY))

        top_score_raw = (

            0.40 * _clamp(f["top_story_score"] / 6.0) +

            0.30 * top_proximity +

            0.20 * _clamp(f["top_up_ratio"]) +

            0.10 * top_flatness

        )



    raw = {"swing": swing_score_raw, "riser": riser_score_raw,

           "down": down_score_raw,   "top":   top_score_raw}



    label = max(raw, key=raw.get)

    score = raw[label]

    if score < EPS:

        label = "none"



    return {

        "ticker":     f["ticker"],

        "price":      f["current_price"],

        "label":      label,

        "score":      round(score, 4),
        "swing_opportunity_score": swing_opportunity_score,
        "swing_components": swing_components,
        "swing_opportunity_label": ("HIGH" if swing_opportunity_score >= 0.80 else
                                     "INTER" if swing_opportunity_score >= 0.60 else
                                     "MIXED" if swing_opportunity_score >= 0.40 else "LOW"),

        "raw":        {k: round(v, 4) for k, v in raw.items()},

        "normalized": {k: round(v, 4) for k, v in _normalize(raw).items()},

        "softmax":    {k: round(v, 4) for k, v in _softmax(raw).items()},

        "regime":     regime,

    }





# --- STAGE 2 FILTER ---

def apply_stage2(result: dict, ma: Optional[dict]) -> dict:

    if ma is None:

        result.update({"ma30w": None, "price_vs_ma30w": None, "ma30w_slope": None,

                       "above_ma30w": False, "ma30w_rising": False,

                       "rs": None, "vol_ratio": None,

                       "rs_ok": False, "vol_ok": False, "stage2": False})

        return result



    result.update(ma)

    result["rs_ok"]  = ma["rs"] is not None and ma["rs"] > MIN_RS

    result["vol_ok"] = ma["vol_ratio"] is not None and ma["vol_ratio"] >= VOL_BREAKOUT_MULT

    result["stage2"] = bool(

        result["label"] == "riser"

        and result["score"] > MIN_RISER_SCORE

        and ma["above_ma30w"]

        and ma["ma30w_rising"]

        and result["rs_ok"]

        and (result["vol_ok"] or not REQUIRE_VOLUME)

    )

    return result





# --- FULL PIPELINE ---

def rank_tickers(

    tickers:      list,

    period:       str = "6mo",

    interval:     str = "1d",

    sort_by:      str = "swing",

    filter_label: Optional[str] = None,

    top_n:        Optional[int] = None,

    stage2_only:  bool = False,

) -> list:



    results = []

    for ticker in tickers:

        print(f"Processing {ticker}...")

        features = fetch_features(ticker, period=period, interval=interval)

        if features:

            scored = score_from_features(features)

            scored = apply_stage2(scored, fetch_ma30w(ticker))

            scored = add_stage2_reliability(scored)

            results.append(scored)



    if stage2_only:

        results = [r for r in results if r["stage2"]]

    if filter_label:

        results = [r for r in results if r["label"] == filter_label]



    if sort_by == "swing":
        results.sort(key=lambda r: r.get("swing_opportunity_score", 0.0), reverse=True)
    elif sort_by in ("riser", "down", "top"):
        results.sort(key=lambda r: r["raw"][sort_by], reverse=True)
    else:
        results.sort(key=lambda r: r["score"], reverse=True)



    if top_n:

        results = results[:top_n]

    return results





def print_ranking(results: list):

    print(f"\n{'#':<4} {'Ticker':<8} {'Price':>9}  {'Label':<7} {'Score':>6}  "

          f"{'vsMA30w':>8} {'MA↑':>4} {'RS':>8} {'Vol×':>5} {'Weinstein Stage2':>16}  Raw scores")

    print("-" * 138)

    for i, r in enumerate(results, 1):

        raw_str = "  ".join(f"{k}={v:.3f}" for k, v in r["raw"].items())

        vs   = f"{r['price_vs_ma30w']*100:+.1f}%" if r["price_vs_ma30w"] is not None else "n/a"

        rise = "Y" if r["ma30w_rising"] else "N"

        rs   = f"{r['rs']*100:+.1f}%" if r.get("rs") is not None else "n/a"

        vol  = f"{r['vol_ratio']:.2f}" if r.get("vol_ratio") is not None else "n/a"

        s2   = "YES" if r["stage2"] else "-"

        print(f"{i:<4} {r['ticker']:<8} {r['price']:>9.2f}  {r['label']:<7} {r['score']:>6.4f}  "

              f"{vs:>8} {rise:>4} {rs:>8} {vol:>5} {s2:>16}  {raw_str}")



# ============================================================


# ============================================================
# FIABILITAT HISTÒRICA STAGE 2
# ============================================================
# Es calcula NOMÉS quan un ticker confirma Stage 2.
# Fiabilitat = % de senyals històrics NO -> SÍ amb retorn
# positiu després de 20 sessions.
# ============================================================

BACKTEST_YEARS = 5
BACKTEST_FORWARD_DAYS = 20
BACKTEST_MIN_SIGNALS = 10
_BACKTEST_CACHE = {}

def _reliability_download_daily(ticker):
    try:
        df = yf.download(ticker, period=f"{BACKTEST_YEARS}y", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception as e:
        print(f"[Reliability download {ticker}] {type(e).__name__}: {e}")
        return None


def _reliability_weekly(df):
    if df is None or df.empty:
        return pd.DataFrame()
    return df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna(subset=["Close"])


def calc_at_date(daily, w, bw, date):
    d = daily.loc[:date].tail(126)
    ww = w.loc[:date]

    if len(d) < MIN_PERIODS:
        return None
    if len(ww) < MA_WEEKS + MA_SLOPE_LOOKBACK:
        return None

    c = d["Close"].astype(float).values
    hi = d["High"].astype(float).values
    lo = d["Low"].astype(float).values

    price = float(c[-1])
    dist = (float(hi.max()) - price) / max(EPS, float(hi.max()))

    recent = c[-REGRESSION_WINDOW:]
    sl = _linear_regression_slope(recent)

    start = len(c) % ERA_SIZE
    eras = [
        c[i:i + ERA_SIZE]
        for i in range(start, len(c) - ERA_SIZE + 1, ERA_SIZE)
    ]
    ers = [
        (e[-1] - e[0]) / max(EPS, e[0])
        for e in eras if len(e) == ERA_SIZE
    ]

    up = sum(x > 0 for x in ers)
    down = sum(x < 0 for x in ers)
    total = len(ers)

    story_up = (up / max(1, total)) * 6
    story_down = (down / max(1, total)) * 6
    var = float(np.std(ers)) if ers else 0.0

    rr = np.diff(recent) / recent[:-1]
    up_ratio = float(np.mean(rr > 0))
    down_ratio = float(np.mean(rr < 0))

    dd = (np.maximum.accumulate(c) - c) / np.maximum(EPS, np.maximum.accumulate(c))
    maxdd = float(dd.max())

    recent_low = lo[-REGRESSION_WINDOW:].min()
    recovery = (price - recent_low) / max(EPS, recent_low) * 6

    streak = 0
    best = 0
    for x in rr:
        if x > 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0

    swing = 0.0
    if 0.3 < up_ratio < 0.7:
        swing = (
            0.55 * _clamp(min(recovery, 6) / 6) +
            0.30 * _clamp(1 - maxdd) +
            0.15 * _clamp(best / 3)
        )

    riser = 0.0
    if sl > 0.002 and up_ratio > 0.55:
        head = _clamp(
            (RISER_MAX_DIST_TO_HIGH - dist) /
            RISER_MAX_DIST_TO_HIGH
        )
        clean = 1 - _clamp(var / RISER_MAX_VARIABILITY)
        riser = (
            0.40 * _clamp(story_up / 6) +
            0.20 * _clamp(up_ratio) +
            0.20 * _clamp(max(0, sl) / 0.01) +
            0.10 * head +
            0.10 * clean
        )

    downscore = 0.0
    if sl < -0.002 and down_ratio > 0.55:
        tail = 1 - _clamp(var / 0.05)
        downscore = (
            0.45 * _clamp(story_down / 6) +
            0.20 * _clamp(down_ratio) +
            0.20 * _clamp(abs(min(0, sl)) / 0.01) +
            0.15 * tail
        )

    topscore = 0.0
    if dist < TOP_MAX_DIST_TO_HIGH and up_ratio > 0.5:
        prox = _clamp(1 - dist / TOP_MAX_DIST_TO_HIGH)
        flat = 1 - _clamp(var / TOP_MAX_VARIABILITY)
        topscore = (
            0.40 * _clamp(story_up / 6) +
            0.30 * prox +
            0.20 * _clamp(up_ratio) +
            0.10 * flat
        )

    raw = {
        "swing": swing,
        "riser": riser,
        "down": downscore,
        "top": topscore
    }

    label = max(raw, key=raw.get)
    score = raw[label]

    if score < EPS:
        label = "none"

    cw = ww["Close"].astype(float)
    ma = cw.rolling(MA_WEEKS).mean()

    ma_now = float(ma.iloc[-1])
    ma_prev = float(ma.iloc[-1 - MA_SLOPE_LOOKBACK])

    b = bw.reindex(cw.index, method="ffill")
    rs = None

    if len(cw) > RS_WEEKS:
        idx = -RS_WEEKS - 1
        if not b.iloc[idx:].isna().any():
            stock_ret = cw.iloc[-1] / cw.iloc[idx] - 1
            bench_ret = b.iloc[-1] / b.iloc[idx] - 1
            rs = float(stock_ret - bench_ret)

    volx = None
    if len(ww) > VOL_BASE_WEEKS + 1:
        vol = ww["Volume"].astype(float)
        base = vol.iloc[-VOL_BASE_WEEKS - 1:-1].mean()
        if base > 0:
            volx = float(vol.iloc[-1] / base)

    stage2 = bool(
        label == "riser"
        and score > MIN_RISER_SCORE
        and price > ma_now
        and ma_now > ma_prev
        and rs is not None
        and rs > 0
        and (
            (volx is not None and volx >= VOL_BREAKOUT_MULT)
            or not REQUIRE_VOLUME
        )
    )

    return {
        "price": price,
        "score": score,
        "label": label,
        "stage2": stage2,
        "vsma": price / ma_now - 1,
        "rs": rs,
        "volx": volx
    }


def backtest(ticker, daily, bench):
    w = weekly(daily)
    bw = weekly(bench)["Close"]

    out = []
    prev = False

    start = max(160, MA_WEEKS * 5)

    for i in range(start, len(daily)):
        date = daily.index[i]
        f = calc_at_date(daily, w, bw, date)

        if f is None:
            continue

        new_signal = f["stage2"] and not prev

        if new_signal:
            future = daily["Close"].iloc[i + 1:]

            r = {
                "Ticker": ticker,
                "SignalDate": date.date(),
                "SignalPrice": f["price"],
                "Score": f["score"],
                "vsMA30w": f["vsma"],
                "RS": f["rs"],
                "VolX": f["volx"]
            }

            for h in FORWARD_DAYS:
                if len(future) >= h:
                    r[f"Ret_{h}d"] = float(
                        future.iloc[h - 1] / f["price"] - 1
                    )
                else:
                    r[f"Ret_{h}d"] = np.nan

            win = future.iloc[:30]

            if len(win):
                r["MaxGain30"] = float(win.max() / f["price"] - 1)
                r["MaxDD30"] = float(win.min() / f["price"] - 1)
            else:
                r["MaxGain30"] = np.nan
                r["MaxDD30"] = np.nan

            out.append(r)

        prev = f["stage2"]

    return pd.DataFrame(out)


def stats(df):
    rows = []

    for h in FORWARD_DAYS:
        s = df[f"Ret_{h}d"].dropna()

        if len(s):
            rows.append({
                "Horitzó": f"{h} sessions",
                "Senyals": len(s),
                "% positius": round((s > 0).mean() * 100, 1),
                "Mitjana %": round(s.mean() * 100, 2),
                "Mediana %": round(s.median() * 100, 2),
                "Millor %": round(s.max() * 100, 2),
                "Pitjor %": round(s.min() * 100, 2)
            })

    return pd.DataFrame(rows)



def _reliability_backtest(ticker, daily, benchmark_daily):
    w = _reliability_weekly(daily)
    bw = _reliability_weekly(benchmark_daily)["Close"]

    out = []
    prev = False
    start = max(160, MA_WEEKS * 5)

    for i in range(start, len(daily)):
        date = daily.index[i]
        f = calc_at_date(daily, w, bw, date)
        if f is None:
            continue

        new_signal = f["stage2"] and not prev
        if new_signal:
            future = daily["Close"].iloc[i + 1:]
            r = {"Ticker": ticker, "SignalDate": date.date(), "SignalPrice": f["price"]}
            if len(future) >= BACKTEST_FORWARD_DAYS:
                r["Ret_20d"] = float(future.iloc[BACKTEST_FORWARD_DAYS - 1] / f["price"] - 1)
            else:
                r["Ret_20d"] = np.nan
            out.append(r)

        prev = f["stage2"]

    return pd.DataFrame(out)


def historical_stage2_reliability(ticker: str) -> Optional[dict]:
    if ticker in _BACKTEST_CACHE:
        return _BACKTEST_CACHE[ticker]

    daily = _reliability_download_daily(ticker)
    benchmark_daily = _reliability_download_daily(RS_BENCHMARK)
    if daily is None or benchmark_daily is None or len(daily) < 300:
        return None

    trades = _reliability_backtest(ticker, daily, benchmark_daily)
    if trades is None or trades.empty or "Ret_20d" not in trades.columns:
        result = {"reliability": None, "reliability_label": "⚪ Sense mostra", "signals": 0, "mean20": None, "median20": None}
        _BACKTEST_CACHE[ticker] = result
        return result

    vals = trades["Ret_20d"].dropna().astype(float)
    if vals.empty:
        result = {"reliability": None, "reliability_label": "⚪ Sense dades", "signals": 0, "mean20": None, "median20": None}
        _BACKTEST_CACHE[ticker] = result
        return result

    reliability = float((vals > 0).mean())
    if len(vals) < BACKTEST_MIN_SIGNALS:
        label = "⚪ Mostra insuficient"
    elif reliability >= 0.70:
        label = "🟢 Alta"
    elif reliability >= 0.60:
        label = "🟢 Bona"
    elif reliability >= 0.50:
        label = "🟡 Moderada"
    else:
        label = "🔴 Baixa"

    result = {
        "reliability": reliability,
        "reliability_label": label,
        "signals": int(len(vals)),
        "mean20": float(vals.mean()),
        "median20": float(vals.median()),
    }
    _BACKTEST_CACHE[ticker] = result
    return result


def add_stage2_reliability(result: dict) -> dict:
    if not result.get("stage2"):
        result["stage2_reliability"] = None
        return result
    try:
        result["stage2_reliability"] = historical_stage2_reliability(result["ticker"])
    except Exception as e:
        print(f"[Reliability {result.get('ticker','?')}] {type(e).__name__}: {e}")
        result["stage2_reliability"] = {"reliability": None, "reliability_label": "⚪ Error backtest", "signals": 0, "mean20": None, "median20": None}
    return result




# ============================================================
# STREAMLIT UI — Swing Score
# ============================================================
import streamlit as st
import io
from datetime import datetime

def get_action(r):
    """Acció orientativa basada en Swing Opportunity; Stage 2 és independent."""
    score = float(r.get("swing_opportunity_score", 0.0) or 0.0)
    if score >= 0.80:
        return "🔎 REVISAR ENTRADA"
    if score >= 0.60:
        return "👀 VIGILAR"
    if score >= 0.40:
        return "⏳ ESPERAR"
    return "⚪ BAIXA PRIORITAT"


st.set_page_config(
    page_title="Swing Score",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Swing Score")
st.caption("Detector d'oportunitats swing · Score heurístic, no probabilitat de benefici")

with st.sidebar:
    st.header("Configuració")
    period = st.selectbox("Historial per a l'anàlisi", ["6mo", "1y", "2y"], index=0)
    st.caption("El filtre Stage 2 i el Swing Opportunity es mostren per separat.")
    st.divider()
    st.markdown("**Pesos Swing Opportunity**")
    st.write("Tendència 25% · Momentum 20% · Força relativa 20% · Volum 20% · Qualitat/risc 15%")

st.subheader("1. Selecció de tickers")
manual_tickers = st.text_area(
    "Introdueix tickers manualment (separats per comes o línies)",
    placeholder="TSLA, PANW, QCOM, NVDA",
    height=90,
)
uploaded = st.file_uploader("O importa una Watchlist CSV de Yahoo Finance", type=["csv"])

uploaded_tickers = []
if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        try:
            watch_df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            watch_df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        ticker_col = None
        accepted = {"symbol", "ticker", "tickers", "ticker symbol", "symbols"}
        for col in watch_df.columns:
            normalized = str(col).strip().lower().replace("_", " ")
            if normalized in accepted:
                ticker_col = col
                break
        if ticker_col is None:
            st.error("No he trobat una columna Symbol, Ticker o Tickers al CSV.")
        else:
            uploaded_tickers = [
                str(v).strip().upper()
                for v in watch_df[ticker_col].dropna()
                if str(v).strip()
            ]
            uploaded_tickers = list(dict.fromkeys(uploaded_tickers))
            st.success(f"Watchlist llegida: {len(uploaded_tickers)} tickers.")
    except Exception as e:
        st.error(f"Error llegint el CSV: {type(e).__name__}: {e}")

manual_list = [
    t.strip().upper()
    for t in manual_tickers.replace("\n", ",").replace(";", ",").split(",")
    if t.strip()
]
tickers = list(dict.fromkeys(uploaded_tickers + manual_list))

if tickers:
    st.write(f"**Tickers preparats ({len(tickers)}):** " + ", ".join(tickers))

analyze = st.button("🔎 ANALITZAR", type="primary", disabled=not tickers, use_container_width=True)

if analyze:
    with st.spinner("Descarregant dades i calculant Swing Score... Pot trigar una estona."):
        try:
            ranked = rank_tickers(
                tickers,
                period=period,
                interval="1d",
                sort_by="swing",
                stage2_only=False,
            )
        except Exception as e:
            st.exception(e)
            ranked = []

    if not ranked:
        st.warning("No s'han trobat resultats. Comprova els tickers i torna-ho a provar.")
    else:
        rows = []
        for idx, r in enumerate(ranked, 1):
            comps = r.get("swing_components") or {}
            reliability = r.get("stage2_reliability") or {}
            vs_ma = r.get("price_vs_ma30w")
            rs = r.get("rs")
            vol = r.get("vol_ratio")
            row = {
                "#": idx,
                "Ticker": r.get("ticker"),
                "Preu": round(float(r.get("price", np.nan)), 2),
                "Score règim": round(float(r.get("score", 0.0)), 4),
                "Swing opportunity": round(float(r.get("swing_opportunity_score", 0.0)), 4),
                "Nivell swing": r.get("swing_opportunity_label", "N/A"),
                "Tendència 25%": round(float(comps.get("trend_structure", np.nan)) * 100, 1) if comps.get("trend_structure") is not None else np.nan,
                "Momentum 20%": round(float(comps.get("momentum", np.nan)) * 100, 1) if comps.get("momentum") is not None else np.nan,
                "Força rel. 20d": round(float(comps.get("relative_strength", np.nan)) * 100, 1) if comps.get("relative_strength") is not None else np.nan,
                "Volum 20%": round(float(comps.get("volume", np.nan)) * 100, 1) if comps.get("volume") is not None else np.nan,
                "Qualitat/risc 15%": round(float(comps.get("quality_risk", np.nan)) * 100, 1) if comps.get("quality_risk") is not None else np.nan,
                "Etiqueta règim": str(r.get("label", "")).upper(),
                "Stage 2": "SÍ" if r.get("stage2") else "—",
                "Acció": get_action(r),
                "Fiabilitat hist.": reliability.get("reliability_label", "—") if r.get("stage2") else "—",
                "vs MA30w": f"{vs_ma * 100:+.1f}%" if vs_ma is not None else "n/a",
                "MA30w pujant": "Sí" if r.get("ma30w_rising") else "No",
                "RS (26 setm.)": f"{rs * 100:+.1f}%" if rs is not None else "n/a",
                "Vol×": f"{vol:.2f}" if vol is not None else "n/a",
            }
            rows.append(row)

        results_df = pd.DataFrame(rows)
        st.subheader(f"2. Resultats — {len(results_df)} tickers")
        st.caption("El valor de cada component es sobre 100. El Swing Score és una puntuació heurística, no una probabilitat de benefici.")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.subheader("3. Candidates Stage 2")
        stage2_df = results_df[results_df["Stage 2"] == "SÍ"].copy()
        if stage2_df.empty:
            st.info("Cap ticker compleix actualment el filtre Stage 2.")
        else:
            st.dataframe(stage2_df, use_container_width=True, hide_index=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descarregar resultats CSV",
            data=csv_bytes,
            file_name=f"swing_score_resultats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
