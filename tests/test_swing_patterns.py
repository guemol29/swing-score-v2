import unittest

import numpy as np
import pandas as pd

from swing_patterns import detect_breakout_throwback, build_pattern_chart, build_pattern_overlays


def make_data(after_breakout):
    # Stable 20-session base: resistance 101, support 99.
    rows = [{"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000}
            for _ in range(20)]
    # A daily close 0.5% above resistance plus the 0.5% breakout buffer.
    rows.append({"Open": 101.0, "High": 103.0, "Low": 100.8, "Close": 102.0, "Volume": 1800})
    rows.extend(after_breakout)
    index = pd.bdate_range("2026-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=index)


def bar(close, low=None, high=None):
    return {
        "Open": close,
        "High": high if high is not None else close + 0.5,
        "Low": low if low is not None else close - 0.5,
        "Close": close,
        "Volume": 1200,
    }


class TestSwingPatternDetector(unittest.TestCase):
    def test_base_forming_without_breakout(self):
        data = pd.DataFrame([bar(100.0, 99.0, 101.0) for _ in range(21)],
                            index=pd.bdate_range("2026-01-01", periods=21))
        result = detect_breakout_throwback(data)
        self.assertTrue(result["base_detected"])
        self.assertFalse(result["breakout_detected"])
        self.assertEqual(result["pattern_status"], "BASE EN FORMACIÓ")

    def test_breakout_waits_for_retest(self):
        data = make_data([bar(105.0, 104.5, 106.0), bar(106.0, 105.0, 107.0)])
        result = detect_breakout_throwback(data)
        self.assertTrue(result["breakout_detected"])
        self.assertFalse(result["throwback_confirmed"])
        self.assertFalse(result["entry_confirmed"])
        self.assertEqual(result["pattern_status"], "ESPERANT THROWBACK")

    def test_close_below_support_invalidates_before_retest(self):
        data = make_data([bar(97.0, 96.5, 98.0)])
        result = detect_breakout_throwback(data)
        self.assertTrue(result["breakout_detected"])
        self.assertTrue(result["invalidated"])
        self.assertFalse(result["throwback_confirmed"])
        self.assertEqual(result["pattern_status"], "PATRÓ INVALIDAT")

    def test_throwback_requires_later_new_high_close(self):
        data = make_data([bar(100.8, 100.2, 101.3), bar(102.0, 101.5, 102.5)])
        result = detect_breakout_throwback(data)
        self.assertTrue(result["throwback_confirmed"])
        self.assertFalse(result["entry_confirmed"])
        self.assertEqual(result["pattern_status"], "ESPERANT NOU MÀXIM")

    def test_entry_after_throwback_and_new_high_close(self):
        data = make_data([bar(100.8, 100.2, 101.3), bar(104.0, 103.2, 104.5)])
        result = detect_breakout_throwback(data)
        self.assertTrue(result["throwback_confirmed"])
        self.assertTrue(result["entry_confirmed"])
        self.assertEqual(result["pattern_status"], "ENTRADA CONFIRMADA")

    def test_losing_support_after_throwback_invalidates_pattern(self):
        data = make_data([
            bar(100.8, 100.2, 101.3),
            bar(97.5, 97.0, 98.0),
        ])
        result = detect_breakout_throwback(data)
        self.assertTrue(result["invalidated"])
        self.assertFalse(result["throwback_confirmed"])
        self.assertFalse(result["entry_confirmed"])
        self.assertEqual(result["pattern_status"], "PATRÓ INVALIDAT")


    def test_pattern_overlays_include_base_breakout_and_throwback(self):
        data = make_data([bar(100.8, 100.2, 101.3), bar(104.0, 103.2, 104.5)])
        result = detect_breakout_throwback(data)
        chart = build_pattern_chart(data, result)
        overlays = build_pattern_overlays(chart, result)

        self.assertEqual(len(overlays["base"]), 1)
        self.assertEqual(
            set(overlays["levels"]["Nivell"]),
            {"Resistència", "Suport d'invalidació", "Màxim breakout"},
        )
        self.assertEqual(
            set(overlays["events"]["Esdeveniment"]),
            {"Breakout", "Throwback"},
        )

    def test_chart_calculates_ma200_from_full_history(self):
        n = 230
        close = np.linspace(90, 110, n)
        frame = pd.DataFrame({
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000,
        }, index=pd.bdate_range("2025-01-01", periods=n))
        chart = build_pattern_chart(frame, {}, max_rows=120)
        self.assertEqual(len(chart), 120)
        self.assertTrue(chart["MM200 diària"].notna().any())


if __name__ == "__main__":
    unittest.main()
