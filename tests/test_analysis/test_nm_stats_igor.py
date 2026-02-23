#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Igor Pro comparison tests for stats().

Validates pyNeuroMatic stats() results against Igor Pro WaveStats output
for RecordA0 from nm02Jul04c0_002.pxp, computed over t=500 to 1000 ms.

Reference values sourced from:
    tests/test_io/fixtures/NM_WaveStats_nm02Jul04c0_002.txt
    (Igor Pro WaveStats, x-range 500–1000 ms)
"""
import math
import unittest
from pathlib import Path

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.io.pxp import read_pxp
import pyneuromatic.analysis.nm_tool_stats as nms


_PXP = Path(__file__).parent.parent / "test_io" / "fixtures" / "nm02Jul04c0_002.pxp"

# Igor WaveStats results for RecordA0, t=500–1000 ms
# Source: NM_WaveStats_nm02Jul04c0_002.txt
_IGOR_R0 = {
    "npnts":  25001,
    "median":  -82.875,
    "avg":     -82.824234,
    "sdev":     0.39073047,
    "sem":      0.0024711471,
    "rms":     82.825157,
    "min":    -83.90625,
    "minLoc": 985.52002,
    "minAvg": -83.588821,
    "minAvgLoc": 985.52002,
    "max":    -81.53125,
    "maxLoc": 516.73999,
    "maxAvg": -81.752808,
    "maxAvgLoc": 516.73999,
    "sum":    -2070688.6,
    "area":    -41412.121,
    "slope":  -0.00064825657,
    "ylevel":  -82.0,
    "value@xbgn":  -82.406258,
    "value@xend":  -82.906258,
}

_X0, _X1 = 500.0, 1000.0  # ms
_N_AVG = 500  # n_avg for mean@max and mean@min (10 ms window at 0.02 ms sampling)

# ---------------------------------------------------------------------------
# Sine wave test parameters
# ---------------------------------------------------------------------------
# Half-sine pulse: y = AMP * sin(π*(t - T0) / TAU) for t in [T0, T0+TAU],
#                  y = 0 elsewhere.
# Baseline: flat at 0 for t in [0, T0).
# Igor replication: make wave with dx=0.02 ms, N=2001 points (t = 0 to 40 ms),
#   fill with the formula above, then run NM stats with:
#     baseline: mean over [0, T0-dx]
#     signal:   [T0, T0+TAU]

_S_DX  = 0.02   # ms, x-scale delta
_S_T0  = 10.0   # ms, pulse start (baseline ends just before this)
_S_TAU = 20.0   # ms, half-sine duration (peak at T0 + TAU/2 = 20 ms)
_S_AMP = 1.0    # amplitude (y-units)
_S_N   = 2001   # points: t = 0 to 40 ms inclusive

# Level-crossing times (ylevel = AMP/2).  Igor NM reference values.
# Analytical: T0 + TAU/π * arcsin(0.5) = 10 + 10/3 ≈ 13.333 ms (level+)
#             T0 + TAU - TAU/π * arcsin(0.5) = 30 - 10/3 ≈ 26.667 ms (level-)
_S_YLEVEL            = _S_AMP / 2.0   # = 0.5
_S_LEVEL_PLUS_X_IGOR  = 13.333337     # Igor NM result (ms)
_S_LEVEL_MINUS_X_IGOR = 26.666662     # Igor NM result (ms)

# Rise/fall time percentages.
# risetime:  p0=10 (lower), p1=90 (higher) — p0 < p1 required
# falltime:  p0=90 (higher, reached first after peak),
#            p1=10 (lower, reached last) — p0 > p1 required
_S_P0_RISE = 10.0   # %
_S_P1_RISE = 90.0   # %
_S_P0_FALL = 90.0   # % (NOTE: p0 > p1 for falltime)
_S_P1_FALL = 10.0   # %
_S_P1_DECAY = 36.79   # %

# Rise/fall time and slope.  Igor NM reference values.
# Analytical rise time: TAU/π * (arcsin(0.9) - arcsin(0.1)) ≈ 6.491 ms
# Analytical slope: (0.8 * AMP) / risetime ≈ 0.123 /ms (endpoint formula)
# Regression slope differs because np.polyfit fits all points in the window.
_S_RISETIME_IGOR  = 6.491004    # Igor NM result (ms); fall time is identical (symmetric)
_S_RISESLOPE_IGOR = 0.12547222  # Igor NM regression slope (positive, rising edge)

_S_DECAYTIME_IGOR = 7.6015363   # Igor NM decay-time result for p1=36.79% (1/e)

# FWHM = 2*TAU/3 (exact for half-sine, p0=p1=50%) ≈ 13.333 ms
_S_FWHM_IGOR = 13.333325        # Igor NM result (ms)

# NMStatsWin win-dict base shared by all rise/fall/FWHM tests
_S_WIN_BASE = {
    "bsln_on":  True,
    "bsln_func": {"name": "mean"},
    "bsln_x0":  0.0,
    "bsln_x1":  _S_T0 - _S_DX,   # 9.98 ms — just before pulse starts
    "x0":       _S_T0,
    "x1":       _S_T0 + _S_TAU,
}


def setUpModule():
    """Load PXP file and build sine-wave NMData once for the whole module."""
    global _DATA, _SINE_DATA
    NM = NMManager(quiet=True)

    folder = read_pxp(_PXP)
    _DATA = folder.data.get("RecordA0")

    t = np.arange(_S_N) * _S_DX
    y = np.zeros(_S_N, dtype=np.float64)
    mask = (t >= _S_T0) & (t <= _S_T0 + _S_TAU)
    y[mask] = _S_AMP * np.sin(np.pi * (t[mask] - _S_T0) / _S_TAU)
    _SINE_DATA = NMData(NM, name="SineWave", nparray=y,
                        xscale={"start": 0.0, "delta": _S_DX})


# ---------------------------------------------------------------------------
# Helper: extract the result dict that carries "Δx" (time interval)
# ---------------------------------------------------------------------------
def _dx_result(results):
    for r in reversed(results):
        if "Δx" in r:
            return r
    return {}


# ===========================================================================
# Igor WaveStats comparison tests (real PXP data)
# ===========================================================================

class TestIgorWaveStats(unittest.TestCase):
    """Compare stats() against Igor Pro WaveStats for RecordA0, t=500–1000 ms."""

    def _stat(self, name, **kwargs):
        return nms.stats(_DATA, {"name": name}, x0=_X0, x1=_X1,
                         xclip=True, **kwargs)

    def test_wave_loaded(self):
        self.assertIsNotNone(_DATA)
        self.assertEqual(_DATA.nparray.size, 60000)
        self.assertAlmostEqual(_DATA.xscale.start, 0.0)
        self.assertAlmostEqual(_DATA.xscale.delta, 0.02)

    def test_point_count(self):
        r = self._stat("count")
        self.assertEqual(r["n"], _IGOR_R0["npnts"])

    def test_mean(self):
        r = self._stat("mean")
        self.assertAlmostEqual(r["s"], _IGOR_R0["avg"], places=4)

    def test_std(self):
        r = self._stat("std")
        self.assertAlmostEqual(r["s"], _IGOR_R0["sdev"], places=4)

    def test_sem(self):
        r = self._stat("sem")
        self.assertAlmostEqual(r["s"], _IGOR_R0["sem"], places=6)

    def test_rms(self):
        r = self._stat("rms")
        self.assertAlmostEqual(r["s"], _IGOR_R0["rms"], places=3)

    def test_max(self):
        r = self._stat("max")
        self.assertAlmostEqual(r["s"], _IGOR_R0["max"], places=4)

    def test_max_location(self):
        r = self._stat("max")
        self.assertAlmostEqual(r["x"], _IGOR_R0["maxLoc"], places=1)

    def test_min(self):
        r = self._stat("min")
        self.assertAlmostEqual(r["s"], _IGOR_R0["min"], places=4)

    def test_min_location(self):
        r = self._stat("min")
        self.assertAlmostEqual(r["x"], _IGOR_R0["minLoc"], places=1)

    def test_sum(self):
        r = self._stat("sum")
        self.assertAlmostEqual(r["s"], _IGOR_R0["sum"], places=0)

    def test_median(self):
        r = self._stat("median")
        self.assertAlmostEqual(r["s"], _IGOR_R0["median"], places=3)

    def test_area(self):
        # Igor may use trapezoidal integration; allow difference up to 2 units
        r = self._stat("area")
        self.assertAlmostEqual(r["s"], _IGOR_R0["area"], delta=2.0)

    def test_slope(self):
        r = self._stat("slope")
        self.assertAlmostEqual(r["s"], _IGOR_R0["slope"], places=6)

    def test_value_at_x0(self):
        r = self._stat("value@x0")
        self.assertAlmostEqual(r["s"], _IGOR_R0["value@xbgn"], places=4)

    def test_value_at_x1(self):
        r = self._stat("value@x1")
        self.assertAlmostEqual(r["s"], _IGOR_R0["value@xend"], places=4)

    def test_mean_at_max(self):
        # n_avg=500 determined by matching Igor minAvg/maxAvg to 3 decimal places
        r = nms.stats(_DATA, {"name": "mean@max", "n_avg": _N_AVG},
                      x0=_X0, x1=_X1, xclip=True)
        self.assertAlmostEqual(r["s"], _IGOR_R0["maxAvg"], places=3)

    def test_mean_at_max_location(self):
        r = nms.stats(_DATA, {"name": "mean@max", "n_avg": _N_AVG},
                      x0=_X0, x1=_X1, xclip=True)
        self.assertAlmostEqual(r["x"], _IGOR_R0["maxAvgLoc"], places=1)

    def test_mean_at_min(self):
        r = nms.stats(_DATA, {"name": "mean@min", "n_avg": _N_AVG},
                      x0=_X0, x1=_X1, xclip=True)
        self.assertAlmostEqual(r["s"], _IGOR_R0["minAvg"], places=3)

    def test_mean_at_min_location(self):
        r = nms.stats(_DATA, {"name": "mean@min", "n_avg": _N_AVG},
                      x0=_X0, x1=_X1, xclip=True)
        self.assertAlmostEqual(r["x"], _IGOR_R0["minAvgLoc"], places=1)

    def test_no_nans(self):
        r = self._stat("count_nans")
        self.assertEqual(r["nans"], 0)

    def test_no_infs(self):
        r = self._stat("count_infs")
        self.assertEqual(r["infs"], 0)


class TestIgorLevelCrossings(unittest.TestCase):
    """Compare find_level_crossings() against Igor FindLevels for RecordA0.

    Igor FindLevels R=(500,600), level=-82.0 mV.
    Igor EDGE=0 found 64, EDGE=1 (level+) found 42, EDGE=2 (level-) found 44.

    Exact-touch crossings (data == level at a grid point) are counted by Igor
    but skipped by pyNeuroMatic's strict > comparison — this is by design.

    TODO: detailed per-crossing comparison pending investigation of differences
    between Igor FindLevels and pyNeuroMatic find_level_crossings algorithms.
    """

    def _crossings(self, func_name):
        # Slice to X0–X1 before calling find_level_crossings
        i0 = round((_X0 - _DATA.xscale.start) / _DATA.xscale.delta)
        i1 = round((_X1 - _DATA.xscale.start) / _DATA.xscale.delta)
        yslice = _DATA.nparray[i0:i1 + 1]
        return nms.find_level_crossings(
            yslice, ylevel=_IGOR_R0["ylevel"],
            func_name=func_name,
            xstart=_X0,
            xdelta=_DATA.xscale.delta,
        )

    def test_level_crossings_in_range(self):
        _, xvalues = self._crossings("level")
        self.assertGreater(len(xvalues), 0)
        self.assertTrue(all(_X0 <= x <= _X1 for x in xvalues))

    def test_level_plus_crossings_in_range(self):
        _, xvalues = self._crossings("level+")
        self.assertGreater(len(xvalues), 0)
        self.assertTrue(all(_X0 <= x <= _X1 for x in xvalues))

    def test_level_minus_crossings_in_range(self):
        _, xvalues = self._crossings("level-")
        self.assertGreater(len(xvalues), 0)
        self.assertTrue(all(_X0 <= x <= _X1 for x in xvalues))

    def test_level_plus_before_level_minus(self):
        _, xp = self._crossings("level+")
        _, xm = self._crossings("level-")
        self.assertLess(xp[0], xm[0])


class TestIgorRiseTime(unittest.TestCase):
    """Compare rise-time stats against Igor NeuroMatic results for RecordA0."""

    # TODO: add Igor rise-time results here

    def test_placeholder(self):
        pass  # replace with real Igor comparison once values are available


class TestIgorFallTime(unittest.TestCase):
    """Compare fall-time stats against Igor NeuroMatic results for RecordA0."""

    # TODO: add Igor fall-time results here

    def test_placeholder(self):
        pass  # replace with real Igor comparison once values are available


class TestIgorFWHM(unittest.TestCase):
    """Compare FWHM stats against Igor NeuroMatic results for RecordA0."""

    # TODO: add Igor FWHM results here

    def test_placeholder(self):
        pass  # replace with real Igor comparison once values are available


# ===========================================================================
# Sine wave analytical tests (synthetic data, analytically known values)
# ===========================================================================

class TestSineWaveLevelCrossings(unittest.TestCase):
    """Level-crossing tests on a half-sine pulse with analytical reference values.

    Data: y = AMP * sin(π*(t - T0) / TAU) for t in [T0, T0+TAU], 0 elsewhere.
    Parameters: AMP=1, T0=10 ms, TAU=20 ms, dx=0.02 ms, N=2001 points.

    Igor replication:
        Make a wave: dx=0.02, N=2001, start=0.
        Set y = sin(π*(x-10)/20) for x in [10, 30], 0 elsewhere.
        Run FindLevel / FindLevels with level=0.5.
    """

    def _crossings(self, func_name):
        return nms.find_level_crossings(
            _SINE_DATA.nparray,
            ylevel=_S_YLEVEL,
            func_name=func_name,
            xstart=0.0,
            xdelta=_S_DX,
        )

    def test_level_plus_count(self):
        # Exactly one rising crossing of 0.5 within the pulse
        _, xvals = self._crossings("level+")
        in_pulse = [x for x in xvals if _S_T0 <= x <= _S_T0 + _S_TAU]
        self.assertEqual(len(in_pulse), 1)

    def test_level_plus_location(self):
        # T0 + TAU/π * arcsin(0.5) = 10 + 10/3 ≈ 13.333 ms
        _, xvals = self._crossings("level+")
        in_pulse = [x for x in xvals if _S_T0 <= x <= _S_T0 + _S_TAU]
        self.assertAlmostEqual(in_pulse[0], _S_LEVEL_PLUS_X_IGOR, places=3)

    def test_level_minus_count(self):
        # Exactly one falling crossing of 0.5 within the pulse
        _, xvals = self._crossings("level-")
        in_pulse = [x for x in xvals if _S_T0 <= x <= _S_T0 + _S_TAU]
        self.assertEqual(len(in_pulse), 1)

    def test_level_minus_location(self):
        # T0 + TAU - TAU/π * arcsin(0.5) = 30 - 10/3 ≈ 26.667 ms
        _, xvals = self._crossings("level-")
        in_pulse = [x for x in xvals if _S_T0 <= x <= _S_T0 + _S_TAU]
        self.assertAlmostEqual(in_pulse[0], _S_LEVEL_MINUS_X_IGOR, places=3)

    def test_level_plus_before_level_minus(self):
        _, xp = self._crossings("level+")
        _, xm = self._crossings("level-")
        in_pulse_p = [x for x in xp if _S_T0 <= x <= _S_T0 + _S_TAU]
        in_pulse_m = [x for x in xm if _S_T0 <= x <= _S_T0 + _S_TAU]
        self.assertLess(in_pulse_p[0], in_pulse_m[0])


class TestSineWaveRiseTime(unittest.TestCase):
    """Rise-time tests on a half-sine pulse with analytical reference values.

    Data: same half-sine as TestSineWaveLevelCrossings.

    Igor replication:
        Same wave. Run NM Stats with:
            baseline: mean over [0, 9.98 ms]
            func: risetime+ (p0=10, p1=90)
            signal: [10, 30 ms]
        Compare Δt and slope results.
    """

    def _compute(self, func_dict):
        win = dict(_S_WIN_BASE)
        win["func"] = func_dict
        w = nms.NMStatsWin(name="w")
        w.bsln_on = win["bsln_on"]
        w.bsln_func = win["bsln_func"]
        w.bsln_x0 = win["bsln_x0"]
        w.bsln_x1 = win["bsln_x1"]
        w.func = win["func"]
        w.x0 = win["x0"]
        w.x1 = win["x1"]
        return w.compute(_SINE_DATA, xclip=True)

    def test_risetime_delta_x(self):
        # TAU/π * (arcsin(0.9) - arcsin(0.1)) ≈ 6.491 ms
        results = self._compute({"name": "risetime+", "p0": _S_P0_RISE, "p1": _S_P1_RISE})
        r = _dx_result(results)
        self.assertAlmostEqual(r["Δx"], _S_RISETIME_IGOR, places=3)

    def test_risetime_p0_location(self):
        # t at 10% of amplitude on rising edge ≈ 10.638 ms
        results = self._compute({"name": "risetime+", "p0": _S_P0_RISE, "p1": _S_P1_RISE})
        r = _dx_result(results)
        t_p0 = _S_T0 + _S_TAU / math.pi * math.asin(_S_P0_RISE / 100.0)
        self.assertAlmostEqual(r["x"] - r["Δx"], t_p0, places=1)

    def test_risetime_p1_location(self):
        # t at 90% of amplitude on rising edge ≈ 17.127 ms
        results = self._compute({"name": "risetime+", "p0": _S_P0_RISE, "p1": _S_P1_RISE})
        r = _dx_result(results)
        t_p1 = _S_T0 + _S_TAU / math.pi * math.asin(_S_P1_RISE / 100.0)
        self.assertAlmostEqual(r["x"], t_p1, places=1)

    def test_risetimeslope_delta_x(self):
        # Slope variant still reports the same Δx
        results = self._compute({"name": "risetimeslope+", "p0": _S_P0_RISE, "p1": _S_P1_RISE})
        r = _dx_result(results)
        self.assertAlmostEqual(r["Δx"], _S_RISETIME_IGOR, places=3)

    def test_risetimeslope_value(self):
        # Linear regression slope of rising half-sine from 10% to 90%.
        # Note: regression slope (≈ 0.12547) differs from the simple endpoint
        # slope (≈ 0.1233) because it fits all points between the two crossings.
        results = self._compute({"name": "risetimeslope+", "p0": _S_P0_RISE, "p1": _S_P1_RISE})
        slope_r = results[-1]   # slope result is appended last
        self.assertGreater(slope_r["s"], 0.0)
        self.assertAlmostEqual(slope_r["s"], _S_RISESLOPE_IGOR, delta=0.0001)


class TestSineWaveFallTime(unittest.TestCase):
    """Fall-time tests on a half-sine pulse with analytical reference values.

    Igor replication:
        Same wave as TestSineWaveRiseTime. Run NM Stats with:
            baseline: mean over [0, 9.98 ms]
            func: falltime+ (p0=90, p1=10)
            signal: [10, 30 ms]
    """

    def _compute(self, func_dict):
        win = dict(_S_WIN_BASE)
        win["func"] = func_dict
        w = nms.NMStatsWin(name="w")
        w.bsln_on = win["bsln_on"]
        w.bsln_func = win["bsln_func"]
        w.bsln_x0 = win["bsln_x0"]
        w.bsln_x1 = win["bsln_x1"]
        w.func = win["func"]
        w.x0 = win["x0"]
        w.x1 = win["x1"]
        return w.compute(_SINE_DATA, xclip=True)

    def test_falltime_delta_x(self):
        # Same magnitude as rise time (symmetric half-sine) ≈ 6.491 ms
        results = self._compute({"name": "falltime+", "p0": _S_P0_FALL, "p1": _S_P1_FALL})
        r = _dx_result(results)
        self.assertAlmostEqual(r["Δx"], _S_RISETIME_IGOR, places=3)

    def test_falltime_p0_location(self):
        # t at 90% of amplitude on falling edge ≈ 22.873 ms
        results = self._compute({"name": "falltime+", "p0": _S_P0_FALL, "p1": _S_P1_FALL})
        r = _dx_result(results)
        # r["x"] is the p1 (10%) crossing; p0 (90%) crossing = r["x"] - Δx
        t_p0_fall = _S_T0 + _S_TAU - _S_TAU / math.pi * math.asin(_S_P0_FALL / 100.0)
        self.assertAlmostEqual(r["x"] - r["Δx"], t_p0_fall, places=1)

    def test_falltime_p1_location(self):
        # t at 10% of amplitude on falling edge ≈ 29.362 ms
        results = self._compute({"name": "falltime+", "p0": _S_P0_FALL, "p1": _S_P1_FALL})
        r = _dx_result(results)
        t_p1_fall = _S_T0 + _S_TAU - _S_TAU / math.pi * math.asin(_S_P1_FALL / 100.0)
        self.assertAlmostEqual(r["x"], t_p1_fall, places=1)

    def test_falltimeslope_delta_x(self):
        results = self._compute({"name": "falltimeslope+", "p0": _S_P0_FALL, "p1": _S_P1_FALL})
        r = _dx_result(results)
        self.assertAlmostEqual(r["Δx"], _S_RISETIME_IGOR, places=3)

    def test_falltimeslope_value(self):
        # Linear regression slope of falling half-sine from 90% to 10%.
        # Symmetric to rise slope: expected ≈ -0.12547 /ms.
        results = self._compute({"name": "falltimeslope+", "p0": _S_P0_FALL, "p1": _S_P1_FALL})
        slope_r = results[-1]
        self.assertLess(slope_r["s"], 0.0)
        self.assertAlmostEqual(slope_r["s"], -_S_RISESLOPE_IGOR, delta=0.0001)


class TestSineWaveFWHM(unittest.TestCase):
    """FWHM tests on a half-sine pulse with analytical reference values.

    Analytical FWHM = 2*TAU/3 ≈ 13.333 ms (for p0=p1=50% on a half-sine).

    Igor replication:
        Same wave as TestSineWaveRiseTime. Run NM Stats with:
            baseline: mean over [0, 9.98 ms]
            func: fwhm+ (p0=50, p1=50)
            signal: [10, 30 ms]
    """

    def _compute(self, func_dict):
        win = dict(_S_WIN_BASE)
        win["func"] = func_dict
        w = nms.NMStatsWin(name="w")
        w.bsln_on = win["bsln_on"]
        w.bsln_func = win["bsln_func"]
        w.bsln_x0 = win["bsln_x0"]
        w.bsln_x1 = win["bsln_x1"]
        w.func = win["func"]
        w.x0 = win["x0"]
        w.x1 = win["x1"]
        return w.compute(_SINE_DATA, xclip=True)

    def test_fwhm_value(self):
        # 2 * TAU / 3 = 40/3 ≈ 13.333 ms
        results = self._compute({"name": "fwhm+", "p0": 50, "p1": 50})
        r = _dx_result(results)
        self.assertAlmostEqual(r["Δx"], _S_FWHM_IGOR, places=3)

    def test_fwhm_left_location(self):
        # Left (rising) half-max crossing ≈ 13.333 ms
        results = self._compute({"name": "fwhm+", "p0": 50, "p1": 50})
        r = _dx_result(results)
        self.assertAlmostEqual(r["x"] - r["Δx"], _S_LEVEL_PLUS_X_IGOR, places=3)

    def test_fwhm_right_location(self):
        # Right (falling) half-max crossing ≈ 26.667 ms
        results = self._compute({"name": "fwhm+", "p0": 50, "p1": 50})
        r = _dx_result(results)
        self.assertAlmostEqual(r["x"], _S_LEVEL_MINUS_X_IGOR, places=3)

    def test_fwhm_symmetry(self):
        # For a symmetric half-sine, left and right are equidistant from peak
        results = self._compute({"name": "fwhm+", "p0": 50, "p1": 50})
        r = _dx_result(results)
        peak_x = _S_T0 + _S_TAU / 2.0  # = 20.0 ms
        left_x = r["x"] - r["Δx"]
        right_x = r["x"]
        self.assertAlmostEqual(peak_x - left_x, right_x - peak_x, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
