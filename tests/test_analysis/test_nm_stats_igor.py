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

from pyneuromatic.io.pxp import read_pxp
import pyneuromatic.analysis.nm_tool_stats as nms


_PXP = Path(__file__).parent.parent / "test_io" / "fixtures" / "nm02Jul04c0_002.pxp"

# Igor WaveStats results for RecordA0, t=500–1000 ms
# Source: NM_WaveStats_nm02Jul04c0_002.txt
_IGOR = {
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

X0, X1 = 500.0, 1000.0  # ms


def setUpModule():
    """Load the PXP file once for the whole module."""
    global _DATA
    folder = read_pxp(_PXP)
    _DATA = folder.data.get("RecordA0")


class TestIgorWaveStats(unittest.TestCase):
    """Compare stats() against Igor Pro WaveStats for RecordA0, t=500–1000 ms."""

    def _stat(self, name, **kwargs):
        return nms.stats(_DATA, {"name": name}, x0=X0, x1=X1,
                         xclip=True, **kwargs)

    def test_wave_loaded(self):
        self.assertIsNotNone(_DATA)
        self.assertEqual(_DATA.nparray.size, 60000)
        self.assertAlmostEqual(_DATA.xscale.start, 0.0)
        self.assertAlmostEqual(_DATA.xscale.delta, 0.02)

    def test_point_count(self):
        r = self._stat("count")
        self.assertEqual(r["n"], _IGOR["npnts"])

    def test_mean(self):
        r = self._stat("mean")
        self.assertAlmostEqual(r["s"], _IGOR["avg"], places=4)

    def test_std(self):
        r = self._stat("std")
        self.assertAlmostEqual(r["s"], _IGOR["sdev"], places=4)

    def test_sem(self):
        r = self._stat("sem")
        self.assertAlmostEqual(r["s"], _IGOR["sem"], places=6)

    def test_rms(self):
        r = self._stat("rms")
        self.assertAlmostEqual(r["s"], _IGOR["rms"], places=3)

    def test_max(self):
        r = self._stat("max")
        self.assertAlmostEqual(r["s"], _IGOR["max"], places=4)

    def test_max_location(self):
        r = self._stat("max")
        self.assertAlmostEqual(r["x"], _IGOR["maxLoc"], places=1)

    def test_min(self):
        r = self._stat("min")
        self.assertAlmostEqual(r["s"], _IGOR["min"], places=4)

    def test_min_location(self):
        r = self._stat("min")
        self.assertAlmostEqual(r["x"], _IGOR["minLoc"], places=1)

    def test_sum(self):
        r = self._stat("sum")
        self.assertAlmostEqual(r["s"], _IGOR["sum"], places=0)

    def test_median(self):
        r = self._stat("median")
        self.assertAlmostEqual(r["s"], _IGOR["median"], places=3)

    def test_area(self):
        # Igor may use trapezoidal integration; allow difference up to 2 units
        r = self._stat("area")
        self.assertAlmostEqual(r["s"], _IGOR["area"], delta=2.0)

    def test_slope(self):
        r = self._stat("slope")
        self.assertAlmostEqual(r["s"], _IGOR["slope"], places=6)

    def test_value_at_x0(self):
        r = self._stat("value@x0")
        self.assertAlmostEqual(r["s"], _IGOR["value@xbgn"], places=4)

    def test_value_at_x1(self):
        r = self._stat("value@x1")
        self.assertAlmostEqual(r["s"], _IGOR["value@xend"], places=4)

    def test_mean_at_max(self):
        # n_avg=500 determined by matching Igor minAvg/maxAvg to 3 decimal places
        r = nms.stats(_DATA, {"name": "mean@max", "n_avg": 500},
                      x0=X0, x1=X1, xclip=True)
        self.assertAlmostEqual(r["s"], _IGOR["maxAvg"], places=3)

    def test_mean_at_max_location(self):
        r = nms.stats(_DATA, {"name": "mean@max", "n_avg": 500},
                      x0=X0, x1=X1, xclip=True)
        self.assertAlmostEqual(r["x"], _IGOR["maxAvgLoc"], places=1)

    def test_mean_at_min(self):
        r = nms.stats(_DATA, {"name": "mean@min", "n_avg": 500},
                      x0=X0, x1=X1, xclip=True)
        self.assertAlmostEqual(r["s"], _IGOR["minAvg"], places=3)

    def test_mean_at_min_location(self):
        r = nms.stats(_DATA, {"name": "mean@min", "n_avg": 500},
                      x0=X0, x1=X1, xclip=True)
        self.assertAlmostEqual(r["x"], _IGOR["minAvgLoc"], places=1)

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
        i0 = round((X0 - _DATA.xscale.start) / _DATA.xscale.delta)
        i1 = round((X1 - _DATA.xscale.start) / _DATA.xscale.delta)
        yslice = _DATA.nparray[i0:i1 + 1]
        return nms.find_level_crossings(
            yslice, ylevel=_IGOR["ylevel"],
            func_name=func_name,
            xstart=X0,
            xdelta=_DATA.xscale.delta,
        )

    def test_level_crossings_in_range(self):
        _, xvalues = self._crossings("level")
        self.assertGreater(len(xvalues), 0)
        self.assertTrue(all(X0 <= x <= X1 for x in xvalues))

    def test_level_plus_crossings_in_range(self):
        _, xvalues = self._crossings("level+")
        self.assertGreater(len(xvalues), 0)
        self.assertTrue(all(X0 <= x <= X1 for x in xvalues))

    def test_level_minus_crossings_in_range(self):
        _, xvalues = self._crossings("level-")
        self.assertGreater(len(xvalues), 0)
        self.assertTrue(all(X0 <= x <= X1 for x in xvalues))

    def test_level_plus_before_level_minus(self):
        _, xp = self._crossings("level+")
        _, xm = self._crossings("level-")
        self.assertLess(xp[0], xm[0])


class TestIgorFWHM(unittest.TestCase):
    """Compare FWHM stats against Igor NeuroMatic results for RecordA0."""

    # TODO: add Igor FWHM results here

    def test_placeholder(self):
        pass  # replace with real Igor comparison once values are available


if __name__ == "__main__":
    unittest.main(verbosity=2)
