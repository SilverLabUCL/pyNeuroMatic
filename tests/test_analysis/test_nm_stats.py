# -*- coding: utf-8 -*-
import math
import numpy
import unittest
from typing import Union

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.analysis.nm_tool_stats as nms
import pyneuromatic.core.nm_utilities as nmu

QUIET = True
NM = NMManager(quiet=QUIET)


class NMStatsTest(unittest.TestCase):

    def setUp(self):
        n = 100
        self.win0 = {
            "on": True,
            "func": {"name": "mean"},
            "x0": 0,
            "x1": n+10,
            "transform": None,
            "bsln_on": True,
            "bsln_func": {"name": "mean"},
            "bsln_x0": 0,
            "bsln_x1": 10,
            }
        self.win1 = {
            "on": True,
            "func": {"name": "max"},
            "x0": 0,
            "x1": n+10,
            "transform": None,
            "bsln_on": True,
            "bsln_func": {"name": "mean"},
            "bsln_x0": 0,
            "bsln_x1": 10,
            }
        self.w0 = nms.NMStatsWin(NM, "w0", win=self.win0)
        self.w1 = nms.NMStatsWin(NM, "w1", win=self.win1)

        ydata = numpy.random.normal(loc=0, scale=1, size=n)
        self.data = NMData(
            NM, name="recordA0", nparray=ydata,
            yscale={"label": "Membrane Current", "units": "pA"},
            xscale={"label": "Time", "units": "ms", "start": 0, "delta": 1},
        )

        ydata = numpy.random.normal(loc=0, scale=1, size=n)
        ydata[3] = math.nan
        ydata[66] = math.nan
        self.datanan = NMData(
            NM, name="recordA0", nparray=ydata,
            yscale={"label": "Membrane Current", "units": "pA"},
            xscale={"label": "Time", "units": "ms", "start": 0, "delta": 1},
        )
    #    self.stats = NMToolStats()

    def test00_statswin_init(self):
        # args: parent, name, win
        for b in nmu.badtypes(ok=[{}, None]):
            with self.assertRaises(TypeError):
                nms.NMStatsWin(win=b)
        with self.assertRaises(TypeError):
            nms.NMStatsWin(copy=NM)  # unexpected kwarg

    def test01_statswin_eq(self):
        self.assertFalse(self.w0 == self.w1)
        self.w1._win_set(self.win0)
        self.assertFalse(self.w0 == self.w1)
        self.w1.name = self.w0.name
        self.assertTrue(self.w0 == self.w1)
        self.w1.x0 = -1
        self.assertFalse(self.w0 == self.w1)

        c = self.w0.copy()
        self.assertTrue(self.w0 == c)

        self.assertFalse(self.w0 == self.w1)

    def xtest02_statswin_copy(self):
        pass  # TODO

    def xtest03_statswin_parameters(self):
        # print(w0.parameters)
        pass  # TODO

    def test04_statswin_to_dict(self):
        keys = ["name", "on", "func", "x0", "x1", "transform",
                "bsln_on", "bsln_func", "bsln_x0", "bsln_x1"]
        self.assertTrue(isinstance(self.w0.to_dict(), dict))
        self.assertEqual(len(keys), len(self.w0.to_dict().keys()))
        for k in keys:
            self.assertTrue(k in self.w0.to_dict())
        self.assertEqual(self.w0.to_dict()["name"], "w0")

        for b in nmu.badtypes(ok=[{}]):
            with self.assertRaises(TypeError):
                self.w0._win_set(b)
        with self.assertRaises(KeyError):
            self.w0._win_set({"badkey": "max"})
        self.w0._win_set({"on": False, "bsln_on": False})
        self.assertFalse(self.w0.on)
        self.assertFalse(self.w0.bsln_on)
        self.w0._win_set(self.win1)
        self.assertEqual(self.w0.to_dict(), {"name": "w0", **self.win1})
        self.w0._win_set(self.win0)

    def test05_statswin_func(self):
        self.assertTrue(isinstance(self.w1.func, dict))
        self.assertTrue("name" in self.w1.func)
        self.w1._func_set(None)
        self.assertEqual(self.w1.func, {})
        self.w1._func_set({"name": None})
        self.assertEqual(self.w1.func, {})

        for b in nmu.badtypes(ok=[None, {}, "string"]):
            with self.assertRaises(TypeError):
                self.w0._func_set(b)
        for b in nmu.badtypes(ok=[None, "string"]):
            with self.assertRaises(TypeError):
                self.w0._func_set({"name": b})

        with self.assertRaises(KeyError):
            self.w1.func = {"badkey": "max"}
        with self.assertRaises(ValueError):
            self.w1.func = {"name": "badname"}

        # min, max, mean@max, mean@min, imean
        # see check_meanatmaxmin()
        self.w1._func_set({"name": "mean@max", "imean": 7})
        self.w1._func_set({"imean": "3"})
        self.assertEqual(self.w1.func["name"], "mean@max")
        self.assertEqual(self.w1.func["imean"], 3)
        self.w1._func_set({"imean": True})  # ok for float()
        self.assertEqual(self.w1.func["imean"], 1)
        with self.assertRaises(KeyError):
            self.w1._func_set({"name": "mean"})
            self.w1._func_set({"imean": 7})

        # level, ylevel
        # see check_level()
        self.w1._func_set({"name": "level+", "ylevel": -10})
        self.assertEqual(self.w1.func["name"], "level+")
        self.assertEqual(self.w1.func["ylevel"], -10)
        self.w1._func_set({"ylevel": "10"})
        self.assertEqual(self.w1.func["ylevel"], 10)
        self.w1._func_set({"ylevel": True})    # ok for float()
        self.assertEqual(self.w1.func["ylevel"], 1)
        with self.assertRaises(KeyError):
            self.w1._func_set({"name": "mean"})
            self.w1._func_set({"ylevel": 10})

        # risetime
        # see check_risefall()
        self.w1._func_set({"name": "risetime+", "p0": 10, "p1": 90})
        self.assertEqual(self.w1.func["name"], "risetime+")
        self.assertEqual(self.w1.func["p0"], 10)
        self.assertEqual(self.w1.func["p1"], 90)
        self.w1._func_set({"p0": "20", "p1": "80"})
        self.assertEqual(self.w1.func["p0"], 20)
        self.assertEqual(self.w1.func["p1"], 80)
        self.w1._func_set({"p0": 25, "p1": 75})
        self.assertEqual(self.w1.func["p0"], 25)
        self.assertEqual(self.w1.func["p1"], 75)
        with self.assertRaises(ValueError):
            self.w1._func_set({"p0": 75, "p1": 25})
        with self.assertRaises(KeyError):
            self.w1._func_set({"name": "mean"})
            self.w1._func_set({"p0": 10})

        # falltime
        # see check_risefall()
        self.w1._func_set({"name": "falltime+", "p0": 90, "p1": 10})
        self.assertEqual(self.w1.func["name"], "falltime+")
        self.assertEqual(self.w1.func["p0"], 90)
        self.assertEqual(self.w1.func["p1"], 10)
        self.w1._func_set({"p0": "80", "p1": "20"})
        self.assertEqual(self.w1.func["p0"], 80)
        self.assertEqual(self.w1.func["p1"], 20)
        self.w1._func_set({"p0": 75, "p1": 25})
        self.assertEqual(self.w1.func["p0"], 75)
        self.assertEqual(self.w1.func["p1"], 25)
        self.w1._func_set({"p0": 36})
        self.assertEqual(self.w1.func["p0"], 36)
        self.assertEqual(self.w1.func["p1"], None)
        with self.assertRaises(ValueError):
            self.w1._func_set({"p0": 25, "p1": 75})
        with self.assertRaises(KeyError):
            self.w1._func_set({"name": "mean"})
            self.w1._func_set({"p0": 10})

        # fwhm
        # see check_fwhm()
        for f in ["fwhm+", "fwhm-"]:
            self.w1._func_set({"name": f})
            self.assertEqual(self.w1.func["name"], f)
            self.assertEqual(self.w1.func["p0"], 50)  # auto set 50
            self.assertEqual(self.w1.func["p1"], 50)  # auto set 50
            self.w1._func_set({"p0": "40", "p1": "40"})
            self.assertEqual(self.w1.func["p0"], 40)
            self.assertEqual(self.w1.func["p1"], 40)

        fnames = ['max', 'min', 'mean@max', 'mean@min',
                  'median', 'mean', 'mean+var', 'mean+std',
                  'mean+sem', 'var', 'std', 'sem',
                  'rms', 'sum', 'pathlength', 'area', 'slope',
                  'level', 'level+', 'level-', 'value@x0', 'value@x1',
                  'count', 'count_nans', 'count_infs',
                  'risetime+', 'falltime+', 'risetimeslope+',
                  'falltimeslope+', 'fwhm+', 'risetime-', 'falltime-',
                  'risetimeslope-', 'falltimeslope-', 'fwhm-']
        self.assertEqual(len(nms.FUNC_NAMES), len(fnames))
        for f in fnames:
            self.assertTrue(f in nms.FUNC_NAMES)
        fnames = ['max', 'min',
                  'median', 'mean', 'mean+var', 'mean+std',
                  'mean+sem', 'var', 'std', 'sem',
                  'rms', 'sum', 'pathlength', 'area', 'slope',
                  'value@x0', 'value@x1',
                  'count', 'count_nans', 'count_infs',
                  'fwhm+', 'fwhm-']
        for f in fnames:
            # functions that dont require extra parameters
            self.w1._func_set({"name": f})
            self.assertEqual(self.w1.func["name"], f)

    def test06_statswin_x(self):
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                self.w0._x_set("x0", b)

        with self.assertRaises(ValueError):
            self.w0._x_set("x0", math.nan)
        with self.assertRaises(ValueError):
            self.w0._x_set("x0", "badvalue")
        self.w0._x_set("x0", "0")
        self.assertEqual(self.w0.x0, 0)
        self.w0._x_set("x0", "inf")  # should set to -inf
        self.assertTrue(math.isinf(self.w0.x0))
        self.assertTrue(self.w0.x0 < 0)
        self.w0._x_set("x1", "inf")
        self.assertTrue(math.isinf(self.w0.x1))
        self.assertTrue(self.w0.x1 > 0)

        self.w0._x_set("x0", -99)
        self.assertEqual(self.w0.x0, -99)
        self.w0._x_set("x1", 99)
        self.assertEqual(self.w0.x1, 99)
        self.w0._x_set("bsln_x0", -99)
        self.assertEqual(self.w0.bsln_x0, -99)
        self.w0._x_set("bsln_x1", 99)
        self.assertEqual(self.w0.bsln_x1, 99)

    def test07_statswin_transform(self):
        from pyneuromatic.core.nm_transform import (
            NMTransform,
            NMTransformInvert,
            NMTransformLog,
            NMTransformLn,
        )

        # Default is None
        self.assertIsNone(self.w0.transform)

        # Set with NMTransform objects
        transforms = [NMTransformInvert(), NMTransformLog()]
        self.w0.transform = transforms
        self.assertEqual(len(self.w0.transform), 2)
        self.assertIsInstance(self.w0.transform[0], NMTransformInvert)
        self.assertIsInstance(self.w0.transform[1], NMTransformLog)

        # Set with dicts (should auto-convert)
        self.w0.transform = [
            {"type": "NMTransformInvert"},
            {"type": "NMTransformLn"},
        ]
        self.assertEqual(len(self.w0.transform), 2)
        self.assertIsInstance(self.w0.transform[0], NMTransformInvert)
        self.assertIsInstance(self.w0.transform[1], NMTransformLn)

        # to_dict should serialize transforms to dicts
        win = self.w0.to_dict()
        self.assertIsInstance(win["transform"], list)
        self.assertEqual(win["transform"][0], {"type": "NMTransformInvert"})
        self.assertEqual(win["transform"][1], {"type": "NMTransformLn"})

        # Round-trip: win dict -> new NMStatsWin
        from pyneuromatic.analysis.nm_tool_stats import NMStatsWin
        w2 = NMStatsWin(win=win)
        self.assertIsNotNone(w2.transform)
        self.assertEqual(len(w2.transform), 2)
        self.assertIsInstance(w2.transform[0], NMTransformInvert)
        self.assertIsInstance(w2.transform[1], NMTransformLn)

        # Set to None
        self.w0.transform = None
        self.assertIsNone(self.w0.transform)

        # to_dict with None transform
        win = self.w0.to_dict()
        self.assertIsNone(win["transform"])

        # Type errors
        for b in nmu.badtypes(ok=[None, []]):
            with self.assertRaises(TypeError):
                self.w0.transform = b

    def test08_statswin_bsln_func(self):
        self.assertTrue(isinstance(self.w1.bsln_func, dict))
        self.assertTrue("name" in self.w1.bsln_func)

        for b in nmu.badtypes(ok=[None, {}, "string"]):
            with self.assertRaises(TypeError):
                self.w0._bsln_func_set(b)
        for b in nmu.badtypes(ok=[None, "string"]):
            with self.assertRaises(TypeError):
                self.w0._bsln_func_set({"name": b})

        with self.assertRaises(KeyError):
            self.w1.bsln_func = {"badkey": "max"}
        with self.assertRaises(ValueError):
            self.w1.bsln_func = {"name": "badname"}
        with self.assertRaises(ValueError):
            self.w1._bsln_func_set({"name": "mean@max", "imean": 7})

        fnames = ('median', 'mean', 'mean+var', 'mean+std',
                  'mean+sem')
        self.assertEqual(len(nms.BSLN_FUNC_NAMES), len(fnames))
        for f in fnames:
            self.assertTrue(f in nms.BSLN_FUNC_NAMES)

    def test09_statswin_results(self):
        self.assertTrue(isinstance(self.w1.results, list))
        # direct acces to this list allows modifications
        self.w1.results.append("test0")
        self.w1.results.append("test1")
        self.assertTrue("test1" in self.w1.results)
        self.w1.results.clear()
        self.assertTrue(len(self.w1.results) == 0)

    def test10_statswin_compute(self):
        for b in nmu.badtypes(ok=[None]):
            with self.assertRaises(TypeError):
                self.w0.compute(b)

        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        # print(r[0])
        # print(r[1])
        keys = ['win', 'id', 'func', 'x0', 'x1', 'data', 'i0', 'i1',
                'n', 'nans', 'infs', 's', 'sunits']
        self.assertEqual(len(r), 2)
        self.assertEqual(len(keys), len(r[0]))
        for k in keys:
            self.assertTrue(k in r[0])
        self.assertEqual(r[0]["id"], "bsln")
        self.assertEqual(r[0]["data"], self.datanan.path_str)
        keys = ['win', 'id', 'func', 'x0', 'x1', 'data', 'i0', 'i1',
                'n', 'nans', 'infs', 's', 'sunits', 'i',
                'x', 'xunits', 'Δs']
        self.assertEqual(len(keys), len(r[1]))
        for k in keys:
            self.assertTrue(k in r[1])
        self.assertEqual(r[0]["id"], "bsln")
        self.assertEqual(r[1]["id"], "main")
        self.assertTrue("Δs" in r[1])
        if r[1]["Δs"]:
            self.assertEqual(r[1]["Δs"], (r[1]["s"]-r[0]["s"]))

        r = self.w1.compute(self.datanan, xclip=False, ignore_nans=True)
        self.assertTrue(r[1]["i1"] is None)
        self.assertTrue("error" in r[1])

        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=False)
        self.assertTrue(numpy.isnan(r[0]["s"]))
        self.assertTrue(numpy.isnan(r[1]["s"]))
        self.assertTrue(numpy.isnan(r[1]["Δs"]))

        # test mean@max, mean@min, imean
        self.w1.func = {"name": "mean@max", "imean": 0}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertTrue("warning" in r[1]["func"])  # not enough points
        self.w1.func = {"name": "mean@max", "imean": 1}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertTrue("warning" in r[1]["func"])  # not enough points
        self.w1.func = {"name": "mean@max", "imean": 5}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertEqual(r[1]["func"]["name"], "mean@max")
        self.assertEqual(r[1]["func"]["imean"], 5)
        self.w1.func = {"name": "mean@min", "imean": 5}
        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        self.assertEqual(r[1]["func"]["name"], "mean@min")
        # print(r[0])
        # print(r[1])

        # test level nstd
        ylevel = 10
        self.w1.func = {"name": "level+", "ylevel": ylevel}
        # r = self.w1.compute(self.datanan, xclip=True, ignore_nans=True)
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        # NaNs: RuntimeWarning: invalid value encountered in greater
        self.assertTrue(len(r), 2)
        self.assertTrue("ylevel" in r[1]["func"])
        self.assertEqual(r[1]["func"]["ylevel"], ylevel)
        nstd = 2
        self.w1.func = {"name": "level+", "nstd": nstd}
        with self.assertRaises(RuntimeError):
            self.w1.bsln_on = False  # need baseline
            self.w1.compute(self.data, xclip=True, ignore_nans=True)
        with self.assertRaises(RuntimeError):
            self.w1.bsln_on = True
            # need baseline 'mean+std'
            self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.w1.bsln_func = "mean+std"
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertTrue("ylevel" in r[1]["func"])
        ds1 = round(r[1]["Δs"] * 1000)
        ds2 = round(nstd * r[0]["std"] * 1000)
        self.assertEqual(ds1, ds2)
        nstd = -2
        self.w1.func = {"name": "level-", "nstd": nstd}
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        ds1 = round(r[1]["Δs"] * 1000)
        ds2 = round(nstd * r[0]["std"] * 1000)
        self.assertEqual(ds1, ds2)

        # test falltime p0 = 36
        self.w1.bsln_x0 = 0
        self.w1.bsln_x1 = 10
        self.w1.x0 = -math.inf
        self.w1.x1 = math.inf
        self.w1.func = {"name": "falltime+", "p0": 36}
        r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
        self.assertEqual(r[0]["id"], "bsln")
        self.assertEqual(r[1]["id"], "falltime+")
        self.assertTrue("Δs" in r[1])
        if math.isnan(r[1]["Δs"]):
            self.assertEqual(len(r), 2)
        else:
            self.assertEqual(len(r), 3)
            self.assertEqual(r[1]["Δs"], (r[1]["s"]-r[0]["s"]))
            self.assertEqual(r[2]["id"], "falltime+")
            self.assertTrue("Δx" in r[2])
            if not math.isnan(r[2]["Δx"]):
                self.assertEqual(r[2]["Δx"], (r[2]["x"]-r[1]["x"]))

        r = self.w1.compute(self.datanan, xclip=True, ignore_nans=False)
        # print(r[0])
        # print(r[1])
        self.assertEqual(len(r), 2)  # missing r[2]
        self.assertTrue(numpy.isnan(r[1]["Δs"]))

        # risetime, falltime, fwhm, p0 and p1
        flist = ["risetime+", "risetime-",
                 "risetimeslope+", "risetimeslope-",
                 "falltime+", "falltime-",
                 "falltimeslope+", "falltimeslope-",
                 "fwhm+", "fwhm-"
                 ]
        for f in flist:

            rise = "rise" in f
            fall = "fall" in f
            slope = "slope" in f
            fwhm = "fwhm" in f

            for b in [-10, math.nan, -math.inf, "badvalue"]:
                with self.assertRaises(ValueError):
                    self.w1.func = {"name": f, "p0": b, "p1": 90}
                    self.w1.compute(self.data, xclip=True)
                with self.assertRaises(ValueError):
                    self.w1.func = {"name": f, "p0": 10, "p1": b}
                    self.w1.compute(self.data, xclip=True)

            if rise:
                p0 = 10
                p1 = 90
            elif fall:
                p0 = 90
                p1 = 10
            elif fwhm:
                p0 = 50
                p1 = 50
            else:
                break
            self.w1.func = {"name": f, "p0": p0, "p1": p1}

            with self.assertRaises(RuntimeError):
                self.w1.bsln_on = False  # need baseline
                r = self.w1.compute(self.data, xclip=True)  # x1 OOB

            self.w1.bsln_on = True
            r = self.w1.compute(self.datanan, xclip=True, ignore_nans=False)
            self.assertEqual(len(r), 2)
            self.assertTrue(math.isnan(r[1]["Δs"]))
            self.assertEqual(r[0]["id"], "bsln")
            if "+" in f:
                f1 = "max"
            elif "-" in f:
                f1 = "min"
            else:
                f1 = "unknown"
            self.assertEqual(r[1]["func"]["name"], f1)

            r = self.w1.compute(self.data, xclip=True, ignore_nans=True)
            if math.isnan(r[1]["Δs"]):
                self.assertEqual(len(r), 2)
                continue
            self.assertEqual(r[1]["Δs"], (r[1]["s"]-r[0]["s"]))
            self.assertTrue(len(r) >= 4)
            self.assertEqual(r[2]["p0"], p0)
            self.assertEqual(r[3]["p1"], p1)
            f2 = "unknown"
            if "+" in f:
                if rise:
                    f2 = "level+"
                    f3 = "level+"
                elif fall:
                    f2 = "level-"
                    f3 = "level-"
                elif fwhm:
                    f2 = "level+"
                    f3 = "level-"
            elif "-" in f:
                if rise:
                    f2 = "level-"
                    f3 = "level-"
                elif fall:
                    f2 = "level+"
                    f3 = "level+"
                elif fwhm:
                    f2 = "level-"
                    f3 = "level+"
            self.assertEqual(r[2]["func"]["name"], f2)
            self.assertEqual(r[3]["func"]["name"], f3)
            self.assertTrue("Δx" in r[3])
            if not math.isnan(r[3]["Δx"]):
                self.assertEqual(r[3]["Δx"], (r[3]["x"]-r[2]["x"]))
            if slope:
                if math.isnan(r[3]["Δx"]):
                    n = 4
                else:
                    n = 5
            else:
                n = 4
            self.assertEqual(len(r), n)
            for i in range(n):
                # print(r[i])
                if i > 0:
                    self.assertEqual(r[i]["id"], f)
            if slope and n == 5:
                if "error" in r[4]:
                    self.assertFalse("s" in r[4])
                    self.assertFalse("b" in r[4])
                else:
                    self.assertTrue("s" in r[4])
                    self.assertTrue("b" in r[4])

    def test11_stats(self):
        func = {"name": "mean"}
        for b in nmu.badtypes(ok=[]):
            with self.assertRaises(TypeError):
                nms.stats(b, {})
        for b in nmu.badtypes(ok=[{}]):
            with self.assertRaises(TypeError):
                nms.stats(self.datanan, b)
        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                nms.stats(self.datanan, {"name": b})
        for b in nmu.badtypes(ok=[3, 3.14]):
            with self.assertRaises(TypeError):
                nms.stats(self.datanan, func, x0=b)
            with self.assertRaises(TypeError):
                nms.stats(self.datanan, func, x1=b)
        for b in nmu.badtypes(ok=[None, {}]):
            with self.assertRaises(TypeError):
                nms.stats(self.datanan, func, results=b)

        ydata = numpy.array([1, 2, 3, 4])
        xdata = numpy.array([1, 2, 3, 4])
        data = NMData(NM, name="recordA0", nparray=ydata, xarray=xdata)
        data.xarray = numpy.array([1, 2, 3])
        with self.assertRaises(RuntimeError):
            nms.stats(data, func)

        func = {"name": "max"}
        r = nms.stats(self.datanan, func, x0=-100, x1=200, xclip=True)
        keys = ['data', 'i0', 'i1', 'n', 'nans', 'infs', 's', 'sunits',
                'i', 'x', 'xunits']
        self.assertEqual(list(r.keys()), keys)
        self.assertEqual(r["data"], self.datanan.path_str)
        self.assertEqual(r["i0"], 0)  # xclip = True
        pnts = len(self.datanan.nparray)
        self.assertEqual(r["i1"], pnts-1)  # xclip = True
        self.assertEqual(r["sunits"], self.datanan.yscale.units)
        self.assertEqual(r["xunits"], self.datanan.xscale.units)
        r = nms.stats(self.datanan, func, x0=-100, xclip=False)
        keys = ['data', 'i0', 'i1', 'error']
        self.assertEqual(list(r.keys()), keys)
        self.assertEqual(r["i0"], None)
        self.assertEqual(r["i1"], pnts-1)
        r = nms.stats(self.datanan, func, x1=200, xclip=False)
        self.assertEqual(r["i0"], 0)
        self.assertEqual(r["i1"], None)

        func = {"name": "value@x0"}
        r = nms.stats(self.datanan, func, x0=10)
        self.assertEqual(r["i0"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])
        func = {"name": "value@x1"}
        r = nms.stats(self.datanan, func, x1=10)
        self.assertEqual(r["i1"], 10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

        func = {"name": "max"}
        r = nms.stats(self.datanan, func, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])
        func = {"name": "min"}
        r = nms.stats(self.datanan, func, x0=10, x1=10)
        self.assertEqual(r["s"], self.datanan.nparray[10])

        func = {"name": "mean"}
        r = nms.stats(self.datanan, func)
        # print(r)
        self.datanan.nparray[3] = math.inf
        self.datanan.nparray[66] = math.inf
        r = nms.stats(self.datanan, func)
        # self.assertTrue("error" in r)
        # r = nms.stats(self.datanan, func, x0=pnts-1, x1=0)
        # self.assertEqual(r["i0"], None)  # switch
        # self.assertEqual(r["i1"], pnts-1)
        # print(r)
        # dx = 0.1
        # xdata = numpy.arange(0, n * dx, dx)

    def test91_find_level_crossings(self):
        # TODO test yarray and xarray
        ydata = self.data.nparray
        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(ydata, ylevel=0, func_name=b)
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(ydata, ylevel=b)
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(ydata, ylevel=0, xstart=b)
        for b in nmu.badtypes(ok=[3, 3.14, True, "string"]):
            with self.assertRaises(TypeError):
                nms.find_level_crossings(ydata, ylevel=0, xdelta=b)
        with self.assertRaises(ValueError):
            nms.find_level_crossings(ydata, ylevel=0, func_name="badname")
        ylevel = 2
        r = nms.find_level_crossings(ydata, ylevel=ylevel,
                                     func_name="level", i_nearest=False)
        self.assertTrue(isinstance(r, tuple))
        print(r)
        indexes = []
        xvalues = []
        yon = False
        for i in range(len(ydata)):
            if i == 0:
                yon = ydata[i] >= ylevel
                continue
            if yon:
                if ydata[i] < ylevel:
                    yon = False
                    indexes.append(i)
            else:
                if ydata[i] >= ylevel:
                    yon = True
                    indexes.append(i)
        
        # for i in indexes:
        #    xinterp = nms.xinterp(ylevel, )

    def test92_linear_regression(self):
        pass  # TODO

    def test93_check_meanatmaxmin(self):
        # mean@max, mean@min
        f = "mean@max"
        for b in nmu.badtypes(ok=[{}, "string"]):
            with self.assertRaises(TypeError):
                nms.check_meanatmaxmin(b)
        with self.assertRaises(KeyError):
            func = {"badname": "max"}
            nms.check_meanatmaxmin(func)
        with self.assertRaises(KeyError):
            func = {"name": "max", "badkey": 0}
            nms.check_meanatmaxmin(func)

        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                func = {"name": b}
                nms.check_meanatmaxmin(func)
        with self.assertRaises(ValueError):
            func = {"name": "badfuncname"}
            nms.check_meanatmaxmin(func)

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                func = {"name": f, "imean": b}
                nms.check_meanatmaxmin(func)
        for b in [-10, math.nan, "badvalue"]:
            with self.assertRaises(ValueError):
                func = {"name": f, "imean": b}
                nms.check_meanatmaxmin(func)
        with self.assertRaises(OverflowError):
            func = {"name": f, "imean": math.inf}
            nms.check_meanatmaxmin(func)
        with self.assertRaises(KeyError):
            func = {"name": f}
            nms.check_meanatmaxmin(func)

        for f in ["min", "max", "mean@max", "mean@min"]:
            func = {"name": f, "imean": 10}
            p = nms.check_meanatmaxmin(func)
            if f == "min" or f == "max":
                self.assertEqual(p["name"], "mean@"+f)  # changes
            else:
                self.assertEqual(p["name"], f)
            self.assertEqual(p["imean"], 10)
            func = {"NAME": f.upper(), "IMEAN": 10}
            p = nms.check_meanatmaxmin(func)
            if f == "min" or f == "max":
                self.assertEqual(p["name"], "mean@"+f)
            else:
                self.assertEqual(p["name"], f)
            self.assertEqual(p["imean"], 10)

    def test95_check_level(self):
        f = "level"
        for b in nmu.badtypes(ok=[{}, "string"]):
            with self.assertRaises(TypeError):
                nms.check_level(b)
        with self.assertRaises(KeyError):
            func = {"badname": f}
            nms.check_level(func)
        with self.assertRaises(KeyError):
            func = {"name": f, "badkey": 0}
            nms.check_level(func)
        with self.assertRaises(KeyError):  # both keys not allowed
            func = {"name": f, "ylevel": 10, "nstd": 2}
            nms.check_level(func)
        with self.assertRaises(KeyError):
            func = {"name": f}
            nms.check_level(func)

        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                func = {"name": b}
                nms.check_level(func)
        for b in ["badfuncname", "ylevel", "max"]:
            with self.assertRaises(ValueError):
                func = {"name": b}
                nms.check_level(func)

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                func = {"name": f, "ylevel": b}
                nms.check_level(func)
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                func = {"name": f, "ylevel": b}
                nms.check_level(func)
        for b in [math.nan, math.inf, "badvalue", 0]:
            with self.assertRaises(ValueError):
                func = {"name": f, "nstd": b}
                nms.check_level(func)
        for b in [None, [], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                func = {"name": f, "option": b}
                nms.check_level(func)
        for b in [math.nan, "badvalue"]:
            with self.assertRaises(ValueError):
                func = {"name": f, "option": b}
                nms.check_level(func)
        with self.assertRaises(OverflowError):
            func = {"name": f, "option": math.inf}
            nms.check_level(func)

        for f in ["level", "level+", "level-"]:
            y = 10
            func = {"name": f, "ylevel": y}
            p = nms.check_level(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["ylevel"], y)
            func = {"NAME": f.upper(), "YLEVEL": y}
            p = nms.check_level(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["ylevel"], y)
            n = 2
            func = {"name": f, "nstd": n}
            p = nms.check_level(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["nstd"], n)
            func = {"NAME": f.upper(), "NSTD": n}
            p = nms.check_level(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["nstd"], n)
            n = -2
            func = {"name": f, "nstd": n}
            p = nms.check_level(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["nstd"], n)
            func = {"NAME": f.upper(), "NSTD": n}
            p = nms.check_level(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["nstd"], n)

        with self.assertRaises(KeyError):
            func = {"name": f}
            nms.check_level(func)
        with self.assertRaises(KeyError):
            func = {"name": f}
            nms.check_level(func, option=1)
        func = {"name": f, "ylevel": 10}
        p = nms.check_level(func, option=3)
        self.assertEqual(p["ylevel"], 10)  # option is ignored

    def test97_check_risefall(self):
        f = "risetime+"
        for b in nmu.badtypes(ok=[{}, "string"]):
            with self.assertRaises(TypeError):
                nms.check_risefall(b)
        with self.assertRaises(KeyError):
            func = {"badname": f}
            nms.check_risefall(func)
        with self.assertRaises(KeyError):
            func = {"name": f, "badkey": 10}
            nms.check_risefall(func)
        with self.assertRaises(KeyError):
            func = {"name": f, "p0": 10, "badkey": 90}
            nms.check_risefall(func)
        with self.assertRaises(KeyError):
            func = {"name": f}  # need p0 and p1 keys
            nms.check_risefall(func)
        with self.assertRaises(KeyError):
            func = {"name": f, "p0": 10}  # need p1 key
            nms.check_risefall(func)
        with self.assertRaises(KeyError):
            func = {"name": f, "p1": 90}  # need p0 key
            nms.check_risefall(func)

        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                func = {"name": b}
                nms.check_risefall(func)
        for b in ["badfuncname", "rise", "fall"]:
            with self.assertRaises(ValueError):
                func = {"name": b}
                nms.check_risefall(func)

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                func = {"name": f, "p0": b}
                nms.check_risefall(func)
            with self.assertRaises(TypeError):
                func = {"name": f, "p0": 10, "p1": b}
                nms.check_risefall(func)
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                func = {"name": f, "p0": b}
                nms.check_risefall(func)
            with self.assertRaises(ValueError):
                func = {"name": f, "p0": 10, "p1": b}
                nms.check_risefall(func)

        with self.assertRaises(KeyError):
            func = {"name": f, "p0": 63}  # need p1
            nms.check_risefall(func)
        for b in [105, -1, "badvalue"]:
            with self.assertRaises(ValueError):
                func = {"name": f, "p0": b}
                nms.check_risefall(func)
        with self.assertRaises(ValueError):
            func = {"name": f, "p0": 90, "p1": 10}  # backwards
            nms.check_risefall(func)
        with self.assertRaises(ValueError):
            func = {"name": "falltime+", "p0": 10, "p1": 90}  # backwards
            nms.check_risefall(func)
        func = {"name": "falltime+", "p0": 36}
        p = nms.check_risefall(func)
        self.assertEqual(p["p0"], 36)

        for f in ["risetime+", "risetime-", "risetimeslope+",
                  "risetimeslope-"]:
            func = {"name": f, "p0": 10, "p1": 90}
            p = nms.check_risefall(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["p0"], 10)
            self.assertEqual(p["p1"], 90)
            func = {"name": f, "p0": 10.5, "p1": 89.5}
            p = nms.check_risefall(func)
            self.assertEqual(p["p0"], 10.5)
            self.assertEqual(p["p1"], 89.5)
        for f in ["falltime+", "falltime-", "falltimeslope+",
                  "falltimeslope-"]:
            func = {"name": f, "p0": 90, "p1": 10}
            p = nms.check_risefall(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["p0"], 90)
            self.assertEqual(p["p1"], 10)
            func = {"name": f, "p0": 36}
            p = nms.check_risefall(func)
            self.assertEqual(p["p0"], 36)
            self.assertEqual(p["p1"], None)

    def test99_check_fwhm(self):
        for b in nmu.badtypes(ok=[{}, "string"]):
            with self.assertRaises(TypeError):
                p = nms.check_fwhm(b)
        with self.assertRaises(KeyError):
            func = {"badname": "fwhm+"}
            p = nms.check_fwhm(func)
        with self.assertRaises(KeyError):
            func = {"name": "fwhm+", "badkey": 0}
            p = nms.check_fwhm(func)

        for b in nmu.badtypes(ok=["string"]):
            with self.assertRaises(TypeError):
                func = {"name": b}
                nms.check_fwhm(func)
        for b in ["badfuncname", "fwhmslope+", "fwhm"]:
            with self.assertRaises(ValueError):
                func = {"name": b}
                nms.check_fwhm(func)

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                func = {"name": "fwhm+", "p0": b, "p1": 50}
                nms.check_fwhm(func)
            with self.assertRaises(TypeError):
                func = {"name": "fwhm+", "p0": 50, "p1": b}
                nms.check_fwhm(func)
        for b in [-10, 110, math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                func = {"name": "fwhm+", "p0": b, "p1": 50}
                nms.check_fwhm(func)
            with self.assertRaises(ValueError):
                func = {"name": "fwhm+", "p0": 50, "p1": b}
                nms.check_fwhm(func)

        for f in ["fwhm+", "fwhm-"]:
            func = {"name": f}
            p = nms.check_fwhm(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["p0"], 50)
            self.assertEqual(p["p1"], 50)
            func = f
            p = nms.check_fwhm(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["p0"], 50)
            self.assertEqual(p["p1"], 50)
            func = {"NAME": f.upper(), "P0": 45, "P1": 55}
            p = nms.check_fwhm(func)
            self.assertEqual(p["name"], f)
            self.assertEqual(p["p0"], 45)
            self.assertEqual(p["p1"], 55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
