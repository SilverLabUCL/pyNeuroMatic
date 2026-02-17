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
        self.assertEqual(len(nms.FUNC_NAMES_BSLN), len(fnames))
        for f in fnames:
            self.assertTrue(f in nms.FUNC_NAMES_BSLN)

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

    def test93_NMStatsFuncMaxMin(self):
        # NMStatsFuncMaxMin constructor validation
        with self.assertRaises(ValueError):
            nms.NMStatsFuncMaxMin("badfuncname")

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncMaxMin("mean@max", imean=b)
        for b in [-10, math.nan, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncMaxMin("mean@max", imean=b)
        with self.assertRaises(OverflowError):
            nms.NMStatsFuncMaxMin("mean@max", imean=math.inf)
        with self.assertRaises(KeyError):
            nms.NMStatsFuncMaxMin("mean@max")  # imean required

        # max/min without imean → stays max/min
        t = nms.NMStatsFuncMaxMin("max")
        self.assertEqual(t.name, "max")
        self.assertEqual(t.to_dict(), {"name": "max"})
        t = nms.NMStatsFuncMaxMin("min")
        self.assertEqual(t.name, "min")

        # max/min + imean → upgrades to mean@max/mean@min
        for f in ["min", "max", "mean@max", "mean@min"]:
            t = nms.NMStatsFuncMaxMin(f, imean=10)
            if f == "min" or f == "max":
                self.assertEqual(t.name, "mean@" + f)
            else:
                self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["imean"], 10)

        # _stats_func_from_dict
        for f in ["min", "max", "mean@max", "mean@min"]:
            t = nms._stats_func_from_dict({"name": f, "imean": 10})
            if f == "min" or f == "max":
                self.assertEqual(t.name, "mean@" + f)
            else:
                self.assertEqual(t.name, f)
            self.assertEqual(t["imean"], 10)

        # unknown key via _stats_func_from_dict
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"name": "max", "badkey": 0})

    def test95_NMStatsFuncLevel(self):
        # NMStatsFuncLevel (ylevel) constructor validation
        with self.assertRaises(ValueError):
            nms.NMStatsFuncLevel("badfuncname", ylevel=10)
        with self.assertRaises(KeyError):  # missing ylevel
            nms.NMStatsFuncLevel("level")

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncLevel("level", ylevel=b)
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncLevel("level", ylevel=b)

        for f in ["level", "level+", "level-"]:
            y = 10
            t = nms.NMStatsFuncLevel(f, ylevel=y)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["ylevel"], y)
            self.assertFalse(t.needs_baseline)

        # NMStatsFuncLevelNstd constructor validation
        with self.assertRaises(ValueError):
            nms.NMStatsFuncLevelNstd("badfuncname", nstd=2)
        with self.assertRaises(KeyError):  # missing nstd
            nms.NMStatsFuncLevelNstd("level")

        for b in [math.nan, math.inf, "badvalue", 0]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncLevelNstd("level", nstd=b)

        for f in ["level", "level+", "level-"]:
            n = 2
            t = nms.NMStatsFuncLevelNstd(f, nstd=n)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["nstd"], n)
            self.assertTrue(t.needs_baseline)

            n = -2
            t = nms.NMStatsFuncLevelNstd(f, nstd=n)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["nstd"], n)

        # validate_baseline
        t = nms.NMStatsFuncLevelNstd("level", nstd=2)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        with self.assertRaises(RuntimeError):
            t.validate_baseline("mean")
        t.validate_baseline("mean+std")  # ok

        # _stats_func_from_dict dispatch
        t = nms._stats_func_from_dict({"name": "level+", "ylevel": 10})
        self.assertIsInstance(t, nms.NMStatsFuncLevel)
        self.assertEqual(t.name, "level+")
        self.assertEqual(t["ylevel"], 10)

        t = nms._stats_func_from_dict({"name": "level-", "nstd": -2})
        self.assertIsInstance(t, nms.NMStatsFuncLevelNstd)
        self.assertEqual(t.name, "level-")
        self.assertEqual(t["nstd"], -2)

        with self.assertRaises(KeyError):  # both keys
            nms._stats_func_from_dict(
                {"name": "level", "ylevel": 10, "nstd": 2})
        with self.assertRaises(KeyError):  # neither key
            nms._stats_func_from_dict({"name": "level"})
        with self.assertRaises(KeyError):  # bad key
            nms._stats_func_from_dict({"name": "level", "badkey": 0})

    def test97_NMStatsFuncRiseFall(self):
        # NMStatsFuncRise constructor validation
        f = "risetime+"
        with self.assertRaises(ValueError):
            nms.NMStatsFuncRiseTime("badfuncname", p0=10, p1=90)
        with self.assertRaises(ValueError):
            nms.NMStatsFuncRiseTime("falltime+", p0=90, p1=10)

        with self.assertRaises(KeyError):
            nms.NMStatsFuncRiseTime(f)  # need p0
        with self.assertRaises(KeyError):
            nms.NMStatsFuncRiseTime(f, p0=10)  # rise needs p1
        with self.assertRaises(KeyError):
            nms.NMStatsFuncRiseTime(f, p1=90)  # need p0

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncRiseTime(f, p0=b, p1=90)
            with self.assertRaises(TypeError):
                nms.NMStatsFuncRiseTime(f, p0=10, p1=b)
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncRiseTime(f, p0=b, p1=90)
            with self.assertRaises(ValueError):
                nms.NMStatsFuncRiseTime(f, p0=10, p1=b)

        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncRiseTime(f, p0=b, p1=90)
        with self.assertRaises(ValueError):
            nms.NMStatsFuncRiseTime(f, p0=90, p1=10)  # backwards

        for f in ["risetime+", "risetime-", "risetimeslope+",
                  "risetimeslope-"]:
            t = nms.NMStatsFuncRiseTime(f, p0=10, p1=90)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 10)
            self.assertEqual(t.to_dict()["p1"], 90)
            t = nms.NMStatsFuncRiseTime(f, p0=10.5, p1=89.5)
            self.assertEqual(t.to_dict()["p0"], 10.5)
            self.assertEqual(t.to_dict()["p1"], 89.5)

        # needs_baseline and validate_baseline
        t = nms.NMStatsFuncRiseTime("risetime+", p0=10, p1=90)
        self.assertTrue(t.needs_baseline)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")  # ok
        t.validate_baseline("median")  # ok

        # _stats_func_from_dict
        t = nms._stats_func_from_dict(
            {"name": "risetime+", "p0": 10, "p1": 90})
        self.assertIsInstance(t, nms.NMStatsFuncRiseTime)
        self.assertEqual(t.name, "risetime+")
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict(
                {"name": "risetime+", "p0": 10, "badkey": 90})

        # NMStatsFuncFall constructor validation
        f = "falltime+"
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFallTime("badfuncname", p0=90)
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFallTime("risetime+", p0=10, p1=90)

        with self.assertRaises(KeyError):
            nms.NMStatsFuncFallTime(f)  # need p0

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncFallTime(f, p0=b)
        for b in [math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFallTime(f, p0=b)

        for b in [105, -1]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFallTime(f, p0=b)
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFallTime(f, p0=10, p1=90)  # backwards

        # falltime p0 without p1 is ok
        t = nms.NMStatsFuncFallTime("falltime+", p0=36)
        self.assertEqual(t.to_dict()["p0"], 36)
        self.assertIsNone(t.to_dict()["p1"])

        for f in ["falltime+", "falltime-", "falltimeslope+",
                  "falltimeslope-"]:
            t = nms.NMStatsFuncFallTime(f, p0=90, p1=10)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 90)
            self.assertEqual(t.to_dict()["p1"], 10)
            t = nms.NMStatsFuncFallTime(f, p0=36)
            self.assertEqual(t.to_dict()["p0"], 36)
            self.assertIsNone(t.to_dict()["p1"])

        # needs_baseline and validate_baseline
        t = nms.NMStatsFuncFallTime("falltime+", p0=90, p1=10)
        self.assertTrue(t.needs_baseline)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")  # ok
        t.validate_baseline("median")  # ok

        # _stats_func_from_dict
        t = nms._stats_func_from_dict(
            {"name": "falltime+", "p0": 90, "p1": 10})
        self.assertIsInstance(t, nms.NMStatsFuncFallTime)
        self.assertEqual(t.name, "falltime+")
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict(
                {"name": "falltime+", "p0": 90, "badkey": 10})

    def test99_NMStatsFuncFWHM(self):
        # NMStatsFuncFWHM constructor validation
        with self.assertRaises(ValueError):
            nms.NMStatsFuncFWHM("badfuncname")

        for b in [[], (), {}, set(), NM]:
            with self.assertRaises(TypeError):
                nms.NMStatsFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(TypeError):
                nms.NMStatsFuncFWHM("fwhm+", p0=50, p1=b)
        for b in [-10, 110, math.nan, math.inf, "badvalue"]:
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFWHM("fwhm+", p0=b, p1=50)
            with self.assertRaises(ValueError):
                nms.NMStatsFuncFWHM("fwhm+", p0=50, p1=b)

        for f in ["fwhm+", "fwhm-"]:
            # defaults to 50/50
            t = nms.NMStatsFuncFWHM(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 50)
            self.assertEqual(t.to_dict()["p1"], 50)
            # custom values
            t = nms.NMStatsFuncFWHM(f, p0=45, p1=55)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict()["p0"], 45)
            self.assertEqual(t.to_dict()["p1"], 55)

        # needs_baseline and validate_baseline
        t = nms.NMStatsFuncFWHM("fwhm+")
        self.assertTrue(t.needs_baseline)
        with self.assertRaises(RuntimeError):
            t.validate_baseline(None)
        t.validate_baseline("mean")  # ok
        t.validate_baseline("median")  # ok

        # _stats_func_from_dict with defaults
        t = nms._stats_func_from_dict({"name": "fwhm+"})
        self.assertEqual(t.name, "fwhm+")
        self.assertEqual(t["p0"], 50)
        self.assertEqual(t["p1"], 50)
        t = nms._stats_func_from_dict({"name": "fwhm-", "p0": 45, "p1": 55})
        self.assertEqual(t["p0"], 45)
        self.assertEqual(t["p1"], 55)
        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"name": "fwhm+", "badkey": 0})

    def test100_NMStatsFunc_base(self):
        # NMStatsFunc base class
        import copy
        t = nms.NMStatsFunc("test")
        self.assertEqual(t.name, "test")
        self.assertFalse(t.needs_baseline)
        self.assertEqual(t.to_dict(), {"name": "test"})
        self.assertEqual(t["name"], "test")
        with self.assertRaises(KeyError):
            t["nonexistent"]
        with self.assertRaises(NotImplementedError):
            t.compute(None, 0, 1, False, False, None, {})

        # deepcopy resets _parent
        parent = object()
        t = nms.NMStatsFuncBasic("mean", parent=parent)
        t2 = copy.deepcopy(t)
        self.assertIsNone(t2._parent)
        self.assertEqual(t, t2)

        # __eq__
        t1 = nms.NMStatsFuncBasic("mean")
        t2 = nms.NMStatsFuncBasic("mean")
        self.assertEqual(t1, t2)
        t3 = nms.NMStatsFuncBasic("median")
        self.assertNotEqual(t1, t3)
        self.assertEqual(t1, {"name": "mean"})
        self.assertEqual(t1.__eq__(42), NotImplemented)

        # __repr__
        t = nms.NMStatsFuncBasic("mean")
        self.assertIn("NMStatsFuncBasic", repr(t))
        self.assertIn("mean", repr(t))

    def test101_NMStatsFuncBasic(self):
        # NMStatsFuncBasic constructor validation
        with self.assertRaises(ValueError):
            nms.NMStatsFuncBasic("badfuncname")
        for f in nms.FUNC_NAMES_BASIC:
            t = nms.NMStatsFuncBasic(f)
            self.assertEqual(t.name, f)
            self.assertEqual(t.to_dict(), {"name": f})

    def test102_stats_func_from_dict(self):
        # _stats_func_from_dict helper
        self.assertIsNone(nms._stats_func_from_dict(None))
        self.assertIsNone(nms._stats_func_from_dict({}))
        self.assertIsNone(nms._stats_func_from_dict({"name": None}))

        with self.assertRaises(KeyError):
            nms._stats_func_from_dict({"badkey": "mean"})
        with self.assertRaises(ValueError):
            nms._stats_func_from_dict({"name": "badname"})
        with self.assertRaises(TypeError):
            nms._stats_func_from_dict(42)
        with self.assertRaises(TypeError):
            nms._stats_func_from_dict({"name": 42})

        # string shorthand
        t = nms._stats_func_from_dict("mean")
        self.assertIsInstance(t, nms.NMStatsFuncBasic)
        self.assertEqual(t.name, "mean")

        # round-trip
        for f in nms.FUNC_NAMES:
            if f in nms.FUNC_NAMES_BASIC:
                t = nms._stats_func_from_dict({"name": f})
                self.assertEqual(t.name, f)
            elif f in ("max", "min"):
                t = nms._stats_func_from_dict({"name": f})
                self.assertEqual(t.name, f)
            elif f in ("mean@max", "mean@min"):
                t = nms._stats_func_from_dict({"name": f, "imean": 5})
                self.assertEqual(t.name, f)
            elif f in nms.FUNC_NAMES_FWHM:
                t = nms._stats_func_from_dict({"name": f})
                self.assertEqual(t.name, f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
