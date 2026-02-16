# -*- coding: utf-8 -*-
"""
[Module description].

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source 
Software Toolkit for Acquisition, Analysis and Simulation of 
Electrophysiological Data. Front. Neuroinform. 12:14. 
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations
import copy
import math
from typing import Any
import numpy as np

from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_preferences as nmp
from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.core.nm_transform import (
    NMTransform,
    apply_transforms,
    _transform_from_dict,
)
import pyneuromatic.core.nm_utilities as nmu

FUNC_NAMES = (
    "max",  # np.argmax
    "min",  # np.argmin
    "mean@max",
    "mean@min",
    "median",  # np.median
    "mean",  # "avg" # np.mean
    "mean+var",
    "mean+std",
    "mean+sem",
    "var",  # np.var
    "std",  # "sdev" # np.std
    "sem",
    "rms",
    "sum",  # np.sum
    "pathlength",
    "area",
    "slope",
    # TODO "onset",  # Igor version requires sigmoid curvefit
    "level",
    "level+",
    "level-",
    "value@x0",
    "value@x1",
    "count",  # "numpnts"
    "count_nans",
    "count_infs",
    # positive peaks
    "risetime+",
    "falltime+",
    "risetimeslope+",
    "falltimeslope+",
    "fwhm+",
    # negative peaks
    "risetime-",
    "falltime-",
    "risetimeslope-",
    "falltimeslope-",
    "fwhm-"
)

BSLN_FUNC_NAMES = (
    "median",
    "mean",
    "mean+var",
    "mean+std",
    "mean+sem",
    # "var",
    # "std",
    # "sem",
    # "rms",
    # "sum",
    # "pathlength",
    # "area",
    # "slope",
)


class NMToolStats(NMTool):
    """
    NM Stats Tool class
    """

    def __init__(self) -> None:

        self.__win_container = NMStatsWinContainer(parent=self)
        self.__win_container.new()

        self.__xclip = True
        # if x0|x1 OOB, clip to data x-scale limits
        # if x0 = -math.inf, then x0 will be clipped to smallest x-value
        # if x1 = math.inf, then x1 will be clipped to largest x-value

        self.__ignore_nans = True
        # NumPy array analysis
        # for example: if ignore_nans,
        # then np.nanmean(array) else np.mean(array)

        self.__results: dict[str, list[Any]] = {}
        # {"w0": [ [{}, {}], [{}, {}]... ],  stats win0
        #  "w1": [ [{}, {}], [{}, {}]... ],  stats win1
        #  ...}
        # for each stats window (e.g. "w0") there is a list
        # containing results [{}, {}] for each data array.
        # for each data array there is a list [{}, {}] containing
        # results for each measure {} made for the stats window
        # e.g. baseline, main, p0, p1, slope, etc.

    @property
    def windows(self) -> NMStatsWinContainer:
        return self.__win_container

    @property
    def xclip(self) -> bool:
        return self.__xclip

    @xclip.setter
    def xclip(self, xclip: bool) -> None:
        return self._xclip_set(xclip)

    def _xclip_set(
        self,
        xclip: bool,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(xclip, bool):
            self.__xclip = xclip
        else:
            e = nmu.type_error_str(xclip, "xclip", "boolean")
            raise TypeError(e)

    @property
    def ignore_nans(self) -> bool:
        return self.__ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, ignore_nans: bool) -> None:
        return self._ignore_nans_set(ignore_nans)

    def _ignore_nans_set(
        self,
        ignore_nans: bool,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(ignore_nans, bool):
            self.__ignore_nans = ignore_nans
        else:
            e = nmu.type_error_str(ignore_nans, "ignore_nans", "boolean")
            raise TypeError(e)

    # override, no super
    def execute_init(self) -> bool:
        if isinstance(self.__results, dict):
            self.__results.clear()
        return True  # ok

    # override, no super
    def execute(self) -> bool:
        if not isinstance(self.data, NMData):
            raise RuntimeError("no data selected")
        for w in self.windows:
            self.windows.selected_name = w.name
            if not w.on:
                continue
            w.compute(self.data, xclip=self.__xclip,
                      ignore_nans=self.__ignore_nans)
            # results saved to w.results
            if not w.results:
                continue
            if w.name in self.__results:
                self.__results[w.name].append(w.results)
            else:
                self.__results[w.name] = [w.results]
        return True  # ok

    # override, no super
    def execute_finish(self) -> bool:
        NMToolStats.results_print(self.__results)
        self.results_save()
        self.results_save_as_numpy()
        return True  # ok

    @staticmethod
    def results_print(results: dict[str, list[Any]]) -> None:
        if not isinstance(results, dict):
            return None
        for kwin, vlist in results.items():  # windows
            print("\n" + "stats results for win '%s':" % kwin)
            if not isinstance(vlist, list):
                return None
            for ilist in vlist:  # NMData
                if not isinstance(ilist, list):
                    return None
                for rdict in ilist:  # stats results
                    print(rdict)
        return None

    def results_save(self) -> str | None:
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")
        return self.folder.toolresults_save("stats", self.__results)

    def results_save_as_numpy(self) -> NMToolFolder | None:
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")
        for kwin, vlist in self.__results.items():  # windows
            if not isinstance(vlist, list) or len(vlist) == 0:
                return None  # error
            data_list = []
            bsln_func = None
            sbsln = []
            func = None
            s = []
            for ilist in vlist:
                if not isinstance(ilist, list):
                    return None  # error
                first = True
                for rdict in ilist:
                    if "data" not in rdict:
                        return None  # error
                    if first:
                        data_list.append(rdict["data"])
                        first = False
                    if "id" in rdict and rdict["id"] == "bsln":
                        if not bsln_func:
                            f = rdict["func"]
                            bsln_func = f["name"]
                        sbsln.append(rdict["s"])
                    else:
                        if not func:
                            f = rdict["func"]
                            func = f["name"]
                        s.append(rdict["s"])
            if len(data_list) != len(sbsln):
                return None  # error
            if len(data_list) != len(s):
                return None  # error

            fname = "stats_test"
            tf = self.folder.toolfolder
            if tf is None:
                return None
            f = tf.new(fname)
            if f is None:
                return None
            print(f.name)

            prefix = "ST_" + kwin

            data_np = np.array(data_list)
            dname = prefix + "_data"

            sbsln_np = np.array(sbsln)
            if f.data is not None:
                dname = prefix + "_bsln_" + str(bsln_func)
                f.data.new(dname, nparray=sbsln_np)

            s_np = np.array(s)
            if f.data is not None:
                dname = prefix + "_" + str(func)
                f.data.new(dname, nparray=s_np)

                print(f.data.content)

            # print(data_list)
            # print(s_bsln)
            # print(s)

        return f

    """
    def results_save(self) -> bool:
        r = {}
        r["tool"] = "stats"
        r["date"] = str(datetime.datetime.now())
        r["results"] = self.__results
        for i in range(99):
            sname = "stats" + str(i)
            if sname not in self.folder.toolresults:
                self.folder.toolresults[sname] = r
                break
        print(self.folder.toolresults)
        fname = "stats_test"
        f = self.folder.toolfolder.new(fname)
        f.data.new("ST_w0_avg_")
        "ST_" + w.name + func_name
        return True
    """


class NMStatsWin:
    """NM Stats Window class.

    Lightweight class (does not inherit NMObject) following the NMScaleY
    pattern. Each window defines a stats measurement with x-range, function,
    optional baseline, and optional transforms.
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMStatsWin0",
        win: dict[str, object] | None = None,
    ) -> None:
        self._parent = parent
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        if not name or not nmu.name_ok(name):
            raise ValueError("name: %s" % name)
        self._name = name

        self.__on = True
        self.__func: dict[str, Any] = {}
        self.__x0 = -math.inf
        self.__x1 = math.inf
        self.__transform: list[NMTransform] | None = None
        self.__results: list[dict[str, Any]] = []  # [ {}, {} ...] list of dictionaries

        # baseline
        self.__bsln_on = False
        self.__bsln_func: dict[str, Any] = {}
        self.__bsln_x0 = -math.inf
        self.__bsln_x1 = math.inf

        if win is None:
            pass  # ok
        elif isinstance(win, dict):
            self._win_set(win, quiet=True)
        else:
            e = nmu.type_error_str(win, "win", "dictionary")
            raise TypeError(e)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMStatsWin):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __deepcopy__(self, memo: dict) -> NMStatsWin:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attr, value in self.__dict__.items():
            if attr == "_parent":
                setattr(result, attr, None)
            else:
                setattr(result, attr, copy.deepcopy(value, memo))
        return result

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, newname: str) -> None:
        if not isinstance(newname, str):
            raise TypeError(nmu.type_error_str(newname, "newname", "string"))
        if not newname or not nmu.name_ok(newname):
            raise ValueError("newname: %s" % newname)
        self._name = newname

    def copy(self) -> NMStatsWin:
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Serialize this stats window to a dict."""
        if self.__transform is not None:
            transform_dicts = [t.to_dict() for t in self.__transform]
        else:
            transform_dicts = None
        return {
            "name": self._name,
            "on": self.__on,
            "func": self.__func,
            "x0": self.__x0,
            "x1": self.__x1,
            "transform": transform_dicts,
            "bsln_on": self.__bsln_on,
            "bsln_func": self.__bsln_func,
            "bsln_x0": self.__bsln_x0,
            "bsln_x1": self.__bsln_x1
        }

    def _win_set(
        self,
        win: dict[str, object],
        quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(win, dict):
            e = nmu.type_error_str(win, "win", "dictionary")
            raise TypeError(e)
        for k, v in win.items():
            if not isinstance(k, str):
                e = nmu.type_error_str(k, "key", "string")
                raise TypeError(e)
            k = k.lower()
            if k == "name":
                continue  # name is set via constructor, not _win_set
            elif k == "on":
                self._on_set(v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "func":
                self._func_set(v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "x0":
                self._x_set("x0", v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "x1":
                self._x_set("x1", v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "transform":
                self._transform_set(v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "bsln_on":
                self._bsln_on_set(v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "bsln_func":
                self._bsln_func_set(v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "bsln_x0":
                self._x_set("bsln_x0", v, quiet=quiet)  # type: ignore[arg-type]
            elif k == "bsln_x1":
                self._x_set("bsln_x1", v, quiet=quiet)  # type: ignore[arg-type]
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def on(self) -> bool:
        return self.__on

    @on.setter
    def on(self, on: bool) -> None:
        return self._on_set(on)

    def _on_set(self, on: bool, quiet: bool = nmp.QUIET) -> None:
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__on = on
        return None

    @property
    def func(self) -> dict:
        return self.__func

    @func.setter
    def func(self, func: dict | str) -> None:
        self._func_set(func)
        return None

    def _func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmp.QUIET
    ) -> None:

        if func is None:
            self.__func.clear()
            return None
        if isinstance(func, dict):
            if len(func) == 0:
                self.__func.clear()
                return None
            if "name" in func:
                func_name = func["name"]
            elif "name" in self.__func:
                func_name = self.__func["name"]
                # allows updating func parameters without passing name
            else:
                e = "missing func key 'name'"
                raise KeyError(e)
        elif isinstance(func, str):
            func_name = func
            func = {"name": func_name}
        else:
            e = nmu.type_error_str(func, "func", "dictionary, string or None")
            raise TypeError(e)

        if func_name is None:
            self.__func.clear()
            return None
        if not isinstance(func_name, str):
            e = nmu.type_error_str(func_name, "func_name", "string")
            raise TypeError(e)

        found = False
        for f in FUNC_NAMES:
            if f.lower() == func_name.lower():
                found = True
                break
        if not found:
            raise ValueError("func_name: %s" % func_name)

        f = func_name.lower()
        func.update({"name": f})  # make sure key "name" exists
        if f in ["min", "max", "mean@max", "mean@min"]:
            parameters = check_meanatmaxmin(func)
        elif f in ["level", "level+", "level-"]:
            parameters = check_level(func)
        elif "risetime" in f or "falltime" in f:
            parameters = check_risefall(func)
        elif f in ["fwhm+", "fwhm-"]:
            parameters = check_fwhm(func)
        else:  # no parameters for remaining functions
            for k in func.keys():
                if k.lower() != "name":
                    e = ("unknown key parameter '%s' for func '%s'"
                         % (k, func_name))
                    raise KeyError(e)
            parameters = {}

        self.__func.clear()
        self.__func.update({"name": f})
        if parameters:
            self.__func.update(parameters)

        return None

    @property
    def x0(self) -> float:
        return self.__x0

    @x0.setter
    def x0(self, x0: float) -> None:
        return self._x_set("x0", x0)

    def _x_set(
        self,
        xname: str,  # e.g. "x0" or "bsln_x0"
        x: float,
        quiet: bool = nmp.QUIET
    ) -> None:
        x = float(x)  # might raise type error
        if math.isnan(x):
            raise ValueError(xname + ": %s" % x)
        n = xname.lower()
        if n not in ("x0", "x1", "bsln_x0", "bsln_x1"):
            raise ValueError("xname: %s" % xname)
        if math.isinf(x):
            x = -math.inf if n.endswith("x0") else math.inf
        if n == "x0":
            self.__x0 = x
        elif n == "x1":
            self.__x1 = x
        elif n == "bsln_x0":
            self.__bsln_x0 = x
        elif n == "bsln_x1":
            self.__bsln_x1 = x
        return None

    @property
    def x1(self) -> float:
        return self.__x1

    @x1.setter
    def x1(self, x1: float) -> None:
        return self._x_set("x1", x1)

    @property
    def transform(self) -> list[NMTransform] | None:
        return self.__transform

    @transform.setter
    def transform(self, transform_list: list) -> None:
        return self._transform_set(transform_list)

    def _transform_set(
        self,
        transform_list: list[NMTransform] | list[dict] | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if transform_list is None:
            self.__transform = None
            return None
        if not isinstance(transform_list, list):
            e = nmu.type_error_str(transform_list, "transform_list", "list")
            raise TypeError(e)
        result = []
        for item in transform_list:
            if isinstance(item, NMTransform):
                result.append(item)
            elif isinstance(item, dict):
                result.append(_transform_from_dict(item))
            else:
                e = nmu.type_error_str(
                    item, "transform item", "NMTransform or dict"
                )
                raise TypeError(e)
        self.__transform = result
        return None

    @property
    def bsln_on(self) -> bool:
        return self.__bsln_on

    @bsln_on.setter
    def bsln_on(self, on: bool) -> None:
        return self._bsln_on_set(on)

    def _bsln_on_set(self, on: bool, quiet: bool = nmp.QUIET) -> None:
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__bsln_on = on
        return None

    @property
    def bsln_func(self) -> dict:
        return self.__bsln_func

    @bsln_func.setter
    def bsln_func(self, func: dict | str) -> None:
        self._bsln_func_set(func)
        return None

    def _bsln_func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmp.QUIET
    ) -> None:

        if func is None:
            self.__bsln_func.clear()
            return None
        if isinstance(func, dict):
            if len(func) == 0:
                self.__bsln_func.clear()
                return None
            if "name" not in func:
                e = "missing func key 'name'"
                raise KeyError(e)
            func_name = func["name"]
        elif isinstance(func, str):
            func_name = func
        else:
            e = nmu.type_error_str(func, "func", "dictionary, string or None")
            raise TypeError(e)

        if func_name is None:
            self.__bsln_func.clear()
            return None
        if not isinstance(func_name, str):
            e = nmu.type_error_str(func_name, "func_name", "string")
            raise TypeError(e)

        found = False
        for f in BSLN_FUNC_NAMES:
            if f.lower() == func_name.lower():
                found = True
                break
        if not found:
            raise ValueError("func_name: %s" % func_name)

        self.__bsln_func.clear()
        self.__bsln_func.update({"name": func_name.lower()})
        return None

    @property
    def bsln_x0(self) -> float:
        return self.__bsln_x0

    @bsln_x0.setter
    def bsln_x0(self, x0: float) -> None:
        return self._x_set("bsln_x0", x0)

    @property
    def bsln_x1(self) -> float:
        return self.__bsln_x1

    @bsln_x1.setter
    def bsln_x1(self, x1: float) -> None:
        return self._x_set("bsln_x1", x1)

    @property
    def results(self) -> list[dict]:
        return self.__results

    def _run_stat(self, data, func, id_str, x0, x1, xclip, ignore_nans,
                  **extra):
        """Create a result dict, append to results, and call stats()."""
        r: dict[str, Any] = {"win": self.name, "id": id_str}
        r.update(extra)
        r["func"] = func
        r["x0"] = x0
        r["x1"] = x1
        self.__results.append(r)
        stats(data, func, x0=x0, x1=x1, xclip=xclip,
              ignore_nans=ignore_nans, results=r)
        return r

    def compute(
        self,
        data: NMData,
        xclip: bool = False,  # if x0|x1 OOB, clip to data x-scale limits
        ignore_nans: bool = False
    ) -> list:

        # Apply transforms to a copy of the data (original never mutated)
        if self.__transform and len(self.__transform) > 0:
            if data is not None and data.nparray is not None:
                transformed = apply_transforms(
                    data.nparray, self.__transform, xscale=data.xscale
                )
                data = NMData(
                    name=data.name,
                    nparray=transformed,
                    xscale=data.xscale.to_dict(),
                    yscale=data.yscale.to_dict(),
                )

        self.__results = []

        if data is None or self.__func is None or "name" not in self.__func:
            return self.__results

        if not isinstance(xclip, bool):
            xclip = False

        if not isinstance(ignore_nans, bool):
            ignore_nans = True

        f = self.__func["name"]

        if not isinstance(f, str):
            e = nmu.type_error_str(f, "func_name", "string")
            raise TypeError(e)

        f = f.lower()
        level_nstd = "level" in f and "nstd" in self.__func
        rise = "risetime" in f
        fall = "falltime" in f
        fwhm = "fwhm" in f
        slope = "slope" in f

        fpeak: dict[str, Any] = {}
        fmaxmin: dict[str, Any] = {}
        flevelnstd: dict[str, Any] = {}

        if rise or fall or fwhm:
            if "+" in f:
                fpeak["name"] = "max"
            elif "-" in f:
                fpeak["name"] = "min"

        if fpeak and "imean" in self.__func:
            fpeak["imean"] = self.__func["imean"]

        if f == "mean@max" or f == "mean@min":
            fmaxmin["name"] = f

        if fmaxmin:
            w = None
            if "imean" in self.__func:
                imean = self.__func["imean"]
            else:
                imean = 0
            if imean <= 1:
                w = "not enough data points to compute a mean"
            fmaxmin["imean"] = imean
            if w:
                fmaxmin["warning"] = w

        if self.__bsln_on:

            if fpeak:
                bf_ok = False
                if "name" in self.__bsln_func:
                    bf = self.__bsln_func["name"]
                    bf = bf.lower()
                    if "mean" in bf or bf == "median":
                        bf_ok = True
                if not bf_ok:
                    e = ("peak func '%s' requires baseline 'mean' computation"
                         % f)
                    raise RuntimeError(e)
            elif level_nstd:
                bf_ok = False
                if "name" in self.__bsln_func:
                    bf = self.__bsln_func["name"]
                    if bf.lower() == "mean+std":
                        bf_ok = True
                if not bf_ok:
                    e = "level nstd requires baseline 'mean+std' computation"
                    raise RuntimeError(e)

            b = self._run_stat(
                data, self.__bsln_func.copy(), "bsln",
                self.__bsln_x0, self.__bsln_x1, xclip, ignore_nans
            )

            if level_nstd:
                nstd = self.__func["nstd"]
                if badvalue(nstd):
                    raise ValueError("level nstd: '%s'" % nstd)
                ylevel = math.nan
                if "s" in b and "std" in b:
                    s = b["s"]
                    std = b["std"]
                    if not badvalue(s) and not badvalue(std):
                        ylevel = s + nstd * std
                flevelnstd["name"] = f
                flevelnstd["ylevel"] = ylevel

        else:
            if fpeak:
                e = "peak func '%s' requires baseline 'mean' computation" % f
                raise RuntimeError(e)
            if level_nstd:
                e = "level nstd requires baseline 'mean+std' computation"
                raise RuntimeError(e)
            b = {}  # empty baseline

        if fpeak:
            main_id = f
            main_func = fpeak
        elif fmaxmin:
            main_id = "main"
            main_func = fmaxmin
        elif flevelnstd:
            main_id = "main"
            main_func = flevelnstd
        else:
            main_id = "main"
            main_func = self.__func.copy()
        r = self._run_stat(
            data, main_func, main_id,
            self.__x0, self.__x1, xclip, ignore_nans
        )

        ds = math.nan

        if self.__bsln_on:
            if "s" in b:
                bs = b["s"]
                if "s" in r:
                    rs = r["s"]
                elif "ylevel" in r["func"]:
                    rs = r["func"]["ylevel"]
                else:
                    rs = math.nan
                if not badvalue(bs) and not badvalue(rs):
                    ds = rs - bs
            r["Δs"] = ds

        if not fpeak:
            return self.__results  # finished

        if badvalue(ds):
            r["error"] = "unable to compute peak height Δs"
            return self.__results  # finished

        flevel: dict[str, Any] = {}
        fslope: dict[str, Any] = {"name": "slope"}

        if rise:

            if "+" in f:
                flevel["name"] = "level+"
            elif "-" in f:
                flevel["name"] = "level-"
            else:
                raise ValueError(
                    "peak func '%s' requires '+' or '-' sign" % f)

            if "p0" in self.__func:
                p0 = self.__func["p0"]
            else:
                raise KeyError("missing key 'p0'")
            if not (p0 > 0 and p0 < 100):
                raise ValueError("bad percent p0: %s" % p0)

            if "p1" in self.__func:
                p1 = self.__func["p1"]
            else:
                raise KeyError("missing key 'p1'")
            if not (p1 > 0 and p1 < 100):
                raise ValueError("bad percent p1: %s" % p1)

            flevel["ylevel"] = 0.01 * p0 * ds
            x1 = r["x"]
            r0 = self._run_stat(
                data, flevel, f, self.__x0, x1, xclip, ignore_nans, p0=p0
            )
            r0_error = "x" not in r0 or badvalue(r0["x"])

            flevel["ylevel"] = 0.01 * p1 * ds
            r1 = self._run_stat(
                data, flevel, f, self.__x0, x1, xclip, ignore_nans, p1=p1
            )
            r1_error = "x" not in r1 or badvalue(r1["x"])

            if r1_error:
                r1["error"] = "unable to locate p1 level"
                r1["Δx"] = math.nan
                return self.__results  # finished

            if r0_error:
                r1["Δx"] = math.nan
                return self.__results  # finished
            else:
                r1["Δx"] = r1["x"] - r0["x"]

            if slope:
                self._run_stat(
                    data, fslope, f, r0["x"], r1["x"], xclip, ignore_nans
                )

            return self.__results  # finished

        if fall:

            p1 = None

            if "+" in f:
                flevel["name"] = "level-"  # opposite sign
            elif "-" in f:
                flevel["name"] = "level+"  # opposite sign
            else:
                raise ValueError(
                    "peak func '%s' requires '+' or '-' sign" % f)

            if "p0" in self.__func:
                p0 = self.__func["p0"]
            else:
                raise KeyError("missing key 'p0'")
            if not (p0 > 0 and p0 < 100):
                raise ValueError("bad percent p0: %s" % p0)

            if "p1" in self.__func:
                p1 = self.__func["p1"]
                if p1 is not None and not (p1 > 0 and p1 < 100):
                    raise ValueError("bad percent p1: %s" % p1)

            flevel["ylevel"] = 0.01 * p0 * ds
            x0 = r["x"]
            r0 = self._run_stat(
                data, flevel, f, x0, self.__x1, xclip, ignore_nans, p0=p0
            )
            r0_error = "x" not in r0 or badvalue(r0["x"])
            if r0_error:
                r0["error"] = "unable to locate p0 level"

            if p1 is None:
                if r0_error:
                    r0["Δx"] = math.nan
                    return self.__results  # finished
                else:
                    r0["Δx"] = r0["x"] - x0  # use time of peak
            else:

                flevel["ylevel"] = 0.01 * p1 * ds
                r1 = self._run_stat(
                    data, flevel, f, x0, self.__x1, xclip, ignore_nans,
                    p1=p1
                )
                r1_error = "x" not in r1 or badvalue(r1["x"])

                if r1_error:
                    r1["error"] = "unable to locate p1 level"
                    r1["Δx"] = math.nan
                    return self.__results  # finished

                if r0_error:
                    r1["Δx"] = math.nan
                    return self.__results  # finished
                else:
                    r1["Δx"] = r1["x"] - r0["x"]

            if slope:
                if p1:
                    self._run_stat(
                        data, fslope, f, r0["x"], r1["x"],
                        xclip, ignore_nans
                    )
                else:
                    self._run_stat(
                        data, fslope, f, x0, r0["x"],
                        xclip, ignore_nans
                    )

            return self.__results  # finished

        if fwhm:

            flevel1: dict[str, Any] = {}
            flevel2: dict[str, Any] = {}

            if "+" in f:
                flevel1["name"] = "level+"
                flevel2["name"] = "level-"  # opposite sign
            elif "-" in f:
                flevel1["name"] = "level-"
                flevel2["name"] = "level+"  # opposite sign
            else:
                raise ValueError(
                    "peak func '%s' requires '+' or '-' sign" % f)

            p0 = self.__func.get("p0", 50)
            if not (p0 > 0 and p0 < 100):
                raise ValueError("bad percent p0: %s" % p0)
            p1 = self.__func.get("p1", 50)
            if not (p1 > 0 and p1 < 100):
                raise ValueError("bad percent p1: %s" % p1)
            w = None
            if p0 != 50 or p1 != 50:
                w = "unusual fwhm %% values: %s-%s" % (p0, p1)

            flevel1["ylevel"] = 0.01 * p0 * ds
            x1 = r["x"]
            extra0 = {"p0": p0}
            if w:
                extra0["warning"] = w
            r0 = self._run_stat(
                data, flevel1, f, self.__x0, x1, xclip, ignore_nans,
                **extra0
            )
            r0_error = "x" not in r0 or badvalue(r0["x"])
            if r0_error:
                r0["error"] = "unable to locate p0 level"

            flevel2["ylevel"] = 0.01 * p1 * ds
            x0 = r["x"]
            extra1 = {"p1": p1}
            if w:
                extra1["warning"] = w
            r1 = self._run_stat(
                data, flevel2, f, x0, self.__x1, xclip, ignore_nans,
                **extra1
            )
            r1_error = "x" not in r1 or badvalue(r1["x"])

            if r1_error:
                r1["error"] = "unable to locate p1 level"
                r1["Δx"] = math.nan
                return self.__results  # finished

            if r0_error:
                r1["Δx"] = math.nan
            else:
                r1["Δx"] = r1["x"] - r0["x"]

            return self.__results  # finished

        e = "unknown peak func '%s'" % f
        raise ValueError(e)

        return {}


class NMStatsWinContainer:
    """Simple container of NMStatsWin objects with auto-naming."""

    def __init__(
        self,
        parent: object | None = None,
        name_prefix: str = "w",
    ) -> None:
        self._parent = parent
        self._prefix = name_prefix
        self._windows: dict[str, NMStatsWin] = {}
        self._count = 0
        self.selected_name: str | None = None

    def new(self) -> NMStatsWin:
        name = "%s%d" % (self._prefix, self._count)
        self._count += 1
        w = NMStatsWin(parent=self._parent, name=name)
        self._windows[name] = w
        if self.selected_name is None:
            self.selected_name = name
        return w

    def __iter__(self):
        return iter(self._windows.values())

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, name: str) -> NMStatsWin:
        return self._windows[name]

    def __contains__(self, name: str) -> bool:
        return name in self._windows


def badvalue(n: float | None) -> bool:
    return n is None or math.isnan(n) or math.isinf(n)


def check_meanatmaxmin(
    func: dict,  # name = "mean@max" or "mean@min"
) -> dict:
    func_name = None
    imean = None
    fnames = ["max", "min", "mean@max", "mean@min"]

    if isinstance(func, dict):
        for key, v in func.items():
            k = key.lower()
            if k == "name":
                if isinstance(v, str):
                    func_name = v
                else:
                    e = nmu.type_error_str(v, "func name", "string")
                    raise TypeError(e)
            elif k == "imean":
                if v is not None:
                    imean = int(v)  # might raise type error
            else:
                raise KeyError("unknown func key '%s'" % k)
    elif isinstance(func, str):
        func_name = func
    else:
        e = nmu.type_error_str(func, "func", "dictionary or string")
        raise TypeError(e)
    if func_name is None:
        raise KeyError("missing func key 'name'")
    f = func_name.lower()
    if f not in fnames:
        e = "func name: '%s'" % func_name
        e += "\n" + "expected 'mean@max' or 'mean@min'"
        raise ValueError(e)
    if f == "max" or f == "min":
        if imean is None:
            return {"name": f}
    elif imean is None:
        raise KeyError("missing func key 'imean'")
    if math.isnan(imean) or math.isinf(imean) or imean < 0:
        raise ValueError("imean: '%s'" % imean)
    if "max" in f:
        return {"name": "mean@max", "imean": imean}
    if "min" in f:
        return {"name": "mean@min", "imean": imean}
    return {}


def check_level(
    func: dict | str,
    option: int = 0,
) -> dict:
    func_name = None
    ylevel = None
    nstd = None
    options = [1, 2, 3]
    fnames = ["level", "level+", "level-"]

    if not isinstance(option, int):
        option = int(option)

    if isinstance(func, dict):
        for key, v in func.items():
            k = key.lower()
            if k == "name":
                if isinstance(v, str):
                    func_name = v
                else:
                    e = nmu.type_error_str(v, "func name", "string")
                    raise TypeError(e)
            elif k == "option":
                option = int(v)  # might raise type error
            elif k == "ylevel":
                if v is not None:
                    ylevel = float(v)  # might raise type error
                option = 1
            elif k == "nstd":
                if v is not None:
                    nstd = float(v)  # might raise type error
                if nstd is not None and nstd > 0:
                    option = 2
                else:
                    option = 3
            else:
                raise KeyError("unknown func key '%s'" % k)
    elif isinstance(func, str):
        func_name = func
    else:
        e = nmu.type_error_str(func, "func", "dictionary or string")
        raise TypeError(e)
    if func_name is None:
        raise KeyError("missing func key 'name'")
    f = func_name.lower()
    if f not in fnames:
        e = "func name: '%s'" % func_name
        e += "\n" + "expected one of the following: %s" % fnames
        raise ValueError(e)

    if ylevel is not None and nstd is not None:
        e = "either 'ylevel' or 'nstd' is allowed, not both"
        raise KeyError(e)

    if option not in options:
        raise KeyError("missing func key 'ylevel' or 'nstd'")
    if option == 1:
        if ylevel is None:
            raise KeyError("missing func key 'ylevel'")
        if math.isnan(ylevel) or math.isinf(ylevel):
            raise ValueError("ylevel: '%s'" % ylevel)
        return {"name": f, "ylevel": ylevel}
    if option == 2:
        if nstd is None:
            raise KeyError("missing func key 'nstd'")
        if math.isnan(nstd) or math.isinf(nstd) or nstd == 0:
            raise ValueError("nstd: '%s'" % nstd)
        return {"name": f, "nstd": abs(nstd)}
    if option == 3:
        if nstd is None:
            raise KeyError("missing func key 'nstd'")
        if math.isnan(nstd) or math.isinf(nstd) or nstd == 0:
            raise ValueError("nstd: '%s'" % nstd)
        return {"name": f, "nstd": -1*abs(nstd)}
    raise ValueError("option: '%s'" % option)


def check_risefall(
    func: dict | str,
) -> dict:
    # check func dictionary is ok
    # if necessary, get input for p0 and p1
    func_name = None
    p0 = None
    p1 = None
    fnames = ["risetime+", "risetime-", "risetimeslope+", "risetimeslope-",
              "falltime+", "falltime-", "falltimeslope+", "falltimeslope-"]

    if isinstance(func, dict):
        for key, v in func.items():
            k = key.lower()
            if k == "name":
                if isinstance(v, str):
                    func_name = v
                else:
                    e = nmu.type_error_str(func_name, "func name", "string")
                    raise TypeError(e)
            elif k == "p0":
                if v is not None:
                    p0 = float(v)  # might raise type error
            elif k == "p1":
                if v is not None:
                    p1 = float(v)  # might raise type error
            else:
                raise KeyError("unknown func key '%s'" % k)
    elif isinstance(func, str):
        func_name = func.lower()
    else:
        e = nmu.type_error_str(func, "func", "dictionary or string")
        raise TypeError(e)
    if func_name is None:
        raise KeyError("missing func key 'name'")
    f = func_name.lower()
    if f not in fnames:
        e = "func name: '%s'" % func_name
        e += "\n" + "expected one of the following: %s" % fnames
        raise ValueError(e)
    rise = "risetime" in f
    fall = "falltime" in f
    if not rise and not fall:
        raise ValueError("expected func name 'risetime' or 'falltime'")

    if p0 is None:
        raise KeyError("missing func key 'p0'")
    elif math.isnan(p0) or math.isinf(p0):
        raise ValueError("p0: '%s'" % p0)
    elif p0 > 0 and p0 < 100:
        pass  # ok
    else:
        e = "bad percent p0: %s" % p0
        raise ValueError(e)
    if p1 is None:
        if rise:
            raise KeyError("missing func key 'p1'")
        return {"name": f, "p0": p0, "p1": p1}
    elif math.isnan(p1) or math.isinf(p1):
        raise ValueError("p1: '%s'" % p1)
    elif p1 > 0 and p1 < 100:
        pass  # ok
    else:
        e = "bad percent p1: %s" % p1
        raise ValueError(e)
    if rise:
        if p0 >= p1:
            e = "for risetime, need p0 < p1 but got %s >= %s" % (p0, p1)
            raise ValueError(e)
    elif fall:
        if p0 <= p1:
            e = "for falltime, need p0 > p1 but got %s <= %s" % (p0, p1)
            raise ValueError(e)
    return {"name": f, "p0": p0, "p1": p1}


def check_fwhm(
    func: dict | str
) -> dict:
    # check func dictionary is ok
    func_name = None
    p0 = None
    p1 = None
    fnames = ["fwhm+", "fwhm-"]

    if isinstance(func, dict):
        for key, v in func.items():
            k = key.lower()
            if k == "name":
                if isinstance(v, str):
                    func_name = v
                else:
                    e = nmu.type_error_str(v, "func name", "string")
                    raise TypeError(e)
            elif k == "p0":
                if v is not None:
                    p0 = float(v)  # might raise type error
            elif k == "p1":
                if v is not None:
                    p1 = float(v)  # might raise type error
            else:
                raise KeyError("unknown func key '%s'" % k)
    elif isinstance(func, str):
        func_name = func
    else:
        e = nmu.type_error_str(func, "func", "dictionary or string")
        raise TypeError(e)
    if func_name is None:
        raise KeyError("missing func key 'name'")
    if p0 is None:
        p0 = 50
    if p1 is None:
        p1 = 50
    f = func_name.lower()
    if f not in fnames:
        e = "func_name: '%s'" % func_name
        e += "\n" + "expected one of the following: %s" % fnames
        raise ValueError(e)
    if p0 > 0 and p0 < 100:
        pass  # ok
    else:
        raise ValueError("bad percent p0: %s" % p0)
    if p1 > 0 and p1 < 100:
        pass  # ok
    else:
        raise ValueError("bad percent p1: %s" % p1)

    return {"name": f, "p0": p0, "p1": p1}


def stats(
    data: NMData,
    func: dict,
    x0: float = -math.inf,
    x1: float = math.inf,  # math.inf denotes xclip = True
    xclip: bool = False,  # if x0|x1 OOB, clip to data x-scale limits
    ignore_nans: bool = False,
    results: dict | None = None
) -> dict:  # returns results

    if not isinstance(data, NMData):
        e = nmu.type_error_str(data, "data", "NMData")
        raise TypeError(e)
    if not isinstance(data.nparray, np.ndarray):
        e = nmu.type_error_str(data.nparray, "nparray", "NumPy.ndarray")
        raise TypeError(e)

    if not isinstance(func, dict):
        e = nmu.type_error_str(func, "func", "dictionary")
        raise TypeError(e)
    if "name" not in func:
        e = "missing key 'name' in func dictionary"
        raise KeyError(e)

    f = func["name"]
    if not isinstance(f, str):
        e = nmu.type_error_str(f, "func_name", "string")
        raise TypeError(e)
    f = f.lower()

    found_xarray = isinstance(data.xarray, np.ndarray)
    ysize = data.nparray.size

    if found_xarray and data.xarray.size != ysize:
        e = ("x-y paired NumPy arrays have different size: %s != %s"
             % (data.xarray.size, ysize))
        raise RuntimeError(e)

    if results is None:
        results = {}
    elif not isinstance(results, dict):
        e = nmu.type_error_str(results, "results", "dictionary")
        raise TypeError(e)

    # results["func"] = f
    results["data"] = data.path_str

    xunits = data.xscale.units
    yunits = data.yscale.units

    i0 = data.get_xindex(x0, clip=xclip)
    i1 = data.get_xindex(x1, clip=xclip)

    results["i0"] = i0
    results["i1"] = i1

    if i0 is None:
        e = "failed to compute i0 from x0"
        # raise ValueError(e)
        results["error"] = e
        return results
    if i1 is None:
        e = "failed to compute i1 from x1"
        # raise ValueError(e)
        results["error"] = e
        return results

    if f == "value@x0":
        results["s"] = data.nparray[i0]
        results["sunits"] = yunits
        return results
    if f == "value@x1":
        results["s"] = data.nparray[i1]
        results["sunits"] = yunits
        return results

    # if i0 == i1:  # 1-point array should be ok
        # e = "i0 = i1: %s = %s" % (i0, i1)
        # raise ValueError("i0 = i1: %s = %s" % (i0, i1))
        # results["error"] = e
        # return results
    if i0 > i1:  # switch
        isave = i0
        i0 = i1
        i1 = isave
        results["i0"] = i0
        results["i1"] = i1

    if i0 == 0 and i1 == ysize - 1:
        yarray = data.nparray
        if found_xarray:
            xarray = data.xarray
        else:
            xstart = data.xscale.start
    else:  # slice
        yarray = data.nparray[i0:i1+1]
        if found_xarray:
            xarray = data.xarray[i0:i1+1]
        else:
            xstart = data.get_xvalue(i0)

    nans = np.count_nonzero(np.isnan(yarray))
    infs = np.count_nonzero(np.isinf(yarray))
    if ignore_nans:
        n = yarray.size - nans
    else:
        n = yarray.size
    results["n"] = n
    results["nans"] = nans
    results["infs"] = infs

    maxmin = False

    if f == "max":
        if ignore_nans:
            index = np.nanargmax(yarray)
        else:
            index = np.argmax(yarray)
            # might be index of nan
        maxmin = True
    elif f == "min":
        if ignore_nans:
            index = np.nanargmin(yarray)
        else:
            index = np.argmin(yarray)
            # might be index of nan
        maxmin = True

    if maxmin:
        # should always get an index
        results["s"] = yarray[index]
        results["sunits"] = yunits
        i = int(index) + int(i0)  # shift due to slicing
        results["i"] = i
        results["x"] = data.get_xvalue(i)
        results["xunits"] = data.xscale.units

        imean = 0

        if "imean" in func and i >= 0 and i < ysize:
            imean = int(func["imean"])  # might raise type error
            if imean <= 1:
                imean = 0

        if imean > 1:
            if imean % 2 == 0:  # even
                i0 = int(i - 0.5 * imean)
                i1 = int(i0 + imean - 1)
            else:  # odd
                i0 = int(i - 0.5 * (imean - 1))
                i1 = int(i + 0.5 * (imean - 1))
            i0 = max(i0, 0)
            i0 = min(i0, ysize - 1)
            i1 = max(i1, 0)
            i1 = min(i1, ysize - 1)
            # print("i0: %s, i1: %s" % (i0, i1))
            if i0 == 0 and i1 == ysize - 1:
                yarray = data.nparray
            else:  # slice
                yarray = data.nparray[i0:i1+1]
            if ignore_nans:
                results["s"] = np.nanmean(yarray)
            else:
                results["s"] = np.mean(yarray)

        return results  # finished max/min

    if "level" in f:
        if "ylevel" in func:
            ylevel = func["ylevel"]
        else:
            e = "missing key 'ylevel'"
            raise KeyError(e)
        if found_xarray:
            i_x = find_level_crossings(
                            yarray,
                            ylevel,
                            func_name=f,
                            xarray=xarray,
                            ignore_nans=ignore_nans
            )
        else:
            xstart_val = xstart if isinstance(xstart, float) else 0.0
            xdelta_val = float(data.xscale.delta)
            i_x = find_level_crossings(
                            yarray,
                            ylevel,
                            func_name=f,
                            xstart=xstart_val,
                            xdelta=xdelta_val,
                            ignore_nans=ignore_nans
            )
        indexes = i_x[0]
        xvalues = i_x[1]
        if "func" in results:
            fxn = results["func"]
            if isinstance(fxn, dict):
                fxn.update({"yunits": yunits})
        # results["s"] = ylevel
        # results["sunits"] = yunits
        if indexes.size > 0:  # return first level crossing
            results["i"] = indexes[0] + i0  # shift due to slicing
            results["x"] = xvalues[0]  # shift not needed for x-values
            results["xunits"] = xunits
        else:
            results["i"] = None
            results["x"] = None
        return results  # finished level

    if f == "slope":
        if found_xarray:
            mb = linear_regression(
                            yarray,
                            xarray=xarray,
                            ignore_nans=ignore_nans
            )
        else:
            xstart_val = xstart if isinstance(xstart, float) else 0.0
            xdelta_val = float(data.xscale.delta)
            mb = linear_regression(
                            yarray,
                            xstart=xstart_val,
                            xdelta=xdelta_val,
                            ignore_nans=ignore_nans
            )
        if mb:
            results["s"] = mb[0]
            if isinstance(xunits, str) and isinstance(yunits, str):
                results["sunits"] = yunits + "/" + xunits
            else:
                results["sunits"] = None
            results["b"] = mb[1]
            if isinstance(yunits, str):
                results["bunits"] = yunits
            else:
                results["bunits"] = None
        else:
            results["s"] = None
            results["b"] = None
        return results

    if f == "median":
        if ignore_nans:
            results["s"] = np.nanmedian(yarray)
        else:
            results["s"] = np.median(yarray)
        results["sunits"] = yunits
        return results

    elif "mean" in f:
        if ignore_nans:
            results["s"] = np.nanmean(yarray)
        else:
            results["s"] = np.mean(yarray)
        results["sunits"] = yunits
        if f == "mean":
            return results
        # else continue to +var, +std, +sem

    elif f == "var":
        if ignore_nans:
            results["s"] = np.nanvar(yarray)
        else:
            results["s"] = np.var(yarray)
        if isinstance(yunits, str):
            results["sunits"] = yunits + "**2"
        else:
            results["sunits"] = None
        return results

    elif f == "std":
        if ignore_nans:
            results["s"] = np.nanstd(yarray)
        else:
            results["s"] = np.std(yarray)
        results["sunits"] = yunits
        return results

    elif f == "sem":
        if ignore_nans:
            std = np.nanstd(yarray)
        else:
            std = np.std(yarray)
        results["s"] = std / math.sqrt(n)
        results["sunits"] = yunits
        return results

    elif f == "rms":
        if ignore_nans:
            sos = np.nansum(np.square(yarray))
        else:
            sos = np.sum(np.square(yarray))
        results["s"] = math.sqrt(sos/n)
        results["sunits"] = yunits
        return results

    elif f == "sum":
        if ignore_nans:
            results["s"] = np.nansum(yarray)
        else:
            results["s"] = np.sum(yarray)
        results["sunits"] = yunits
        return results

    elif f == "pathlength":
        w = None
        if isinstance(xunits, str) and isinstance(yunits, str):
            if xunits != yunits:
                e = ("pathlength: x- and y-scales have " +
                     "different units: %s != %s" % (xunits, yunits))
                raise ValueError(e)
        else:
            w = "pathlength assumes x- and y-scales have the same units"
        if found_xarray:
            dx2 = np.square(np.diff(xarray))
            dy2 = np.square(np.diff(yarray))
            h = np.sqrt(np.add(dx2, dy2))
        else:
            dx = float(data.xscale.delta)
            dx2 = dx**2
            dy2 = np.square(np.diff(yarray))
            h = np.sqrt(dx2 + dy2)
        if ignore_nans:
            results["s"] = np.nansum(h)
        else:
            results["s"] = np.sum(h)
        results["sunits"] = yunits
        if w:
            results["warning"] = w
        return results

    elif f == "area":
        if found_xarray:
            if ignore_nans:
                results["s"] = np.nansum(np.multiply(xarray, yarray))
            else:
                results["s"] = np.sum(np.multiply(xarray, yarray))
        else:
            if ignore_nans:
                sum_y = np.nansum(yarray)
            else:
                sum_y = np.sum(yarray)
            results["s"] = sum_y * data.xscale.delta
        if isinstance(xunits, str) and isinstance(yunits, str):
            if xunits == yunits:
                results["sunits"] = xunits + "**2"
            else:
                results["sunits"] = xunits + "*" + yunits
        else:
            results["sunits"] = None
        return results

    elif f == "count" or f == "count_nans" or f == "count_infs":
        return results

    else:
        e = "unknown function '%s'" % func
        raise ValueError(e)

    if "+var" in f:
        if ignore_nans:
            results["var"] = np.nanvar(yarray)
        else:
            results["var"] = np.var(yarray)

    if "+std" in f:
        if ignore_nans:
            results["std"] = np.nanstd(yarray)
        else:
            results["std"] = np.std(yarray)

    if "+sem" in f:
        if ignore_nans:
            std = np.nanstd(yarray)
        else:
            std = np.std(yarray)
        results["sem"] = std / math.sqrt(n)

    return results


def find_level_crossings(
    yarray,
    ylevel: float,  # the y-axis level (yl) to search for
    func_name: str = "level",
    # "level":  find all level crossings (both pos and neg slopes)
    # "level+": find level crossings on positive slopes
    # "level-": find level crossings on negative slopes
    xarray=None,  # NumPy array containing x-scale
    xstart: float = 0,  # x-scale start value, used if xarray=None
    xdelta: float = 1,  # x-scale delta increment, used if xarray=None
    i_nearest: bool = True,
    # return array indexes (i-values) that are nearest to ylevel crossings
    # method uses linear interpoloation
    # otherwise returns array indexes immediately after level crossing (i1)
    x_interp: bool = True,
    # return estimated x-value at location of ylevel crossing
    # method uses linear interpoloation
    # otherwise returns x-values at corresponding i-values (e.g. x1 at i1)
    ignore_nans: bool = True
) -> tuple:
    #                         y y y y y
    #                    y1 y
    #
    #  yl-->
    #        y y y y y y0
    #
    #  y y y
    #
    #  i 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
    #  x 0 2 4 6 8 0 2 4 6 8 0 2 4 6 8 0 (x0 = 0, dx = 2)
    #  results:
    #    y0, i0=7, x0=14   before y-level crossing
    #    y1, i1=8, x1=16   after y-level crossing
    #    i-nearest = 8
    #    x-interp = 14.3333

    if not isinstance(func_name, str):
        e = nmu.type_error_str(func_name, "func_name", "string")
        raise TypeError(e)

    f = func_name.lower()
    if "level" not in f:
        e = "func_name: '%s'" % func_name
        raise ValueError(e)

    if isinstance(ylevel, float):
        if math.isinf(ylevel) or math.isnan(ylevel):
            raise ValueError("ylevel: '%s'" % ylevel)
    else:
        ylevel = float(ylevel)  # might raise type error

    if not isinstance(yarray, np.ndarray):
        e = nmu.type_error_str(yarray, "yarray", "NumPy.ndarray")
        raise TypeError(e)

    found_xarray = False

    if xarray is None:
        pass
    elif isinstance(xarray, np.ndarray):
        if xarray.size == yarray.size:
            found_xarray = True
        else:
            e = ("x-y paired NumPy arrays have different size: %s != %s"
                 % (xarray.size, yarray.size))
            raise RuntimeError(e)
    else:
        e = nmu.type_error_str(xarray, "xarray", "NumPy.ndarray")
        raise TypeError(e)

    if isinstance(xstart, float):
        if math.isinf(xstart) or math.isnan(xstart):
            raise ValueError("xstart: '%s'" % xstart)
    else:
        xstart = float(xstart)  # might raise type error

    if isinstance(xdelta, float):
        if math.isinf(xdelta) or math.isnan(xdelta):
            raise ValueError("xdelta: '%s'" % xdelta)
    else:
        xdelta = float(xdelta)  # might raise type error

    if not isinstance(i_nearest, bool):
        i_nearest = True

    if not isinstance(x_interp, bool):
        x_interp = True

    if not isinstance(ignore_nans, bool):
        ignore_nans = True

    level_crossings = np.diff(yarray > ylevel, prepend=False)
    # diff output array is one point smaller than yarray
    # so prepend 1 False to beginning
    # True denotes transition through ylevel, either pos or neg direction
    # RuntimeWarning: invalid value encountered in greater
    # warning due to NaNs
    """
    with warnings.catch_warnings(record=True) as wlist:
        level_crossings = np.diff(yarray > ylevel, prepend=False)
        for w in wlist:
            print(w)
            """
    locations = np.argwhere(level_crossings)  # grab True locations
    if len(locations.shape) != 2:  # shape = (N, 1)
        raise RuntimeError("locations shape should be 2")
    locations = locations[:, 0]
    indexes = []
    xvalues = []

    for i in locations:

        if i == 0:
            continue
            # e = "location index should be great than 0: %s" % i
            # raise ValueError(e)
            # should not occur since False is prepended to level_crossings

        # transition occurs between y0 and y1
        y0 = yarray[i-1]
        y1 = yarray[i]

        if f == "level+":
            if y1 <= y0:
                continue  # wrong slope
        elif f == "level-":
            if y1 >= y0:
                continue  # wrong slope

        if found_xarray:
            x0 = xarray[i-1]
            x1 = xarray[i]
            dx = x1 - x0
        else:
            x0 = xstart + (i - 1) * xdelta
            x1 = xstart + i * xdelta
            dx = xdelta

        if not i_nearest:  # save index just after y-level crossings
            indexes.append(i)
            xvalues.append(x1)
            continue

        # find closest index via linear interpolation
        # compute x-location via linear interpolation (x-interp)
        dy = y1 - y0
        m = dy / dx
        b = y1 - m * x1
        x = (ylevel - b) / m

        if abs(x - x0) <= abs(x - x1):
            indexes.append(i-1)
            if x_interp:
                xvalues.append(x)
            else:
                xvalues.append(x0)
        else:
            indexes.append(i)
            if x_interp:
                xvalues.append(x)
            else:
                xvalues.append(x1)

    return (np.array(indexes), np.array(xvalues))


def xinterp(ylevel, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    m = dy / dx
    b = y1 - m * x1
    x = (ylevel - b) / m
    return x


def linear_regression(
    yarray,
    xarray=None,
    xstart: float = 0,
    xdelta: float = 1,
    ignore_nans: bool = True
) -> tuple:  # (m, b)

    if not isinstance(yarray, np.ndarray):
        e = nmu.type_error_str(yarray, "yarray", "NumPy.ndarray")
        raise TypeError(e)

    found_xarray = False

    if xarray is None:
        pass
    elif isinstance(xarray, np.ndarray):
        if xarray.size == yarray.size:
            found_xarray = True
        else:
            e = ("x-y paired NumPy arrays have different size: %s != %s"
                 % (xarray.size, yarray.size))
            raise RuntimeError(e)
    else:
        e = nmu.type_error_str(xarray, "xarray", "NumPy.ndarray")
        raise TypeError(e)

    if not found_xarray:

        if isinstance(xstart, float):
            if math.isinf(xstart) or math.isnan(xstart):
                raise ValueError("xstart: '%s'" % xstart)
        elif isinstance(xstart, int) and not isinstance(xstart, bool):
            pass
        else:
            e = nmu.type_error_str(xstart, "xstart", "float")
            raise TypeError(e)

        if isinstance(xdelta, float):
            if math.isinf(xdelta) or math.isnan(xdelta):
                raise ValueError("xdelta: '%s'" % xdelta)
        elif isinstance(xdelta, int) and not isinstance(xdelta, bool):
            pass
        else:
            e = nmu.type_error_str(xdelta, "xdelta", "float")
            raise TypeError(e)

        x0 = xstart
        x1 = xstart + (yarray.size - 1) * xdelta
        xarray = np.linspace(x0, x1, yarray.size)

    if ignore_nans:
        xavg = np.nanmean(xarray)
        xsum = np.nansum(xarray)
        yavg = np.nanmean(yarray)
        ysum = np.nansum(yarray)
        n = float(np.count_nonzero(~np.isnan(yarray)))
        xdelta = xarray - xavg
        ydelta = yarray - yavg
        sumxy = np.nansum(np.multiply(xdelta, ydelta))
        sumxsqr = np.nansum(xdelta**2)
    else:
        xavg = np.mean(xarray)
        xsum = np.sum(xarray)
        yavg = np.mean(yarray)
        ysum = np.sum(yarray)
        n = float(yarray.size)
        xdelta = xarray - xavg
        ydelta = yarray - yavg
        sumxy = np.sum(np.multiply(xdelta, ydelta))
        sumxsqr = np.sum(xdelta**2)

    m = sumxy / sumxsqr
    b = (ysum - m * xsum) / n

    return (m, b)
