# -*- coding: utf-8 -*-
"""
NMToolStats and NMToolStats2: high-level stat tool classes.

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
import math
from typing import Any

import numpy as np

from pyneuromatic.analysis.nm_stat_func import FUNC_NAMES_BSLN  # noqa: F401
from pyneuromatic.analysis.nm_stat_utilities import stats
from pyneuromatic.analysis.nm_stat_win import NMStatWin, NMStatWinContainer  # noqa: F401
from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


class NMToolStats(NMTool):
    """
    NM Stats Tool class
    """

    def __init__(self) -> None:

        self.__win_container = NMStatWinContainer(parent=self)
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
        # {"w0": [ [{}, {}], [{}, {}]... ],  stat win0
        #  "w1": [ [{}, {}], [{}, {}]... ],  stat win1
        #  ...}
        # for each stat window (e.g. "w0") there is a list
        # containing results [{}, {}] for each data array.
        # for each data array there is a list [{}, {}] containing
        # results for each measure {} made for the stat window
        # e.g. baseline, main, p0, p1, slope, etc.

        self.__results_to_history = False
        self.__results_to_cache = True
        self.__results_to_numpy = False

    @property
    def windows(self) -> NMStatWinContainer:
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
        nmh.history("set xclip=%s" % xclip, quiet=quiet)

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
        nmh.history("set ignore_nans=%s" % ignore_nans, quiet=quiet)

    @property
    def results_to_history(self) -> bool:
        return self.__results_to_history

    @results_to_history.setter
    def results_to_history(self, value: bool) -> None:
        self._results_to_history_set(value)

    def _results_to_history_set(
        self,
        value: bool,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_history", "boolean"))
        self.__results_to_history = value
        nmh.history("set results_to_history=%s" % value, quiet=quiet)

    @property
    def results_to_cache(self) -> bool:
        return self.__results_to_cache

    @results_to_cache.setter
    def results_to_cache(self, value: bool) -> None:
        self._results_to_cache_set(value)

    def _results_to_cache_set(
        self,
        value: bool,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_cache", "boolean"))
        self.__results_to_cache = value
        nmh.history("set results_to_cache=%s" % value, quiet=quiet)

    @property
    def results_to_numpy(self) -> bool:
        return self.__results_to_numpy

    @results_to_numpy.setter
    def results_to_numpy(self, value: bool) -> None:
        self._results_to_numpy_set(value)

    def _results_to_numpy_set(
        self,
        value: bool,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_numpy", "boolean"))
        self.__results_to_numpy = value
        nmh.history("set results_to_numpy=%s" % value, quiet=quiet)

    # override, no super
    def run_init(self) -> bool:
        if isinstance(self.__results, dict):
            self.__results.clear()
        return True  # ok

    # override, no super
    def run(self) -> bool:
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
    def run_finish(self) -> bool:
        if self.__results_to_history:
            self._results_to_history()
        if self.__results_to_cache:
            self._results_to_cache()
        if self.__results_to_numpy:
            self._results_to_numpy()
        return True  # ok

    def _results_to_history(self, quiet: bool = False) -> None:
        if not isinstance(self.__results, dict):
            return None
        for kwin, vlist in self.__results.items():  # windows
            nmh.history(
                "stat results for win '%s':" % kwin,
                quiet=quiet,
            )
            if not isinstance(vlist, list):
                return None
            for ilist in vlist:  # NMData
                if not isinstance(ilist, list):
                    return None
                for rdict in ilist:  # stat results
                    nmh.history(str(rdict), quiet=quiet)
        return None

    def _results_to_cache(self) -> int | None:
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")
        return self.folder.toolresults_save("stats", self.__results)

    # Numeric keys extracted from result dicts into NMData arrays.
    # Maps result dict key → NMData name suffix and units source key.
    _NUMERIC_KEYS: dict[str, tuple[str, str | None]] = {
        "s":    ("s",    "sunits"),
        "x":    ("x",    "xunits"),
        "i":    ("i",    None),
        "i0":   ("i0",   None),
        "i1":   ("i1",   None),
        "n":    ("n",    None),
        "nans": ("nans", None),
        "infs": ("infs", None),
        "var":  ("var",  "sunits"),
        "std":  ("std",  "sunits"),
        "sem":  ("sem",  "sunits"),
        "b":    ("b",    "bunits"),
        "dx":   ("dx",   "xunits"),
    }

    def _results_to_numpy(self) -> NMToolFolder | None:
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")

        # Find next unused folder name stats0, stats1, ...
        tf = self.folder.toolfolder
        i = 0
        f = None
        while f is None:
            try:
                f = tf.new(name="stats%d" % i)
            except KeyError:
                i += 1

        for wname, vlist in self.__results.items():
            # vlist: list of lists, one inner list per data wave
            # Each inner list contains result dicts for that wave

            # Collect data path strings (one per wave, from first rdict)
            data_paths: list[str] = []
            # Collect result dicts grouped by id: {id_str: [rdict, ...]}
            id_rdicts: dict[str, list[dict]] = {}

            for ilist in vlist:
                wave_path: str | None = None
                for rdict in ilist:
                    id_str = rdict.get("id", "main")
                    if wave_path is None:
                        wave_path = rdict.get("data", "")
                    if id_str not in id_rdicts:
                        id_rdicts[id_str] = []
                    id_rdicts[id_str].append(rdict)
                if wave_path is not None:
                    data_paths.append(wave_path)

            # Save data path strings
            if data_paths and f.data is not None:
                f.data.new(
                    "ST_%s_data" % wname,
                    nparray=np.array(data_paths, dtype=object),
                )

            # Save numeric arrays per id
            for id_str, rdicts in id_rdicts.items():
                for rkey, (suffix, units_key) in self._NUMERIC_KEYS.items():
                    values = [rdict.get(rkey) for rdict in rdicts]
                    if all(v is None for v in values):
                        continue  # key not present for this func
                    arr = np.array(
                        [v if v is not None else math.nan for v in values],
                        dtype=float,
                    )
                    units = rdicts[0].get(units_key) if units_key else None
                    yscale = {"units": units} if units else None
                    dname = "ST_%s_%s_%s" % (wname, id_str, suffix)
                    if f.data is not None:
                        f.data.new(dname, nparray=arr, yscale=yscale)

                # Save warnings if any occurred
                warnings = [rdict.get("warning") for rdict in rdicts]
                if any(w is not None for w in warnings):
                    dname = "ST_%s_%s_warning" % (wname, id_str)
                    if f.data is not None:
                        f.data.new(
                            dname,
                            nparray=np.array(
                                [w or "" for w in warnings], dtype=object
                            ),
                        )

        return f


class NMToolStats2:
    """Compute summary statistics of Stats results (ST_ arrays).

    Operates on a NMToolFolder produced by NMToolStats._results_to_numpy(),
    computing stats() on each selected ST_ array and saving results as
    ST2_ arrays in the same folder.
    """

    # Keys written to ST2_ arrays (in order)
    _ST2_KEYS = ("mean", "std", "sem", "N", "NaNs", "INFs", "rms", "min", "max")

    def __init__(self) -> None:
        self.__results: dict[str, Any] = {}
        self.__ignore_nans: bool = True
        self.__results_to_history: bool = False
        self.__results_to_cache: bool = True
        self.__results_to_numpy: bool = False

    # --- ignore_nans (same pattern as NMToolStats) ---

    @property
    def ignore_nans(self) -> bool:
        return self.__ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, value: bool) -> None:
        self._ignore_nans_set(value)

    def _ignore_nans_set(
        self, value: bool, quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self.__ignore_nans = value
        nmh.history("set ignore_nans=%s" % value, quiet=quiet)

    # --- save flags (same pattern as NMToolStats) ---

    @property
    def results_to_history(self) -> bool:
        return self.__results_to_history

    @results_to_history.setter
    def results_to_history(self, value: bool) -> None:
        self._results_to_history_set(value)

    def _results_to_history_set(
        self, value: bool, quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_history", "boolean"))
        self.__results_to_history = value
        nmh.history("set results_to_history=%s" % value, quiet=quiet)

    @property
    def results_to_cache(self) -> bool:
        return self.__results_to_cache

    @results_to_cache.setter
    def results_to_cache(self, value: bool) -> None:
        self._results_to_cache_set(value)

    def _results_to_cache_set(
        self, value: bool, quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_cache", "boolean"))
        self.__results_to_cache = value
        nmh.history("set results_to_cache=%s" % value, quiet=quiet)

    @property
    def results_to_numpy(self) -> bool:
        return self.__results_to_numpy

    @results_to_numpy.setter
    def results_to_numpy(self, value: bool) -> None:
        self._results_to_numpy_set(value)

    def _results_to_numpy_set(
        self, value: bool, quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_numpy", "boolean"))
        self.__results_to_numpy = value
        nmh.history("set results_to_numpy=%s" % value, quiet=quiet)

    @property
    def results(self) -> dict:
        return self.__results

    def compute(
        self,
        toolfolder: NMToolFolder,
        select: str = "all",
        ignore_nans: bool | None = None,
    ) -> dict[str, Any]:
        """Compute summary statistics on ST_ arrays in a NMToolFolder.

        Args:
            toolfolder: NMToolFolder containing ST_ arrays from NMToolStats.
            select: Name of a specific ST_ array, or "all" to process all
                numeric ST_ arrays.
            ignore_nans: If True, exclude NaN values from calculations.
                Defaults to None, which uses self.ignore_nans.

        Returns:
            Results dict keyed by source array name.
        """
        if not isinstance(toolfolder, NMToolFolder):
            raise TypeError(
                nmu.type_error_str(toolfolder, "toolfolder", "NMToolFolder")
            )
        if not isinstance(select, str):
            raise TypeError(nmu.type_error_str(select, "select", "string"))
        if ignore_nans is None:
            ignore_nans = self.__ignore_nans
        elif not isinstance(ignore_nans, bool):
            ignore_nans = self.__ignore_nans

        self.__results = {}

        # Collect target ST_ arrays
        targets: list[NMData] = []
        if select.lower() == "all":
            for name in toolfolder.data:
                if name.startswith("ST_") and not name.endswith("_data"):
                    d = toolfolder.data.get(name)
                    if (d is not None
                            and isinstance(d.nparray, np.ndarray)
                            and d.nparray.dtype.kind in ("f", "i", "u")):
                        targets.append(d)
        else:
            d = toolfolder.data.get(select)
            if d is None:
                raise KeyError("array not found in toolfolder: %s" % select)
            if not isinstance(d.nparray, np.ndarray):
                raise ValueError("array has no nparray: %s" % select)
            targets.append(d)

        if not targets:
            return self.__results

        for d in targets:
            r: dict[str, Any] = {}
            stats(d.nparray, ignore_nans=ignore_nans, results=r)
            self.__results[d.name] = r

        if self.__results_to_history:
            self._results_to_history()
        if self.__results_to_cache:
            self._results_to_cache(toolfolder)
        if self.__results_to_numpy:
            self._results_to_numpy(toolfolder)

        return self.__results

    def _results_to_history(self, quiet: bool = False) -> None:
        for src_name, r in self.__results.items():
            nmh.history("stats2 %s: %s" % (src_name, r), quiet=quiet)

    def _results_to_cache(self, toolfolder: NMToolFolder) -> None:
        # Store in the folder's parent NMFolder toolresults if available
        parent = toolfolder._parent
        while parent is not None:
            if hasattr(parent, "toolresults_save"):
                parent.toolresults_save("stats2", self.__results)
                return
            parent = getattr(parent, "_parent", None)

    def _results_to_numpy(self, toolfolder: NMToolFolder) -> None:
        if not self.__results:
            return
        src_names = list(self.__results.keys())
        # ST2_data: text array of source array names
        toolfolder.data.new(
            "ST2_data",
            nparray=np.array(src_names, dtype=object),
        )
        # One ST2_ array per metric
        for key in self._ST2_KEYS:
            values = [self.__results[n].get(key, math.nan) for n in src_names]
            toolfolder.data.new(
                "ST2_%s" % key,
                nparray=np.array(values, dtype=float),
            )
