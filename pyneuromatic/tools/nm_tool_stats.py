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

from pyneuromatic.tools.nm_stat_utilities import stat
from pyneuromatic.tools.nm_stat_win import NMStatWinContainer  # noqa: F401
from pyneuromatic.tools.nm_tool import NMTool
from pyneuromatic.tools.nm_tool_config import NMToolConfig
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData, nm_ref
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_math as nm_math
import pyneuromatic.core.nm_utilities as nmu


class NMToolStatsConfig(NMToolConfig):
    """Configuration for NMToolStats.

    Parameters:
        ignore_nans: If True, use NaN-ignoring numpy functions (e.g.
            ``np.nanmean``).  Defaults to True.
        results_to_history: If True, print results to history log after run.
            Defaults to False.
        results_to_cache: If True, save results to NMFolder tool-results cache
            (raw result dicts in memory, for scripts; the app doesn't use
            it). Defaults to False (#480).
        results_to_numpy: If True, write results as ST_ NMData arrays.
            Defaults to True.
        results_all_epochs: If True, ST_ arrays of a data series run have one
            point per epoch of the series (NaN where an epoch wasn't
            analysed), so they line up with the series. Defaults to False:
            one point per analysed epoch.
    """

    _TOML_TYPE = "stats_config"
    _schema = {
        "ignore_nans":        {"type": bool, "default": True},
        "overwrite":          {"type": bool, "default": False},
        "results_to_history": {"type": bool, "default": False},
        "results_to_cache":   {"type": bool, "default": False},  # memory; scripts (#480)
        "results_to_numpy":   {"type": bool, "default": True},
        "results_all_epochs": {"type": bool, "default": False},
    }


class NMToolStats(NMTool):
    """High-level stats tool that runs NMStatWin windows over a data folder.

    Iterates over all active NMStatWin windows in sequence, calling
    ``window.compute(data)`` for each selected NMData object, and collects
    results into an internal dict.  After the run loop completes,
    ``run_finish()`` optionally writes those results to the history log,
    the folder cache, or new NMData numpy arrays (ST_ prefix).

    Attributes:
        windows: Container of NMStatWin objects (auto-named w0, w1, …).
        ignore_nans: If True, use NaN-ignoring numpy functions (e.g.
            ``np.nanmean``).
        results_to_history: If True, print results to history log after run.
        results_to_cache: If True, save results to NMFolder tool-results cache.
        results_to_numpy: If True, write results as ST_ NMData arrays.
        results_all_epochs: If True, ST_ arrays of a data series run have one
            point per epoch of the series (NaN where not analysed).
    """

    def __init__(self) -> None:
        super().__init__(name="stats")
        self._config = NMToolStatsConfig()

        # Initialize output flags from config defaults
        self._ignore_nans = self._config.ignore_nans
        self._overwrite = self._config.overwrite
        self._results_to_history = self._config.results_to_history
        self._results_to_cache = self._config.results_to_cache
        self._results_to_numpy = self._config.results_to_numpy
        self._results_all_epochs = self._config.results_all_epochs

        self.__win_container = NMStatWinContainer(nm_path="%s.windows" % self._name)
        self.__win_container.new()

        self.__results: dict[str, list[Any]] = {}
        # The run's rows (#476): (data array path, its epoch's position in its
        # data series and its number (E3: 3), or None), one per data array
        # analysed; each window's results' rows; and the series' epoch count
        self.__rows: list[tuple[str, int | None, int | None]] = []
        self.__row_of: dict[str, list[int]] = {}
        self.__n_epochs: int | None = None
        self.__run_series: NMDataSeries | None = None  # and its channel run on
        self.__run_channel = None
        self.__transforms: list[str] = []  # channel filters/transforms used
        self.__n_analysed = 0  # data arrays the run measured
        self.__toolfolder: NMToolFolder | None = None

    @property
    def results_all_epochs(self) -> bool:
        """If True, ST_ arrays of a data series run have one point per epoch
        of the series (NaN where an epoch wasn't analysed)."""
        return self._results_all_epochs

    @results_all_epochs.setter
    @nmch.records_command
    def results_all_epochs(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_all_epochs", "boolean"))
        self._results_all_epochs = value
        nmch.add_nm_command("%s.results_all_epochs = %r" % (self._name, value))

    @nmch.records_command
    def histogram(
        self,
        toolfolder: NMToolFolder,
        name: str,
        bins: int | str | list = "auto",
        binwidth: float | None = None,
        xrange: tuple | None = None,
        pad: int = 1,
        output: str = "count",
    ) -> dict[str, Any]:
        """Histogram of ST_ array *name*, saved in *toolfolder* as ``H_{name}``
        (replacing an earlier one), as one command (#448, #492), e.g.::

            nm.stats.histogram(nm.folders['Demo'].toolfolders['Stats_Record_A_All_0'],
                               'ST_w0_max_y', binwidth=2)

        See :meth:`NMToolStats2.histogram` for the arguments.
        """
        result = NMToolStats2.histogram(toolfolder, name, bins=bins, binwidth=binwidth,
                                        xrange=xrange, pad=pad, output=output)
        call = nmu.kwargs_repr(self.histogram,
                               {"bins": bins, "binwidth": binwidth, "xrange": xrange,
                                "pad": pad, "output": output})
        nmch.add_nm_command("%s.histogram(%s, %r%s)"
                            % (self._name, nm_ref(toolfolder), name, call))
        return result

    @property
    def windows(self) -> NMStatWinContainer:
        """Return the container of NMStatWin objects for this tool."""
        return self.__win_container

    @property
    def results(self) -> dict[str, list[list[dict[str, Any]]]]:
        """Results from the most recent run, grouped by window then data item."""
        return {
            window_name: [list(data_results) for data_results in window_results]
            for window_name, window_results in self.__results.items()
        }

    @property
    def toolfolder(self) -> NMToolFolder | None:
        """Toolfolder of ST_ arrays written by the most recent run, if any."""
        return self.__toolfolder

    def _add_note(self, data: NMData, text: str) -> None:
        """Append a note to *data*.notes if available."""
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    # override, no super
    def run_init(self) -> bool:
        """Clear results dict before the run loop starts.

        Returns:
            True on success.
        """
        if isinstance(self.__results, dict):
            self.__results.clear()
        self.__rows = []
        self.__row_of = {}
        self.__n_epochs = None
        self.__run_series = self.__run_channel = None
        self.__transforms = []
        self.__n_analysed = 0
        return True  # ok

    # override, no super
    def run(self) -> bool:
        """Compute stats for the current NMData object across all active windows.

        Called once per NMData object by the NMTool run loop.  Skips windows
        where ``on`` is False, then appends each window's results to the
        internal results dict keyed by window name.

        Returns:
            True on success.
        """
        if not isinstance(self.data, NMData):
            raise RuntimeError("no data selected")
        # The channel's filters and transforms (a copy, as the plot shows)
        data = self.transformed_data()
        if data is not self.data and self.channel is not None:
            steps = self.channel.transform_str()
            if steps not in self.__transforms:
                self.__transforms.append(steps)
        self.__n_analysed += 1
        position = self._epoch_position()
        self.__last_windows: list[str] = []  # for discard_last
        for w in self.windows:
            if not w.on:
                continue
            w.compute(data, ignore_nans=self._ignore_nans)
            # results saved to w.results
            if not w.results:
                continue
            if w.name in self.__results:
                self.__results[w.name].append(w.results)
            else:
                self.__results[w.name] = [w.results]
            self.__row_of.setdefault(w.name, []).append(len(self.__rows))
            self.__last_windows.append(w.name)
        number = self.epoch.number if position is not None else None
        self.__rows.append((self.data.path_str, position, number))  # whatever the windows
        return True  # ok

    def discard_last(self) -> None:
        """Drop the results of the data array :meth:`run` just measured (an
        epoch omitted from the run, #472)."""
        for name in getattr(self, "_NMToolStats__last_windows", []):
            self.__results[name].pop()
            self.__row_of[name].pop()
            if not self.__results[name]:
                del self.__results[name]
                del self.__row_of[name]
        self.__last_windows = []
        if self.__rows:
            self.__rows.pop()
        self.__n_analysed -= 1

    def _epoch_position(self) -> int | None:
        """The current epoch's position in its data series (E0 is 0), or None
        when not running on a data series."""
        series, epoch = self.dataseries, self.epoch
        if series is None or epoch is None:
            return None
        names = list(series.epochs.keys())
        if epoch.name not in names:
            return None
        self.__n_epochs = len(names)
        self.__run_series, self.__run_channel = series, self.channel
        return names.index(epoch.name)

    # override, no super
    def run_finish(self) -> bool:
        """Save results after the run loop completes.

        Dispatches to the enabled output sinks: history log, NMFolder cache,
        and/or ST_ numpy arrays, based on the ``results_to_*`` flags.

        Returns:
            True on success.
        """
        if self._results_to_history:
            self._write_results_to_history()
        if self._results_to_cache:
            self._write_results_to_cache()
        self.__toolfolder = None
        if self._results_to_numpy:
            self.__toolfolder = self._write_results_to_numpy()
        if self.__transforms:
            nmh.history("channel transform: %s" % self._transform_note(),
                        quiet=nmc.QUIET)
        return True  # ok

    def _run_note(self) -> str:
        """How the run was made, for its results folder, e.g. "NMStats:
        Record, channel A, epochs all (n=3); w0: mean, x 0 to inf, baseline
        mean x 0 to 10; channel transform: Invert()"."""
        meta = self._run_meta if isinstance(getattr(self, "_run_meta", None), dict) else {}
        keys = meta.get("run_keys") or {}
        n = self.__n_analysed
        series, channels = meta.get("dataseries") or [], meta.get("channels") or []
        if series:
            what = ", ".join(series)
            if channels:
                what += ", channel%s %s" % ("s" * (len(channels) > 1), ", ".join(channels))
            epochs = keys.get("epoch")
            if keys.get("group") is not None:
                group = "group %s" % keys["group"]
                epochs = ("%s %s %s" % (epochs, keys.get("group_operator") or "AND", group)
                          if epochs else group)
            what += ", epochs %s (n=%d)" % (epochs or "selected", n)
        else:
            what = "%d data array%s" % (n, "s" * (n != 1))
            if keys.get("data"):
                what += " (data set %s)" % keys["data"]
        parts = ["NMStats: " + what]

        def x_range(xbgn, xend) -> str:
            return "x %g to %g" % (xbgn, xend)

        for w in self.windows:
            if not w.on:
                continue
            text = "%s: %s, %s" % (w.name, w.func.get("name", "?"), x_range(w.xbgn, w.xend))
            if w.bsln_on:
                text += ", baseline %s %s" % (w.bsln_func.get("name", "?"),
                                              x_range(w.bsln_xbgn, w.bsln_xend))
            parts.append(text)
        transform = self._transform_note()
        if transform:
            parts.append("channel transform: %s" % transform)
        return "; ".join(parts)

    def _transform_note(self) -> str:
        """The channel filters and transforms the run used ("" if none)."""
        return "; ".join(self.__transforms)

    def _write_results_to_history(self, quiet: bool = False) -> None:
        """Print all results to the history log.

        Args:
            quiet: If True, suppress output.
        """
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

    def _write_results_to_cache(self) -> int | None:
        """Save results to the NMFolder tool-results cache.

        Returns:
            Cache slot index on success, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")
        return self.folder.toolresults_save("stats", self.__results)

    # Numeric keys extracted from result dicts into NMData arrays.
    # Maps result dict key → (NMData name suffix, units source key or None).
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
        "Δs":   ("dy",   "sunits"),  # the value minus the baseline's (#488)
    }

    # Maps stat function names containing special chars or long names to
    # NMData-safe abbreviations (alphanumeric + underscores only).
    _FUNC_NAME_MAP: dict[str, str] = {
        "mean@max":       "mean_max",
        "mean@min":       "mean_min",
        "value@xbgn":     "val_xbgn",
        "value@xend":     "val_xend",
        "level+":         "level_p",
        "level-":         "level_m",
        "risetime+":      "rt_p",
        "risetime-":      "rt_m",
        "risetimeslope+": "rtslope_p",
        "risetimeslope-": "rtslope_m",
        "falltime+":      "ft_p",
        "falltime-":      "ft_m",
        "falltimeslope+": "ftslope_p",
        "falltimeslope-": "ftslope_m",
        "decaytime+":     "decay_p",
        "decaytime-":     "decay_m",
        "fwhm+":          "fwhm_p",
        "fwhm-":          "fwhm_m",
        "pathlength":     "pathlen",
    }

    # Compound functions that compute two outputs from one call.
    # Maps func_name → (base name used for "s" key, secondary result key).
    # e.g. "mean+std" splits into ST_w0_mean_y and ST_w0_std_y.
    _COMPOUND_FUNCS: dict[str, tuple[str, str]] = {
        "mean+std": ("mean", "std"),
        "mean+var": ("mean", "var"),
        "mean+sem": ("mean", "sem"),
    }

    @staticmethod
    def _sanitize_func_name(name: str) -> str:
        """Return an NMData-safe version of a stat function name.

        Looks up ``_FUNC_NAME_MAP`` first; falls back to replacing ``+``,
        ``-``, and ``@`` with underscored abbreviations.

        Args:
            name: Raw function name (e.g. ``"risetime+"``).

        Returns:
            NMData-safe string (e.g. ``"rt_p"``).
        """
        if name in NMToolStats._FUNC_NAME_MAP:
            return NMToolStats._FUNC_NAME_MAP[name]
        return name.replace("+", "_p").replace("-", "_m").replace("@", "_")

    @staticmethod
    def _result_step(rdict: dict[str, Any]) -> str:
        """Return the pipeline step of a result dict within its id.

        Multi-step funcs (risetime, falltime, decaytime, fwhm) record several
        results under one id per data array: the peak (``""``), the p0 and p1
        level crossings (``"p0"``, ``"p1"``) and, for slope variants, the
        regression between them (``"slope"``).
        """
        if "p0" in rdict:
            return "p0"
        if "p1" in rdict:
            return "p1"
        func = rdict.get("func")
        if (rdict.get("id") not in ("main", "bsln")
                and isinstance(func, dict) and func.get("name") == "slope"):
            return "slope"
        return ""

    @staticmethod
    def _st_array_name(
        wname: str, func_name: str, id_str: str, rkey: str, step: str = "",
    ) -> str:
        """Build the ST_ NMData array name for a single result key.

        Naming rules:

        * ``"s"`` result key → ``"y"`` suffix (y-axis stat value).
        * ``"main"`` id → omitted; sanitized func name embedded instead.
        * ``"bsln"`` id → ``"bsln"`` component; func name omitted.
        * Compound funcs (``mean+std/var/sem``) → split into separate arrays:
          ``ST_{w}_mean_y`` and ``ST_{w}_std_y`` (etc.); their other keys
          are the mean's, e.g. ``ST_{w}_mean_dy``, ``ST_{w}_mean_n``.
        * Complex func ids (``risetime+``, etc.) → sanitized and used directly.

        Args:
            wname: Window name (e.g. ``"w0"``).
            func_name: Stat function name from the result dict.
            id_str: Computation stage (``"main"``, ``"bsln"``, or func name
                for complex pipelines).
            rkey: Result dict key (e.g. ``"s"``, ``"x"``, ``"dx"``).
            step: Pipeline step from ``_result_step`` (e.g. ``"p0"``). Added
                after the func name, except for ``"dx"`` which only one step
                records (e.g. ``ST_w0_rt_p_p0_x`` but ``ST_w0_rt_p_dx``).

        Returns:
            Array name string (e.g. ``"ST_w0_mean_y"``).
        """
        suffix = "y" if rkey == "s" else "dy" if rkey == "Δs" else rkey

        if id_str == "bsln":
            return "ST_%s_bsln_%s" % (wname, suffix)

        if id_str == "main" and func_name in NMToolStats._COMPOUND_FUNCS:
            base, sec_key = NMToolStats._COMPOUND_FUNCS[func_name]
            if rkey == "s":
                return "ST_%s_%s_y" % (wname, base)
            if rkey == sec_key:
                return "ST_%s_%s_y" % (wname, sec_key)
            # other keys (Δs, n, nans, etc.): the mean's, e.g. ST_w0_mean_dy
            return "ST_%s_%s_%s" % (wname, base, suffix)

        safe = NMToolStats._sanitize_func_name(
            func_name if id_str == "main" else id_str
        )
        if step and rkey != "dx":
            safe = "%s_%s" % (safe, step)
        return "ST_%s_%s_%s" % (wname, safe, suffix)

    def _run_rows(self) -> tuple[list[str], list[int] | None, list[int], bool]:
        """The ST_ arrays' rows (#476): the data array names, their epoch
        numbers (None outside a data series), each run row's index, and
        whether there's a row per epoch of the series.

        One row per data array analysed; with results_all_epochs (on a data
        series), one per epoch of the series, analysed or not."""
        positions = [position for _path, position, _number in self.__rows]
        if (self._results_all_epochs and self.__n_epochs
                and all(p is not None for p in positions)):
            names = [""] * self.__n_epochs
            for path, position, _number in self.__rows:
                names[position] = path
            self._fill_epoch_paths(names)
            series = self.__run_series
            numbers = ([epoch.number for epoch in series.epochs.values()]
                       if series is not None else list(range(self.__n_epochs)))
            return names, numbers, positions, True
        numbers = [number for _path, _position, number in self.__rows]
        epochs = numbers if numbers and all(n is not None for n in numbers) else None
        return [path for path, _p, _n in self.__rows], epochs, list(range(len(self.__rows))), False

    @staticmethod
    def _result_value(rdict: dict[str, Any], rkey: str) -> Any:
        """A result's value for ST_ array key *rkey*; a crossing (no "s") gives
        its level as "s", so it's saved as its step's _y (#476)."""
        if rkey == "s" and "s" not in rdict:
            return (rdict.get("func") or {}).get("ylevel")
        return rdict.get(rkey)

    @staticmethod
    def _note_params(rdict: dict[str, Any], win: Any, id_str: str) -> str:
        """What a step's note adds (#476): its percentage (p0=10), the
        function that made it (stepfunc=max, stepfunc=level+), and a level's
        ylevel or n_std, e.g. ", p0=10, stepfunc=level+"."""
        parts = []
        for key in ("p0", "p1"):
            if key in rdict:
                parts.append("%s=%g" % (key, rdict[key]))
        func = rdict.get("func") or {}
        if id_str not in ("main", "bsln") and func.get("name"):
            parts.append("stepfunc=%s" % func["name"])
        win_func = win.func if win is not None else {}
        if id_str == "main":
            for key in ("ylevel", "n_std"):
                if win_func.get(key) is not None:
                    parts.append("%s=%g" % (key, win_func[key]))
        return "".join(", " + part for part in parts)

    def _fill_epoch_paths(self, data_paths: list[str]) -> None:
        """Fill the blank entries of a per-epoch ST_data_names with the names of
        the epochs' arrays (in the channel run on) that weren't analysed."""
        series, channel = self.__run_series, self.__run_channel
        if series is None or channel is None:
            return
        arrays = series.epoch_data_map(channel.name)
        for index, name in enumerate(series.epochs.keys()):
            data = arrays.get(name)
            if index < len(data_paths) and not data_paths[index] and data is not None:
                data_paths[index] = data.path_str

    def _write_results_to_numpy(self) -> NMToolFolder | None:
        """Write results as ST_ NMData arrays in a new NMToolFolder.

        Creates a subfolder named ``Stats_{dataseries}_{channel}_{epoch_set}_N``
        (first unused N) under the current folder's toolfolder.  Components are
        omitted when not set; epoch set is included only when present in
        ``run_keys``.  Examples: ``Stats_0``, ``Stats_Record_A_0``,
        ``Stats_Record_A_Set1_0``.

        Writes one NMData array per numeric result key per window/func
        combination.  Array naming rules:

        * Primary stat value (``"s"`` key): ``ST_{w}_{func}_y``
          e.g. ``ST_w0_mean_y``, ``ST_w0_max_y``.
        * Compound funcs split into two: ``mean+std`` →
          ``ST_{w}_mean_y`` and ``ST_{w}_std_y``; their other keys are
          the mean's (``ST_{w}_mean_dy``, ``ST_{w}_mean_n``, ...).
        * Baseline results: ``ST_{w}_bsln_y``, ``ST_{w}_bsln_std``, etc.
        * Other keys keep their descriptive suffix: ``ST_{w}_{func}_x``,
          ``ST_{w}_{func}_dx``, ``ST_{w}_{func}_n``, etc.
        * Long/special-char func names are abbreviated via
          ``_FUNC_NAME_MAP`` (e.g. ``risetime+`` → ``rt_p``).
        * The rows, shared by all windows (#476): ``ST_data_names`` (the
          data arrays' paths) and, for a data series, ``ST_epoch_num`` (their
          epoch numbers). A window with no result for a row has NaN there.
        * A crossing's level is saved as its step's ``_y`` (e.g.
          ``ST_w0_rt_p_p0_y``); its percentage, the function that made the
          step, and a level's ylevel / n_std are in each array's note.

        Returns:
            The newly created NMToolFolder, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")

        f = self._make_toolfolder("Stats", overwrite=self._overwrite)
        transform = self._transform_note()  # on the folder and each array's note
        f.notes.add(self._run_note())  # how the results were made
        transform_str = ", transform=%s" % transform if transform else ""

        # The rows, shared by all windows (#476). Results set directly (not by
        # run(), e.g. tests): a row per result, from their data names
        if any(len(self.__row_of.get(w, [])) != len(v) for w, v in self.__results.items()):
            longest = max(self.__results.values(), key=len)
            self.__rows = [((ilist[0].get("data", "") if ilist else ""), None, None)
                           for ilist in longest]
            self.__row_of = {w: list(range(len(v))) for w, v in self.__results.items()}
        data_names, epoch_nums, slot_of_row, per_epoch_rows = self._run_rows()
        n_arrays = len(data_names)
        if f.data is not None:
            f.data.new("ST_data_names", nparray=np.array(data_names, dtype=object))
            if epoch_nums is not None:
                f.data.new("ST_epoch_num", nparray=np.array(epoch_nums, dtype=int))

        for wname, vlist in self.__results.items():
            # vlist: one list of result dicts per data array the window measured.
            # Result dicts grouped by (id, step), a slot per row, so every ST_
            # array lines up with ST_data_names; a row the window (or step)
            # has no result for stays None and becomes NaN.
            rows = self.__row_of.get(wname, [])
            step_rdicts: dict[tuple[str, str], list[dict | None]] = {}
            for result_index, ilist in enumerate(vlist):
                index = slot_of_row[rows[result_index]]
                for rdict in ilist:
                    key = (rdict.get("id", "main"), self._result_step(rdict))
                    if key not in step_rdicts:
                        step_rdicts[key] = [None] * n_arrays
                    step_rdicts[key][index] = rdict

            # Save numeric arrays per (id, step)
            for (id_str, step), slots in step_rdicts.items():
                present = [rdict for rdict in slots if rdict is not None]
                func_name = present[0].get("func", {}).get("name", "")
                for rkey, (_suffix, units_key) in self._NUMERIC_KEYS.items():
                    values = [
                        self._result_value(rdict, rkey) if rdict is not None else None
                        for rdict in slots
                    ]
                    if all(v is None for v in values):
                        continue  # key not present for this func
                    arr = np.array(
                        [v if v is not None else math.nan for v in values],
                        dtype=float,
                    )
                    units = present[0].get(units_key) if units_key else None
                    if rkey in ("s", "Δs") and not units:  # a crossing's level (and Δ): y units
                        units = present[0].get("func", {}).get("yunits")
                    yscale = {"units": units} if units else None
                    dname = self._st_array_name(
                        wname, func_name, id_str, rkey, step
                    )
                    if f.data is not None:
                        d = f.data.new(dname, nparray=arr, yscale=yscale)
                        win = self.windows[wname] if wname in self.windows else None
                        if id_str == "bsln" and win is not None:
                            xbgn = win.bsln_xbgn
                            xend = win.bsln_xend
                            note_func = win.bsln_func.get("name", func_name)
                        else:
                            xbgn = win.xbgn if win is not None else None
                            xend = win.xend if win is not None else None
                            # Steps of multi-step funcs run helper funcs
                            # (max, level+, ...); name the window's func
                            note_func = func_name if id_str == "main" else id_str
                        step_str = ", step=%s" % step if step else ""
                        per_epoch = ", per epoch" if per_epoch_rows else ""
                        self._add_note(
                            d,
                            "NMStats(win=%s, func=%s, id=%s%s%s, xbgn=%s, xend=%s, n=%d%s%s)"
                            % (wname, note_func, id_str, step_str,
                               self._note_params(present[0], win, id_str),
                               xbgn, xend, len(present), per_epoch, transform_str),
                        )

                # Save warnings if any occurred
                warnings = [
                    rdict.get("warning") if rdict is not None else None
                    for rdict in slots
                ]
                if any(w is not None for w in warnings):
                    dname = self._st_array_name(
                        wname, func_name, id_str, "warning", step
                    )
                    if f.data is not None:
                        f.data.new(
                            dname,
                            nparray=np.array(
                                [w or "" for w in warnings], dtype=object
                            ),
                        )

        return f


class NMToolStats2:
    """Compute summary statistics and histograms of Stats results (ST_ arrays).

    Operates on a NMToolFolder produced by NMToolStats._write_results_to_numpy().
    All options are passed as method parameters rather than stored as instance
    state.

    Methods:
        stats: Compute summary statistics on ST_ arrays.
        histogram: Compute a histogram of a single ST_ array.
    """

    # Keys written to ST2_ arrays (in order)
    _ST2_KEYS = ("mean", "std", "sem", "N", "NaNs", "INFs", "rms", "min", "max")

    @staticmethod
    def stats(
        toolfolder: NMToolFolder,
        select: str = "all",
        ignore_nans: bool = True,
        results_to_history: bool = False,
        results_to_cache: bool = False,
        results_to_numpy: bool = False,
    ) -> dict[str, Any]:
        """Compute summary statistics on ST_ arrays in a NMToolFolder.

        Args:
            toolfolder: NMToolFolder containing ST_ arrays from NMToolStats.
            select: Name of a specific ST_ array, or "all" to process all
                numeric ST_ arrays.
            ignore_nans: If True, exclude NaN values from calculations.
                Defaults to True.
            results_to_history: If True, print results to the history log.
                Defaults to False.
            results_to_cache: If True, save results to the NMFolder
                tool-results cache. Defaults to False.
            results_to_numpy: If True, write results as ST2_ NMData arrays
                in toolfolder. Defaults to False.

        Returns:
            Results dict keyed by source array name.
        """
        if not isinstance(toolfolder, NMToolFolder):
            raise TypeError(
                nmu.type_error_str(toolfolder, "toolfolder", "NMToolFolder")
            )
        if not isinstance(select, str):
            raise TypeError(nmu.type_error_str(select, "select", "string"))
        if not isinstance(ignore_nans, bool):
            ignore_nans = True

        results: dict[str, Any] = {}

        # Collect target ST_ arrays
        targets: list[NMData] = []
        if select.lower() == "all":
            for name in toolfolder.data:
                if (name.startswith("ST_") and not name.endswith("_data")
                        and name not in ("ST_data_names", "ST_epoch_num")):
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
            return results

        for d in targets:
            r: dict[str, Any] = {}
            nm_math.array_stats(d.nparray, ignore_nans=ignore_nans, results=r)
            results[d.name] = r

        if results_to_history:
            NMToolStats2._stats_results_to_history(results)
        if results_to_cache:
            NMToolStats2._stats_results_to_cache(toolfolder, results)
        if results_to_numpy:
            NMToolStats2._stats_results_to_numpy(toolfolder, results)

        return results

    @staticmethod
    def _stats_results_to_history(
        results: dict[str, Any], quiet: bool = False
    ) -> None:
        """Print all stats results to the history log.

        Args:
            results: Results dict keyed by source array name.
            quiet: If True, suppress output.
        """
        for src_name, r in results.items():
            nmh.history("stats2 %s: %s" % (src_name, r), quiet=quiet)

    @staticmethod
    def _stats_results_to_cache(
        toolfolder: NMToolFolder, results: dict[str, Any]
    ) -> None:
        """Save stats results to the nearest NMFolder tool-results cache.

        Args:
            toolfolder: The NMToolFolder whose parent hierarchy is searched for
                a ``toolresults_save`` method.
            results: Results dict keyed by source array name.
        """
        parent = toolfolder._parent
        while parent is not None:
            if hasattr(parent, "toolresults_save"):
                parent.toolresults_save("stats2", results)
                return
            parent = getattr(parent, "_parent", None)

    @staticmethod
    def _stats_results_to_numpy(
        toolfolder: NMToolFolder, results: dict[str, Any]
    ) -> None:
        """Write stats results as ST2_ NMData arrays in the given NMToolFolder.

        Saves a ``ST2_data`` text array of source array names, then one
        ``ST2_{key}`` float array per metric in ``_ST2_KEYS``.

        Args:
            toolfolder: Destination NMToolFolder for the ST2_ arrays.
            results: Results dict keyed by source array name.
        """
        if not results:
            return
        src_names = list(results.keys())
        # ST2_data: text array of source array names
        toolfolder.data.new(
            "ST2_data",
            nparray=np.array(src_names, dtype=object),
        )
        # One ST2_ array per metric
        for key in NMToolStats2._ST2_KEYS:
            values = [results[n].get(key, math.nan) for n in src_names]
            toolfolder.data.new(
                "ST2_%s" % key,
                nparray=np.array(values, dtype=float),
            )

    @staticmethod
    def histogram(
        toolfolder: NMToolFolder,
        name: str,
        bins: int | str | list = "auto",
        binwidth: float | None = None,
        xrange: tuple | None = None,
        pad: int = 1,
        output: str = "count",
        save_to_numpy: bool = True,
    ) -> dict[str, Any]:
        """Histogram of ST_ array *name* (#492), saved in *toolfolder* as
        ``H_{name}`` (replacing an earlier one).

        NaN and Inf values are excluded. ``H_{name}`` holds the counts (or
        probability, or density), its x scale the first bin edge and the bin
        width, in *name*'s units. The bin edges are saved too, as
        ``H_{name}_edges``, only if the bins differ in width (*bins* a list of
        edges); otherwise an earlier ``H_{name}_edges`` is removed.

        Args:
            toolfolder: NMToolFolder containing the ST_ array.
            name: Name of the ST_ array to histogram.
            bins: ``"auto"`` (numpy's "auto" width), a number of bins, or a
                list of bin edges.
            binwidth: The bins' width (instead of *bins*).
            xrange: ``(min, max)`` the bins span, or None: the values' range,
                the bin edges multiples of the width (round numbers).
            pad: Empty bins added at each end (not with *xrange* or a list of
                edges). Defaults to 1.
            output: ``"count"``, ``"probability"`` (the heights add up to 1)
                or ``"density"`` (the areas add up to 1).
            save_to_numpy: If True, save ``H_{name}`` in toolfolder.

        Returns:
            Dict with keys ``"counts"`` (the output), ``"edges"`` (length
            bins + 1), ``"width"`` (None for unequal bins) and ``"n"`` (the
            number of values counted). See :func:`nm_math.histogram_round_edges`.

        Raises:
            TypeError: If toolfolder is not an NMToolFolder or name is not
                a string.
            KeyError: If name is not found in toolfolder.
            ValueError: If the named array has no nparray data, or no finite
                values, or for bad bins or output.
        """
        if not isinstance(toolfolder, NMToolFolder):
            raise TypeError(
                nmu.type_error_str(toolfolder, "toolfolder", "NMToolFolder")
            )
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        d = toolfolder.data.get(name)
        if d is None:
            raise KeyError("array not found in toolfolder: %s" % name)
        if not isinstance(d.nparray, np.ndarray):
            raise ValueError("array has no nparray: %s" % name)

        result = nm_math.histogram_round_edges(d.nparray, bins=bins, binwidth=binwidth,
                                           xrange=xrange, pad=pad, output=output)

        if save_to_numpy:
            edges, width = result["edges"], result["width"]
            units = d.yscale.units
            xscale = {"start": float(edges[0]),
                      "delta": float(width if width is not None else edges[1] - edges[0]),
                      "label": d.yscale.label, "units": units}
            yscale = {"label": output,
                      "units": "1/%s" % units if output == "density" and units else None}
            hname = "H_%s" % name
            for old in (hname, hname + "_edges"):  # replace an earlier histogram's
                if old in toolfolder.data:
                    toolfolder.data.pop(old)
            h = toolfolder.data.new(hname, nparray=result["counts"], xscale=xscale,
                                    yscale=yscale)
            how = ("binwidth=%g" % binwidth if binwidth is not None
                   else "bins=%s" % (bins if isinstance(bins, (str, int, np.integer))
                                     else "edges"))
            h.notes.add("NMHistogram(array=%s, %s, nbins=%d, xbgn=%g, xend=%g, pad=%d, "
                        "output=%s, n=%d)" % (name, how, len(result["counts"]), edges[0],
                                              edges[-1], pad if xrange is None else 0,
                                              output, result["n"]))
            if width is None:  # unequal bins: the x scale can't hold them
                toolfolder.data.new(hname + "_edges", nparray=np.asarray(edges, dtype=float),
                                    yscale={"label": d.yscale.label, "units": units})

        return result

    @staticmethod
    def inequality(
        toolfolder: NMToolFolder,
        name: str,
        op: str,
        a: float,
        b: float | None = None,
        binary_output: bool = True,
        dataseries: NMDataSeries | None = None,
        set_name_success: str | None = None,
        set_name_failure: str | None = None,
        save_to_numpy: bool = True,
    ) -> dict:
        """Filter an ST_ array by an inequality condition.

        Applies a threshold condition to each value in the named ST_ array
        and returns a result array plus success/failure counts.  Optionally
        creates epoch sets from the passing and failing epochs.

        This mirrors the Igor NeuroMatic ``NMStatsInequality()`` function.

        Args:
            toolfolder: NMToolFolder containing the ST_ array.
            name: Name of the ST_ array to filter (e.g. ``"ST_w0_max_y"``).
            op: Comparison operator.  Single-threshold: ``">"``, ``">="``,
                ``"<"``, ``"<="``, ``"=="``, ``"!="``.  Range (requires
                ``b``): ``"<<"`` (a < y < b), ``"<=<="`` (a <= y <= b),
                ``"<=<"`` (a <= y < b), ``"<<="`` (a < y <= b).
            a: Threshold value (or lower bound for range operators).
            b: Upper bound; required for range operators.
            binary_output: If True (default), result array contains 1.0
                (pass) or 0.0 (fail).  If False, result contains the
                original value where the condition is met and NaN elsewhere.
            dataseries: NMDataSeries to use when creating epoch sets.
                Required if set_name_success or set_name_failure is given.
            set_name_success: Name of the epoch set to populate with passing
                epochs.  Requires dataseries.
            set_name_failure: Name of the epoch set to populate with failing
                epochs.  Requires dataseries.
            save_to_numpy: If True (default), save the result array as
                ``IQ_{name}`` in toolfolder.

        Returns:
            Dict with keys:

            - ``"result"``: numpy array (binary 0/1 or value/NaN).
            - ``"mask"``: boolean numpy array, True where condition is met.
            - ``"successes"``: number of values that passed.
            - ``"failures"``: number of values that failed.
            - ``"condition"``: human-readable condition string.

        Raises:
            TypeError: If toolfolder is not NMToolFolder or name is not str.
            KeyError: If name is not found in toolfolder.
            ValueError: If the named array has no data, op is unrecognised,
                or a range op is given without b.
        """
        if not isinstance(toolfolder, NMToolFolder):
            raise TypeError(
                nmu.type_error_str(toolfolder, "toolfolder", "NMToolFolder")
            )
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        if op not in nm_math.VALID_INEQUALITY_OPS:
            raise ValueError(
                "unknown operator %r. Single: %s; range: %s"
                % (op, sorted(nm_math._SINGLE_INEQUALITY_OPS), sorted(nm_math._RANGE_INEQUALITY_OPS))
            )
        if op in nm_math._RANGE_INEQUALITY_OPS and b is None:
            raise ValueError(
                "range operator %r requires b to be specified" % op
            )

        d = toolfolder.data.get(name)
        if d is None:
            raise KeyError("array not found in toolfolder: %s" % name)
        if not isinstance(d.nparray, np.ndarray):
            raise ValueError("array has no nparray: %s" % name)

        arr = d.nparray.astype(float)

        mask = nm_math.inequality_mask(arr, op, a, b)

        if binary_output:
            result = mask.astype(float)
        else:
            result = np.where(mask, arr, np.nan)

        successes = int(np.sum(mask))
        failures = len(mask) - successes

        condition = nm_math.inequality_condition_str(op, a, b)

        if save_to_numpy and toolfolder.data is not None:
            toolfolder.data.new("IQ_%s" % name, nparray=result)

        # Create epoch sets if dataseries and set names are given
        if isinstance(dataseries, NMDataSeries) and (
            set_name_success or set_name_failure
        ):
            NMToolStats2._add_epoch_sets_from_mask(
                toolfolder, name, dataseries, mask,
                set_name_true=set_name_success,
                set_name_false=set_name_failure,
            )

        return {
            "result": result,
            "mask": mask,
            "successes": successes,
            "failures": failures,
            "condition": condition,
        }

    @staticmethod
    def _add_epoch_sets_from_mask(
        toolfolder: NMToolFolder,
        name: str,
        dataseries: NMDataSeries,
        mask: np.ndarray,
        set_name_true: str | None = None,
        set_name_false: str | None = None,
    ) -> None:
        """Populate epoch sets from a boolean mask aligned to an ST_ array.

        Looks up the ``ST_{wname}_data`` array name in *toolfolder*,
        maps each index to an epoch via *dataseries*, and adds matching
        epoch names to *set_name_true* (where mask is True) and/or
        *set_name_false* (where mask is False).

        Args:
            toolfolder: NMToolFolder containing the ST_ arrays.
            name: Name of the ST_ result array (e.g. ``"ST_w0_mean_y"``).
                Used to derive *wname* and locate ``ST_{wname}_data``.
            dataseries: NMDataSeries whose epoch sets will be updated.
            mask: Boolean array aligned to the ST_ array (True = passes).
            set_name_true: Epoch set name for mask-True indices.  Skipped
                if None or if no matching epochs are found.
            set_name_false: Epoch set name for mask-False indices.  Skipped
                if None or if no matching epochs are found.
        """
        parts = name.split("_")
        if len(parts) < 2:
            return
        wname = parts[1]
        epoch_map: dict[int, str] = {
            ep.number: ep.name
            for ep in dataseries.epochs.values()
            if isinstance(ep, NMEpoch)
        }
        # Each row's epoch number: ST_epoch_num (#476), else parsed from the
        # rows' data names (ST_data_names; or, saved before, ST_{w}_data)
        numbers_arr = toolfolder.data.get("ST_epoch_num")
        if numbers_arr is not None and isinstance(numbers_arr.nparray, np.ndarray):
            numbers = [int(n) for n in numbers_arr.nparray]
        else:
            data_arr = (toolfolder.data.get("ST_data_names")
                        or toolfolder.data.get("ST_%s_data" % wname))
            if data_arr is None or not isinstance(data_arr.nparray, np.ndarray):
                return
            parsed_names = [nmu.parse_data_name(str(n)) for n in data_arr.nparray]
            numbers = [p[2] if p is not None else None for p in parsed_names]

        true_epochs: list[str] = []
        false_epochs: list[str] = []
        for i, number in enumerate(numbers):
            if i >= len(mask):
                break
            if number is None:
                continue
            ep_name = epoch_map.get(number)
            if ep_name is None:
                continue
            if mask[i]:
                true_epochs.append(ep_name)
            else:
                false_epochs.append(ep_name)

        if set_name_true and true_epochs:
            dataseries.epochs.sets.add(set_name_true, true_epochs)
        if set_name_false and false_epochs:
            dataseries.epochs.sets.add(set_name_false, false_epochs)

    @staticmethod
    def ks_test(
        toolfolder: NMToolFolder,
        name1: str,
        name2: str,
        alpha: float = 0.05,
        method: str = "auto",
        save_to_numpy: bool = False,
    ) -> dict:
        """Two-sample Kolmogorov-Smirnov test on two ST_ arrays.

        Determines whether two distributions of ST_ results are significantly
        different.  NaN and Inf values are excluded before testing, matching
        the Igor ``NMKSTest()`` behaviour of removing NaNs before sorting.

        This mirrors the Igor NeuroMatic ``NMKSTest()`` function
        (``NM_StatsTabKolmogorov.ipf``), originally written by Dr. Angus
        Silver (UCL) based on *Numerical Recipes*.

        Args:
            toolfolder: NMToolFolder containing the ST_ arrays.
            name1: Name of the first ST_ array (e.g. ``"ST_w0_peak_y"``).
            name2: Name of the second ST_ array.
            alpha: Significance level; ``significant`` is True when
                ``pvalue <= alpha``.  Defaults to 0.05.
            method: P-value calculation method forwarded to
                ``scipy.stats.ks_2samp``.  One of ``"auto"`` (default),
                ``"exact"``, or ``"asymp"``.
            save_to_numpy: If True (default), save empirical CDF arrays to
                toolfolder:

                - ``KS_{name1}_sort``, ``KS_{name1}_ecdf``
                - ``KS_{name2}_sort``, ``KS_{name2}_ecdf``

        Returns:
            Dict with keys:

            - ``"d"``: KS statistic — max |CDF1 - CDF2|.
            - ``"pvalue"``: p-value.
            - ``"alpha"``: significance level (echoed from param).
            - ``"significant"``: True if ``pvalue <= alpha``.
            - ``"message"``: ``"different populations"`` or
              ``"same population"``.
            - ``"n1"``: sample size of arr1 after NaN/Inf removal.
            - ``"n2"``: sample size of arr2 after NaN/Inf removal.

        Raises:
            TypeError: If toolfolder is not NMToolFolder or name1/name2 are
                not strings.
            KeyError: If name1 or name2 is not found in toolfolder.
            ValueError: If the named array has no nparray data.
            ImportError: If scipy is not installed.
        """
        if not isinstance(toolfolder, NMToolFolder):
            raise TypeError(
                nmu.type_error_str(toolfolder, "toolfolder", "NMToolFolder")
            )
        if not isinstance(name1, str):
            raise TypeError(nmu.type_error_str(name1, "name1", "string"))
        if not isinstance(name2, str):
            raise TypeError(nmu.type_error_str(name2, "name2", "string"))

        d1 = toolfolder.data.get(name1)
        if d1 is None:
            raise KeyError("array not found in toolfolder: %s" % name1)
        if not isinstance(d1.nparray, np.ndarray):
            raise ValueError("array has no nparray: %s" % name1)

        d2 = toolfolder.data.get(name2)
        if d2 is None:
            raise KeyError("array not found in toolfolder: %s" % name2)
        if not isinstance(d2.nparray, np.ndarray):
            raise ValueError("array has no nparray: %s" % name2)

        result = nm_math.ks_test(d1.nparray, d2.nparray, alpha=alpha,
                                 method=method)

        if save_to_numpy and toolfolder.data is not None:
            toolfolder.data.new("KS_%s_sort" % name1, nparray=result["sort1"])
            toolfolder.data.new("KS_%s_ecdf" % name1, nparray=result["ecdf1"])
            toolfolder.data.new("KS_%s_sort" % name2, nparray=result["sort2"])
            toolfolder.data.new("KS_%s_ecdf" % name2, nparray=result["ecdf2"])

        return {k: result[k] for k in
                ("d", "pvalue", "alpha", "significant", "message", "n1", "n2")}

    @staticmethod
    def stability_test(
        toolfolder: NMToolFolder,
        name: str,
        alpha: float = 0.05,
        min_window: int = 10,
        dataseries: NMDataSeries | None = None,
        set_name_stable: str | None = None,
        save_to_numpy: bool = False,
    ) -> dict:
        """Find the largest stable (trend-free) window in an ST_ array.

        Tests whether consecutive subsets of values have no significant
        monotonic trend over time using the Spearman rank-order correlation
        between values and their array indices.  Searches from the largest
        possible window down to ``min_window``, stopping as soon as the
        largest stable window is identified.

        This mirrors the Igor NeuroMatic ``NMStabilityRankOrderTest()``
        function (``NM_StatsTabStability.ipf``, pass 1 only), originally
        written by Dr. Angus Silver and Simon Mitchell (UCL), based on
        *Numerical Recipes in C*.

        Args:
            toolfolder: NMToolFolder containing the ST_ array.
            name: Name of the ST_ array (e.g. ``"ST_w0_mean_y"``).
            alpha: Significance level.  A window is "stable" when its
                Spearman p-value exceeds this threshold.  Defaults to 0.05.
            min_window: Minimum window size in data points.  Must be >= 3
                (scipy requires at least 3 points to compute a p-value for
                non-constant data).  Defaults to 10.
            dataseries: NMDataSeries to use when creating an epoch set.
                Required if set_name_stable is given.
            set_name_stable: Name of the epoch set to populate with epochs
                inside the stable region.  Requires dataseries.
            save_to_numpy: If True, save ``STAB_{name}_mask`` (float 0/1
                array) to toolfolder.  Defaults to False.

        Returns:
            Dict with keys:

            - ``"stable"``: True if a stable region was found.
            - ``"start"``: Index into original array of first stable point,
              or None.
            - ``"end"``: Index into original array of last stable point
              (inclusive), or None.
            - ``"n"``: Window size (``end - start + 1``), or None.
            - ``"rs"``: Spearman rs at best window, or None.
            - ``"pvalue"``: p-value at best window, or None.
            - ``"alpha"``: Significance level (echoed from param).
            - ``"mask"``: Boolean numpy array (length = original array
              length), True where the stable region falls.

        Raises:
            TypeError: If toolfolder is not NMToolFolder or name is not str.
            KeyError: If name is not found in toolfolder.
            ValueError: If the array has no data, or min_window is out of
                range.
        """
        if not isinstance(toolfolder, NMToolFolder):
            raise TypeError(
                nmu.type_error_str(toolfolder, "toolfolder", "NMToolFolder")
            )
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))

        d = toolfolder.data.get(name)
        if d is None:
            raise KeyError("array not found in toolfolder: %s" % name)
        if not isinstance(d.nparray, np.ndarray):
            raise ValueError("array has no nparray: %s" % name)

        result = nm_math.stability_test(d.nparray, alpha=alpha,
                                        min_window=min_window)

        if save_to_numpy and toolfolder.data is not None:
            toolfolder.data.new("STAB_%s_mask" % name,
                                nparray=result["mask"].astype(float))

        if isinstance(dataseries, NMDataSeries) and set_name_stable:
            NMToolStats2._add_epoch_sets_from_mask(
                toolfolder, name, dataseries, result["mask"],
                set_name_true=set_name_stable,
            )

        return result
