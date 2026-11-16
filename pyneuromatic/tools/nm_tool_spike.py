# -*- coding: utf-8 -*-
"""
NMToolSpike: spike detection tool.

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

import numpy as np

from pyneuromatic.tools.nm_tool import NMTool
from pyneuromatic.tools.nm_tool_config import NMToolConfig
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_math as nm_math
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.tools.nm_tool_utilities import find_level_crossings_nmdata

_VALID_FUNC_NAMES: frozenset[str] = frozenset({"level", "level+", "level-"})

# How a run's spike times are saved (#526): two raster arrays, SP_RX (the
# times) and SP_RY (each spike's epoch), or an SP_<data> array per epoch
RESULTS_FORMATS: tuple[str, ...] = ("raster", "epochs")

# The per-epoch arrays (one row per data array analysed), as Stats' ST_ arrays
SP_ROW_ARRAYS: tuple[str, ...] = (
    "SP_count", "SP_rate", "SP_first", "SP_isi", "SP_epoch_num", "SP_data_names")
# Arrays a Spike subfolder holds that aren't an epoch's spike times
_SP_RESERVED: frozenset[str] = frozenset(
    {"SP_RX", "SP_RY", "SP_PST", "SP_ISI", "SP_epoch_names", *SP_ROW_ARRAYS})


def _epoch_array_name(name: str) -> str:
    """The ``"epochs"`` format's array of data *name*'s spike times:
    ``SP_{name}``, or ``SP_{name}_times`` if that's a per-epoch array's
    name (e.g. data named "count")."""
    array_name = "SP_" + name
    return array_name + "_times" if array_name in _SP_RESERVED else array_name


def _rate_units(xunits: str | None) -> tuple[float, str | None]:
    """(factor, units) of a rate of spikes per *xunits*: Hz for s or ms."""
    if xunits == "s":
        return 1.0, "Hz"
    if xunits == "ms":
        return 1000.0, "Hz"
    return 1.0, ("1/%s" % xunits if xunits else None)


class NMToolSpikeConfig(NMToolConfig):
    """Configuration for NMToolSpike.

    Parameters:
        ylevel: Y-axis detection threshold. Default 0.0.
        func_name: Crossing direction — ``"level+"`` (rising, default),
            ``"level-"`` (falling), or ``"level"`` (both).
        xbgn: X-axis window start for detection. Default ``-inf`` (no lower
            bound). If *xbgn* > *xend*, a backwards search is performed.
        xend: X-axis window end for detection. Default ``+inf`` (no upper
            bound).
        ignore_nans: If True (default), detect crossings across NaN gaps
            via linear interpolation.
        results_to_history: If True, print spike counts to the history log
            after each run. Default False.
        results_to_cache: If True, save spike times dict to
            ``folder.toolresults`` after each run. Default True.
        results_to_numpy: If True, write ``SP_`` NMData arrays to a new
            Spike subfolder after each run. Default True.
        results_format: How spike times are saved: ``"raster"`` (default;
            ``SP_RX`` and ``SP_RY``) or ``"epochs"`` (an ``SP_{data}`` array
            per epoch).
    """

    _TOML_TYPE = "spike_config"
    _schema = {
        "ylevel":             {"type": float, "default": 0.0},
        "func_name":          {"type": str,   "default": "level+",
                               "choices": ["level", "level+", "level-"]},
        "xbgn":                 {"type": float, "default": -math.inf},
        "xend":                 {"type": float, "default":  math.inf},
        "ignore_nans":        {"type": bool,  "default": True},
        "overwrite":          {"type": bool,  "default": True},
        "results_to_history": {"type": bool,  "default": False},
        "results_to_cache":   {"type": bool,  "default": True},
        "results_to_numpy":   {"type": bool,  "default": True},
        "results_format":     {"type": str,   "default": "raster",
                               "choices": list(RESULTS_FORMATS)},
    }


class NMToolSpike(NMTool):
    """Spike detection tool using threshold crossings.

    Detects action potentials (or other threshold-crossing events) in
    NMData arrays via :func:`~pyneuromatic.core.nm_math.find_level_crossings`.
    After each run, the results are written to a new Spike subfolder
    (:meth:`_write_results_to_numpy`, #526): the spike times as a raster
    (``SP_RX``, ``SP_RY``) or as an array per epoch, and a row per epoch of
    measures (``SP_count``, ``SP_rate``, ``SP_first``, ``SP_isi``), as
    Stats' ST_ arrays.

    After detection, call :meth:`pst` or :meth:`isi` to compute histograms
    from the most recent run's spike times, or from a subfolder's.

    Attributes:
        ylevel: Y-axis detection threshold. Default 0.0.
        func_name: Crossing direction — ``"level+"`` (rising, default),
            ``"level-"`` (falling), or ``"level"`` (both).
        xbgn: X-axis window start. Default ``-inf`` (no lower bound).
        xend: X-axis window end. Default ``+inf`` (no upper bound).
        ignore_nans: If True (default), detect crossings across NaN gaps
            via linear interpolation.
        results_to_history: If True, print spike counts to the history log
            after each run. Default False.
        results_to_cache: If True, save spike times dict to
            ``folder.toolresults`` after each run. Default True.
        results_to_numpy: If True, write ``SP_`` NMData arrays to a new
            Spike subfolder after each run. Default True.
        results_format: ``"raster"`` (default) or ``"epochs"``.
    """

    TOOLFOLDER_PREFIX = "Spike_"  # its results subfolders

    def __init__(self) -> None:
        super().__init__(name="spike")
        self._config = NMToolSpikeConfig()

        # Initialize output flags from config defaults
        self._ignore_nans = self._config.ignore_nans
        self._overwrite = self._config.overwrite
        self._results_to_history = self._config.results_to_history
        self._results_to_cache = self._config.results_to_cache
        self._results_to_numpy = self._config.results_to_numpy
        self.__results_format: str = self._config.results_format

        self._xbgn = self._config.xbgn
        self._xend = self._config.xend

        self.__ylevel: float = 0.0
        self.__func_name: str = "level+"

        # Internal run state — reset by run_init()
        self._spike_times: list[np.ndarray] = []
        self._epoch_names: list[str] = []
        self._epoch_numbers: list[int | None] = []  # in its data series
        self._durations: list[float] = []  # the x range searched
        self._detected_xunits: str | None = None
        self._toolfolder: NMToolFolder | None = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def ylevel(self) -> float:
        """Y-axis detection threshold."""
        return self.__ylevel

    @ylevel.setter
    def ylevel(self, value: float) -> None:
        self._ylevel_set(value)

    @nmch.records_command
    def _ylevel_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        """Set ylevel.

        Args:
            value: Threshold y-value (float, bool rejected).
            quiet: If True, suppress history log output.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "ylevel", "float"))
        self.__ylevel = float(value)
        nmh.history("set ylevel=%g" % self.__ylevel, quiet=quiet)
        nmch.add_nm_command("%s.ylevel = %r" % (self._name, self.__ylevel))

    @property
    def func_name(self) -> str:
        """Crossing direction: ``'level+'``, ``'level-'``, or ``'level'``."""
        return self.__func_name

    @func_name.setter
    def func_name(self, value: str) -> None:
        self._func_name_set(value)

    @nmch.records_command
    def _func_name_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        """Set func_name.

        Args:
            value: One of ``'level'``, ``'level+'``, ``'level-'``.
            quiet: If True, suppress history log output.
        """
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "func_name", "string"))
        if value not in _VALID_FUNC_NAMES:
            raise ValueError(
                "func_name must be one of %s, got %r"
                % (sorted(_VALID_FUNC_NAMES), value)
            )
        self.__func_name = value
        nmh.history("set func_name=%r" % self.__func_name, quiet=quiet)
        nmch.add_nm_command("%s.func_name = %r" % (self._name, self.__func_name))

    @property
    def results_format(self) -> str:
        """How spike times are saved (#526): ``"raster"`` (``SP_RX``, the
        times, and ``SP_RY``, each spike's epoch) or ``"epochs"`` (an
        ``SP_{data}`` array of times per epoch)."""
        return self.__results_format

    @results_format.setter
    def results_format(self, value: str) -> None:
        self._results_format_set(value)

    @nmch.records_command
    def _results_format_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "results_format", "string"))
        if value not in RESULTS_FORMATS:
            raise ValueError("results_format must be one of %s, got %r"
                             % (list(RESULTS_FORMATS), value))
        self.__results_format = value
        nmh.history("set results_format=%r" % value, quiet=quiet)
        nmch.add_nm_command("%s.results_format = %r" % (self._name, value))

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> bool:
        """Reset internal state before the run loop."""
        self._spike_times = []
        self._epoch_names = []
        self._epoch_numbers = []
        self._durations = []
        self._source_data: list[NMData] = []
        self._detected_xunits = None
        self._toolfolder = None
        return True

    def run(self) -> bool:
        """Detect threshold crossings in the current NMData array.

        Appends the detected spike times (interpolated x-values), the data
        name, its epoch number (in a data series) and the x range searched
        to the internal lists.  Silently skips arrays with no numpy data
        (``nparray`` is None).

        Returns:
            True on success.
        """
        data = self.data
        if not isinstance(data, NMData):
            raise RuntimeError("no data selected")
        if data.nparray is None:
            return True
        if self._detected_xunits is None:
            self._detected_xunits = data.xscale.units
        _indexes, x_times = find_level_crossings_nmdata(
            data, self.__ylevel,
            func_name=self.__func_name,
            xbgn=self._xbgn,
            xend=self._xend,
            ignore_nans=self._ignore_nans,
        )
        self._spike_times.append(np.asarray(x_times, dtype=float))
        self._epoch_names.append(data.name)
        epoch = self.epoch if self.dataseries is not None else None
        self._epoch_numbers.append(epoch.number if epoch is not None else None)
        self._durations.append(self._searched(data))
        self._source_data.append(data)
        return True

    def _searched(self, data: NMData) -> float:
        """The length of *data*'s x range searched: its x range within xbgn
        to xend (NaN if unknown)."""
        size = data.nparray.size
        if size == 0:
            return math.nan
        first, last = data.get_xvalue(0), data.get_xvalue(size - 1)
        if first is None or last is None:
            return math.nan
        lo, hi = sorted((float(first), float(last)))
        bgn, end = sorted((self._xbgn, self._xend))
        return max(min(hi, end) - max(lo, bgn), 0.0)

    def run_finish(self) -> bool:
        """Persist results via the enabled output sinks.

        Dispatches to :meth:`_results_to_history`, :meth:`_results_to_cache`,
        and/or :meth:`_results_to_numpy` based on the ``results_to_*`` flags.

        Returns:
            True on success.
        """
        if not self._epoch_names:
            return True
        if self._results_to_history:
            self._write_results_to_history()
        if self._results_to_cache:
            self._write_results_to_cache()
        if self._results_to_numpy:
            self._write_results_to_numpy()
        return True

    # ------------------------------------------------------------------
    # Output sinks

    def _add_note(self, data: NMData, text: str) -> None:
        """Append a note to *data*.notes if available."""
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text, readonly=True)  # a record of what was done (#505)

    def _row_numbers(self) -> list[int]:
        """Each row's epoch number (SP_epoch_num, SP_RY): its epoch's in the
        data series, or the row's index when not all rows have one (or two
        rows share one, e.g. two channels)."""
        numbers = self._epoch_numbers
        if (numbers and all(n is not None for n in numbers)
                and len(set(numbers)) == len(numbers)):
            return [int(n) for n in numbers]
        return list(range(len(self._epoch_names)))

    def _row_measures(self) -> dict[str, np.ndarray]:
        """Each row's measures: spike count, rate, first spike time, mean ISI."""
        factor, _units = _rate_units(self._detected_xunits)
        count, rate, first, isi = [], [], [], []
        for times, duration in zip(self._spike_times, self._durations):
            n = len(times)
            count.append(n)
            rate.append(n / duration * factor
                        if math.isfinite(duration) and duration > 0 else math.nan)
            first.append(float(np.min(times)) if n else math.nan)
            isi.append(float(np.mean(np.diff(np.sort(times)))) if n > 1 else math.nan)
        return {"SP_count": np.array(count, dtype=float),
                "SP_rate": np.array(rate, dtype=float),
                "SP_first": np.array(first, dtype=float),
                "SP_isi": np.array(isi, dtype=float)}

    def _run_note(self) -> str:
        """How the run was made, for its results folder, e.g. "NMSpike:
        Record, channel A, epochs all (n=3); level+ 0, x -inf to inf;
        format raster"."""
        parts = ["NMSpike: " + self._run_scope(len(self._epoch_names)),
                 "%s %g, x %g to %g" % (self.__func_name, self.__ylevel,
                                        self._xbgn, self._xend),
                 "format %s" % self.__results_format]
        return "; ".join(parts)

    def _write_results_to_numpy(self) -> NMToolFolder | None:
        """Write SP_ NMData arrays to a new Spike subfolder (#526).

        Creates a subfolder named ``Spike_{dataseries}_{channel}_{epochs}_N``
        under the current folder's toolfolder, notes how the run was made
        (one note, on the subfolder), then writes:

        * The spike times. Format ``"raster"`` (default): ``SP_RX``, every
          spike's time, and ``SP_RY``, its epoch number (as
          ``SP_epoch_num``), for a raster plot. Format ``"epochs"``: an
          ``SP_{data}`` array of times per epoch.
        * A row per data array analysed, as Stats' ST_ arrays:
          ``SP_count`` (spikes), ``SP_rate`` (spikes / x range searched; Hz
          for s or ms), ``SP_first`` (the first spike's time), ``SP_isi``
          (the mean interspike interval), ``SP_epoch_num`` and
          ``SP_data_names``.

        Returns:
            The newly created NMToolFolder, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None
        self._toolfolder = self._make_toolfolder(
            self.TOOLFOLDER_PREFIX.rstrip("_"), overwrite=self._overwrite)
        f = self._toolfolder
        f.notes.add(self._run_note(), readonly=True)  # how the results were made
        xunits = self._detected_xunits
        numbers = self._row_numbers()
        if self.__results_format == "raster":
            rx = (np.concatenate(self._spike_times) if self._spike_times
                  else np.array([], dtype=float))
            ry = np.concatenate([np.full(len(t), n, dtype=float)
                                 for t, n in zip(self._spike_times, numbers)])
            f.data.new("SP_RX", nparray=rx, yscale={"label": "Time", "units": xunits})
            f.data.new("SP_RY", nparray=ry, yscale={"label": "Epoch"})
        else:
            for name, times in zip(self._epoch_names, self._spike_times):
                f.data.new(_epoch_array_name(name), nparray=times,
                           yscale={"label": "Time", "units": xunits})
        _factor, rate_units = _rate_units(xunits)
        yscales = {"SP_count": {"label": "Spikes"},
                   "SP_rate": {"label": "Rate", "units": rate_units},
                   "SP_first": {"label": "First spike", "units": xunits},
                   "SP_isi": {"label": "Mean ISI", "units": xunits}}
        for name, values in self._row_measures().items():
            f.data.new(name, nparray=values, yscale=yscales[name])
        f.data.new("SP_epoch_num", nparray=np.array(numbers, dtype=int))
        f.data.new("SP_data_names", nparray=np.array(self._epoch_names, dtype=object))
        return f

    def _write_results_to_cache(self) -> None:
        """Save spike times dict to folder.toolresults."""
        if not isinstance(self.folder, NMFolder):
            return
        results = {
            name: times
            for name, times in zip(self._epoch_names, self._spike_times)
        }
        self.folder.toolresults_save("spike", results)

    def _write_results_to_history(self) -> None:
        """Print spike counts to the history log."""
        for name, times in zip(self._epoch_names, self._spike_times):
            nmh.history("spike: %s: %d spike(s)" % (name, len(times)))

    # ------------------------------------------------------------------
    # Convenience methods (called after run_all)

    @staticmethod
    def _values(toolfolder: NMToolFolder, name: str) -> np.ndarray | None:
        d = toolfolder.data.get(name)
        return d.nparray if d is not None and d.nparray is not None else None

    def _spike_times_from_toolfolder(
        self, toolfolder: NMToolFolder
    ) -> tuple[list[np.ndarray], list[str], str | None]:
        """Reconstruct spike times from the SP_ arrays stored in *toolfolder*.

        Reads, in turn (#526):

        * the raster, ``SP_RX`` / ``SP_RY``: each row's times are the spikes
          whose ``SP_RY`` is its ``SP_epoch_num``; rows named by
          ``SP_data_names``;
        * an ``SP_{data}`` array per name in ``SP_data_names`` (format
          ``"epochs"``), or in ``SP_epoch_names`` (saved before #526);
        * failing those, every other ``SP_*`` array, in order.

        Returns:
            ``(spike_times, epoch_names, detected_xunits)``
        """
        names_arr = self._values(toolfolder, "SP_data_names")
        rx_d, ry = toolfolder.data.get("SP_RX"), self._values(toolfolder, "SP_RY")
        if rx_d is not None and rx_d.nparray is not None and ry is not None:
            rx = rx_d.nparray
            numbers = self._values(toolfolder, "SP_epoch_num")
            if numbers is None:
                numbers = np.unique(ry)
            names = ([str(n) for n in names_arr] if names_arr is not None
                     and len(names_arr) == len(numbers)
                     else ["E%d" % int(n) for n in numbers])
            spike_times = [rx[ry == n] for n in numbers]
            return spike_times, names, rx_d.yscale.units
        if names_arr is None:
            names_arr = self._values(toolfolder, "SP_epoch_names")  # before #526
            array_name = lambda name: "SP_" + name  # noqa: E731
        else:
            array_name = _epoch_array_name
        if names_arr is not None:
            epoch_names = [str(n) for n in names_arr]
            spike_times: list[np.ndarray] = []
            detected_xunits: str | None = None
            for en in epoch_names:
                d = toolfolder.data.get(array_name(en))
                arr = d.nparray if d is not None else None
                spike_times.append(arr if arr is not None else np.array([]))
                if detected_xunits is None and d is not None:
                    detected_xunits = d.yscale.units
            return spike_times, epoch_names, detected_xunits
        # Fallback: scan by name (saved without SP_data_names or SP_epoch_names)
        spike_times = []
        epoch_names = []
        detected_xunits = None
        for d in toolfolder.data.values():
            if not isinstance(d, NMData):
                continue
            name = d.name
            if not name.startswith("SP_"):
                continue
            if any(name == p or name.startswith(p + "_") for p in _SP_RESERVED):
                continue
            arr = d.nparray
            spike_times.append(arr if arr is not None else np.array([]))
            epoch_names.append(name[3:])
            if detected_xunits is None:
                detected_xunits = d.yscale.units
        return spike_times, epoch_names, detected_xunits

    def raster(
        self,
        xbgn: float | None = None,
        xend: float | None = None,
        toolfolder: NMToolFolder | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Return spike times and epoch labels ready for raster plotting.

        By default returns the internal spike times collected during the most
        recent :meth:`run_all` call.  Pass *toolfolder* to reconstruct from
        the ``SP_`` NMData arrays saved in a previous run's subfolder instead.

        Example::

            times_list, labels = tool.raster()
            for i, (times, label) in enumerate(zip(times_list, labels)):
                ax.scatter(times, np.full_like(times, i), marker="|")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

        Args:
            xbgn: Lower bound for filtering spike times. Default None (no lower
                bound).
            xend: Upper bound for filtering spike times. Default None (no upper
                bound).
            toolfolder: Optional Spike toolfolder from a previous run.  When
                provided, spike times are read from its ``SP_*`` arrays rather
                than from in-memory state.

        Returns:
            ``(spike_times, epoch_names)`` where *spike_times* is a list of
            numpy arrays (one per epoch, possibly empty) and *epoch_names*
            is the corresponding list of data array names.

        Raises:
            RuntimeError: If called before :meth:`run_all` (no toolfolder)
                or if the toolfolder contains no ``SP_`` arrays.
        """
        if toolfolder is not None:
            spike_times, epoch_names, _ = self._spike_times_from_toolfolder(toolfolder)
            if not epoch_names:
                raise RuntimeError(
                    "NMToolSpike.raster: no SP_ arrays found in toolfolder"
                )
        else:
            if not self._epoch_names:
                raise RuntimeError(
                    "NMToolSpike.raster: no spike data — run detection first"
                )
            spike_times = list(self._spike_times)
            epoch_names = list(self._epoch_names)
        if xbgn is not None or xend is not None:
            lo = float(xbgn) if xbgn is not None else -math.inf
            hi = float(xend) if xend is not None else math.inf
            spike_times = [t[(t >= lo) & (t <= hi)] for t in spike_times]
        return spike_times, list(epoch_names)

    def _result_array_name(
        self, prefix: str, folder: NMToolFolder, overwrite: bool
    ) -> str:
        """The name of a PST / ISI histogram, e.g. ``H_SP_PST`` (#526).

        With *overwrite* True returns *prefix* (replacing an earlier one).
        With *overwrite* False, the first unused of *prefix*, ``{prefix}_1``,
        ``{prefix}_2``, … in *folder*.
        """
        if overwrite or prefix not in folder.data:
            return prefix
        i = 1
        while ("%s_%d" % (prefix, i)) in folder.data:
            i += 1
        return "%s_%d" % (prefix, i)

    def intervals(
        self,
        xbgn: float | None = None,
        xend: float | None = None,
        toolfolder: NMToolFolder | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Return per-epoch interspike intervals ready for analysis or plotting.

        For each epoch computes ``numpy.diff`` of the (optionally filtered)
        spike times.  Epochs with fewer than two spikes produce an empty array.

        By default reads from the most recent :meth:`run_all` call.  Pass
        *toolfolder* to reconstruct from a previous run's ``SP_*`` arrays.

        Example::

            ivs, labels = tool.intervals()
            for ivs_epoch, label in zip(ivs, labels):
                print(label, ivs_epoch.mean())

        Args:
            xbgn: Lower bound for filtering spike times before computing
                intervals. Default None (no lower bound).
            xend: Upper bound for filtering spike times before computing
                intervals. Default None (no upper bound).
            toolfolder: Optional Spike toolfolder from a previous run.  When
                provided, spike times are read from its ``SP_*`` arrays.

        Returns:
            ``(interval_arrays, epoch_names)`` where *interval_arrays* is a
            list of numpy arrays (one per epoch, possibly empty) and
            *epoch_names* is the corresponding list of data array names.

        Raises:
            RuntimeError: If called before :meth:`run_all` (no toolfolder)
                or if the toolfolder contains no ``SP_`` arrays.
        """
        if toolfolder is not None:
            spike_times, epoch_names, _ = self._spike_times_from_toolfolder(toolfolder)
            if not epoch_names:
                raise RuntimeError(
                    "NMToolSpike.intervals: no SP_ arrays found in toolfolder"
                )
        else:
            if not self._spike_times:
                raise RuntimeError(
                    "NMToolSpike.intervals: no spike data — run detection first"
                )
            spike_times = list(self._spike_times)
            epoch_names = list(self._epoch_names)
        if xbgn is not None or xend is not None:
            lo = float(xbgn) if xbgn is not None else -math.inf
            hi = float(xend) if xend is not None else math.inf
            spike_times = [t[(t >= lo) & (t <= hi)] for t in spike_times]
        return [np.diff(t) for t in spike_times], list(epoch_names)

    def extract_spike_waveforms(
        self,
        pre: float,
        post: float,
        xbgn: float | None = None,
        xend: float | None = None,
        clip_to_next_spike: bool = False,
        edge: str = "skip",
        align: str = "zero",
        toolfolder: NMToolFolder | None = None,
    ) -> list[NMData]:
        """Extract fixed-width waveform snippets around each detected spike.

        For each spike detected during the most recent :meth:`run_all` call,
        extracts a ``[−pre, +post]`` window of samples from the source epoch
        array and writes the snippet as a new NMData array (``SPK_{ch}{n}``)
        in the Spike subfolder.

        Pass *toolfolder* to instead reconstruct spike times from a previous
        run's ``SP_*`` arrays.  Source recordings are looked up by epoch name
        from ``self.folder.data``; epochs whose source array cannot be found
        are silently skipped.  Snippets are written back into *toolfolder*.

        Args:
            pre:               Time before spike in source x-units. Must be > 0.
            post:              Time after spike in source x-units. Must be > 0.
            xbgn:                Lower bound for filtering spike times before
                extracting waveforms. Default None (no lower bound). Spikes
                with times < *xbgn* are skipped.
            xend:                Upper bound for filtering spike times before
                extracting waveforms. Default None (no upper bound). Spikes
                with times > *xend* are skipped.
            clip_to_next_spike: If True, samples in the post window that fall
                beyond the next spike time are replaced with ``NaN``, so the
                snippet does not contain data from the next spike's waveform.
                All snippets remain the same length (``pre + post`` samples).
                The last spike in each epoch is never clipped.
            edge:              How to handle spikes whose window extends
                beyond the recording edge (case-insensitive):

                * ``"skip"`` (default) — omit the spike; no array is written.
                * ``"pad"``  — write a full-length snippet, filling
                  out-of-bounds samples with ``NaN``.

            align:             X-axis alignment of output snippets
                (case-insensitive):

                * ``"zero"`` (default) — x starts at 0 (runs 0 to
                  ``pre + post``).
                * ``"spike"`` — x=0 at the threshold crossing (runs
                  ``−pre`` to ``+post``).
                * ``"source"`` — x values preserved from the source
                  recording; no shift is applied.

            toolfolder:        Optional Spike toolfolder from a previous run.
                When provided, spike times are read from its ``SP_*`` arrays,
                source recordings are looked up from ``self.folder.data`` by
                epoch name, and snippets are written into *toolfolder*.

        Returns:
            List of newly created NMData arrays, one per non-skipped spike
            across all processed epochs.

        Note:
            When the source array has a non-uniform xarray (e.g. a recording
            with two different sample rates), the number of samples extracted
            is determined by ``numpy.median(numpy.diff(xarray))`` — a
            pragmatic single-pass estimate of the typical sample interval.
            This means the actual duration covered by *pre* and *post* may
            differ from requested for spikes in regions whose local sample
            rate differs from the median.  The output xarray is always a
            direct slice of the source (nonuniform spacing is preserved);
            only the sample count is affected.

        Raises:
            RuntimeError: If called before :meth:`run_all` (no toolfolder)
                or if the toolfolder contains no ``SP_`` arrays.
            ValueError:   If *pre* or *post* <= 0, or *edge* is not
                          ``"skip"`` or ``"pad"``.
            ValueError:   If *align* is not ``"zero"``, ``"spike"``, or
                          ``"source"``.
            TypeError:    If *pre*, *post*, or *clip_to_next_spike* have
                          wrong types.
        """
        _VALID_EDGES = ("skip", "pad")
        edge = edge.lower()
        if edge not in _VALID_EDGES:
            raise ValueError(
                "edge must be one of %s, got %r" % (list(_VALID_EDGES), edge)
            )
        if isinstance(pre, bool) or not isinstance(pre, (int, float)):
            raise TypeError(nmu.type_error_str(pre, "pre", "float"))
        if pre <= 0:
            raise ValueError("pre must be > 0, got %g" % pre)
        if isinstance(post, bool) or not isinstance(post, (int, float)):
            raise TypeError(nmu.type_error_str(post, "post", "float"))
        if post <= 0:
            raise ValueError("post must be > 0, got %g" % post)
        if not isinstance(clip_to_next_spike, bool):
            raise TypeError(
                nmu.type_error_str(clip_to_next_spike, "clip_to_next_spike", "boolean")
            )
        _VALID_ALIGNS = ("zero", "spike", "source")
        if not isinstance(align, str):
            raise TypeError(nmu.type_error_str(align, "align", "str"))
        align = align.lower()
        if align not in _VALID_ALIGNS:
            raise ValueError(
                "align must be one of %s, got %r" % (list(_VALID_ALIGNS), align)
            )
        if toolfolder is not None:
            spike_times_list, epoch_names_list, _ = self._spike_times_from_toolfolder(toolfolder)
            if not epoch_names_list:
                raise RuntimeError(
                    "NMToolSpike.extract_spike_waveforms: no SP_ arrays found in toolfolder"
                )
            source_data_list: list[NMData | None] = []
            for en in epoch_names_list:
                src = (
                    self.folder.data.get(en)
                    if isinstance(self.folder, NMFolder)
                    else None
                )
                source_data_list.append(src)
            out_folder: NMToolFolder | None = toolfolder
        else:
            if not self._epoch_names:
                raise RuntimeError(
                    "NMToolSpike.extract_spike_waveforms: no spike data — run detection first"
                )
            spike_times_list = self._spike_times
            epoch_names_list = self._epoch_names
            source_data_list = self._source_data
            out_folder = None

        output: list[NMData] = []
        ch_char = self.channel.name if self.channel is not None else "A"
        spike_counter: int = 0
        matches: dict[tuple[str, int], NMData] = {}

        for epoch_name, spike_times_arr, source in zip(
            epoch_names_list, spike_times_list, source_data_list
        ):
            if source is None:
                continue
            ydata = source.nparray
            if ydata is None:
                continue
            n_total = ydata.size

            # Sample interval for converting pre/post to sample counts
            if source.xarray is not None and len(source.xarray) >= 2:
                delta_for_samples = float(np.median(np.diff(source.xarray)))
            else:
                delta_for_samples = float(source.xscale.delta)

            if xbgn is not None or xend is not None:
                lo = float(xbgn) if xbgn is not None else -math.inf
                hi = float(xend) if xend is not None else math.inf
                spike_times_arr = spike_times_arr[
                    (spike_times_arr >= lo) & (spike_times_arr <= hi)
                ]

            pre_samples = int(round(pre / abs(delta_for_samples)))

            post_samples = max(int(round(post / abs(delta_for_samples))), 1)

            for n, spike_x in enumerate(spike_times_arr):
                # Nearest sample index to spike
                i_spike = source.get_xindex(spike_x, clip=False)
                if i_spike is None:
                    continue

                i0 = i_spike - pre_samples
                i1 = i_spike + post_samples

                # Edge handling: build full-length y snippet (and x if needed)
                use_xarray = source.xarray is not None
                if i0 < 0 or i1 > n_total:
                    if edge == "skip":
                        continue
                    # "pad": full-length NaN array, fill valid region
                    snippet = np.full(pre_samples + post_samples, np.nan)
                    src_start = max(i0, 0)
                    src_end   = min(i1, n_total)
                    dst_start = src_start - i0
                    dst_end   = dst_start + (src_end - src_start)
                    snippet[dst_start:dst_end] = ydata[src_start:src_end]
                    n_padded = (pre_samples + post_samples) - (src_end - src_start)
                    if use_xarray:
                        # Extrapolate out-of-bounds x-values using delta
                        x_snippet = np.full(pre_samples + post_samples, np.nan)
                        x_snippet[dst_start:dst_end] = source.xarray[src_start:src_end]
                        # fill left pad by stepping backwards from first valid x
                        if dst_start > 0:
                            xbgn_valid = source.xarray[src_start]
                            for k in range(dst_start - 1, -1, -1):
                                x_snippet[k] = xbgn_valid - (dst_start - k) * delta_for_samples
                        # fill right pad by stepping forwards from last valid x
                        if dst_end < len(x_snippet):
                            xend_valid = source.xarray[src_end - 1]
                            for k in range(dst_end, len(x_snippet)):
                                x_snippet[k] = xend_valid + (k - dst_end + 1) * delta_for_samples
                else:
                    snippet = ydata[i0:i1].copy()
                    n_padded = 0
                    if use_xarray:
                        x_snippet = source.xarray[i0:i1].copy()

                # clip_to_next_spike: NaN-fill samples beyond next spike time
                n_nan_clipped = 0
                if clip_to_next_spike and n + 1 < len(spike_times_arr):
                    next_interval = float(spike_times_arr[n + 1]) - float(spike_x)
                    clip_samples = int(round(next_interval / abs(delta_for_samples)))
                    clip_samples = max(clip_samples, 1)
                    if clip_samples < post_samples:
                        snippet = snippet.copy()
                        snippet[pre_samples + clip_samples:] = np.nan
                        n_nan_clipped = post_samples - clip_samples

                # Build output x-scale or xarray
                if use_xarray:
                    if align == "spike":
                        x_snippet = x_snippet - float(spike_x)
                    elif align == "zero":
                        x_snippet = x_snippet - float(x_snippet[0])
                    # align == "source": no shift
                    out_xscale = {
                        "label": source.xscale.label,
                        "units": source.xscale.units,
                    }
                else:
                    x_snippet = None
                    if align == "spike":
                        xstart = -float(pre)
                    elif align == "zero":
                        xstart = 0.0
                    else:  # "source"
                        xstart = float(source.xscale.start) + i0 * float(source.xscale.delta)
                    out_xscale = {
                        "start": xstart,
                        "delta": float(source.xscale.delta),
                        "label": source.xscale.label,
                        "units": source.xscale.units,
                    }
                out_yscale = {
                    "label": source.yscale.label,
                    "units": source.yscale.units,
                }
                if out_folder is None:
                    out_folder = self._make_toolfolder("Spike", overwrite=self._overwrite)
                array_name = "SPK_%s%d" % (ch_char, spike_counter)
                d = out_folder.data.new(
                    array_name,
                    nparray=snippet,
                    xarray=x_snippet,
                    xscale=out_xscale,
                    yscale=out_yscale,
                )
                matches[ch_char, spike_counter] = d
                spike_counter += 1
                note = (
                    "NMSpike.extract_spike_waveforms(source=%s, spike_x=%.6g, "
                    "pre=%g, post=%g, clip_to_next_spike=%s, edge=%r, align=%s"
                    % (epoch_name, float(spike_x), pre, post,
                       clip_to_next_spike, edge, align)
                )
                if n_padded:
                    note += ", n_padded=%d" % n_padded
                if n_nan_clipped:
                    note += ", n_nan_clipped=%d" % n_nan_clipped
                note += ")"
                self._add_note(d, note)
                output.append(d)

        if out_folder is not None:
            out_folder.build_dataseries("SPK_", matches)
        return output

    # ------------------------------------------------------------------
    # Histogram methods (called after run_all)

    def pst(
        self,
        bins: int = 100,
        xbgn: float | None = None,
        xend: float | None = None,
        output_mode: str = "count",
        overwrite: bool = True,
        toolfolder: NMToolFolder | None = None,
    ) -> NMData | None:
        """Compute a peri-stimulus time (PST) histogram from spike times.

        Pools all spike times across epochs, computes a histogram via
        :func:`~pyneuromatic.core.nm_math.histogram`, and writes the result
        as ``H_SP_PST`` in the Spike subfolder (named as Stats' histograms).

        By default reads from the most recent :meth:`run_all` call's in-memory
        state and writes to ``self._toolfolder``.  Pass *toolfolder* to instead
        reconstruct spike times from a previous run's ``SP_*`` arrays and write
        ``H_SP_PST`` back into that same subfolder.

        Args:
            bins: Number of histogram bins. Default 100.
            xbgn: Lower edge of the first bin (inclusive). Default None
                (use minimum spike time).
            xend: Upper edge of the last bin (exclusive). Default None
                (use maximum spike time).
            output_mode: Controls the y-axis values:

                * ``"count"`` (default) — raw spike count per bin.
                * ``"rate"`` — mean firing rate in Hz:
                  ``count / (n_epochs × bin_width)``.
                * ``"probability"`` — fraction of epochs with a spike in
                  each bin: ``count / n_epochs`` (values in ``[0, 1]``).

            overwrite: If True (default), result is written as ``H_SP_PST``,
                replacing any existing array of that name.  If False, a new
                ``H_SP_PST_1``, ``H_SP_PST_2``, … is created so previous
                results are preserved.
            toolfolder: Optional Spike toolfolder from a previous run.  When
                provided, spike times are read from its ``SP_*`` arrays and
                the result is written back into it.

        Returns:
            The new ``H_SP_PST`` NMData, or None if no spikes were detected.

        Raises:
            RuntimeError: If called before :meth:`run_all` (no toolfolder)
                or if the toolfolder contains no ``SP_`` arrays.
            ValueError: If *output_mode* is not ``"count"``, ``"rate"``,
                or ``"probability"``.
        """
        _VALID_MODES = ("count", "rate", "probability")
        output_mode = output_mode.lower()
        if output_mode not in _VALID_MODES:
            raise ValueError(
                "output_mode must be one of %s, got %r"
                % (list(_VALID_MODES), output_mode)
            )
        if toolfolder is not None:
            spike_times, _, detected_xunits = self._spike_times_from_toolfolder(toolfolder)
            if not spike_times:
                raise RuntimeError(
                    "NMToolSpike.pst: no SP_ arrays found in toolfolder"
                )
            out_folder = toolfolder
        else:
            if not self._spike_times:
                raise RuntimeError(
                    "NMToolSpike.pst: no spike data — run detection first"
                )
            spike_times = self._spike_times
            detected_xunits = self._detected_xunits
            if self._toolfolder is None:
                self._toolfolder = self._make_toolfolder("Spike", overwrite=self._overwrite)
            out_folder = self._toolfolder
        all_times = np.concatenate(spike_times)
        if len(all_times) == 0:
            return None
        xrange: tuple[float, float] | None = None
        if xbgn is not None or xend is not None:
            lo = float(xbgn) if xbgn is not None else float(all_times.min())
            hi = float(xend) if xend is not None else float(all_times.max())
            xrange = (lo, hi)
        result = nm_math.histogram(all_times, bins=bins, xrange=xrange)
        counts, edges = result["counts"], result["edges"]
        delta = float(edges[1] - edges[0])
        n_epochs = len(spike_times)
        if output_mode == "rate":
            ydata = counts.astype(float) / (n_epochs * delta)
            ylabel = "Spike rate (Hz)"
        elif output_mode == "probability":
            ydata = counts.astype(float) / n_epochs
            ylabel = "Spike probability"
        else:
            ydata = counts.astype(float)
            ylabel = "Spike count"
        pst_name = self._result_array_name("H_SP_PST", out_folder, overwrite)
        if pst_name in out_folder.data:
            del out_folder.data[pst_name]
        d = out_folder.data.new(
            pst_name,
            nparray=ydata,
            xscale={
                "start": float(edges[0]),
                "delta": delta,
                "label": "Time",
                "units": detected_xunits,
            },
            yscale={"label": ylabel},
        )
        self._add_note(
            d,
            "NMSpike.pst(bins=%d, xbgn=%s, xend=%s, output_mode=%r, n_epochs=%d, n_spikes=%d)"
            % (bins, xbgn, xend, output_mode, n_epochs, int(counts.sum())),
        )
        return d

    def isi(
        self,
        bins: int = 100,
        xbgn: float | None = None,
        xend: float | None = None,
        min_isi: float | None = None,
        max_isi: float | None = None,
        output_mode: str = "count",
        overwrite: bool = True,
        toolfolder: NMToolFolder | None = None,
    ) -> NMData | None:
        """Compute an interspike interval (ISI) histogram from spike times.

        Optionally filters spike times to a window [*xbgn*, *xend*] before
        computing intervals. Computes ``numpy.diff`` of each epoch's
        (filtered) spike times, pools the intervals across all epochs, and
        writes the histogram as ``H_SP_ISI`` in the Spike subfolder.
        Epochs with fewer than two spikes (after filtering) contribute no
        intervals.

        By default reads from the most recent :meth:`run_all` call's in-memory
        state.  Pass *toolfolder* to instead reconstruct spike times from a
        previous run's ``SP_*`` arrays and write ``H_SP_ISI`` back into that
        same subfolder.

        Args:
            bins:        Number of histogram bins. Default 100.
            xbgn:          Lower bound for filtering spike times before
                         computing intervals. Default None (no lower bound).
            xend:          Upper bound for filtering spike times before
                         computing intervals. Default None (no upper bound).
            min_isi:     Lower edge of the histogram x-axis. Default None
                         (use the minimum interval). Useful for excluding
                         refractory-period artefacts.
            max_isi:     Upper edge of the histogram x-axis. Default None
                         (use the maximum interval).
            output_mode: Controls the y-axis values (case-insensitive):

                         * ``"count"`` (default) — raw interval counts.
                         * ``"probability"`` — count / total_intervals;
                           normalised to a probability distribution
                           (sums to 1.0).

            overwrite:   If True (default), result is written as ``H_SP_ISI``,
                         replacing any existing array of that name.  If False,
                         a new ``H_SP_ISI_1``, ``H_SP_ISI_2``, … is created so
                         previous results are preserved.
            toolfolder:  Optional Spike toolfolder from a previous run.  When
                         provided, spike times are read from its ``SP_*``
                         arrays and the result is written back into it.

        Returns:
            The new ``H_SP_ISI`` NMData, or None if fewer than 2 spikes total
            (after xbgn/xend filtering).

        Raises:
            RuntimeError: If called before :meth:`run_all` (no toolfolder)
                or if the toolfolder contains no ``SP_`` arrays.
            ValueError:   If *output_mode* is not ``"count"`` or
                          ``"probability"``.
        """
        _VALID_MODES = ("count", "probability")
        output_mode = output_mode.lower()
        if output_mode not in _VALID_MODES:
            raise ValueError(
                "output_mode must be one of %s, got %r"
                % (list(_VALID_MODES), output_mode)
            )
        # intervals() validates spike data and applies xbgn/xend filtering
        interval_arrays, _ = self.intervals(xbgn=xbgn, xend=xend, toolfolder=toolfolder)
        valid = [iv for iv in interval_arrays if len(iv) > 0]
        if not valid:
            return None
        all_isis = np.concatenate(valid)
        # output folder and x-units
        if toolfolder is not None:
            _, _, detected_xunits = self._spike_times_from_toolfolder(toolfolder)
            out_folder = toolfolder
        else:
            detected_xunits = self._detected_xunits
            if self._toolfolder is None:
                self._toolfolder = self._make_toolfolder("Spike", overwrite=self._overwrite)
            out_folder = self._toolfolder
        lo_isi = float(min_isi) if min_isi is not None else None
        hi_isi = float(max_isi) if max_isi is not None else None
        if lo_isi is not None or hi_isi is not None:
            xrange: tuple[float, float] | None = (
                lo_isi if lo_isi is not None else float(all_isis.min()),
                hi_isi if hi_isi is not None else float(all_isis.max()),
            )
        else:
            xrange = None
        result = nm_math.histogram(all_isis, bins=bins, xrange=xrange)
        counts, edges = result["counts"], result["edges"]
        delta = float(edges[1] - edges[0])
        n_intervals = int(counts.sum())
        if output_mode == "probability":
            total = float(len(all_isis))
            ydata = counts.astype(float) / total if total > 0 else counts.astype(float)
            ylabel = "ISI probability"
        else:
            ydata = counts.astype(float)
            ylabel = "Count"
        isi_name = self._result_array_name("H_SP_ISI", out_folder, overwrite)
        if isi_name in out_folder.data:
            del out_folder.data[isi_name]
        d = out_folder.data.new(
            isi_name,
            nparray=ydata,
            xscale={
                "start": float(edges[0]),
                "delta": delta,
                "label": "ISI",
                "units": detected_xunits,
            },
            yscale={"label": ylabel},
        )
        self._add_note(
            d,
            "NMSpike.isi(bins=%d, xbgn=%s, xend=%s, min_isi=%s, max_isi=%s, "
            "output_mode=%r, n_intervals=%d)"
            % (bins, xbgn, xend, min_isi, max_isi, output_mode, n_intervals),
        )
        return d
