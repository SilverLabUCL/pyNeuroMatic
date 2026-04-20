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

import numpy as np

from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.analysis.nm_tool_config import NMToolConfig
from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_math as nm_math
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.analysis.nm_tool_utilities import find_level_crossings_nmdata

_VALID_FUNC_NAMES: frozenset[str] = frozenset({"level", "level+", "level-"})


class NMToolSpikeConfig(NMToolConfig):
    """Configuration for NMToolSpike.

    Parameters:
        ylevel: Y-axis detection threshold. Default 0.0.
        func_name: Crossing direction — ``"level+"`` (rising, default),
            ``"level-"`` (falling), or ``"level"`` (both).
        results_to_history: If True, print spike counts to the history log
            after each run. Default False.
        results_to_cache: If True, save spike times dict to
            ``folder.toolresults`` after each run. Default True.
        results_to_numpy: If True, write ``SP_`` NMData arrays to a new
            Spike subfolder after each run. Default True.
    """

    _TOML_TYPE = "spike_config"
    _schema = {
        "ylevel":             {"type": float, "default": 0.0},
        "func_name":          {"type": str,   "default": "level+",
                               "choices": ["level", "level+", "level-"]},
        "overwrite":          {"type": bool,  "default": True},
        "results_to_history": {"type": bool,  "default": False},
        "results_to_cache":   {"type": bool,  "default": True},
        "results_to_numpy":   {"type": bool,  "default": True},
    }


class NMToolSpike(NMTool):
    """Spike detection tool using threshold crossings.

    Detects action potentials (or other threshold-crossing events) in
    NMData arrays via :func:`~pyneuromatic.core.nm_math.find_level_crossings`.
    Per-epoch spike time arrays (``SP_`` prefix) and a spike-count array
    (``SP_count``) are written to a new Spike subfolder after each run.

    After detection, call :meth:`pst` or :meth:`isi` to compute histograms
    from the most recent run's spike times.

    Attributes:
        ylevel: Y-axis detection threshold. Default 0.0.
        func_name: Crossing direction — ``"level+"`` (rising, default),
            ``"level-"`` (falling), or ``"level"`` (both).
        results_to_history: If True, print spike counts to the history log
            after each run. Default False.
        results_to_cache: If True, save spike times dict to
            ``folder.toolresults`` after each run. Default True.
        results_to_numpy: If True, write ``SP_`` NMData arrays to a new
            Spike subfolder after each run. Default True.
    """

    def __init__(self) -> None:
        super().__init__(name="spike")
        self._config = NMToolSpikeConfig()

        self.__ylevel: float = 0.0
        self.__func_name: str = "level+"
        self.__overwrite: bool = True

        self.__results_to_history: bool = False
        self.__results_to_cache: bool = True
        self.__results_to_numpy: bool = True

        # Internal run state — reset by run_init()
        self._spike_times: list[np.ndarray] = []
        self._epoch_names: list[str] = []
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
    def overwrite(self) -> bool:
        """If True, reuse ``{base}_0`` subfolder (clearing old arrays) on each run.

        If False, each run creates a new numbered subfolder
        (``{base}_0``, ``{base}_1``, …) preserving previous results.
        """
        return self.__overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        self._overwrite_set(value)

    def _overwrite_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "overwrite", "boolean"))
        self.__overwrite = value
        nmh.history("set overwrite=%s" % value, quiet=quiet)
        nmch.add_nm_command("%s.overwrite = %r" % (self._name, self.__overwrite))

    @property
    def results_to_history(self) -> bool:
        """If True, print spike counts to the history log after each run."""
        return self.__results_to_history

    @results_to_history.setter
    def results_to_history(self, value: bool) -> None:
        self._results_to_history_set(value)

    def _results_to_history_set(
        self, value: bool, quiet: bool = nmc.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                nmu.type_error_str(value, "results_to_history", "boolean")
            )
        self.__results_to_history = value
        nmh.history("set results_to_history=%s" % value, quiet=quiet)
        nmch.add_nm_command(
            "%s.results_to_history = %r" % (self._name, self.__results_to_history)
        )

    @property
    def results_to_cache(self) -> bool:
        """If True, save spike times dict to folder.toolresults after each run."""
        return self.__results_to_cache

    @results_to_cache.setter
    def results_to_cache(self, value: bool) -> None:
        self._results_to_cache_set(value)

    def _results_to_cache_set(
        self, value: bool, quiet: bool = nmc.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                nmu.type_error_str(value, "results_to_cache", "boolean")
            )
        self.__results_to_cache = value
        nmh.history("set results_to_cache=%s" % value, quiet=quiet)
        nmch.add_nm_command(
            "%s.results_to_cache = %r" % (self._name, self.__results_to_cache)
        )

    @property
    def results_to_numpy(self) -> bool:
        """If True, write SP_ NMData arrays to a Spike subfolder after each run."""
        return self.__results_to_numpy

    @results_to_numpy.setter
    def results_to_numpy(self, value: bool) -> None:
        self._results_to_numpy_set(value)

    def _results_to_numpy_set(
        self, value: bool, quiet: bool = nmc.QUIET
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(
                nmu.type_error_str(value, "results_to_numpy", "boolean")
            )
        self.__results_to_numpy = value
        nmh.history("set results_to_numpy=%s" % value, quiet=quiet)
        nmch.add_nm_command(
            "%s.results_to_numpy = %r" % (self._name, self.__results_to_numpy)
        )

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> bool:
        """Reset internal state before the run loop."""
        self._spike_times = []
        self._epoch_names = []
        self._detected_xunits = None
        self._toolfolder = None
        return True

    def run(self) -> bool:
        """Detect threshold crossings in the current NMData array.

        Appends the detected spike times (interpolated x-values) and the
        data name to the internal lists.  Silently skips arrays with no
        numpy data (``nparray`` is None).

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
            data, self.__ylevel, func_name=self.__func_name
        )
        self._spike_times.append(x_times)
        self._epoch_names.append(data.name)
        return True

    def run_finish(self) -> bool:
        """Persist results via the enabled output sinks.

        Dispatches to :meth:`_results_to_history`, :meth:`_results_to_cache`,
        and/or :meth:`_results_to_numpy` based on the ``results_to_*`` flags.

        Returns:
            True on success.
        """
        if not self._epoch_names:
            return True
        if self.__results_to_history:
            self._results_to_history()
        if self.__results_to_cache:
            self._results_to_cache()
        if self.__results_to_numpy:
            self._results_to_numpy()
        return True

    # ------------------------------------------------------------------
    # Output sinks

    def _add_note(self, data: NMData, text: str) -> None:
        """Append a note to *data*.notes if available."""
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    def _results_to_numpy(self) -> NMToolFolder | None:
        """Write SP_ NMData arrays to a new Spike subfolder.

        Creates a subfolder named ``spike_{dataseries}_{channel}_N``
        (first unused N) under the current folder's toolfolder, then writes:

        * One ``SP_{epoch_name}`` array per epoch containing the spike times.
        * One ``SP_count`` array (length = number of epochs) containing the
          spike count per epoch.

        Returns:
            The newly created NMToolFolder, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None
        self._toolfolder = self._make_toolfolder()
        f = self._toolfolder
        for name, times in zip(self._epoch_names, self._spike_times):
            d = f.data.new(
                "SP_" + name,
                nparray=times,
                yscale={"label": "Time", "units": self._detected_xunits},
            )
            self._add_note(
                d,
                "NMSpike(source=%s, ylevel=%g, func_name=%r, n=%d)"
                % (name, self.__ylevel, self.__func_name, len(times)),
            )
        counts = np.array([len(t) for t in self._spike_times], dtype=float)
        d_count = f.data.new("SP_count", nparray=counts)
        self._add_note(
            d_count,
            "NMSpike(ylevel=%g, func_name=%r, n_epochs=%d)"
            % (self.__ylevel, self.__func_name, len(self._epoch_names)),
        )
        return f

    def _results_to_cache(self) -> None:
        """Save spike times dict to folder.toolresults."""
        if not isinstance(self.folder, NMFolder):
            return
        results = {
            name: times
            for name, times in zip(self._epoch_names, self._spike_times)
        }
        self.folder.toolresults_save("spike", results)

    def _results_to_history(self) -> None:
        """Print spike counts to the history log."""
        for name, times in zip(self._epoch_names, self._spike_times):
            nmh.history("spike: %s: %d spike(s)" % (name, len(times)))

    def _make_toolfolder(self) -> NMToolFolder:
        """Return the target ``spike_{dataseries}_{channel}_N`` subfolder."""
        parts = ["Spike"]
        if self.dataseries is not None:
            parts.append(self.dataseries.name)
        if self.channel is not None:
            parts.append(self.channel.name)
        base = "_".join(parts)
        return self.folder.toolfolder.get_or_create(base, overwrite=self.__overwrite)

    # ------------------------------------------------------------------
    # Convenience methods (called after run_all)

    def raster(self) -> tuple[list[np.ndarray], list[str]]:
        """Return spike times and epoch labels ready for raster plotting.

        Returns the internal spike times collected during the most recent
        :meth:`run_all` call as a pair of parallel lists — one entry per
        epoch — without requiring the caller to know the subfolder structure.

        Example::

            times_list, labels = tool.raster()
            for i, (times, label) in enumerate(zip(times_list, labels)):
                ax.scatter(times, np.full_like(times, i), marker="|")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

        Returns:
            ``(spike_times, epoch_names)`` where *spike_times* is a list of
            numpy arrays (one per epoch, possibly empty) and *epoch_names*
            is the corresponding list of data array names.

        Raises:
            RuntimeError: If called before :meth:`run_all`.
        """
        if self._toolfolder is None and not self._epoch_names:
            raise RuntimeError(
                "NMToolSpike.raster: no spike data — run detection first"
            )
        return list(self._spike_times), list(self._epoch_names)

    # ------------------------------------------------------------------
    # Histogram methods (called after run_all)

    def pst(
        self,
        bins: int = 100,
        tstart: float | None = None,
        tend: float | None = None,
    ) -> NMData | None:
        """Compute a peri-stimulus time (PST) histogram from spike times.

        Pools all spike times across epochs from the most recent
        :meth:`run_all` call, computes a histogram via
        :func:`~pyneuromatic.core.nm_math.histogram`, and writes the result
        as ``SP_PST`` in the current Spike subfolder.

        Args:
            bins: Number of histogram bins. Default 100.
            tstart: Lower edge of the first bin (inclusive). Default None
                (use minimum spike time).
            tend: Upper edge of the last bin (exclusive). Default None
                (use maximum spike time).

        Returns:
            The new ``SP_PST`` NMData, or None if no spikes were detected.

        Raises:
            RuntimeError: If called before :meth:`run_all`.
        """
        if self._toolfolder is None:
            raise RuntimeError(
                "NMToolSpike.pst: no spike data — run detection first"
            )
        if not self._spike_times:
            return None
        all_times = np.concatenate(self._spike_times)
        if len(all_times) == 0:
            return None
        xrange: tuple[float, float] | None = None
        if tstart is not None or tend is not None:
            lo = float(tstart) if tstart is not None else float(all_times.min())
            hi = float(tend) if tend is not None else float(all_times.max())
            xrange = (lo, hi)
        result = nm_math.histogram(all_times, bins=bins, xrange=xrange)
        counts, edges = result["counts"], result["edges"]
        delta = float(edges[1] - edges[0])
        d = self._toolfolder.data.new(
            "SP_PST",
            nparray=counts.astype(float),
            xscale={
                "start": float(edges[0]),
                "delta": delta,
                "label": "Time",
                "units": self._detected_xunits,
            },
            yscale={"label": "Spike count"},
        )
        self._add_note(
            d,
            "NMSpike.pst(bins=%d, tstart=%s, tend=%s, n_spikes=%d)"
            % (bins, tstart, tend, int(counts.sum())),
        )
        return d

    def isi(
        self,
        bins: int = 100,
        max_isi: float | None = None,
    ) -> NMData | None:
        """Compute an interspike interval (ISI) histogram from spike times.

        Computes ``numpy.diff`` of each epoch's spike times, pools the
        intervals across all epochs, and writes the histogram as ``SP_ISI``
        in the current Spike subfolder.  Epochs with fewer than two spikes
        contribute no intervals.

        Args:
            bins: Number of histogram bins. Default 100.
            max_isi: Upper bound of the last bin. Default None (use the
                maximum interval).

        Returns:
            The new ``SP_ISI`` NMData, or None if fewer than 2 spikes total.

        Raises:
            RuntimeError: If called before :meth:`run_all`.
        """
        if self._toolfolder is None:
            raise RuntimeError(
                "NMToolSpike.isi: no spike data — run detection first"
            )
        intervals = [
            np.diff(t) for t in self._spike_times if len(t) >= 2
        ]
        if not intervals:
            return None
        all_isis = np.concatenate(intervals)
        xrange: tuple[float, float] | None = (
            (0.0, float(max_isi)) if max_isi is not None else None
        )
        result = nm_math.histogram(all_isis, bins=bins, xrange=xrange)
        counts, edges = result["counts"], result["edges"]
        delta = float(edges[1] - edges[0])
        d = self._toolfolder.data.new(
            "SP_ISI",
            nparray=counts.astype(float),
            xscale={
                "start": float(edges[0]),
                "delta": delta,
                "label": "ISI",
                "units": self._detected_xunits,
            },
            yscale={"label": "Count"},
        )
        self._add_note(
            d,
            "NMSpike.isi(bins=%d, max_isi=%s, n_intervals=%d)"
            % (bins, max_isi, int(counts.sum())),
        )
        return d
