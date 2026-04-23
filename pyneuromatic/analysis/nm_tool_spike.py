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
        x0: X-axis window start for detection. Default ``-inf`` (no lower
            bound). If *x0* > *x1*, a backwards search is performed.
        x1: X-axis window end for detection. Default ``+inf`` (no upper
            bound).
        ignore_nans: If True (default), detect crossings across NaN gaps
            via linear interpolation.
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
        "x0":                 {"type": float, "default": -math.inf},
        "x1":                 {"type": float, "default":  math.inf},
        "ignore_nans":        {"type": bool,  "default": True},
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
        x0: X-axis window start. Default ``-inf`` (no lower bound).
        x1: X-axis window end. Default ``+inf`` (no upper bound).
        ignore_nans: If True (default), detect crossings across NaN gaps
            via linear interpolation.
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
        self.__x0: float = -math.inf
        self.__x1: float = math.inf
        self.__ignore_nans: bool = True
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
    def x0(self) -> float:
        """X-axis window start for detection. Default ``-inf`` (no lower bound)."""
        return self.__x0

    @x0.setter
    def x0(self, value: float) -> None:
        self._x0_set(value)

    def _x0_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0", "float"))
        if math.isnan(value):
            raise ValueError("x0 cannot be NaN")
        self.__x0 = float(value)
        nmh.history("set x0=%g" % self.__x0, quiet=quiet)
        nmch.add_nm_command("%s.x0 = %r" % (self._name, self.__x0))

    @property
    def x1(self) -> float:
        """X-axis window end for detection. Default ``+inf`` (no upper bound)."""
        return self.__x1

    @x1.setter
    def x1(self, value: float) -> None:
        self._x1_set(value)

    def _x1_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1", "float"))
        if math.isnan(value):
            raise ValueError("x1 cannot be NaN")
        self.__x1 = float(value)
        nmh.history("set x1=%g" % self.__x1, quiet=quiet)
        nmch.add_nm_command("%s.x1 = %r" % (self._name, self.__x1))

    @property
    def ignore_nans(self) -> bool:
        """If True (default), detect crossings across NaN gaps."""
        return self.__ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, value: bool) -> None:
        self._ignore_nans_set(value)

    def _ignore_nans_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self.__ignore_nans = value
        nmh.history("set ignore_nans=%s" % value, quiet=quiet)
        nmch.add_nm_command("%s.ignore_nans = %r" % (self._name, self.__ignore_nans))

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
        self._source_data: list[NMData] = []
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
            data, self.__ylevel,
            func_name=self.__func_name,
            x0=self.__x0,
            x1=self.__x1,
            ignore_nans=self.__ignore_nans,
        )
        self._spike_times.append(x_times)
        self._epoch_names.append(data.name)
        self._source_data.append(data)
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
        self._toolfolder = self._make_toolfolder("Spike", overwrite=self.__overwrite)
        f = self._toolfolder
        for name, times in zip(self._epoch_names, self._spike_times):
            d = f.data.new(
                "SP_" + name,
                nparray=times,
                yscale={"label": "Time", "units": self._detected_xunits},
            )
            self._add_note(
                d,
                "NMSpike(source=%s, ylevel=%g, func_name=%r, x0=%s, x1=%s, n=%d)"
                % (name, self.__ylevel, self.__func_name,
                   self.__x0, self.__x1, len(times)),
            )
        counts = np.array([len(t) for t in self._spike_times], dtype=float)
        d_count = f.data.new("SP_count", nparray=counts)
        self._add_note(
            d_count,
            "NMSpike(ylevel=%g, func_name=%r, x0=%s, x1=%s, n_epochs=%d)"
            % (self.__ylevel, self.__func_name,
               self.__x0, self.__x1, len(self._epoch_names)),
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
        if not self._epoch_names:
            raise RuntimeError(
                "NMToolSpike.raster: no spike data — run detection first"
            )
        return list(self._spike_times), list(self._epoch_names)

    def extract_spike_waveforms(
        self,
        pre: float,
        post: float,
        clip_to_next_spike: bool = False,
        edge: str = "skip",
        align: str = "zero",
    ) -> list[NMData]:
        """Extract fixed-width waveform snippets around each detected spike.

        For each spike detected during the most recent :meth:`run_all` call,
        extracts a ``[−pre, +post]`` window of samples from the source epoch
        array and writes the snippet as a new NMData array (``SPK_{ch}{n}``)
        in the Spike subfolder.

        Args:
            pre:               Time before spike in source x-units. Must be > 0.
            post:              Time after spike in source x-units. Must be > 0.
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

        Returns:
            List of newly created NMData arrays, one per non-skipped spike
            across all processed epochs.

        Raises:
            RuntimeError: If called before :meth:`run_all`.
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
        if not self._epoch_names:
            raise RuntimeError(
                "NMToolSpike.extract_spike_waveforms: no spike data — run detection first"
            )

        output: list[NMData] = []
        ch_char = self.channel.name if self.channel is not None else "A"
        toolfolder: NMToolFolder | None = None
        spike_counter: int = 0
        matches: dict[tuple[str, int], NMData] = {}

        for epoch_name, spike_times_arr, source in zip(
            self._epoch_names, self._spike_times, self._source_data
        ):
            ydata = source.nparray
            if ydata is None:
                continue
            n_total = ydata.size

            # Sample interval for converting pre/post to sample counts
            if source.xarray is not None and len(source.xarray) >= 2:
                delta_for_samples = float(np.median(np.diff(source.xarray)))
            else:
                delta_for_samples = float(source.xscale.delta)

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
                            x0_valid = source.xarray[src_start]
                            for k in range(dst_start - 1, -1, -1):
                                x_snippet[k] = x0_valid - (dst_start - k) * delta_for_samples
                        # fill right pad by stepping forwards from last valid x
                        if dst_end < len(x_snippet):
                            x1_valid = source.xarray[src_end - 1]
                            for k in range(dst_end, len(x_snippet)):
                                x_snippet[k] = x1_valid + (k - dst_end + 1) * delta_for_samples
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
                if toolfolder is None:
                    toolfolder = self._make_toolfolder("Spike", overwrite=self.__overwrite)
                array_name = "SPK_%s%d" % (ch_char, spike_counter)
                d = toolfolder.data.new(
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

        if toolfolder is not None:
            toolfolder.build_dataseries("SPK_", matches)
        return output

    # ------------------------------------------------------------------
    # Histogram methods (called after run_all)

    def pst(
        self,
        bins: int = 100,
        x0: float | None = None,
        x1: float | None = None,
        output_mode: str = "count",
    ) -> NMData | None:
        """Compute a peri-stimulus time (PST) histogram from spike times.

        Pools all spike times across epochs from the most recent
        :meth:`run_all` call, computes a histogram via
        :func:`~pyneuromatic.core.nm_math.histogram`, and writes the result
        as ``SP_PST`` in the current Spike subfolder.

        Args:
            bins: Number of histogram bins. Default 100.
            x0: Lower edge of the first bin (inclusive). Default None
                (use minimum spike time).
            x1: Upper edge of the last bin (exclusive). Default None
                (use maximum spike time).
            output_mode: Controls the y-axis values:

                * ``"count"`` (default) — raw spike count per bin.
                * ``"rate"`` — mean firing rate in Hz:
                  ``count / (n_epochs × bin_width)``.
                * ``"probability"`` — fraction of epochs with a spike in
                  each bin: ``count / n_epochs`` (values in ``[0, 1]``).

        Returns:
            The new ``SP_PST`` NMData, or None if no spikes were detected.

        Raises:
            RuntimeError: If called before :meth:`run_all`.
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
        if not self._spike_times:
            raise RuntimeError(
                "NMToolSpike.pst: no spike data — run detection first"
            )
        all_times = np.concatenate(self._spike_times)
        if len(all_times) == 0:
            return None
        xrange: tuple[float, float] | None = None
        if x0 is not None or x1 is not None:
            lo = float(x0) if x0 is not None else float(all_times.min())
            hi = float(x1) if x1 is not None else float(all_times.max())
            xrange = (lo, hi)
        result = nm_math.histogram(all_times, bins=bins, xrange=xrange)
        counts, edges = result["counts"], result["edges"]
        delta = float(edges[1] - edges[0])
        n_epochs = len(self._spike_times)
        if output_mode == "rate":
            ydata = counts.astype(float) / (n_epochs * delta)
            ylabel = "Spike rate (Hz)"
        elif output_mode == "probability":
            ydata = counts.astype(float) / n_epochs
            ylabel = "Spike probability"
        else:
            ydata = counts.astype(float)
            ylabel = "Spike count"
        if self._toolfolder is None:
            self._toolfolder = self._make_toolfolder("Spike", overwrite=self.__overwrite)
        d = self._toolfolder.data.new(
            "SP_PST",
            nparray=ydata,
            xscale={
                "start": float(edges[0]),
                "delta": delta,
                "label": "Time",
                "units": self._detected_xunits,
            },
            yscale={"label": ylabel},
        )
        self._add_note(
            d,
            "NMSpike.pst(bins=%d, x0=%s, x1=%s, output_mode=%r, n_epochs=%d, n_spikes=%d)"
            % (bins, x0, x1, output_mode, n_epochs, int(counts.sum())),
        )
        return d

    def isi(
        self,
        bins: int = 100,
        x0: float | None = None,
        x1: float | None = None,
        min_isi: float | None = None,
        max_isi: float | None = None,
        output_mode: str = "count",
    ) -> NMData | None:
        """Compute an interspike interval (ISI) histogram from spike times.

        Optionally filters spike times to a window [*x0*, *x1*] before
        computing intervals. Computes ``numpy.diff`` of each epoch's
        (filtered) spike times, pools the intervals across all epochs, and
        writes the histogram as ``SP_ISI`` in the current Spike subfolder.
        Epochs with fewer than two spikes (after filtering) contribute no
        intervals.

        Args:
            bins:        Number of histogram bins. Default 100.
            x0:          Lower bound for filtering spike times before
                         computing intervals. Default None (no lower bound).
            x1:          Upper bound for filtering spike times before
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

        Returns:
            The new ``SP_ISI`` NMData, or None if fewer than 2 spikes total
            (after x0/x1 filtering).

        Raises:
            RuntimeError: If called before :meth:`run_all`.
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
        if not self._spike_times:
            raise RuntimeError(
                "NMToolSpike.isi: no spike data — run detection first"
            )
        spike_times = self._spike_times
        if x0 is not None or x1 is not None:
            lo = float(x0) if x0 is not None else -math.inf
            hi = float(x1) if x1 is not None else math.inf
            spike_times = [t[(t >= lo) & (t <= hi)] for t in spike_times]
        intervals = [np.diff(t) for t in spike_times if len(t) >= 2]
        if not intervals:
            return None
        all_isis = np.concatenate(intervals)
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
        if self._toolfolder is None:
            self._toolfolder = self._make_toolfolder("Spike", overwrite=self.__overwrite)
        d = self._toolfolder.data.new(
            "SP_ISI",
            nparray=ydata,
            xscale={
                "start": float(edges[0]),
                "delta": delta,
                "label": "ISI",
                "units": self._detected_xunits,
            },
            yscale={"label": ylabel},
        )
        self._add_note(
            d,
            "NMSpike.isi(bins=%d, x0=%s, x1=%s, min_isi=%s, max_isi=%s, "
            "output_mode=%r, n_intervals=%d)"
            % (bins, x0, x1, min_isi, max_isi, output_mode, n_intervals),
        )
        return d
