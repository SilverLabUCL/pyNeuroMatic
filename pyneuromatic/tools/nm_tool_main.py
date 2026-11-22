# -*- coding: utf-8 -*-
"""
NMToolMain - Main tool that is always loaded.

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

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.tools.nm_main_op import (
    NMMainOp, NMMainOpAccumulate, NMMainOpAverage, op_from_name)
from pyneuromatic.tools.nm_tool import NMTool
from pyneuromatic.tools.nm_tool_config import NMToolConfig
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_command_history import records_command
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu


class NMToolMainConfig(NMToolConfig):
    """Settings › Main: the Main tool's options (#553).

    Parameters:
        ignore_nans: Functions leave NaN values out (otherwise a NaN in any
            array gives NaN there).
        overwrite: Functions run on the same selection share its first Main
            subfolder (``..._0``), each replacing only the outputs it makes;
            off: a new subfolder each run.
        xbgn, xend: The x range functions use (default: all of each array).
    """

    _TOML_TYPE = "main_config"
    _schema = {
        "ignore_nans": {"type": bool, "default": True},
        "overwrite": {"type": bool, "default": False},
        "xbgn": {"type": float, "default": -math.inf},
        "xend": {"type": float, "default": math.inf},
    }


class NMToolMain(NMTool):
    """Main NM Tool — always loaded by default.

    Provides core array operations (Average, Scale, …) and display ops
    (Plot) via a pluggable ``op`` property.  Setting ``op`` to a string
    name (e.g. ``"average"``, ``"plot"``) looks up and instantiates the
    corresponding :class:`NMMainOp` subclass from the registry.  Setting
    it to an :class:`NMMainOp` instance allows full parameter control::

        tool = NMToolMain()
        tool.op = NMMainOpArithmetic(factor=2.0)
        nm.run_tool(tool)

    Delegates to the current ``op`` via the standard ``run_init / run /
    run_finish`` lifecycle inherited from :class:`NMTool`.

    A function (Average, Sum, SumSqr, Min, Max, Concatenate: an
    :class:`NMMainOpAccumulate`) saves its outputs in a Main subfolder of the
    folder, named as Stats' and Spike's, e.g. ``Main_Record_A_All_0`` (#553):
    a note says how they were made, and each kind of output is a data series
    there, a channel each (e.g. ``MN_Avg_Record``: ``MN_Avg_RecordA``,
    ``MN_Avg_RecordB``). Without :attr:`overwrite`, each run makes a new
    subfolder; with it, runs on the same selection share ``..._0``, each
    replacing only the outputs it makes (an Average doesn't remove a Max).
    Other operations change the arrays themselves, or write to the folder.
    """

    OUTPUT_PREFIX = "MN_"  # before its outputs' names, as Stats' ST_

    TOOLFOLDER_PREFIX = "Main_"  # its outputs' subfolders

    def __init__(self) -> None:
        super().__init__(name="main")  # its commands: nm.main...
        self._config = NMToolMainConfig()
        self._ignore_nans = self._config.ignore_nans
        self._overwrite = self._config.overwrite
        self._xbgn = self._config.xbgn
        self._xend = self._config.xend
        self._op: NMMainOp = NMMainOpAverage()
        self._toolfolder: NMToolFolder | None = None
        self._outputs: list[str] = []

    @property
    def toolfolder(self) -> NMToolFolder | None:
        """The Main subfolder the most recent function wrote, if any."""
        return self._toolfolder

    @property
    def outputs(self) -> list[str]:
        """The names of the arrays the most recent function made (in
        :attr:`toolfolder`; with Overwrite, beside earlier runs' outputs)."""
        return list(self._outputs)

    # ------------------------------------------------------------------
    # op property

    @property
    def op(self) -> NMMainOp:
        """Current operation (NMMainOp instance)."""
        return self._op

    @op.setter
    @records_command
    def op(self, value: NMMainOp | str) -> None:
        """Set the current operation.

        Args:
            value: An :class:`NMMainOp` instance, or a string name looked up
                in the op registry (e.g. ``"average"``, ``"scale"``).

        Raises:
            TypeError: If value is neither an NMMainOp nor a string.
            ValueError: If value is a string not in the registry.
        """
        if isinstance(value, str):
            self._op = op_from_name(value)
        elif isinstance(value, NMMainOp):
            self._op = value
        else:
            raise TypeError(
                nmu.type_error_str(value, "op", "NMMainOp or string")
            )
        from pyneuromatic.core.nm_command_history import add_nm_command
        params = self._op._op_params_str()
        if params is None:
            params = ""
        add_nm_command("%s.op = %s(%s)" % (self._name, type(self._op).__name__, params))

    # ------------------------------------------------------------------
    # Lifecycle hooks

    def run_init(self) -> bool:
        """Reset per-run context and initialise the current op."""
        self._run_folder: NMFolder | None = None
        self._run_prefix: str | None = None
        self._run_channels: list[str] = []  # in the order run
        self._run_count = 0
        self._sources: dict[str, list[NMData]] = {}  # channel: arrays used
        self._last: tuple[str, bool] | None = None  # (channel, added): discard_last
        self._toolfolder = None
        self._outputs = []
        self._op.run_init()
        return True

    def run(self) -> bool:
        """Process the currently selected data item via the current op."""
        if (self.dataseries is not None
                and self.channel is not None
                and self.epoch is not None):
            d = self.dataseries.get_data(self.channel.name, self.epoch.name)
        else:
            d = self.data
        if d is None:
            return True  # skip silently, don't stop the loop

        channel_name = self.channel.name if self.channel is not None else None

        if self._run_folder is None and self.folder is not None:
            self._run_folder = self.folder
        if self._run_prefix is None and self.dataseries is not None:
            self._run_prefix = self.dataseries.name
        if channel_name is not None and channel_name not in self._run_channels:
            self._run_channels.append(channel_name)
        self._run_count += 1

        if isinstance(self._op, NMMainOpAccumulate):  # a function
            key = channel_name
            if key is None:
                parsed = nmu.parse_data_name(d.name)
                key = parsed[1] if parsed is not None else "A"
            self._sources.setdefault(key, []).append(d)
            d = self._in_x_range(d)  # only xbgn to xend (#553)
            self._last = (key, d is not None and d.nparray is not None)
            if d is None:
                return True
        self._op.run(d, channel_name)
        return True

    @property
    def can_omit(self) -> bool:
        """A function can leave out the epoch just added (Omit, at the run's
        prompt); an operation can't undo what it changed."""
        return isinstance(self._op, NMMainOpAccumulate)

    def discard_last(self) -> None:
        """Leave the array :meth:`run` just added out of the function's
        outputs and their sources (an epoch omitted from the run, #472)."""
        if self._last is None:
            return
        key, added = self._last
        self._last = None
        sources = self._sources.get(key)
        if sources:
            sources.pop()
            if not sources:
                del self._sources[key]
        self._run_count -= 1
        op = self._op
        if added and op._accum.get(key):
            op._accum[key].pop()
            op._data_names[key].pop()
            if not op._accum[key]:  # the channel's only array: no output for it
                for store in (op._accum, op._data_names, op._xscales, op._yscales):
                    store.pop(key, None)

    def _in_x_range(self, data: NMData) -> NMData | None:
        """*data* from xbgn to xend (a copy, its X start where it begins), or
        *data* itself with no range; None if no points are in it."""
        y = data.nparray
        if y is None or (self._xbgn == -math.inf and self._xend == math.inf):
            return data
        xarray = data.xarray
        if xarray is not None and len(xarray) == len(y) and not data.xarray_problem():
            x = np.asarray(xarray, dtype=float)
        else:
            x = data.xscale.start + data.xscale.delta * np.arange(len(y))
        inside = np.nonzero((x >= self._xbgn) & (x <= self._xend))[0]
        if inside.size == 0:
            return None
        first, last = int(inside[0]), int(inside[-1])
        xscale = data.xscale.to_dict()
        xscale["start"] = float(x[first])
        return NMData(name=data.name, nparray=np.array(y[first:last + 1], copy=True),
                      xscale=xscale, yscale=data.yscale.to_dict())

    def run_finish(self) -> bool:
        """Finalise the current op with the folder and prefix captured during
        run(): a function's outputs in a Main subfolder (#553)."""
        if isinstance(self._op, NMMainOpAccumulate) and isinstance(self._run_folder, NMFolder):
            self._write_outputs()
        else:
            self._op.run_finish(self._run_folder, self._run_prefix)
        meta = self.run_meta
        nmh.history(
            "%s: folders=%s, dataseries=%s, channels=%s, epochs=%s"
            % (
                self._op.__class__.__name__,
                meta.get("folders", []),
                meta.get("dataseries", []),
                meta.get("channels", []),
                meta.get("epochs", []),
            ),
            path="NMToolMain.run_finish",
        )
        return True

    # ------------------------------------------------------------------
    # A function's outputs (#553)

    def _subfolder_base(self) -> str:
        """e.g. ``Main_Record_A_All``, as Stats' and Spike's: the data series,
        its channels (``AB`` for several) and the epochs' code."""
        parts = ["Main"]
        if self.dataseries is not None:
            parts.append(self.dataseries.name)
        if self._run_channels:
            parts.append("".join(self._run_channels))
        code = self._selection_code()
        if code:
            parts.append(code)
        return "_".join(parts)

    def _write_outputs(self) -> None:
        """The function's outputs in a Main subfolder, noted, and made a
        data series per kind of output."""
        base = self._subfolder_base()
        toolfolders = self._run_folder.toolfolders
        if self._overwrite:  # ..._0, shared: only this run's outputs replaced
            folder = toolfolders.get("%s_0" % base) or toolfolders.new(name="%s_0" % base)
        else:
            folder = toolfolders.get_or_create(base, overwrite=False)  # a new one
        self._toolfolder = folder
        made = getattr(self, "_made_toolfolders", None)
        if made is not None and folder not in made:
            made.append(folder)
        self._op.name_prefix = self.OUTPUT_PREFIX  # MN_Avg_RecordA...
        try:
            self._op.run_finish(folder, self._run_prefix)
        finally:
            self._op.name_prefix = ""
        folder.notes.add(self._run_note(), readonly=True)  # how they were made
        outputs = [n for n in folder.data.keys()
                   if n in self._op.results.values() or self._is_companion(n)]
        self._outputs = outputs
        self._make_output_series(folder, outputs)
        # Which arrays each output was made from (e.g. to plot them behind it)
        for name in outputs:
            channel = self._output_channel(name)
            if channel in self._sources:
                folder.sources[name] = list(self._sources[channel])

    def _is_companion(self, name: str) -> bool:
        """An output written beside the function's own, this run: an
        Average's Stdv, Var or SEM, if it computed them (e.g.
        MN_SEM_RecordA); not an earlier run's (with Overwrite)."""
        op = self._op
        if not isinstance(op, NMMainOpAverage):
            return False
        companions = [prefix for prefix, on in (("Stdv_", op.compute_stdv),
                                                ("Var_", op.compute_var),
                                                ("SEM_", op.compute_sem)) if on]
        return any(name == main.replace(op.out_prefix, companion, 1)
                   for companion in companions for main in op.results.values())

    def _output_channel(self, name: str) -> str | None:
        channels = list(self._op.results) or self._run_channels
        return next((c for c in channels if name.endswith(c) and len(name) > len(c)), None)

    def _make_output_series(self, folder: NMToolFolder, outputs: list[str]) -> None:
        """Each kind of output a data series, a channel each: e.g.
        Avg_RecordA and Avg_RecordB make Avg_Record (as SPK_, #530)."""
        series: dict[str, dict] = {}
        for name in outputs:
            channel = self._output_channel(name)
            if channel is None or len(channel) != 1 or not channel.isupper():
                continue  # not one output per channel
            series.setdefault(name[: -len(channel)], {})[(channel, 0)] = folder.data[name]
        for prefix, matches in series.items():
            folder.build_dataseries(prefix, matches)

    def _run_note(self) -> str:
        """How the outputs were made, e.g. "NMMain: Average(ignore_nans=True,
        compute_stdv=True, ...); Record, channels A, B, epochs all (n=6)"."""
        params = self._op._op_params_str() or ""
        note = "NMMain: %s(%s); %s" % (
            type(self._op).__name__.replace("NMMainOp", ""), params,
            self._run_scope(self._run_count))
        if self._xbgn != -math.inf or self._xend != math.inf:
            note += "; x %g to %g" % (self._xbgn, self._xend)
        return note
