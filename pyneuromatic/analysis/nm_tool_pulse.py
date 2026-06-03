# -*- coding: utf-8 -*-
"""
NMToolPulse: synthetic waveform generator tool.

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
from pyneuromatic.analysis.nm_pulse import NMPulse, NMPulseContainer
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu


class NMToolPulseConfig(NMToolConfig):
    """Configuration for NMToolPulse.

    Parameters:
        n_points: Number of samples in the output waveform. Default 100.
        xstart: X-value of the first sample. Default 0.0.
        xdelta: Sample interval (must be > 0). Default 1.0.
        prefix: Prefix for output array names (non-empty). Default ``"PG_"``.
        channel: Channel string inserted between prefix and epoch index.
            Default ``""`` (no channel). Combined with prefix:
            ``{prefix}{channel}{epoch_idx}``.
        overwrite: Overwrite existing arrays in NMFolder.data. Default True.
    """

    _TOML_TYPE = "pulse_config"
    _schema = {
        "n_points":  {"type": int,   "default": 100, "min": 1},
        "xstart":    {"type": float, "default": 0.0},
        "xdelta":    {"type": float, "default": 1.0, "min": 0},
        "prefix":    {"type": str,   "default": "PG_"},
        "chan":       {"type": str,   "default": ""},
        "overwrite": {"type": bool,  "default": True},
    }


class NMToolPulse(NMTool):
    """Synthetic waveform generator tool.

    Manages a container of :class:`~pyneuromatic.analysis.nm_pulse.NMPulse`
    objects. On each ``run()`` call all pulses that target the current epoch
    are summed to produce a 1-D output array written directly to
    ``NMFolder.data`` (not a subfolder). The prefix acts as a namespace to
    avoid collisions with other data in the folder.

    Output array naming: ``{prefix}{channel}{epoch_idx}``

    Examples::

        t = NMToolPulse()
        t.pulses.new({"pulse": "square", "amp": 1, "onset": 5, "width": 10})
        t.pulses.new({"pulse": "exp", "amp": 0.5, "onset": 5, "tau": 20})
        # drive directly (no NMManager)
        t.run_all([{}])
        # → NMFolder.data contains PG_0 (sum of both pulses)
    """

    def __init__(self, name: str = "pulse") -> None:
        super().__init__(name=name)
        self._config = NMToolPulseConfig()
        self.__n_points: int   = self._config.n_points
        self.__xstart:   float = self._config.xstart
        self.__xdelta:   float = self._config.xdelta
        self.__prefix:   str   = self._config.prefix
        self.__chan:  str   = self._config.chan
        self._overwrite        = self._config.overwrite
        self._results_to_numpy = True

        self.__pulses: NMPulseContainer = NMPulseContainer(
            nm_path=self._name + ".pulses"
        )
        self._waveforms:   list[np.ndarray] = []
        self._epoch_names: list[str]        = []

    # ------------------------------------------------------------------
    # Properties

    @property
    def n_points(self) -> int:
        """Number of samples in each output waveform."""
        return self.__n_points

    @n_points.setter
    def n_points(self, value: int) -> None:
        self._n_points_set(value)

    def _n_points_set(self, value: int, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_points", "int"))
        if value < 1:
            raise ValueError("n_points must be >= 1, got %d" % value)
        self.__n_points = value
        nmh.history("set n_points=%d" % self.__n_points, quiet=quiet)
        nmch.add_nm_command("%s.n_points = %r" % (self._name, self.__n_points))

    @property
    def xstart(self) -> float:
        """X-value of the first sample."""
        return self.__xstart

    @xstart.setter
    def xstart(self, value: float) -> None:
        self._xstart_set(value)

    def _xstart_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "xstart", "float"))
        self.__xstart = float(value)
        nmh.history("set xstart=%g" % self.__xstart, quiet=quiet)
        nmch.add_nm_command("%s.xstart = %r" % (self._name, self.__xstart))

    @property
    def xdelta(self) -> float:
        """Sample interval (must be > 0)."""
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, value: float) -> None:
        self._xdelta_set(value)

    def _xdelta_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "xdelta", "float"))
        if value <= 0:
            raise ValueError("xdelta must be > 0, got %s" % value)
        self.__xdelta = float(value)
        nmh.history("set xdelta=%g" % self.__xdelta, quiet=quiet)
        nmch.add_nm_command("%s.xdelta = %r" % (self._name, self.__xdelta))

    @property
    def prefix(self) -> str:
        """Output array name prefix (e.g. ``"PG_"``, ``"DAC_0_"``, ``"Record"``)."""
        return self.__prefix

    @prefix.setter
    def prefix(self, value: str) -> None:
        self._prefix_set(value)

    def _prefix_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "prefix", "string"))
        if not value:
            raise ValueError("prefix must be a non-empty string")
        self.__prefix = value
        nmh.history("set prefix=%r" % self.__prefix, quiet=quiet)
        nmch.add_nm_command("%s.prefix = %r" % (self._name, self.__prefix))

    @property
    def chan(self) -> str:
        """Channel string inserted between prefix and epoch index (e.g. ``"A"``)."""
        return self.__chan

    @chan.setter
    def chan(self, value: str) -> None:
        self._chan_set(value)

    def _chan_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "chan", "string"))
        self.__chan = value
        nmh.history("set chan=%r" % self.__chan, quiet=quiet)
        nmch.add_nm_command("%s.chan = %r" % (self._name, self.__chan))

    @property
    def pulses(self) -> NMPulseContainer:
        """The container of NMPulse objects."""
        return self.__pulses

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> bool:
        """Validate parameters and reset accumulators."""
        if self.__n_points < 1:
            raise ValueError("n_points must be >= 1")
        if self.__xdelta <= 0:
            raise ValueError("xdelta must be > 0")
        if not self.__prefix:
            raise ValueError("prefix must be non-empty")
        self._waveforms = []
        self._epoch_names = []
        return True

    def run(self) -> bool:
        """Generate and accumulate the waveform for the current epoch.

        All pulses targeting this epoch index are summed. The result is
        appended to ``_waveforms``; the epoch name from ``self.data.name``
        (or a fallback) is appended to ``_epoch_names``.

        Returns:
            True.
        """
        epoch_idx = len(self._epoch_names)
        epoch_name = self.data.name if self.data else "wave_%d" % epoch_idx

        y = np.zeros(self.__n_points, dtype=float)
        for p in self.__pulses:
            if p.targets_epoch(epoch_idx):
                y += p.waveform(
                    self.__n_points, self.__xstart, self.__xdelta, epoch_idx
                )

        self._waveforms.append(y)
        self._epoch_names.append(epoch_name)
        return True

    def run_finish(self) -> bool:
        """Write output arrays to NMFolder.data.

        Returns:
            True.
        """
        if not self._epoch_names:
            return True
        if self._results_to_numpy:
            self._write_results_to_numpy()
        return True

    # ------------------------------------------------------------------
    # Output

    def _note_str(self) -> str:
        pulse_strs = []
        for p in self.__pulses:
            d = p.func.to_dict()
            parts = [d["pulse"]]
            parts.append("amp=%g" % p.amp)
            parts.append("onset=%g" % p.onset)
            if not math.isinf(p.width):
                parts.append("width=%g" % p.width)
            for key in ("tau", "freq", "phase", "f0", "f1"):
                if key in d:
                    parts.append("%s=%g" % (key, d[key]))
            if p.epoch != 0 or p.epoch_delta != 0:
                parts.append("epoch=%d" % p.epoch)
                parts.append("epoch_delta=%d" % p.epoch_delta)
            pulse_strs.append("(%s)" % ", ".join(parts))
        parts = [
            "NMPulse(n_points=%d" % self.__n_points,
            "xstart=%g" % self.__xstart,
            "xdelta=%g" % self.__xdelta,
        ]
        if self.__chan:
            parts.append("chan=%r" % self.__chan)
        parts.append("pulses=[%s])" % ", ".join(pulse_strs))
        return ", ".join(parts)

    def _add_note(self, data: NMData, text: str) -> None:
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    def _write_results_to_numpy(self) -> NMFolder | None:
        """Write ``{prefix}{channel}{epoch_idx}`` NMData arrays to NMFolder.data.

        Also writes a ``{prefix}{channel}epoch_names`` object array of source
        epoch name strings.

        Returns:
            The NMFolder written to, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None

        f = self.folder
        note = self._note_str()
        xscale = {"start": self.__xstart, "delta": self.__xdelta, "units": ""}
        stem = self.__prefix + self.__chan

        for idx, (epoch_name, waveform) in enumerate(
            zip(self._epoch_names, self._waveforms)
        ):
            name = stem + str(idx)
            if name in f.data and self._overwrite:
                del f.data[name]
            d = f.data.new(
                name,
                nparray=waveform,
                xscale=xscale,
            )
            self._add_note(d, note)

        epoch_names_key = stem.rstrip("_") + "_epoch_names"
        if epoch_names_key in f.data and self._overwrite:
            del f.data[epoch_names_key]
        f.data.new(
            epoch_names_key,
            nparray=np.array(self._epoch_names, dtype=object),
        )

        return f
