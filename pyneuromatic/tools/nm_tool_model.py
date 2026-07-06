# -*- coding: utf-8 -*-
"""
NMToolModel: neural ODE simulation tool.

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
from pyneuromatic.tools.nm_model import NMModelHH
from pyneuromatic.tools.nm_pulse import NMPulseContainer
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu


class NMToolModelConfig(NMToolConfig):
    """Configuration for NMToolModel.

    Parameters:
        n_points:    Number of time samples per simulation.  Default 10000.
        xstart:      Start time (ms).                         Default 0.0.
        xdelta:      Time step (ms; must be > 0).             Default 0.025.
        prefix:      Output array name prefix.                Default ``"VM_"``.
        chan:         Channel string inserted between prefix and epoch index.
                     Default ``""`` (no channel).
        overwrite:   Overwrite existing arrays in NMFolder.data.  Default True.
        save_states: Write gate variable arrays to a toolfolder.  Default False.

    The stimulus waveform is defined by adding :class:`~pyneuromatic.tools.nm_pulse.NMPulse`
    objects to :attr:`NMToolModel.pulses`.  Any pulse shape supported by NMPulse
    (square, ramp, sine, alpha, …) can be used.
    """

    _TOML_TYPE = "model_config"
    _schema = {
        "n_points":    {"type": int,   "default": 10000, "min": 1},
        "xstart":      {"type": float, "default": 0.0},
        "xdelta":      {"type": float, "default": 0.025, "min": 0},
        "prefix":      {"type": str,   "default": "VM_"},
        "chan":         {"type": str,   "default": ""},
        "overwrite":   {"type": bool,  "default": True},
        "save_states": {"type": bool,  "default": False},
    }


class NMToolModel(NMTool):
    """Neural ODE simulation tool.

    Runs a :class:`~pyneuromatic.tools.nm_model.NMModelHH` simulation for
    each epoch and writes the membrane-potential trajectory to
    ``NMFolder.data``.  The prefix acts as a namespace to avoid collisions
    with other data in the folder.

    Output array naming: ``{prefix}{channel}{epoch_idx}``

    The injected current waveform is built by summing all enabled
    :class:`~pyneuromatic.tools.nm_pulse.NMPulse` objects in :attr:`pulses`
    that target the current epoch — identical to how :class:`NMToolPulse`
    generates its output.  Add pulses via ``t.pulses.new(...)``.

    If ``save_states=True``, gate variable arrays (m, h, n for the default
    HH model) are also written to a ``Model_*`` toolfolder.

    Examples::

        t = NMToolModel()
        t.pulses.new({"pulse": "square", "amp": 200.0, "onset": 5.0, "duration": 100.0})
        t.run_all([{}])   # one epoch → VM_0 in NMFolder.data
    """

    def __init__(self, name: str = "model") -> None:
        super().__init__(name=name)
        self._config = NMToolModelConfig()
        self.__n_points:    int   = self._config.n_points
        self.__xstart:      float = self._config.xstart
        self.__xdelta:      float = self._config.xdelta
        self.__prefix:      str   = self._config.prefix
        self.__chan:        str   = self._config.chan
        self._overwrite           = self._config.overwrite
        self.__save_states: bool  = self._config.save_states
        self._results_to_numpy = True

        self.__model: NMModelHH = NMModelHH(nm_path=self._name + ".model")
        self.__pulses: NMPulseContainer = NMPulseContainer(
            nm_path=self._name + ".pulses"
        )

        self._waveforms:     list[np.ndarray]            = []
        self._epoch_names:   list[str]                   = []
        self._state_arrays:  list[dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Properties

    @property
    def n_points(self) -> int:
        """Number of time samples per simulation."""
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
        """Start time (ms)."""
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
        """Time step (ms; must be > 0)."""
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
        """Output array name prefix (e.g. ``"VM_"``)."""
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
        """Channel string inserted between prefix and epoch index."""
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
    def save_states(self) -> bool:
        """If True, write gate variable arrays to a toolfolder."""
        return self.__save_states

    @save_states.setter
    def save_states(self, value: bool) -> None:
        self._save_states_set(value)

    def _save_states_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "save_states", "boolean"))
        self.__save_states = value
        nmh.history("set save_states=%s" % value, quiet=quiet)
        nmch.add_nm_command(
            "%s.save_states = %r" % (self._name, self.__save_states)
        )

    @property
    def model(self) -> NMModelHH:
        """The HH model instance."""
        return self.__model

    @property
    def pulses(self) -> NMPulseContainer:
        """Container of NMPulse objects summed to form the injected current."""
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
        self._state_arrays = []
        return True

    def run(self) -> bool:
        """Simulate one epoch and accumulate the Vm waveform.

        Sums all enabled :class:`~pyneuromatic.tools.nm_pulse.NMPulse` objects
        that target this epoch to form ``i_ext``, then calls
        :meth:`~pyneuromatic.tools.nm_model.NMModelHH.simulate`.

        Returns:
            True.
        """
        epoch_idx = len(self._epoch_names)
        epoch_name = self.data.name if self.data else "sim_%d" % epoch_idx

        i_ext = np.zeros(self.__n_points, dtype=float)
        for p in self.__pulses:
            if p.enabled and p.targets_epoch(epoch_idx):
                i_ext += p.waveform(
                    self.__n_points, self.__xstart, self.__xdelta, epoch_idx
                )

        result = self.__model.simulate(
            self.__n_points, self.__xstart, self.__xdelta, i_ext
        )

        self._waveforms.append(result["V"])
        self._epoch_names.append(epoch_name)
        self._state_arrays.append(
            {k: v for k, v in result.items() if k != "V"}
        )
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
        cond_strs = []
        for cname, cond in self.__model.conductances:
            cond_strs.append(
                "%s(g=%g, e=%g)" % (cname, cond.g_density, cond.e_rev)
            )
        parts = [
            "NMModelHH(v0=%g" % self.__model.v0,
            "cm_density=%g" % self.__model.cm_density,
            "diameter=%g" % self.__model.diameter,
            "temperature=%g" % self.__model.temperature,
            "tau_q10=%g" % self.__model.tau_q10,
            "conductances=[%s]" % ", ".join(cond_strs),
        ]
        pulse_strs = []
        for p in self.__pulses:
            pulse_parts = [p.pulse]
            pulse_parts.append("amp=%g" % p.amp)
            pulse_parts.append("onset=%g" % p.onset)
            if math.isinf(p.duration):
                pulse_parts.append("duration=inf")
            else:
                pulse_parts.append("duration=%g" % p.duration)
            if p.amp_delta != 0.0:
                pulse_parts.append("amp_delta=%g" % p.amp_delta)
            pulse_strs.append("(%s)" % ", ".join(pulse_parts))
        parts.append("pulses=[%s]" % ", ".join(pulse_strs))
        parts.append("n_points=%d" % self.__n_points)
        parts.append("xstart=%g" % self.__xstart)
        parts.append("xdelta=%g" % self.__xdelta)
        if self.__chan:
            parts.append("chan=%r" % self.__chan)
        return ", ".join(parts) + ")"

    def _add_note(self, data: NMData, text: str) -> None:
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    def _write_results_to_numpy(self) -> NMFolder | None:
        """Write ``{prefix}{channel}{epoch_idx}`` NMData arrays to NMFolder.data.

        Also writes a ``{prefix}{channel}epoch_names`` object array of source
        epoch name strings.  If ``save_states=True``, gate variable arrays are
        written to a ``Model_*`` toolfolder.

        Returns:
            The NMFolder written to, or None if no folder is set.
        """
        if not isinstance(self.folder, NMFolder):
            return None

        f = self.folder
        note = self._note_str()
        xscale = {"start": self.__xstart, "delta": self.__xdelta, "units": "ms"}
        yscale_vm = {"label": "V", "units": "mV"}
        stem = self.__prefix + self.__chan

        for idx, (epoch_name, waveform) in enumerate(
            zip(self._epoch_names, self._waveforms)
        ):
            name = stem + str(idx)
            if name in f.data and self._overwrite:
                del f.data[name]
            d = f.data.new(name, nparray=waveform, xscale=xscale, yscale=yscale_vm)
            self._add_note(d, note)

        epoch_names_key = stem.rstrip("_") + "_epoch_names"
        if epoch_names_key in f.data and self._overwrite:
            del f.data[epoch_names_key]
        f.data.new(
            epoch_names_key,
            nparray=np.array(self._epoch_names, dtype=object),
        )

        if self.__save_states and self._state_arrays:
            self._write_states_to_toolfolder(note, xscale)

        return f

    def _write_states_to_toolfolder(self, note: str, xscale: dict) -> None:
        """Write gate variable arrays to a Model toolfolder."""
        tf = self._make_toolfolder("Model", overwrite=self._overwrite)
        chan = self.__chan
        for idx, states in enumerate(self._state_arrays):
            for gate_name, arr in states.items():
                name = gate_name + "_" + chan + str(idx)
                yscale = {"label": gate_name, "units": ""}
                d = tf.data.new(name, nparray=arr, xscale=xscale, yscale=yscale)
                self._add_note(d, note)
