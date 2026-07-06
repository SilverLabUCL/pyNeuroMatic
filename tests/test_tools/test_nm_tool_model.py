# -*- coding: utf-8 -*-
"""Tests for NMToolModel and NMToolModelConfig."""
import math
import numpy as np
import pytest

from pyneuromatic.tools.nm_tool_model import NMToolModel, NMToolModelConfig
from pyneuromatic.tools.nm_model import NMModelHH
from pyneuromatic.tools.nm_pulse import NMPulseContainer
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager

NM = NMManager(quiet=True)

# Suprathreshold amplitude for the default HH neuron (SA=1200 µm², Cm=12 pF)
I_AMP_SUPRA = 300.0   # pA


def _make_folder(name="TestFolder"):
    return NMFolder(NM, name=name)


def _run_tool(tool, n_epochs=1, folder=None):
    """Run the tool for n_epochs, all sharing the same folder."""
    if folder is None:
        folder = _make_folder()
    targets = [{"folder": folder} for _ in range(n_epochs)]
    tool.run_all(targets)
    return folder


_TOOL_KEYS = {"n_points", "prefix", "chan", "xdelta", "xstart", "save_states"}


def _supra_tool(**kwargs):
    """Create a NMToolModel configured to fire an AP.

    kwargs are split: tool-level keys (n_points, save_states, …) are set on the
    tool; everything else is passed into the pulse config dict.
    """
    t = NMToolModel()
    t.n_points = 4000   # 100 ms at 0.025 ms/step
    for k, v in kwargs.items():
        if k in _TOOL_KEYS:
            setattr(t, k, v)
    pulse_cfg = {"pulse": "square", "amp": I_AMP_SUPRA, "onset": 5.0,
                 "duration": 20.0, "epoch": "all"}
    pulse_cfg.update({k: v for k, v in kwargs.items() if k not in _TOOL_KEYS})
    t.pulses.new(pulse_cfg)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# NMToolModelConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelConfig:
    def test_defaults(self):
        c = NMToolModelConfig()
        assert c.n_points == 10000
        assert c.xstart == pytest.approx(0.0)
        assert c.xdelta == pytest.approx(0.025)
        assert c.prefix == "VM_"
        assert c.chan == ""
        assert c.overwrite is True
        assert c.save_states is False

    def test_set_valid_n_points(self):
        c = NMToolModelConfig()
        c.n_points = 5000
        assert c.n_points == 5000

    def test_n_points_below_min_raises(self):
        c = NMToolModelConfig()
        with pytest.raises(ValueError):
            c.n_points = 0

    def test_n_points_bool_raises(self):
        c = NMToolModelConfig()
        with pytest.raises(TypeError):
            c.n_points = True

    def test_xdelta_non_negative_required(self):
        c = NMToolModelConfig()
        with pytest.raises(ValueError):
            c.xdelta = -0.1

    def test_unknown_key_raises(self):
        c = NMToolModelConfig()
        with pytest.raises(AttributeError):
            c.not_a_key = 1

    def test_to_dict_round_trip(self):
        c = NMToolModelConfig()
        c.n_points = 2000
        c2 = NMToolModelConfig.from_dict(c.to_dict())
        assert c == c2


# ──────────────────────────────────────────────────────────────────────────────
# NMToolModel construction and properties
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelConstruct:
    def test_defaults(self):
        t = NMToolModel()
        assert t.prefix == "VM_"
        assert t.chan == ""
        assert t.xdelta == pytest.approx(0.025)
        assert t.save_states is False
        assert len(t.pulses) == 0

    def test_model_property(self):
        t = NMToolModel()
        assert isinstance(t.model, NMModelHH)

    def test_pulses_property(self):
        t = NMToolModel()
        assert isinstance(t.pulses, NMPulseContainer)
        assert len(t.pulses) == 0

    def test_add_pulse(self):
        t = NMToolModel()
        t.pulses.new({"pulse": "square", "amp": 10.0, "onset": 5.0})
        assert len(t.pulses) == 1

    def test_prefix_setter(self):
        t = NMToolModel()
        t.prefix = "HH_"
        assert t.prefix == "HH_"

    def test_prefix_empty_raises(self):
        t = NMToolModel()
        with pytest.raises(ValueError):
            t.prefix = ""

    def test_n_points_setter(self):
        t = NMToolModel()
        t.n_points = 500
        assert t.n_points == 500

    def test_n_points_zero_raises(self):
        t = NMToolModel()
        with pytest.raises(ValueError):
            t.n_points = 0

    def test_xdelta_zero_raises(self):
        t = NMToolModel()
        with pytest.raises(ValueError):
            t.xdelta = 0.0

    def test_save_states_bool_required(self):
        t = NMToolModel()
        with pytest.raises(TypeError):
            t.save_states = 1


# ──────────────────────────────────────────────────────────────────────────────
# Basic output — single epoch
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelSingleEpoch:
    def setup_method(self):
        self.t = _supra_tool()
        self.folder = _run_tool(self.t, n_epochs=1)

    def test_vm_array_in_folder(self):
        assert "VM_0" in self.folder.data

    def test_vm_array_length(self):
        d = self.folder.data["VM_0"]
        assert len(d.nparray) == self.t.n_points

    def test_vm_xscale_start(self):
        d = self.folder.data["VM_0"]
        assert d.xscale.start == pytest.approx(0.0)

    def test_vm_xscale_delta(self):
        d = self.folder.data["VM_0"]
        assert d.xscale.delta == pytest.approx(0.025)

    def test_vm_xscale_units(self):
        d = self.folder.data["VM_0"]
        assert d.xscale.units == "ms"

    def test_vm_yscale_label(self):
        d = self.folder.data["VM_0"]
        assert d.yscale.label == "V"

    def test_vm_yscale_units(self):
        d = self.folder.data["VM_0"]
        assert d.yscale.units == "mV"

    def test_no_ap_in_output(self):
        V = self.folder.data["VM_0"].nparray
        assert V.max() > 0.0, "No AP detected in VM_0 output"

    def test_epoch_names_written(self):
        assert "VM_epoch_names" in self.folder.data

    def test_epoch_names_length(self):
        names = self.folder.data["VM_epoch_names"].nparray
        assert len(names) == 1

    def test_note_on_data(self):
        d = self.folder.data["VM_0"]
        note_text = " ".join(str(n) for n in d.notes)
        assert "NMModelHH" in note_text


# ──────────────────────────────────────────────────────────────────────────────
# Multi-epoch sweep (amp_delta)
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelMultiEpoch:
    def setup_method(self):
        self.t = _supra_tool(amp_delta=50.0)
        self.folder = _run_tool(self.t, n_epochs=3)

    def test_three_vm_arrays(self):
        for i in range(3):
            assert "VM_%d" % i in self.folder.data

    def test_epoch_names_length(self):
        names = self.folder.data["VM_epoch_names"].nparray
        assert len(names) == 3

    def test_different_amps_produce_different_peaks(self):
        """Higher amp_delta epochs should produce APs at least as large as lower epochs."""
        v0 = self.folder.data["VM_0"].nparray
        v1 = self.folder.data["VM_1"].nparray
        v2 = self.folder.data["VM_2"].nparray
        assert v0.max() > 0.0
        assert v1.max() > 0.0
        assert v2.max() > 0.0
        assert v2.max() >= v0.max()


# ──────────────────────────────────────────────────────────────────────────────
# Channel string in output names
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelChanNaming:
    def test_chan_in_name(self):
        t = _supra_tool()
        t.chan = "A"
        folder = _run_tool(t, n_epochs=2)
        assert "VM_A0" in folder.data
        assert "VM_A1" in folder.data

    def test_chan_in_epoch_names_key(self):
        t = _supra_tool()
        t.chan = "A"
        folder = _run_tool(t, n_epochs=1)
        assert "VM_A_epoch_names" in folder.data

    def test_custom_prefix(self):
        t = _supra_tool()
        t.prefix = "HH_"
        folder = _run_tool(t, n_epochs=1)
        assert "HH_0" in folder.data


# ──────────────────────────────────────────────────────────────────────────────
# save_states — gate variables in toolfolder
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelSaveStates:
    def setup_method(self):
        self.t = _supra_tool(save_states=True)
        self.folder = _run_tool(self.t, n_epochs=1)
        tf_names = list(self.folder.toolfolders)
        self.tf_name = tf_names[0] if tf_names else None

    def test_toolfolder_created(self):
        assert self.tf_name is not None, "No toolfolder created"

    def test_toolfolder_starts_with_model(self):
        assert self.tf_name.startswith("Model")

    def test_gate_arrays_present(self):
        tf = self.folder.toolfolders[self.tf_name]
        assert "m_0" in tf.data
        assert "h_0" in tf.data
        assert "n_0" in tf.data

    def test_gate_array_length(self):
        tf = self.folder.toolfolders[self.tf_name]
        m = tf.data["m_0"].nparray
        assert len(m) == self.t.n_points

    def test_gate_array_xscale(self):
        tf = self.folder.toolfolders[self.tf_name]
        m = tf.data["m_0"]
        assert m.xscale.delta == pytest.approx(0.025)
        assert m.xscale.units == "ms"

    def test_gate_array_yscale(self):
        tf = self.folder.toolfolders[self.tf_name]
        m = tf.data["m_0"]
        assert m.yscale.label == "m"

    def test_m_gate_opens_during_ap(self):
        tf = self.folder.toolfolders[self.tf_name]
        m_arr = tf.data["m_0"].nparray
        assert m_arr.max() > 0.5

    def test_no_toolfolder_when_save_states_false(self):
        t = _supra_tool(save_states=False)
        folder = _run_tool(t, n_epochs=1)
        tf_names = list(folder.toolfolders)
        assert len(tf_names) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Overwrite behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelOverwrite:
    def test_overwrite_replaces_existing(self):
        t = _supra_tool()
        folder = _run_tool(t, n_epochs=1)
        v_before = folder.data["VM_0"].nparray.copy()
        t2 = NMToolModel()
        t2.n_points = t.n_points
        t2.pulses.new({"pulse": "square", "amp": I_AMP_SUPRA * 2,
                       "onset": 5.0, "duration": 20.0, "epoch": "all"})
        _run_tool(t2, n_epochs=1, folder=folder)
        v_after = folder.data["VM_0"].nparray
        assert "VM_0" in folder.data
        assert not np.array_equal(v_before, v_after), (
            "VM_0 was not overwritten: data unchanged after second run"
        )

    def test_run_init_resets_accumulators(self):
        t = _supra_tool()
        _run_tool(t, n_epochs=2)
        folder2 = _run_tool(t, n_epochs=1)
        assert "VM_0" in folder2.data
        assert "VM_1" not in folder2.data, (
            "run_init did not reset accumulators: VM_1 present after 1-epoch run"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Note string
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelNoteStr:
    def _note(self, tool, n_epochs=1):
        folder = _run_tool(tool, n_epochs=n_epochs)
        return " ".join(str(n) for n in folder.data["VM_0"].notes)

    def test_note_contains_model_name(self):
        assert "NMModelHH" in self._note(_supra_tool())

    def test_note_contains_amp(self):
        assert "amp" in self._note(_supra_tool())

    def test_note_contains_conductances(self):
        assert "conductances" in self._note(_supra_tool())

    def test_note_inf_duration(self):
        t = NMToolModel()
        t.n_points = 4000
        t.pulses.new({"pulse": "square", "amp": I_AMP_SUPRA, "onset": 5.0,
                      "duration": math.inf, "epoch": "all"})
        assert "inf" in self._note(t)


# ──────────────────────────────────────────────────────────────────────────────
# run_finish with no epochs
# ──────────────────────────────────────────────────────────────────────────────

class TestNMToolModelEmpty:
    def test_run_all_no_targets(self):
        t = NMToolModel()
        result = t.run_all([])
        assert result is True

    def test_no_output_when_no_epochs(self):
        t = NMToolModel()
        folder = _make_folder()
        t.run_all([{"folder": folder}])  # one epoch, no pulses → resting Vm
        count_before = len(folder.data)
        t.run_all([])
        assert len(folder.data) == count_before
