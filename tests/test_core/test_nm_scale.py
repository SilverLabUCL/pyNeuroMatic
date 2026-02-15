# -*- coding: utf-8 -*-
"""Tests for NMScaleY and NMScaleX classes."""
import copy
import math
import unittest

from pyneuromatic.core.nm_scale import (
    NMScaleX,
    NMScaleY,
    _xscale_from_dict,
    _yscale_from_dict,
)


class TestNMScaleY(unittest.TestCase):
    """Tests for NMScaleY."""

    def test_init_defaults(self):
        s = NMScaleY()
        self.assertIsNone(s._parent)
        self.assertEqual(s.label, "")
        self.assertEqual(s.units, "")

    def test_init_values(self):
        s = NMScaleY(label="voltage", units="mV")
        self.assertEqual(s.label, "voltage")
        self.assertEqual(s.units, "mV")

    def test_label_setter(self):
        s = NMScaleY()
        s.label = "current"
        self.assertEqual(s.label, "current")

    def test_label_setter_strips(self):
        s = NMScaleY()
        s.label = "  current  "
        self.assertEqual(s.label, "current")

    def test_label_setter_type_error(self):
        s = NMScaleY()
        with self.assertRaises(TypeError):
            s.label = 123

    def test_units_setter(self):
        s = NMScaleY()
        s.units = "pA"
        self.assertEqual(s.units, "pA")

    def test_units_setter_type_error(self):
        s = NMScaleY()
        with self.assertRaises(TypeError):
            s.units = 42

    def test_to_dict(self):
        s = NMScaleY(label="voltage", units="mV")
        self.assertEqual(s.to_dict(), {"label": "voltage", "units": "mV"})

    def test_getitem(self):
        s = NMScaleY(label="voltage", units="mV")
        self.assertEqual(s["label"], "voltage")
        self.assertEqual(s["units"], "mV")
        with self.assertRaises(KeyError):
            s["nonexistent"]

    def test_eq_same_type(self):
        s1 = NMScaleY(label="voltage", units="mV")
        s2 = NMScaleY(label="voltage", units="mV")
        self.assertEqual(s1, s2)

    def test_eq_different_values(self):
        s1 = NMScaleY(label="voltage", units="mV")
        s2 = NMScaleY(label="current", units="pA")
        self.assertNotEqual(s1, s2)

    def test_eq_dict(self):
        s = NMScaleY(label="voltage", units="mV")
        self.assertEqual(s, {"label": "voltage", "units": "mV"})
        self.assertNotEqual(s, {"label": "current", "units": "pA"})

    def test_eq_wrong_type(self):
        s = NMScaleY()
        self.assertEqual(s.__eq__(42), NotImplemented)

    def test_deepcopy(self):
        parent = object()
        s = NMScaleY(parent=parent, label="voltage", units="mV")
        s2 = copy.deepcopy(s)
        self.assertEqual(s2.label, "voltage")
        self.assertEqual(s2.units, "mV")
        self.assertIsNone(s2._parent)  # parent reset to None

    def test_path_str_no_parent(self):
        s = NMScaleY()
        self.assertEqual(s.path_str, "yscale")

    def test_path_str_with_parent(self):
        class FakeParent:
            @property
            def path_str(self):
                return "folder0.data0"

        s = NMScaleY(parent=FakeParent())
        self.assertEqual(s.path_str, "folder0.data0.yscale")

    def test_repr(self):
        s = NMScaleY(label="voltage", units="mV")
        r = repr(s)
        self.assertIn("NMScaleY", r)
        self.assertIn("voltage", r)
        self.assertIn("mV", r)

    def test_noop_on_same_value(self):
        s = NMScaleY(label="voltage")
        # Setting the same value should be a no-op (no error)
        s.label = "voltage"
        self.assertEqual(s.label, "voltage")


class TestNMScaleX(unittest.TestCase):
    """Tests for NMScaleX."""

    def test_init_defaults(self):
        s = NMScaleX()
        self.assertEqual(s.label, "")
        self.assertEqual(s.units, "")
        self.assertEqual(s.start, 0)
        self.assertEqual(s.delta, 1)

    def test_init_values(self):
        s = NMScaleX(label="time", units="ms", start=0.5, delta=0.02)
        self.assertEqual(s.label, "time")
        self.assertEqual(s.units, "ms")
        self.assertEqual(s.start, 0.5)
        self.assertEqual(s.delta, 0.02)

    def test_start_setter(self):
        s = NMScaleX()
        s.start = 1.5
        self.assertEqual(s.start, 1.5)

    def test_start_setter_int(self):
        s = NMScaleX()
        s.start = 10
        self.assertEqual(s.start, 10)

    def test_start_setter_type_error_bool(self):
        s = NMScaleX()
        with self.assertRaises(TypeError):
            s.start = True

    def test_start_setter_type_error_str(self):
        s = NMScaleX()
        with self.assertRaises(TypeError):
            s.start = "bad"

    def test_start_setter_inf(self):
        s = NMScaleX()
        with self.assertRaises(ValueError):
            s.start = float("inf")

    def test_start_setter_nan(self):
        s = NMScaleX()
        with self.assertRaises(ValueError):
            s.start = float("nan")

    def test_delta_setter(self):
        s = NMScaleX()
        s.delta = 0.05
        self.assertEqual(s.delta, 0.05)

    def test_delta_setter_zero(self):
        s = NMScaleX()
        with self.assertRaises(ValueError):
            s.delta = 0

    def test_delta_setter_type_error_bool(self):
        s = NMScaleX()
        with self.assertRaises(TypeError):
            s.delta = False

    def test_delta_setter_inf(self):
        s = NMScaleX()
        with self.assertRaises(ValueError):
            s.delta = float("-inf")

    def test_delta_setter_nan(self):
        s = NMScaleX()
        with self.assertRaises(ValueError):
            s.delta = float("nan")

    def test_to_dict(self):
        s = NMScaleX(label="time", units="ms", start=0.5, delta=0.02)
        expected = {"label": "time", "units": "ms", "start": 0.5, "delta": 0.02}
        self.assertEqual(s.to_dict(), expected)

    def test_getitem(self):
        s = NMScaleX(start=0.5, delta=0.02)
        self.assertEqual(s["start"], 0.5)
        self.assertEqual(s["delta"], 0.02)
        self.assertEqual(s["label"], "")
        with self.assertRaises(KeyError):
            s["nonexistent"]

    def test_eq_same_type(self):
        s1 = NMScaleX(label="time", units="ms", start=0.5, delta=0.02)
        s2 = NMScaleX(label="time", units="ms", start=0.5, delta=0.02)
        self.assertEqual(s1, s2)

    def test_eq_different_start(self):
        s1 = NMScaleX(start=0)
        s2 = NMScaleX(start=1)
        self.assertNotEqual(s1, s2)

    def test_eq_dict(self):
        s = NMScaleX(label="time", units="ms", start=0.5, delta=0.02)
        expected = {"label": "time", "units": "ms", "start": 0.5, "delta": 0.02}
        self.assertEqual(s, expected)

    def test_eq_wrong_type(self):
        s = NMScaleX()
        self.assertEqual(s.__eq__(42), NotImplemented)

    def test_deepcopy(self):
        parent = object()
        s = NMScaleX(parent=parent, label="time", units="ms", start=0.5, delta=0.02)
        s2 = copy.deepcopy(s)
        self.assertEqual(s2.label, "time")
        self.assertEqual(s2.units, "ms")
        self.assertEqual(s2.start, 0.5)
        self.assertEqual(s2.delta, 0.02)
        self.assertIsNone(s2._parent)

    def test_path_str(self):
        s = NMScaleX()
        self.assertEqual(s.path_str, "xscale")

    def test_repr(self):
        s = NMScaleX(start=0.5, delta=0.02)
        r = repr(s)
        self.assertIn("NMScaleX", r)
        self.assertIn("0.5", r)
        self.assertIn("0.02", r)

    def test_noop_on_same_value(self):
        s = NMScaleX(start=5.0, delta=0.1)
        s.start = 5.0
        s.delta = 0.1
        self.assertEqual(s.start, 5.0)
        self.assertEqual(s.delta, 0.1)

    def test_set_log_false(self):
        """_set_* with log=False should still set value."""
        s = NMScaleX()
        s._set_start(2.0, log=False)
        self.assertEqual(s.start, 2.0)
        s._set_delta(0.5, log=False)
        self.assertEqual(s.delta, 0.5)
        s._set_label("time", log=False)
        self.assertEqual(s.label, "time")
        s._set_units("ms", log=False)
        self.assertEqual(s.units, "ms")


class TestHelperFunctions(unittest.TestCase):
    """Tests for _xscale_from_dict and _yscale_from_dict."""

    def test_xscale_from_none(self):
        s = _xscale_from_dict(None)
        self.assertIsInstance(s, NMScaleX)
        self.assertEqual(s.start, 0)
        self.assertEqual(s.delta, 1)
        self.assertEqual(s.label, "")
        self.assertEqual(s.units, "")

    def test_xscale_from_dict(self):
        d = {"start": 0.5, "delta": 0.02, "label": "time", "units": "ms"}
        s = _xscale_from_dict(d)
        self.assertEqual(s.start, 0.5)
        self.assertEqual(s.delta, 0.02)
        self.assertEqual(s.label, "time")
        self.assertEqual(s.units, "ms")

    def test_xscale_from_partial_dict(self):
        d = {"start": 1.0}
        s = _xscale_from_dict(d)
        self.assertEqual(s.start, 1.0)
        self.assertEqual(s.delta, 1)  # default

    def test_xscale_parent(self):
        parent = object()
        s = _xscale_from_dict(None, parent=parent)
        self.assertIs(s._parent, parent)

    def test_yscale_from_none(self):
        s = _yscale_from_dict(None)
        self.assertIsInstance(s, NMScaleY)
        self.assertEqual(s.label, "")
        self.assertEqual(s.units, "")

    def test_yscale_from_dict(self):
        d = {"label": "current", "units": "pA"}
        s = _yscale_from_dict(d)
        self.assertEqual(s.label, "current")
        self.assertEqual(s.units, "pA")


if __name__ == "__main__":
    unittest.main()
