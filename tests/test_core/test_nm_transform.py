# -*- coding: utf-8 -*-
"""Tests for NMTransform classes."""
import copy
import unittest

import numpy as np

from pyneuromatic.core.nm_scale import NMScaleX
from pyneuromatic.core.nm_transform import (
    NMTransform,
    NMTransformInvert,
    NMTransformDifferentiate,
    NMTransformDoubleDifferentiate,
    NMTransformIntegrate,
    NMTransformLog,
    NMTransformLn,
    NMTransformSmooth,
    _transform_from_dict,
    _transforms_from_list,
    apply_transforms,
)


class TestNMTransform(unittest.TestCase):
    """Tests for the NMTransform base class (tested via concrete subclasses)."""

    def test_init_defaults(self):
        t = NMTransformInvert()
        self.assertIsNone(t._parent)

    def test_init_with_parent(self):
        parent = object()
        t = NMTransformInvert(parent=parent)
        self.assertIs(t._parent, parent)

    def test_base_apply_raises(self):
        t = NMTransform()
        with self.assertRaises(NotImplementedError):
            t.apply(np.array([1.0, 2.0]))

    def test_repr(self):
        t = NMTransformInvert()
        self.assertEqual(repr(t), "NMTransformInvert()")

    def test_type_str(self):
        t = NMTransformInvert()
        self.assertEqual(t.type_str, "NMTransformInvert")

    def test_to_dict(self):
        t = NMTransformInvert()
        self.assertEqual(t.to_dict(), {"type": "NMTransformInvert"})

    def test_getitem(self):
        t = NMTransformInvert()
        self.assertEqual(t["type"], "NMTransformInvert")
        with self.assertRaises(KeyError):
            t["nonexistent"]

    def test_eq_same_type(self):
        t1 = NMTransformInvert()
        t2 = NMTransformInvert()
        self.assertEqual(t1, t2)

    def test_eq_different_type(self):
        t1 = NMTransformInvert()
        t2 = NMTransformLog()
        self.assertNotEqual(t1, t2)

    def test_eq_dict(self):
        t = NMTransformInvert()
        self.assertEqual(t, {"type": "NMTransformInvert"})
        self.assertNotEqual(t, {"type": "NMTransformLog"})

    def test_eq_wrong_type(self):
        t = NMTransformInvert()
        self.assertEqual(t.__eq__(42), NotImplemented)

    def test_deepcopy(self):
        parent = object()
        t = NMTransformInvert(parent=parent)
        t2 = copy.deepcopy(t)
        self.assertIsInstance(t2, NMTransformInvert)
        self.assertIsNone(t2._parent)

    def test_path_str_no_parent(self):
        t = NMTransformInvert()
        self.assertEqual(t.path_str, "transform")

    def test_path_str_with_parent(self):
        class FakeParent:
            @property
            def path_str(self):
                return "folder0.data0"

        t = NMTransformInvert(parent=FakeParent())
        self.assertEqual(t.path_str, "folder0.data0.transform")


class TestNMTransformInvert(unittest.TestCase):
    """Tests for NMTransformInvert."""

    def test_apply_basic(self):
        t = NMTransformInvert()
        y = np.array([1.0, -2.0, 3.0, 0.0])
        result = t.apply(y)
        np.testing.assert_array_equal(result, np.array([-1.0, 2.0, -3.0, -0.0]))

    def test_apply_does_not_modify_input(self):
        t = NMTransformInvert()
        y = np.array([1.0, 2.0, 3.0])
        y_orig = y.copy()
        t.apply(y)
        np.testing.assert_array_equal(y, y_orig)

    def test_apply_with_nans(self):
        t = NMTransformInvert()
        y = np.array([1.0, np.nan, 3.0])
        result = t.apply(y)
        self.assertEqual(result[0], -1.0)
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], -3.0)

    def test_apply_type_error(self):
        t = NMTransformInvert()
        with self.assertRaises(TypeError):
            t.apply([1.0, 2.0])

    def test_apply_ignores_xscale(self):
        t = NMTransformInvert()
        y = np.array([1.0, 2.0])
        xs = NMScaleX(start=0, delta=0.1)
        result = t.apply(y, xscale=xs)
        np.testing.assert_array_equal(result, np.array([-1.0, -2.0]))


class TestNMTransformDifferentiate(unittest.TestCase):
    """Tests for NMTransformDifferentiate."""

    def test_apply_preserves_length(self):
        t = NMTransformDifferentiate()
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        result = t.apply(y)
        self.assertEqual(result.shape, y.shape)

    def test_apply_linear(self):
        t = NMTransformDifferentiate()
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = t.apply(y)
        np.testing.assert_allclose(result, 1.0)

    def test_apply_with_xscale(self):
        t = NMTransformDifferentiate()
        y = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        xs = NMScaleX(start=0, delta=0.5)
        result = t.apply(y, xscale=xs)
        # dy/dx = 0.5/0.5 = 1.0
        np.testing.assert_allclose(result, 1.0)

    def test_apply_type_error(self):
        t = NMTransformDifferentiate()
        with self.assertRaises(TypeError):
            t.apply("not an array")


class TestNMTransformDoubleDifferentiate(unittest.TestCase):
    """Tests for NMTransformDoubleDifferentiate."""

    def test_apply_quadratic(self):
        t = NMTransformDoubleDifferentiate()
        # unit spacing: x = 0, 1, 2, ... so d2(x^2)/dx2 = 2
        x = np.arange(0, 20, dtype=float)
        y = x ** 2
        result = t.apply(y)
        # interior points should be exactly 2.0
        np.testing.assert_allclose(result[2:-2], 2.0, atol=0.01)

    def test_apply_with_xscale(self):
        t = NMTransformDoubleDifferentiate()
        x = np.arange(0, 5, 0.1)
        y = x ** 2
        xs = NMScaleX(start=0, delta=0.1)
        result = t.apply(y, xscale=xs)
        np.testing.assert_allclose(result[5:-5], 2.0, atol=0.1)

    def test_apply_preserves_length(self):
        t = NMTransformDoubleDifferentiate()
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        result = t.apply(y)
        self.assertEqual(result.shape, y.shape)


class TestNMTransformIntegrate(unittest.TestCase):
    """Tests for NMTransformIntegrate."""

    def test_apply_basic(self):
        t = NMTransformIntegrate()
        y = np.array([1.0, 1.0, 1.0, 1.0])
        result = t.apply(y)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_apply_with_xscale(self):
        t = NMTransformIntegrate()
        y = np.array([1.0, 1.0, 1.0, 1.0])
        xs = NMScaleX(start=0, delta=0.5)
        result = t.apply(y, xscale=xs)
        np.testing.assert_array_equal(result, np.array([0.5, 1.0, 1.5, 2.0]))

    def test_apply_preserves_length(self):
        t = NMTransformIntegrate()
        y = np.array([1.0, 2.0, 3.0])
        result = t.apply(y)
        self.assertEqual(result.shape, y.shape)

    def test_apply_type_error(self):
        t = NMTransformIntegrate()
        with self.assertRaises(TypeError):
            t.apply(42)


class TestNMTransformLog(unittest.TestCase):
    """Tests for NMTransformLog (base-10)."""

    def test_apply_basic(self):
        t = NMTransformLog()
        y = np.array([1.0, 10.0, 100.0, 1000.0])
        result = t.apply(y)
        np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0, 3.0]))

    def test_apply_zero_produces_neginf(self):
        t = NMTransformLog()
        y = np.array([0.0, 1.0])
        result = t.apply(y)
        self.assertTrue(np.isneginf(result[0]))

    def test_apply_negative_produces_nan(self):
        t = NMTransformLog()
        y = np.array([-1.0, 1.0])
        result = t.apply(y)
        self.assertTrue(np.isnan(result[0]))

    def test_apply_type_error(self):
        t = NMTransformLog()
        with self.assertRaises(TypeError):
            t.apply(42)


class TestNMTransformLn(unittest.TestCase):
    """Tests for NMTransformLn (natural log)."""

    def test_apply_basic(self):
        t = NMTransformLn()
        y = np.array([1.0, np.e, np.e ** 2])
        result = t.apply(y)
        np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0]))

    def test_apply_zero_produces_neginf(self):
        t = NMTransformLn()
        y = np.array([0.0])
        result = t.apply(y)
        self.assertTrue(np.isneginf(result[0]))

    def test_apply_negative_produces_nan(self):
        t = NMTransformLn()
        y = np.array([-1.0])
        result = t.apply(y)
        self.assertTrue(np.isnan(result[0]))


class TestTransformFromDict(unittest.TestCase):
    """Tests for _transform_from_dict."""

    def test_invert(self):
        t = _transform_from_dict({"type": "NMTransformInvert"})
        self.assertIsInstance(t, NMTransformInvert)

    def test_all_types(self):
        types = [
            ("NMTransformInvert", NMTransformInvert),
            ("NMTransformDifferentiate", NMTransformDifferentiate),
            ("NMTransformDoubleDifferentiate", NMTransformDoubleDifferentiate),
            ("NMTransformIntegrate", NMTransformIntegrate),
            ("NMTransformLog", NMTransformLog),
            ("NMTransformLn", NMTransformLn),
        ]
        for type_str, cls in types:
            t = _transform_from_dict({"type": type_str})
            self.assertIsInstance(t, cls)

    def test_missing_type_key(self):
        with self.assertRaises(KeyError):
            _transform_from_dict({"name": "invert"})

    def test_unknown_type(self):
        with self.assertRaises(ValueError):
            _transform_from_dict({"type": "UnknownTransform"})

    def test_type_error(self):
        with self.assertRaises(TypeError):
            _transform_from_dict("not a dict")

    def test_parent_set(self):
        parent = object()
        t = _transform_from_dict({"type": "NMTransformInvert"}, parent=parent)
        self.assertIs(t._parent, parent)

    def test_round_trip(self):
        original = NMTransformDifferentiate()
        d = original.to_dict()
        restored = _transform_from_dict(d)
        self.assertEqual(original, restored)


class TestTransformsFromList(unittest.TestCase):
    """Tests for _transforms_from_list."""

    def test_none_returns_none(self):
        self.assertIsNone(_transforms_from_list(None))

    def test_empty_list(self):
        result = _transforms_from_list([])
        self.assertEqual(result, [])

    def test_list_of_dicts(self):
        dicts = [
            {"type": "NMTransformInvert"},
            {"type": "NMTransformLog"},
        ]
        result = _transforms_from_list(dicts)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], NMTransformInvert)
        self.assertIsInstance(result[1], NMTransformLog)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            _transforms_from_list("not a list")


class TestApplyTransforms(unittest.TestCase):
    """Tests for apply_transforms."""

    def test_none_returns_original(self):
        y = np.array([1.0, 2.0, 3.0])
        result = apply_transforms(y, None)
        np.testing.assert_array_equal(result, y)

    def test_empty_list_returns_original(self):
        y = np.array([1.0, 2.0, 3.0])
        result = apply_transforms(y, [])
        np.testing.assert_array_equal(result, y)

    def test_single_transform(self):
        y = np.array([1.0, -2.0, 3.0])
        result = apply_transforms(y, [NMTransformInvert()])
        np.testing.assert_array_equal(result, np.array([-1.0, 2.0, -3.0]))

    def test_pipeline_double_invert(self):
        y = np.array([10.0, 100.0])
        transforms = [NMTransformInvert(), NMTransformInvert()]
        result = apply_transforms(y, transforms)
        np.testing.assert_array_equal(result, y)

    def test_does_not_modify_input(self):
        y = np.array([1.0, 2.0, 3.0])
        y_orig = y.copy()
        apply_transforms(y, [NMTransformInvert()])
        np.testing.assert_array_equal(y, y_orig)

    def test_type_error_ydata(self):
        with self.assertRaises(TypeError):
            apply_transforms([1, 2, 3], [NMTransformInvert()])

    def test_type_error_transform_item(self):
        y = np.array([1.0, 2.0])
        with self.assertRaises(TypeError):
            apply_transforms(y, ["not a transform"])

    def test_with_xscale(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        xs = NMScaleX(start=0, delta=0.5)
        result = apply_transforms(y, [NMTransformDifferentiate()], xscale=xs)
        # dy/dx = 1/0.5 = 2.0
        np.testing.assert_allclose(result, 2.0)


class TestNMTransformSmooth(unittest.TestCase):
    """Tests for NMTransformSmooth."""

    def setUp(self):
        self.y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_default_params(self):
        t = NMTransformSmooth()
        self.assertEqual(t.method, "boxcar")
        self.assertEqual(t.window, 5)
        self.assertEqual(t.passes, 1)
        self.assertEqual(t.polyorder, 2)

    def test_boxcar_output_length(self):
        t = NMTransformSmooth(method="boxcar", window=3)
        result = t.apply(self.y)
        self.assertEqual(len(result), len(self.y))

    def test_binomial_output_length(self):
        t = NMTransformSmooth(method="binomial")
        result = t.apply(self.y)
        self.assertEqual(len(result), len(self.y))

    def test_savgol_output_length(self):
        t = NMTransformSmooth(method="savgol", window=5, polyorder=2)
        result = t.apply(self.y)
        self.assertEqual(len(result), len(self.y))

    def test_flat_signal_boxcar(self):
        t = NMTransformSmooth(method="boxcar", window=3)
        y = np.ones(10)
        result = t.apply(y)
        np.testing.assert_allclose(result[1:-1], 1.0)

    def test_to_dict(self):
        t = NMTransformSmooth(method="savgol", window=7, passes=2, polyorder=3)
        d = t.to_dict()
        self.assertEqual(d["type"], "NMTransformSmooth")
        self.assertEqual(d["method"], "savgol")
        self.assertEqual(d["window"], 7)
        self.assertEqual(d["passes"], 2)
        self.assertEqual(d["polyorder"], 3)

    def test_equality(self):
        t1 = NMTransformSmooth(method="boxcar", window=5)
        t2 = NMTransformSmooth(method="boxcar", window=5)
        t3 = NMTransformSmooth(method="binomial")
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_from_dict_registered(self):
        t = NMTransformSmooth(method="binomial", passes=3)
        d = t.to_dict()
        t2 = _transform_from_dict(d)
        self.assertIsInstance(t2, NMTransformSmooth)

    def test_rejects_invalid_method(self):
        with self.assertRaises(ValueError):
            NMTransformSmooth(method="unknown")

    def test_rejects_even_window(self):
        with self.assertRaises(ValueError):
            NMTransformSmooth(window=4)

    def test_rejects_bool_passes(self):
        with self.assertRaises(TypeError):
            NMTransformSmooth(passes=True)

    def test_savgol_passes_gt1_runs_without_error(self):
        # passes > 1 is silently ignored for savgol (nmh.history ALERT issued)
        t = NMTransformSmooth(method="savgol", window=5, passes=3)
        result = t.apply(self.y)
        self.assertEqual(len(result), len(self.y))

    def test_repr(self):
        t = NMTransformSmooth(method="boxcar", window=5, passes=1, polyorder=2)
        self.assertIn("NMTransformSmooth", repr(t))
        self.assertIn("boxcar", repr(t))


if __name__ == "__main__":
    unittest.main()
