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
    NMTransformFilter,
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


class TestNMTransformFilter(unittest.TestCase):
    """Tests for NMTransformFilter."""

    SR = 10000.0  # sample rate Hz used across tests

    def setUp(self):
        # 1000-point DC signal at sample rate 10 kHz
        self.y_dc = np.ones(1000) * 5.0
        # Low-frequency sine (100 Hz) — passes through a 1 kHz lowpass
        t = np.arange(1000) / self.SR
        self.y_low = np.sin(2 * np.pi * 100 * t)
        # High-frequency sine (4 kHz) — attenuated by a 1 kHz lowpass
        self.y_high = np.sin(2 * np.pi * 4000 * t)

    # --- Constructor / property validation ---

    def test_default_params(self):
        t = NMTransformFilter()
        self.assertEqual(t.filter_type, "butterworth")
        self.assertEqual(t.cutoff, 1000.0)
        self.assertIsNone(t.sample_rate)
        self.assertEqual(t.order, 4)
        self.assertEqual(t.btype, "low")
        self.assertEqual(t.q, 30.0)

    def test_sample_rate_explicit(self):
        t = NMTransformFilter(sample_rate=self.SR)
        self.assertEqual(t.sample_rate, self.SR)

    def test_rejects_invalid_filter_type(self):
        with self.assertRaises(ValueError):
            NMTransformFilter(filter_type="fir", sample_rate=self.SR)

    def test_rejects_non_string_filter_type(self):
        with self.assertRaises(TypeError):
            NMTransformFilter(filter_type=1, sample_rate=self.SR)

    def test_rejects_zero_cutoff(self):
        with self.assertRaises(ValueError):
            NMTransformFilter(cutoff=0.0, sample_rate=self.SR)

    def test_rejects_bool_cutoff(self):
        with self.assertRaises(TypeError):
            NMTransformFilter(cutoff=True, sample_rate=self.SR)

    def test_rejects_non_positive_sample_rate(self):
        with self.assertRaises(ValueError):
            NMTransformFilter(sample_rate=0.0)

    def test_rejects_bool_sample_rate(self):
        with self.assertRaises(TypeError):
            NMTransformFilter(sample_rate=True)

    def test_rejects_invalid_btype(self):
        with self.assertRaises(ValueError):
            NMTransformFilter(btype="bandstop", sample_rate=self.SR)

    def test_rejects_non_positive_order(self):
        with self.assertRaises(ValueError):
            NMTransformFilter(order=0, sample_rate=self.SR)

    def test_rejects_bool_order(self):
        with self.assertRaises(TypeError):
            NMTransformFilter(order=True, sample_rate=self.SR)

    def test_rejects_non_positive_q(self):
        with self.assertRaises(ValueError):
            NMTransformFilter(q=0.0, sample_rate=self.SR)

    def test_rejects_bool_q(self):
        with self.assertRaises(TypeError):
            NMTransformFilter(q=True, sample_rate=self.SR)

    # --- Output shape ---

    def test_butterworth_output_length(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_dc)
        self.assertEqual(result.shape, self.y_dc.shape)

    def test_bessel_output_length(self):
        t = NMTransformFilter(filter_type="bessel", cutoff=1000.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_dc)
        self.assertEqual(result.shape, self.y_dc.shape)

    def test_notch_output_length(self):
        t = NMTransformFilter(filter_type="notch", cutoff=60.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_dc)
        self.assertEqual(result.shape, self.y_dc.shape)

    # --- Functional behaviour ---

    def test_butterworth_dc_preserved(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_dc)
        np.testing.assert_allclose(result[100:-100], 5.0, atol=1e-6)

    def test_bessel_dc_preserved(self):
        t = NMTransformFilter(filter_type="bessel", cutoff=1000.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_dc)
        np.testing.assert_allclose(result[100:-100], 5.0, atol=1e-6)

    def test_butterworth_lowpass_passes_low_freq(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_low)
        # 100 Hz should pass — RMS should be close to input RMS
        rms_in = np.sqrt(np.mean(self.y_low[100:-100] ** 2))
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        self.assertAlmostEqual(rms_in, rms_out, delta=0.05)

    def test_butterworth_lowpass_attenuates_high_freq(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                              sample_rate=self.SR)
        result = t.apply(self.y_high)
        # 4 kHz should be substantially attenuated
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        self.assertLess(rms_out, 0.1)

    def test_highpass_attenuates_low_freq(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                              sample_rate=self.SR, btype="high")
        result = t.apply(self.y_low)
        # 100 Hz should be attenuated by a 1 kHz highpass
        rms_out = np.sqrt(np.mean(result[100:-100] ** 2))
        self.assertLess(rms_out, 0.1)

    def test_notch_attenuates_target_frequency(self):
        # Use a longer signal so the high-Q notch transient decays
        n = 10000
        t_arr = np.arange(n) / self.SR
        y_60 = np.sin(2 * np.pi * 60 * t_arr)
        tf = NMTransformFilter(filter_type="notch", cutoff=60.0,
                               sample_rate=self.SR, q=30.0)
        result = tf.apply(y_60)
        rms_out = np.sqrt(np.mean(result[2000:-2000] ** 2))
        self.assertLess(rms_out, 0.1)

    def test_apply_rejects_non_ndarray(self):
        t = NMTransformFilter(sample_rate=self.SR)
        with self.assertRaises(TypeError):
            t.apply([1.0, 2.0, 3.0])

    def test_apply_raises_without_sample_rate_or_xscale(self):
        t = NMTransformFilter()  # sample_rate=None
        with self.assertRaises(ValueError):
            t.apply(self.y_dc)

    def test_apply_derives_sample_rate_from_xscale(self):
        from pyneuromatic.core.nm_scale import NMScaleX
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0)
        xscale = NMScaleX()
        xscale.delta = 1.0 / self.SR  # delta in seconds → SR = 10 kHz
        result = t.apply(self.y_dc, xscale=xscale)
        self.assertEqual(result.shape, self.y_dc.shape)
        np.testing.assert_allclose(result[100:-100], 5.0, atol=1e-6)

    # --- Serialisation / equality ---

    def test_to_dict_butterworth(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=500.0,
                              sample_rate=self.SR, order=2, btype="high")
        d = t.to_dict()
        self.assertEqual(d["type"], "NMTransformFilter")
        self.assertEqual(d["filter_type"], "butterworth")
        self.assertEqual(d["cutoff"], 500.0)
        self.assertEqual(d["sample_rate"], self.SR)
        self.assertEqual(d["order"], 2)
        self.assertEqual(d["btype"], "high")

    def test_to_dict_notch(self):
        t = NMTransformFilter(filter_type="notch", cutoff=60.0,
                              sample_rate=self.SR, q=20.0)
        d = t.to_dict()
        self.assertEqual(d["filter_type"], "notch")
        self.assertEqual(d["q"], 20.0)

    def test_equality(self):
        t1 = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                               sample_rate=self.SR, order=4, btype="low")
        t2 = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                               sample_rate=self.SR, order=4, btype="low")
        t3 = NMTransformFilter(filter_type="bessel", cutoff=1000.0,
                               sample_rate=self.SR)
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_from_dict_registered(self):
        t = NMTransformFilter(filter_type="bessel", cutoff=500.0,
                              sample_rate=self.SR, order=3)
        d = t.to_dict()
        t2 = _transform_from_dict(d)
        self.assertIsInstance(t2, NMTransformFilter)
        self.assertEqual(t, t2)

    def test_repr_butterworth(self):
        t = NMTransformFilter(filter_type="butterworth", cutoff=1000.0,
                              sample_rate=self.SR)
        r = repr(t)
        self.assertIn("NMTransformFilter", r)
        self.assertIn("butterworth", r)
        self.assertIn("btype", r)

    def test_repr_notch(self):
        t = NMTransformFilter(filter_type="notch", cutoff=60.0,
                              sample_rate=self.SR)
        r = repr(t)
        self.assertIn("NMTransformFilter", r)
        self.assertIn("notch", r)
        self.assertIn("q=", r)
        self.assertNotIn("btype", r)


if __name__ == "__main__":
    unittest.main()
