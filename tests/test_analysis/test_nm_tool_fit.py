#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_fit: NMToolFit.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import math
import unittest

import numpy as np

from pyneuromatic.analysis.nm_tool_fit import NMToolFit, NMToolFitConfig
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager, HIERARCHY_SELECT_KEYS

NM = NMManager(quiet=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 200
_XSTART = 0.0
_XDELTA = 0.5   # 0.5 ms steps → 100 ms total


def _make_linear_data(slope, intercept, n=_N, xstart=_XSTART, xdelta=_XDELTA, name="recordA0"):
    x = xstart + np.arange(n) * xdelta
    y = slope * x + intercept
    return NMData(NM, name=name, nparray=y,
                  xscale={"start": xstart, "delta": xdelta})


def _make_poly_data(coeffs, n=_N, xstart=_XSTART, xdelta=_XDELTA, name="recordA0"):
    """coeffs in ascending order: [C0, C1, ..., Cn]."""
    x = xstart + np.arange(n) * xdelta
    y = sum(c * x**k for k, c in enumerate(coeffs))
    return NMData(NM, name=name, nparray=y,
                  xscale={"start": xstart, "delta": xdelta})


def _make_exp_data(A, B, C, n=_N, xstart=_XSTART, xdelta=_XDELTA, name="recordA0"):
    x = xstart + np.arange(n) * xdelta
    y = A * np.exp(-x / B) + C
    return NMData(NM, name=name, nparray=y,
                  xscale={"start": xstart, "delta": xdelta})


def _make_gauss_data(A, mu, sigma_g, C, n=_N, xstart=_XSTART, xdelta=_XDELTA, name="recordA0"):
    x = xstart + np.arange(n) * xdelta
    y = A * np.exp(-0.5 * ((x - mu) / sigma_g) ** 2) + C
    return NMData(NM, name=name, nparray=y,
                  xscale={"start": xstart, "delta": xdelta})


def _make_exp2_data(A1, Tau1, A2, Tau2, C, n=_N, xstart=_XSTART, xdelta=_XDELTA, name="recordA0"):
    x = xstart + np.arange(n) * xdelta
    y = A1 * np.exp(-x / Tau1) + A2 * np.exp(-x / Tau2) + C
    return NMData(NM, name=name, nparray=y,
                  xscale={"start": xstart, "delta": xdelta})


def _make_boltzmann_data(A, X50, k, C, n=_N, xstart=_XSTART, xdelta=_XDELTA, name="recordA0"):
    x = xstart + np.arange(n) * xdelta
    y = A / (1.0 + np.exp(-(x - X50) / k)) + C
    return NMData(NM, name=name, nparray=y,
                  xscale={"start": xstart, "delta": xdelta})


def _make_targets(data_list, folder=None):
    if folder is None:
        folder = NMFolder(NM, name="TestFolder")
    return [
        {k: None for k in HIERARCHY_SELECT_KEYS} | {"folder": folder, "data": d}
        for d in data_list
    ], folder


def _run(tool, data_list, folder=None):
    targets, folder = _make_targets(data_list, folder=folder)
    tool.run_all(targets)
    return folder


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestNMToolFitDefaults(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolFit()

    def test_func_name_default(self):
        self.assertEqual(self.tool.func_name, "line")

    def test_no_degree_property(self):
        self.assertFalse(hasattr(self.tool, "degree"))

    def test_p0_default(self):
        self.assertIsNone(self.tool.p0)

    def test_maxfev_default(self):
        self.assertEqual(self.tool.maxfev, 10000)

    def test_sigma_default(self):
        self.assertIsNone(self.tool.sigma)

    def test_ignore_nans_default(self):
        self.assertTrue(self.tool.ignore_nans)

    def test_overwrite_default(self):
        self.assertTrue(self.tool.overwrite)

    def test_results_to_history_default(self):
        self.assertFalse(self.tool.results_to_history)

    def test_results_to_cache_default(self):
        self.assertTrue(self.tool.results_to_cache)

    def test_results_to_numpy_default(self):
        self.assertTrue(self.tool.results_to_numpy)

    def test_results_errors_default(self):
        self.assertFalse(self.tool.results_errors)

    def test_results_residuals_default(self):
        self.assertFalse(self.tool.results_residuals)

    def test_x_origin_default(self):
        self.assertEqual(self.tool.x_origin, 0.0)

    def test_param_names_default(self):
        self.assertIsNone(self.tool.param_names)

    def test_results_fit_array_default(self):
        self.assertFalse(self.tool.results_fit_array)

    def test_results_fit_npts_default(self):
        self.assertEqual(self.tool.results_fit_npts, 0)


# ---------------------------------------------------------------------------
# Property validation
# ---------------------------------------------------------------------------

class TestNMToolFitProperties(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolFit()

    def test_func_name_valid(self):
        for fn in ("line", "poly2", "poly3", "poly9", "exp", "gauss"):
            self.tool.func_name = fn
            self.assertEqual(self.tool.func_name, fn)

    def test_func_name_invalid_poly1_raises(self):
        with self.assertRaises(ValueError):
            self.tool.func_name = "poly1"

    def test_func_name_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.tool.func_name = "quadratic"

    def test_func_name_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.func_name = 3

    def test_p0_dict_accepted(self):
        self.tool.p0 = {"A": 1.0, "Tau": 5.0}
        self.assertEqual(self.tool.p0, {"A": 1.0, "Tau": 5.0})

    def test_p0_none_accepted(self):
        self.tool.p0 = {"A": 1.0}
        self.tool.p0 = None
        self.assertIsNone(self.tool.p0)

    def test_p0_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.p0 = [1.0, 2.0]

    def test_sigma_array_accepted(self):
        arr = np.ones(10)
        self.tool.sigma = arr
        np.testing.assert_array_equal(self.tool.sigma, arr)

    def test_sigma_none_accepted(self):
        self.tool.sigma = np.ones(5)
        self.tool.sigma = None
        self.assertIsNone(self.tool.sigma)

    def test_sigma_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.sigma = [1.0, 2.0]

    def test_maxfev_valid(self):
        self.tool.maxfev = 500
        self.assertEqual(self.tool.maxfev, 500)

    def test_maxfev_zero_raises(self):
        with self.assertRaises(ValueError):
            self.tool.maxfev = 0

    def test_maxfev_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.maxfev = 1000.0

    def test_maxfev_bool_rejected(self):
        with self.assertRaises(TypeError):
            self.tool.maxfev = True

    def test_results_errors_valid(self):
        self.tool.results_errors = True
        self.assertTrue(self.tool.results_errors)
        self.tool.results_errors = False
        self.assertFalse(self.tool.results_errors)

    def test_results_errors_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.results_errors = 1

    def test_results_residuals_valid(self):
        self.tool.results_residuals = True
        self.assertTrue(self.tool.results_residuals)
        self.tool.results_residuals = False
        self.assertFalse(self.tool.results_residuals)

    def test_results_residuals_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.results_residuals = "yes"

    def test_results_fit_array_valid(self):
        self.tool.results_fit_array = True
        self.assertTrue(self.tool.results_fit_array)

    def test_results_fit_array_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.results_fit_array = 1

    def test_results_fit_npts_valid(self):
        self.tool.results_fit_npts = 500
        self.assertEqual(self.tool.results_fit_npts, 500)
        self.tool.results_fit_npts = 0
        self.assertEqual(self.tool.results_fit_npts, 0)

    def test_results_fit_npts_negative_raises(self):
        with self.assertRaises(ValueError):
            self.tool.results_fit_npts = -1

    def test_results_fit_npts_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.results_fit_npts = 100.0

    def test_results_fit_npts_bool_rejected(self):
        with self.assertRaises(TypeError):
            self.tool.results_fit_npts = True

    def test_x_origin_valid(self):
        self.tool.x_origin = 10.5
        self.assertAlmostEqual(self.tool.x_origin, 10.5)

    def test_x_origin_nan_raises(self):
        with self.assertRaises(ValueError):
            self.tool.x_origin = float("nan")

    def test_x_origin_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.x_origin = "t0"

    def test_param_names_dict_accepted(self):
        self.tool.param_names = {"A": "amplitude", "Tau": "decay"}
        self.assertEqual(self.tool.param_names, {"A": "amplitude", "Tau": "decay"})

    def test_param_names_none_accepted(self):
        self.tool.param_names = {"A": "amp"}
        self.tool.param_names = None
        self.assertIsNone(self.tool.param_names)

    def test_param_names_type_error(self):
        with self.assertRaises(TypeError):
            self.tool.param_names = ["A", "B"]


# ---------------------------------------------------------------------------
# Line fitting
# ---------------------------------------------------------------------------

class TestNMToolFitLine(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolFit()
        self.tool.func_name = "line"

    def test_line_slope_intercept(self):
        slope, intercept = 2.5, -3.0
        data = _make_linear_data(slope, intercept)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIsNotNone(tf)
        m = tf.data.get("FT_M")
        b = tf.data.get("FT_B")
        self.assertIsNotNone(m)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(float(m.nparray[0]), slope, places=5)
        self.assertAlmostEqual(float(b.nparray[0]), intercept, places=5)

    def test_line_r2_near_one(self):
        data = _make_linear_data(1.0, 0.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Line_0")
        r2 = tf.data.get("FT_R2")
        self.assertAlmostEqual(float(r2.nparray[0]), 1.0, places=6)

    def test_line_output_arrays_exist(self):
        data = _make_linear_data(1.0, 0.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Line_0")
        for name in ("FT_M", "FT_B", "FT_R2", "FT_ChiSqr"):
            self.assertIn(name, tf.data, msg="missing %s" % name)


# ---------------------------------------------------------------------------
# Polynomial fitting
# ---------------------------------------------------------------------------

class TestNMToolFitPoly(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolFit()
        self.tool.func_name = "poly2"

    def test_poly_coefficients(self):
        coeffs = [1.0, -0.5, 0.02]  # C0 + C1*x + C2*x^2
        data = _make_poly_data(coeffs)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Poly2_0")
        self.assertIsNotNone(tf)
        for k, expected in enumerate(coeffs):
            arr = tf.data.get("FT_C%d" % k)
            self.assertIsNotNone(arr, msg="FT_C%d missing" % k)
            self.assertAlmostEqual(float(arr.nparray[0]), expected, places=4)

    def test_poly_r2_near_one(self):
        data = _make_poly_data([1.0, 2.0, 0.5])
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Poly2_0")
        r2 = tf.data.get("FT_R2")
        self.assertAlmostEqual(float(r2.nparray[0]), 1.0, places=6)

    def test_poly_output_arrays_exist(self):
        data = _make_poly_data([1.0, 0.0, 0.0])
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Poly2_0")
        for name in ("FT_C0", "FT_C1", "FT_C2", "FT_R2", "FT_ChiSqr"):
            self.assertIn(name, tf.data, msg="missing %s" % name)

    def test_poly3_has_four_coefficients(self):
        self.tool.func_name = "poly3"
        coeffs = [1.0, -0.5, 0.02, -0.001]
        data = _make_poly_data(coeffs)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Poly3_0")
        for k in range(4):
            self.assertIn("FT_C%d" % k, tf.data)


# ---------------------------------------------------------------------------
# Exponential fitting
# ---------------------------------------------------------------------------

class TestNMToolFitExp(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolFit()
        self.tool.func_name = "exp"

    def test_exp_parameters(self):
        A, B, C = 50.0, 20.0, 5.0
        data = _make_exp_data(A, B, C)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        self.assertIsNotNone(tf)
        self.assertAlmostEqual(float(tf.data["FT_A"].nparray[0]), A, delta=0.5)
        self.assertAlmostEqual(float(tf.data["FT_Tau"].nparray[0]), B, delta=0.5)
        self.assertAlmostEqual(float(tf.data["FT_Y0"].nparray[0]), C, delta=0.5)

    def test_exp_r2_near_one(self):
        data = _make_exp_data(50.0, 20.0, 5.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        r2 = tf.data.get("FT_R2")
        self.assertGreater(float(r2.nparray[0]), 0.999)

    def test_exp_converged(self):
        data = _make_exp_data(50.0, 20.0, 5.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        self.assertEqual(float(tf.data["FT_Converged"].nparray[0]), 1.0)

    def test_exp_output_arrays_exist(self):
        data = _make_exp_data(50.0, 20.0, 5.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        for name in ("FT_A", "FT_Tau", "FT_X0", "FT_Y0", "FT_R2", "FT_ChiSqr", "FT_Converged"):
            self.assertIn(name, tf.data, msg="missing %s" % name)


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

class TestNMToolFitGauss(unittest.TestCase):

    def setUp(self):
        self.tool = NMToolFit()
        self.tool.func_name = "gauss"

    def test_gauss_parameters(self):
        A, mu, sigma_g, C = 100.0, 50.0, 10.0, 2.0
        data = _make_gauss_data(A, mu, sigma_g, C)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        self.assertIsNotNone(tf)
        self.assertAlmostEqual(float(tf.data["FT_A"].nparray[0]),     A,       delta=1.0)
        self.assertAlmostEqual(float(tf.data["FT_Mu"].nparray[0]),    mu,      delta=0.5)
        self.assertAlmostEqual(float(tf.data["FT_Sigma"].nparray[0]), sigma_g, delta=0.5)
        self.assertAlmostEqual(float(tf.data["FT_Y0"].nparray[0]),     C,       delta=0.5)

    def test_gauss_r2_near_one(self):
        data = _make_gauss_data(100.0, 50.0, 10.0, 0.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        self.assertGreater(float(tf.data["FT_R2"].nparray[0]), 0.999)

    def test_gauss_converged(self):
        data = _make_gauss_data(100.0, 50.0, 10.0, 0.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        self.assertEqual(float(tf.data["FT_Converged"].nparray[0]), 1.0)

    def test_gauss_output_arrays_exist(self):
        data = _make_gauss_data(100.0, 50.0, 10.0, 0.0)
        folder = _run(self.tool, [data])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        for name in ("FT_A", "FT_Mu", "FT_Sigma", "FT_Y0", "FT_R2", "FT_ChiSqr", "FT_Converged"):
            self.assertIn(name, tf.data, msg="missing %s" % name)


# ---------------------------------------------------------------------------
# p0 override
# ---------------------------------------------------------------------------

class TestNMToolFitP0Override(unittest.TestCase):

    def test_exp_p0_override_converges(self):
        A, B, C = 50.0, 20.0, 5.0
        data = _make_exp_data(A, B, C)
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.p0 = {"A": 45.0, "Tau": 18.0, "Y0": 4.0}
        folder = _run(tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        self.assertEqual(float(tf.data["FT_Converged"].nparray[0]), 1.0)
        self.assertAlmostEqual(float(tf.data["FT_A"].nparray[0]), A, delta=0.5)

    def test_gauss_p0_override_converges(self):
        A, mu, sigma_g, C = 100.0, 50.0, 10.0, 2.0
        data = _make_gauss_data(A, mu, sigma_g, C)
        tool = NMToolFit()
        tool.func_name = "gauss"
        tool.p0 = {"A": 95.0, "Mu": 48.0, "Sigma": 9.0, "Y0": 1.0}
        folder = _run(tool, [data])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        self.assertEqual(float(tf.data["FT_Converged"].nparray[0]), 1.0)
        self.assertAlmostEqual(float(tf.data["FT_Mu"].nparray[0]), mu, delta=0.5)


# ---------------------------------------------------------------------------
# xbgn/xend range restriction
# ---------------------------------------------------------------------------

class TestNMToolFitXRange(unittest.TestCase):

    def test_x_range_restricts_fit(self):
        # Two-segment data: steep slope in [0,50], flat in (50,100]
        n = 200
        xdelta = 0.5
        x = np.arange(n) * xdelta
        y = np.where(x <= 50, 2.0 * x + 1.0, 101.0)
        data = NMData(NM, name="recordA0", nparray=y,
                      xscale={"start": 0.0, "delta": xdelta})
        tool = NMToolFit()
        tool.func_name = "line"
        tool.xbgn = 0.0
        tool.xend = 50.0
        folder = _run(tool, [data])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertAlmostEqual(float(tf.data["FT_M"].nparray[0]), 2.0, delta=0.01)


# ---------------------------------------------------------------------------
# Output array structure
# ---------------------------------------------------------------------------

class TestNMToolFitOutputArrays(unittest.TestCase):

    def test_toolfolder_name_line(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(1.0, 0.0)])
        self.assertIn("Fit_Line_0", folder.toolfolders)

    def test_toolfolder_name_poly(self):
        tool = NMToolFit()
        tool.func_name = "poly2"
        folder = _run(tool, [_make_poly_data([1.0, 0.0, 0.0])])
        self.assertIn("Fit_Poly2_0", folder.toolfolders)

    def test_toolfolder_name_exp(self):
        tool = NMToolFit()
        tool.func_name = "exp"
        folder = _run(tool, [_make_exp_data(50.0, 20.0, 0.0)])
        self.assertIn("Fit_Exp_0", folder.toolfolders)

    def test_toolfolder_name_gauss(self):
        tool = NMToolFit()
        tool.func_name = "gauss"
        folder = _run(tool, [_make_gauss_data(100.0, 50.0, 10.0, 0.0)])
        self.assertIn("Fit_Gauss_0", folder.toolfolders)

    def test_epoch_names_array_written(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(1.0, 0.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        en = tf.data.get("FT_epoch_names")
        self.assertIsNotNone(en)
        self.assertEqual(str(en.nparray[0]), "recordA0")

    def test_output_array_notes_contain_func_name(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(1.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data.get("FT_M")
        notes_text = " ".join(str(n) for n in d.notes)
        self.assertIn("line", notes_text)

    def test_output_xscale_units_epoch(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(1.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data.get("FT_M")
        self.assertEqual(d.xscale.units, "epoch")


# ---------------------------------------------------------------------------
# Multi-epoch
# ---------------------------------------------------------------------------

class TestNMToolFitMultiEpoch(unittest.TestCase):

    def test_multi_epoch_array_length(self):
        tool = NMToolFit()
        tool.func_name = "line"
        d0 = _make_linear_data(1.0, 0.0, name="recordA0")
        d1 = _make_linear_data(2.0, 5.0, name="recordA1")
        folder = _run(tool, [d0, d1])
        tf = folder.toolfolders.get("Fit_Line_0")
        m = tf.data.get("FT_M")
        self.assertEqual(len(m.nparray), 2)
        self.assertAlmostEqual(float(m.nparray[0]), 1.0, places=5)
        self.assertAlmostEqual(float(m.nparray[1]), 2.0, places=5)

    def test_multi_epoch_b_values(self):
        tool = NMToolFit()
        tool.func_name = "line"
        d0 = _make_linear_data(0.0, 3.0, name="recordA0")
        d1 = _make_linear_data(0.0, 7.0, name="recordA1")
        folder = _run(tool, [d0, d1])
        tf = folder.toolfolders.get("Fit_Line_0")
        b = tf.data.get("FT_B")
        self.assertAlmostEqual(float(b.nparray[0]), 3.0, places=5)
        self.assertAlmostEqual(float(b.nparray[1]), 7.0, places=5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestNMToolFitConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = NMToolFitConfig()
        self.assertEqual(cfg.func_name, "line")
        self.assertEqual(cfg.xbgn, -math.inf)
        self.assertEqual(cfg.xend, math.inf)
        self.assertEqual(cfg.maxfev, 10000)

    def test_degree_not_in_config(self):
        cfg = NMToolFitConfig()
        with self.assertRaises(AttributeError):
            cfg.degree = 3

    def test_invalid_func_name_raises(self):
        cfg = NMToolFitConfig()
        with self.assertRaises(ValueError):
            cfg.func_name = "cubic"

    def test_maxfev_too_small_raises(self):
        cfg = NMToolFitConfig()
        with self.assertRaises(ValueError):
            cfg.maxfev = 0

    def test_to_dict_round_trip(self):
        cfg = NMToolFitConfig()
        cfg.func_name = "exp"
        cfg.maxfev = 500
        d = cfg.to_dict()
        cfg2 = NMToolFitConfig.from_dict(d)
        self.assertEqual(cfg2.func_name, "exp")
        self.assertEqual(cfg2.maxfev, 500)

    def test_toml_type(self):
        self.assertEqual(NMToolFitConfig._TOML_TYPE, "fit_config")


# ---------------------------------------------------------------------------
# Overwrite flag
# ---------------------------------------------------------------------------

class TestNMToolFitOverwrite(unittest.TestCase):

    def test_overwrite_true_reuses_folder(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.overwrite = True
        folder = NMFolder(NM, name="TestFolder")
        data = _make_linear_data(1.0, 0.0)
        targets, _ = _make_targets([data], folder=folder)
        tool.run_all(targets)
        tool.run_all(targets)
        # With overwrite=True the toolfolder name ends in _0 both times
        self.assertIn("Fit_Line_0", folder.toolfolders)

    def test_overwrite_false_creates_new_folder(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.overwrite = False
        folder = NMFolder(NM, name="TestFolder")
        data = _make_linear_data(1.0, 0.0)
        targets, _ = _make_targets([data], folder=folder)
        tool.run_all(targets)
        tool.run_all(targets)
        self.assertIn("Fit_Line_0", folder.toolfolders)
        self.assertIn("Fit_Line_1", folder.toolfolders)


# ---------------------------------------------------------------------------
# results_to_cache
# ---------------------------------------------------------------------------

class TestNMToolFitResultsToCache(unittest.TestCase):

    def test_cache_contains_slope(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = NMFolder(NM, name="TestFolder")
        data = _make_linear_data(3.0, 1.0, name="recordA0")
        targets, _ = _make_targets([data], folder=folder)
        tool.run_all(targets)
        self.assertIn("fit", folder.toolresults)
        # toolresults["fit"] is a list of entries; most recent is last
        cached = folder.toolresults["fit"][-1]["results"]
        self.assertIn("recordA0", cached)
        self.assertAlmostEqual(cached["recordA0"]["slope"], 3.0, places=5)


# ---------------------------------------------------------------------------
# results_errors
# ---------------------------------------------------------------------------

class TestNMToolFitResultsErrors(unittest.TestCase):

    def test_line_errors_not_written_by_default(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(2.0, 1.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertNotIn("FT_err_A", tf.data)
        self.assertNotIn("FT_err_B", tf.data)

    def test_line_errors_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_errors = True
        folder = _run(tool, [_make_linear_data(2.0, 1.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_err_M", tf.data)
        self.assertIn("FT_err_B", tf.data)

    def test_line_errors_near_zero_for_exact_data(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_errors = True
        folder = _run(tool, [_make_linear_data(2.0, 1.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        err_a = float(tf.data["FT_err_M"].nparray[0])
        err_b = float(tf.data["FT_err_B"].nparray[0])
        self.assertLess(err_a, 1e-6)
        self.assertLess(err_b, 1e-6)

    def test_poly_errors_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "poly2"
        tool.results_errors = True
        folder = _run(tool, [_make_poly_data([1.0, -0.5, 0.02])])
        tf = folder.toolfolders.get("Fit_Poly2_0")
        for k in range(3):
            self.assertIn("FT_err_C%d" % k, tf.data, msg="FT_err_C%d missing" % k)

    def test_poly_errors_not_written_by_default(self):
        tool = NMToolFit()
        tool.func_name = "poly2"
        folder = _run(tool, [_make_poly_data([1.0, 0.0, 0.0])])
        tf = folder.toolfolders.get("Fit_Poly2_0")
        self.assertNotIn("FT_err_C0", tf.data)

    def test_exp_errors_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.results_errors = True
        folder = _run(tool, [_make_exp_data(50.0, 20.0, 5.0)])
        tf = folder.toolfolders.get("Fit_Exp_0")
        for name in ("FT_err_A", "FT_err_Tau", "FT_err_Y0"):
            self.assertIn(name, tf.data, msg="%s missing" % name)

    def test_gauss_errors_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "gauss"
        tool.results_errors = True
        folder = _run(tool, [_make_gauss_data(100.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        for name in ("FT_err_A", "FT_err_Mu", "FT_err_Sigma", "FT_err_Y0"):
            self.assertIn(name, tf.data, msg="%s missing" % name)

    def test_gauss_errors_near_zero_for_exact_data(self):
        tool = NMToolFit()
        tool.func_name = "gauss"
        tool.results_errors = True
        folder = _run(tool, [_make_gauss_data(100.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        for name in ("FT_err_A", "FT_err_Mu", "FT_err_Sigma", "FT_err_Y0"):
            err = float(tf.data[name].nparray[0])
            self.assertLess(err, 1e-3, msg="%s=%g unexpectedly large" % (name, err))


# ---------------------------------------------------------------------------
# results_residuals
# ---------------------------------------------------------------------------

class TestNMToolFitResultsResiduals(unittest.TestCase):

    def test_residuals_not_written_by_default(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(2.0, 1.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertNotIn("FT_resid_recordA0", tf.data)

    def test_residuals_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_residuals = True
        folder = _run(tool, [_make_linear_data(2.0, 1.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_resid_recordA0", tf.data)

    def test_residuals_near_zero_for_exact_line(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_residuals = True
        folder = _run(tool, [_make_linear_data(2.0, 1.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        resid = tf.data["FT_resid_recordA0"].nparray
        np.testing.assert_allclose(resid, 0.0, atol=1e-9)

    def test_residuals_correct_length(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_residuals = True
        folder = _run(tool, [_make_linear_data(1.0, 0.0, n=_N, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        resid = tf.data["FT_resid_recordA0"].nparray
        self.assertEqual(len(resid), _N)

    def test_residuals_xarray_set(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_residuals = True
        folder = _run(tool, [_make_linear_data(1.0, 0.0, n=_N, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data["FT_resid_recordA0"]
        self.assertIsNotNone(d.xarray)
        self.assertEqual(len(d.xarray), _N)

    def test_residuals_multi_epoch(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_residuals = True
        d0 = _make_linear_data(1.0, 0.0, name="recordA0")
        d1 = _make_linear_data(2.0, 5.0, name="recordA1")
        folder = _run(tool, [d0, d1])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_resid_recordA0", tf.data)
        self.assertIn("FT_resid_recordA1", tf.data)

    def test_residuals_windowed_length(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.xbgn = 0.0
        tool.xend = 50.0
        tool.results_residuals = True
        folder = _run(tool, [_make_linear_data(1.0, 0.0, n=_N, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        resid = tf.data["FT_resid_recordA0"].nparray
        # x range [0, 50] with xdelta=0.5 → 101 points (0.0 to 50.0 inclusive)
        self.assertEqual(len(resid), 101)

    def test_residuals_near_zero_for_exact_exp(self):
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.results_residuals = True
        folder = _run(tool, [_make_exp_data(50.0, 20.0, 5.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Exp_0")
        resid = tf.data["FT_resid_recordA0"].nparray
        np.testing.assert_allclose(resid, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# results_fit_array
# ---------------------------------------------------------------------------

class TestNMToolFitResultsFitArray(unittest.TestCase):

    def test_fit_array_not_written_by_default(self):
        tool = NMToolFit()
        tool.func_name = "line"
        folder = _run(tool, [_make_linear_data(2.0, 1.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertNotIn("Fit_recordA0", tf.data)

    def test_fit_array_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_fit_array = True
        folder = _run(tool, [_make_linear_data(2.0, 1.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("Fit_recordA0", tf.data)

    def test_fit_array_xarray_set(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_fit_array = True
        folder = _run(tool, [_make_linear_data(2.0, 1.0, n=_N, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data["Fit_recordA0"]
        self.assertIsNotNone(d.xarray)
        self.assertEqual(len(d.xarray), _N)

    def test_fit_array_values_match_model_line(self):
        slope, intercept = 2.0, 1.0
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_fit_array = True
        folder = _run(tool, [_make_linear_data(slope, intercept, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data["Fit_recordA0"]
        expected = slope * d.xarray + intercept
        np.testing.assert_allclose(d.nparray, expected, rtol=1e-5)

    def test_fit_array_values_match_model_exp(self):
        A, B, C = 50.0, 20.0, 5.0
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.results_fit_array = True
        folder = _run(tool, [_make_exp_data(A, B, C, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Exp_0")
        d = tf.data["Fit_recordA0"]
        expected = A * np.exp(-d.xarray / B) + C
        np.testing.assert_allclose(d.nparray, expected, rtol=1e-4)

    def test_fit_array_values_match_model_gauss(self):
        A, mu, sg, C = 100.0, 50.0, 10.0, 0.0
        tool = NMToolFit()
        tool.func_name = "gauss"
        tool.results_fit_array = True
        folder = _run(tool, [_make_gauss_data(A, mu, sg, C, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        d = tf.data["Fit_recordA0"]
        expected = A * np.exp(-0.5 * ((d.xarray - mu) / sg) ** 2) + C
        np.testing.assert_allclose(d.nparray, expected, rtol=1e-4)

    def test_fit_array_npts_changes_length(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_fit_array = True
        tool.results_fit_npts = 1000
        folder = _run(tool, [_make_linear_data(2.0, 1.0, n=_N, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data["Fit_recordA0"]
        self.assertEqual(len(d.nparray), 1000)
        self.assertEqual(len(d.xarray), 1000)

    def test_fit_array_npts_xrange(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_fit_array = True
        tool.results_fit_npts = 50
        folder = _run(tool, [_make_linear_data(1.0, 0.0, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Line_0")
        d = tf.data["Fit_recordA0"]
        # x-range should span [xstart, xstart + (N-1)*xdelta]
        self.assertAlmostEqual(float(d.xarray[0]),  _XSTART, places=6)
        self.assertAlmostEqual(float(d.xarray[-1]), _XSTART + (_N - 1) * _XDELTA, places=6)

    def test_fit_array_multi_epoch(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_fit_array = True
        d0 = _make_linear_data(1.0, 0.0, name="recordA0")
        d1 = _make_linear_data(2.0, 5.0, name="recordA1")
        folder = _run(tool, [d0, d1])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("Fit_recordA0", tf.data)
        self.assertIn("Fit_recordA1", tf.data)

    def test_fit_array_poly_values(self):
        coeffs = [1.0, -0.5, 0.02]
        tool = NMToolFit()
        tool.func_name = "poly2"
        tool.results_fit_array = True
        folder = _run(tool, [_make_poly_data(coeffs, name="recordA0")])
        tf = folder.toolfolders.get("Fit_Poly2_0")
        d = tf.data["Fit_recordA0"]
        x = d.xarray
        expected = coeffs[0] + coeffs[1] * x + coeffs[2] * x ** 2
        np.testing.assert_allclose(d.nparray, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# x_origin (exp model shift)
# ---------------------------------------------------------------------------

class TestNMToolFitXOrigin(unittest.TestCase):

    def test_x_origin_written_to_ft_X0(self):
        A, B, C = 50.0, 20.0, 5.0
        xbgn = 10.0
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.x_origin = xbgn
        data = _make_exp_data(A, B, C, xstart=xbgn)
        folder = _run(tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        self.assertAlmostEqual(float(tf.data["FT_X0"].nparray[0]), xbgn, places=6)

    def test_x_origin_zero_matches_default_result(self):
        A, B, C = 50.0, 20.0, 5.0
        data = _make_exp_data(A, B, C)
        tool_default = NMToolFit()
        tool_default.func_name = "exp"
        folder1 = _run(tool_default, [data])

        tool_explicit = NMToolFit()
        tool_explicit.func_name = "exp"
        tool_explicit.x_origin = 0.0
        folder2 = _run(tool_explicit, [data])

        tf1 = folder1.toolfolders.get("Fit_Exp_0")
        tf2 = folder2.toolfolders.get("Fit_Exp_0")
        self.assertAlmostEqual(
            float(tf1.data["FT_A"].nparray[0]),
            float(tf2.data["FT_A"].nparray[0]),
            places=5,
        )

    def test_x_origin_shifts_fit_correctly(self):
        # Exponential decay starting at x=5: A*exp(-(x-5)/B) + Y0
        A, B, Y0, X0 = 40.0, 15.0, 3.0, 5.0
        n = 200
        xdelta = 0.5
        x = X0 + np.arange(n) * xdelta
        y = A * np.exp(-(x - X0) / B) + Y0
        data = NMData(NM, name="recordA0", nparray=y,
                      xscale={"start": float(x[0]), "delta": xdelta})
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.x_origin = X0
        folder = _run(tool, [data])
        tf = folder.toolfolders.get("Fit_Exp_0")
        self.assertAlmostEqual(float(tf.data["FT_A"].nparray[0]),  A,  delta=0.5)
        self.assertAlmostEqual(float(tf.data["FT_Tau"].nparray[0]),  B,  delta=0.5)
        self.assertAlmostEqual(float(tf.data["FT_Y0"].nparray[0]), Y0, delta=0.5)


# ---------------------------------------------------------------------------
# param_names (output array name remapping)
# ---------------------------------------------------------------------------

class TestNMToolFitParamNames(unittest.TestCase):

    def test_param_names_renames_line_output(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.param_names = {"M": "slope", "B": "intercept"}
        folder = _run(tool, [_make_linear_data(2.0, 1.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_slope",     tf.data)
        self.assertIn("FT_intercept", tf.data)
        self.assertNotIn("FT_M", tf.data)
        self.assertNotIn("FT_B", tf.data)

    def test_param_names_partial_override(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.param_names = {"M": "slope"}   # only rename M
        folder = _run(tool, [_make_linear_data(2.0, 1.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_slope", tf.data)   # renamed
        self.assertIn("FT_B",     tf.data)   # default kept

    def test_param_names_renames_exp_output(self):
        tool = NMToolFit()
        tool.func_name = "exp"
        tool.param_names = {"A": "amplitude", "Tau": "decay", "Y0": "baseline"}
        folder = _run(tool, [_make_exp_data(50.0, 20.0, 5.0)])
        tf = folder.toolfolders.get("Fit_Exp_0")
        self.assertIn("FT_amplitude", tf.data)
        self.assertIn("FT_decay",     tf.data)
        self.assertIn("FT_baseline",  tf.data)
        self.assertNotIn("FT_A",   tf.data)
        self.assertNotIn("FT_Tau", tf.data)
        self.assertNotIn("FT_Y0",  tf.data)

    def test_param_names_renames_gauss_output(self):
        tool = NMToolFit()
        tool.func_name = "gauss"
        tool.param_names = {"A": "amp", "Mu": "mean", "Sigma": "std", "Y0": "offset"}
        folder = _run(tool, [_make_gauss_data(100.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Gauss_0")
        for expected in ("FT_amp", "FT_mean", "FT_std", "FT_offset"):
            self.assertIn(expected, tf.data, msg="%s missing" % expected)

    def test_param_names_renames_error_arrays_too(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.results_errors = True
        tool.param_names = {"M": "slope", "B": "intercept"}
        folder = _run(tool, [_make_linear_data(2.0, 1.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_err_slope",     tf.data)
        self.assertIn("FT_err_intercept", tf.data)
        self.assertNotIn("FT_err_M", tf.data)

    def test_param_names_none_uses_defaults(self):
        tool = NMToolFit()
        tool.func_name = "line"
        tool.param_names = {"M": "slope"}
        tool.param_names = None
        folder = _run(tool, [_make_linear_data(1.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Line_0")
        self.assertIn("FT_M", tf.data)
        self.assertIn("FT_B", tf.data)


# ---------------------------------------------------------------------------
# Double exponential
# ---------------------------------------------------------------------------

class TestNMToolFitExp2(unittest.TestCase):

    def test_func_name_exp2_accepted(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        self.assertEqual(tool.func_name, "exp2")

    def test_toolfolder_name_exp2(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        folder = _run(tool, [_make_exp2_data(30.0, 5.0, 20.0, 20.0, 2.0)])
        self.assertIn("Fit_Exp2_0", folder.toolfolders)

    def test_output_arrays_present(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        folder = _run(tool, [_make_exp2_data(30.0, 5.0, 20.0, 20.0, 2.0)])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        for name in ("FT_A1", "FT_Tau1", "FT_A2", "FT_Tau2", "FT_X0", "FT_Y0",
                     "FT_R2", "FT_ChiSqr", "FT_Converged"):
            self.assertIn(name, tf.data, msg="%s missing" % name)

    def test_params_match_known_values(self):
        A1, Tau1, A2, Tau2, C = 30.0, 5.0, 20.0, 20.0, 2.0
        tool = NMToolFit()
        tool.func_name = "exp2"
        folder = _run(tool, [_make_exp2_data(A1, Tau1, A2, Tau2, C)])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        fit_a1   = float(tf.data["FT_A1"].nparray[0])
        fit_tau1 = float(tf.data["FT_Tau1"].nparray[0])
        fit_a2   = float(tf.data["FT_A2"].nparray[0])
        fit_tau2 = float(tf.data["FT_Tau2"].nparray[0])
        fit_y0   = float(tf.data["FT_Y0"].nparray[0])
        # Tau1 <= Tau2 after sorting
        self.assertLessEqual(fit_tau1, fit_tau2)
        self.assertAlmostEqual(fit_tau1, Tau1, places=3)
        self.assertAlmostEqual(fit_tau2, Tau2, places=3)
        self.assertAlmostEqual(fit_y0,   C,    places=3)
        # A1 corresponds to smaller Tau1, A2 to larger Tau2
        self.assertAlmostEqual(fit_a1, A1, places=2)
        self.assertAlmostEqual(fit_a2, A2, places=2)

    def test_r2_near_one_for_exact_data(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        folder = _run(tool, [_make_exp2_data(30.0, 5.0, 20.0, 20.0, 2.0)])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        r2 = float(tf.data["FT_R2"].nparray[0])
        self.assertGreater(r2, 0.9999)

    def test_tau_sorted_tau1_le_tau2(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        # Deliberately give larger Tau first in p0 to ensure sort works
        tool.p0 = {"Tau1": 20.0, "Tau2": 5.0}
        folder = _run(tool, [_make_exp2_data(30.0, 5.0, 20.0, 20.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        tau1 = float(tf.data["FT_Tau1"].nparray[0])
        tau2 = float(tf.data["FT_Tau2"].nparray[0])
        self.assertLessEqual(tau1, tau2)

    def test_errors_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        tool.results_errors = True
        folder = _run(tool, [_make_exp2_data(30.0, 5.0, 20.0, 20.0, 2.0)])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        for name in ("FT_err_A1", "FT_err_Tau1", "FT_err_A2", "FT_err_Tau2", "FT_err_Y0"):
            self.assertIn(name, tf.data, msg="%s missing" % name)

    def test_converged_is_one_for_exact_data(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        folder = _run(tool, [_make_exp2_data(30.0, 5.0, 20.0, 20.0, 2.0)])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        self.assertEqual(float(tf.data["FT_Converged"].nparray[0]), 1.0)

    def test_multi_epoch_array_length(self):
        tool = NMToolFit()
        tool.func_name = "exp2"
        d0 = _make_exp2_data(30.0, 5.0, 20.0, 20.0, 2.0, name="recordA0")
        d1 = _make_exp2_data(15.0, 5.0, 10.0, 20.0, 1.0, name="recordA1")
        folder = _run(tool, [d0, d1])
        tf = folder.toolfolders.get("Fit_Exp2_0")
        self.assertEqual(len(tf.data["FT_A1"].nparray), 2)

    def test_config_accepts_exp2(self):
        cfg = NMToolFitConfig()
        cfg.func_name = "exp2"
        self.assertEqual(cfg.func_name, "exp2")


# ---------------------------------------------------------------------------
# Boltzmann (sigmoid)
# ---------------------------------------------------------------------------

class TestNMToolFitBoltzmann(unittest.TestCase):

    def test_func_name_boltzmann_accepted(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        self.assertEqual(tool.func_name, "boltzmann")

    def test_toolfolder_name_boltzmann(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, 10.0, 0.0)])
        self.assertIn("Fit_Boltzmann_0", folder.toolfolders)

    def test_output_arrays_present(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        for name in ("FT_A", "FT_X50", "FT_K", "FT_Y0",
                     "FT_R2", "FT_ChiSqr", "FT_Converged"):
            self.assertIn(name, tf.data, msg="%s missing" % name)

    def test_params_match_known_values(self):
        A, X50, K, C = 1.0, 50.0, 10.0, 0.0
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        folder = _run(tool, [_make_boltzmann_data(A, X50, K, C)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        self.assertAlmostEqual(float(tf.data["FT_A"].nparray[0]),   A,   places=4)
        self.assertAlmostEqual(float(tf.data["FT_X50"].nparray[0]), X50, places=3)
        self.assertAlmostEqual(float(tf.data["FT_K"].nparray[0]),   K,   places=3)
        self.assertAlmostEqual(float(tf.data["FT_Y0"].nparray[0]),  C,   places=4)

    def test_r2_near_one_for_exact_data(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        r2 = float(tf.data["FT_R2"].nparray[0])
        self.assertGreater(r2, 0.9999)

    def test_falling_sigmoid_negative_k(self):
        # Negative k → falling sigmoid
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, -10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        K_fit = float(tf.data["FT_K"].nparray[0])
        self.assertLess(K_fit, 0.0)

    def test_errors_written_when_enabled(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        tool.results_errors = True
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        for name in ("FT_err_A", "FT_err_X50", "FT_err_K", "FT_err_Y0"):
            self.assertIn(name, tf.data, msg="%s missing" % name)

    def test_converged_is_one_for_exact_data(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        self.assertEqual(float(tf.data["FT_Converged"].nparray[0]), 1.0)

    def test_multi_epoch_array_length(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        d0 = _make_boltzmann_data(1.0, 50.0, 10.0, 0.0, name="recordA0")
        d1 = _make_boltzmann_data(2.0, 40.0,  8.0, 0.5, name="recordA1")
        folder = _run(tool, [d0, d1])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        self.assertEqual(len(tf.data["FT_A"].nparray), 2)

    def test_p0_override_used(self):
        tool = NMToolFit()
        tool.func_name = "boltzmann"
        tool.p0 = {"X50": 50.0, "K": 10.0}
        folder = _run(tool, [_make_boltzmann_data(1.0, 50.0, 10.0, 0.0)])
        tf = folder.toolfolders.get("Fit_Boltzmann_0")
        self.assertGreater(float(tf.data["FT_R2"].nparray[0]), 0.999)

    def test_config_accepts_boltzmann(self):
        cfg = NMToolFitConfig()
        cfg.func_name = "boltzmann"
        self.assertEqual(cfg.func_name, "boltzmann")


if __name__ == "__main__":
    unittest.main()
