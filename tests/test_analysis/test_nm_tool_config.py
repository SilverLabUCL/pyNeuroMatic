#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_tool_config: NMToolConfig and NMToolStatsConfig.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import tempfile
import unittest
from pathlib import Path

from pyneuromatic.analysis.nm_tool_config import NMToolConfig


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing the base class in isolation
# ---------------------------------------------------------------------------

class _SampleConfig(NMToolConfig):
    _TOML_TYPE = "sample_config"
    _schema = {
        "flag":   {"type": bool,  "default": True},
        "count":  {"type": int,   "default": 5, "min": 0, "max": 100},
        "ratio":  {"type": float, "default": 0.5, "min": 0.0, "max": 1.0},
        "method": {"type": str,   "default": "auto",
                   "choices": ["auto", "fast", "slow"]},
    }


# =========================================================================
# NMToolConfig base class
# =========================================================================

class TestNMToolConfigDefaults(unittest.TestCase):
    """NMToolConfig sets defaults from schema on construction."""

    def setUp(self):
        self.cfg = _SampleConfig()

    def test_bool_default(self):
        self.assertTrue(self.cfg.flag)

    def test_int_default(self):
        self.assertEqual(self.cfg.count, 5)

    def test_float_default(self):
        self.assertAlmostEqual(self.cfg.ratio, 0.5)

    def test_str_default(self):
        self.assertEqual(self.cfg.method, "auto")


class TestNMToolConfigSetAttr(unittest.TestCase):
    """NMToolConfig validates values on set."""

    def setUp(self):
        self.cfg = _SampleConfig()

    def test_set_valid_bool(self):
        self.cfg.flag = False
        self.assertFalse(self.cfg.flag)

    def test_set_valid_int(self):
        self.cfg.count = 10
        self.assertEqual(self.cfg.count, 10)

    def test_set_valid_float(self):
        self.cfg.ratio = 0.9
        self.assertAlmostEqual(self.cfg.ratio, 0.9)

    def test_set_valid_str_choice(self):
        self.cfg.method = "fast"
        self.assertEqual(self.cfg.method, "fast")

    def test_unknown_key_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            self.cfg.nonexistent = 1

    def test_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.cfg.count = "ten"

    def test_bool_rejected_for_int_param(self):
        # bool is subclass of int — must be rejected when int expected
        with self.assertRaises(TypeError):
            self.cfg.count = True

    def test_bool_rejected_for_float_param(self):
        with self.assertRaises(TypeError):
            self.cfg.ratio = True

    def test_below_min_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cfg.count = -1

    def test_above_max_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cfg.count = 101

    def test_invalid_choice_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cfg.method = "turbo"


class TestNMToolConfigToDict(unittest.TestCase):
    """NMToolConfig.to_dict() produces correct output."""

    def setUp(self):
        self.cfg = _SampleConfig()

    def test_pyneuromatic_header_present(self):
        d = self.cfg.to_dict()
        self.assertIn("pyneuromatic", d)

    def test_pyneuromatic_type(self):
        d = self.cfg.to_dict()
        self.assertEqual(d["pyneuromatic"]["type"], "sample_config")

    def test_pyneuromatic_version(self):
        d = self.cfg.to_dict()
        self.assertIn("version", d["pyneuromatic"])

    def test_all_schema_keys_present(self):
        d = self.cfg.to_dict()
        for key in _SampleConfig._schema:
            self.assertIn(key, d)

    def test_values_match_attributes(self):
        self.cfg.count = 42
        d = self.cfg.to_dict()
        self.assertEqual(d["count"], 42)


class TestNMToolConfigFromDict(unittest.TestCase):
    """NMToolConfig.from_dict() reconstructs and validates."""

    def test_round_trip(self):
        cfg = _SampleConfig()
        cfg.count = 77
        cfg.method = "slow"
        cfg2 = _SampleConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg2.count, 77)
        self.assertEqual(cfg2.method, "slow")

    def test_unknown_keys_ignored(self):
        d = _SampleConfig().to_dict()
        d["unknown_param"] = "ignored"
        cfg = _SampleConfig.from_dict(d)   # no exception
        self.assertFalse(hasattr(cfg, "unknown_param"))

    def test_missing_keys_use_defaults(self):
        cfg = _SampleConfig.from_dict({})
        self.assertEqual(cfg.count, 5)

    def test_invalid_value_raises(self):
        d = _SampleConfig().to_dict()
        d["count"] = -99
        with self.assertRaises(ValueError):
            _SampleConfig.from_dict(d)


class TestNMToolConfigEquality(unittest.TestCase):

    def test_equal_configs(self):
        a = _SampleConfig()
        b = _SampleConfig()
        self.assertEqual(a, b)

    def test_unequal_configs(self):
        a = _SampleConfig()
        b = _SampleConfig()
        b.count = 99
        self.assertNotEqual(a, b)

    def test_different_type_not_equal(self):
        a = _SampleConfig()
        self.assertNotEqual(a, "not a config")


class TestNMToolConfigRepr(unittest.TestCase):

    def test_repr_contains_class_name(self):
        self.assertIn("_SampleConfig", repr(_SampleConfig()))

    def test_repr_contains_param(self):
        self.assertIn("count=5", repr(_SampleConfig()))


class TestNMToolConfigSaveLoad(unittest.TestCase):
    """NMToolConfig save/load TOML round-trip."""

    def test_save_creates_file(self):
        cfg = _SampleConfig()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.toml"
            cfg.save(path)
            self.assertTrue(path.exists())

    def test_save_returns_path(self):
        cfg = _SampleConfig()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.toml"
            result = cfg.save(path)
            self.assertEqual(result, path)

    def test_load_round_trips_values(self):
        cfg = _SampleConfig()
        cfg.count = 42
        cfg.method = "slow"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.toml"
            cfg.save(path)
            cfg2 = _SampleConfig.load(path)
        self.assertEqual(cfg2.count, 42)
        self.assertEqual(cfg2.method, "slow")

    def test_load_wrong_type_raises(self):
        import tomli_w
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.toml"
            with open(path, "wb") as f:
                tomli_w.dump({"pyneuromatic": {"type": "wrong_type"}}, f)
            with self.assertRaises(ValueError):
                _SampleConfig.load(path)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            _SampleConfig.load("/nonexistent/cfg.toml")


if __name__ == "__main__":
    unittest.main(verbosity=2)
