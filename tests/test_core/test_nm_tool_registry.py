#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMToolRegistry.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import unittest

from pyneuromatic.core.nm_tool_registry import (
    DEFAULT_TOOL_REGISTRY,
    NMToolRegistry,
    get_global_registry,
    reset_global_registry,
)


class TestNMToolRegistry(unittest.TestCase):
    """Tests for NMToolRegistry class."""

    def setUp(self):
        self.registry = NMToolRegistry()

    def test_register_tool(self):
        self.registry.register(
            "test",
            "pyneuromatic.analysis.nm_tool_stats",
            "NMToolStats"
        )
        self.assertIn("test", self.registry)
        self.assertEqual(len(self.registry), 1)

    def test_register_type_errors(self):
        bad_types = [None, 3, True, [], {}]
        for bad in bad_types:
            with self.assertRaises(TypeError):
                self.registry.register(bad, "module", "Class")
            with self.assertRaises(TypeError):
                self.registry.register("name", bad, "Class")
            with self.assertRaises(TypeError):
                self.registry.register("name", "module", bad)

    def test_unregister_tool(self):
        self.registry.register("test", "mod", "Class")
        self.assertTrue(self.registry.unregister("test"))
        self.assertFalse(self.registry.unregister("test"))
        self.assertNotIn("test", self.registry)

    def test_load_tool(self):
        self.registry.register(
            "stats",
            "pyneuromatic.analysis.nm_tool_stats",
            "NMToolStats"
        )
        tool = self.registry.load("stats")
        from pyneuromatic.analysis.nm_tool_stats import NMToolStats
        self.assertIsInstance(tool, NMToolStats)

    def test_load_creates_new_instance(self):
        self.registry.register(
            "stats",
            "pyneuromatic.analysis.nm_tool_stats",
            "NMToolStats"
        )
        tool1 = self.registry.load("stats")
        tool2 = self.registry.load("stats")
        self.assertIsNot(tool1, tool2)  # Each load creates new instance

    def test_load_nonexistent_tool(self):
        with self.assertRaises(KeyError):
            self.registry.load("nonexistent")

    def test_load_invalid_module(self):
        self.registry.register("bad", "nonexistent.module", "Class")
        with self.assertRaises(ModuleNotFoundError):
            self.registry.load("bad")

    def test_load_invalid_class(self):
        self.registry.register(
            "bad",
            "pyneuromatic.analysis.nm_tool_stats",
            "NonexistentClass"
        )
        with self.assertRaises(AttributeError):
            self.registry.load("bad")

    def test_case_insensitivity(self):
        self.registry.register("Stats", "module", "Class")
        self.assertIn("stats", self.registry)
        self.assertIn("STATS", self.registry)
        self.assertIn("Stats", self.registry)

    def test_keys(self):
        self.registry.register("a", "mod", "Class")
        self.registry.register("b", "mod", "Class")
        self.assertEqual(sorted(self.registry.keys()), ["a", "b"])

    def test_len(self):
        self.assertEqual(len(self.registry), 0)
        self.registry.register("a", "mod", "Class")
        self.assertEqual(len(self.registry), 1)
        self.registry.register("b", "mod", "Class")
        self.assertEqual(len(self.registry), 2)

    def test_iter(self):
        self.registry.register("a", "mod", "Class")
        self.registry.register("b", "mod", "Class")
        names = list(self.registry)
        self.assertEqual(sorted(names), ["a", "b"])

    def test_is_loaded(self):
        self.registry.register(
            "stats",
            "pyneuromatic.analysis.nm_tool_stats",
            "NMToolStats"
        )
        self.assertFalse(self.registry.is_loaded("stats"))
        self.registry.load("stats")
        self.assertTrue(self.registry.is_loaded("stats"))

    def test_get_class(self):
        self.registry.register(
            "stats",
            "pyneuromatic.analysis.nm_tool_stats",
            "NMToolStats"
        )
        from pyneuromatic.analysis.nm_tool_stats import NMToolStats
        cls = self.registry.get_class("stats")
        self.assertIs(cls, NMToolStats)

    def test_get_info(self):
        self.registry.register(
            "stats",
            "pyneuromatic.analysis.nm_tool_stats",
            "NMToolStats"
        )
        info = self.registry.get_info("stats")
        self.assertEqual(info["name"], "stats")
        self.assertEqual(info["module"], "pyneuromatic.analysis.nm_tool_stats")
        self.assertEqual(info["class"], "NMToolStats")

    def test_get_info_nonexistent(self):
        with self.assertRaises(KeyError):
            self.registry.get_info("nonexistent")

    def test_contains_non_string(self):
        self.assertFalse(None in self.registry)
        self.assertFalse(123 in self.registry)
        self.assertFalse([] in self.registry)


class TestGlobalRegistry(unittest.TestCase):
    """Tests for global registry functions."""

    def setUp(self):
        reset_global_registry()

    def tearDown(self):
        reset_global_registry()

    def test_global_registry_initialized_with_defaults(self):
        registry = get_global_registry()
        for name in DEFAULT_TOOL_REGISTRY:
            self.assertIn(name, registry)

    def test_global_registry_singleton(self):
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        self.assertIs(registry1, registry2)

    def test_reset_global_registry(self):
        registry1 = get_global_registry()
        reset_global_registry()
        registry2 = get_global_registry()
        self.assertIsNot(registry1, registry2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
