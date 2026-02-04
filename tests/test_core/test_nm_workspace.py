#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for NMWorkspace and NMWorkspaceManager.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from pyneuromatic.core.nm_workspace import (
    NMWorkspace,
    NMWorkspaceManager,
)


class TestNMWorkspace(unittest.TestCase):
    """Tests for NMWorkspace dataclass."""

    def test_default_workspace(self):
        ws = NMWorkspace()
        self.assertEqual(ws.name, "default")
        self.assertEqual(ws.description, "")
        self.assertEqual(ws.tools, ["main"])
        self.assertEqual(ws.tool_configs, {})

    def test_custom_workspace(self):
        ws = NMWorkspace(
            name="test",
            description="Test workspace",
            tools=["stats", "spike"],
            tool_configs={"spike": {"threshold": -20.0}}
        )
        self.assertEqual(ws.name, "test")
        self.assertEqual(ws.description, "Test workspace")
        self.assertEqual(ws.tools, ["stats", "spike"])
        self.assertEqual(ws.tool_configs["spike"]["threshold"], -20.0)

    def test_to_dict(self):
        ws = NMWorkspace(
            name="test",
            description="Test workspace",
            tools=["stats", "spike"]
        )
        d = ws.to_dict()
        self.assertEqual(d["workspace"]["name"], "test")
        self.assertEqual(d["workspace"]["description"], "Test workspace")
        self.assertEqual(d["tools"]["enabled"], ["stats", "spike"])

    def test_to_dict_with_tool_configs(self):
        ws = NMWorkspace(
            name="test",
            tools=["stats"],
            tool_configs={"stats": {"xclip": True}}
        )
        d = ws.to_dict()
        self.assertIn("tool_config", d)
        self.assertEqual(d["tool_config"]["stats"]["xclip"], True)

    def test_to_dict_without_tool_configs(self):
        ws = NMWorkspace(name="test", tools=["stats"])
        d = ws.to_dict()
        self.assertNotIn("tool_config", d)

    def test_from_dict(self):
        data = {
            "workspace": {"name": "test", "description": "desc"},
            "tools": {"enabled": ["stats", "spike"]},
            "tool_config": {"spike": {"threshold": -20.0}},
        }
        ws = NMWorkspace.from_dict(data)
        self.assertEqual(ws.name, "test")
        self.assertEqual(ws.description, "desc")
        self.assertEqual(ws.tools, ["stats", "spike"])
        self.assertEqual(ws.tool_configs["spike"]["threshold"], -20.0)

    def test_from_dict_minimal(self):
        data = {}
        ws = NMWorkspace.from_dict(data)
        self.assertEqual(ws.name, "default")
        self.assertEqual(ws.tools, ["main"])

    def test_from_dict_partial(self):
        data = {
            "workspace": {"name": "partial"},
            # Missing tools section
        }
        ws = NMWorkspace.from_dict(data)
        self.assertEqual(ws.name, "partial")
        self.assertEqual(ws.tools, ["main"])  # Default


class TestNMWorkspaceManager(unittest.TestCase):
    """Tests for NMWorkspaceManager class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = NMWorkspaceManager(config_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_dir_property(self):
        self.assertEqual(self.manager.config_dir, Path(self.temp_dir))

    def test_config_dir_string(self):
        manager = NMWorkspaceManager(config_dir=self.temp_dir)
        self.assertEqual(manager.config_dir, Path(self.temp_dir))

    def test_config_dir_path(self):
        manager = NMWorkspaceManager(config_dir=Path(self.temp_dir))
        self.assertEqual(manager.config_dir, Path(self.temp_dir))

    def test_config_dir_none_uses_default(self):
        manager = NMWorkspaceManager(config_dir=None)
        # Should not raise, uses default
        self.assertIsInstance(manager.config_dir, Path)

    def test_config_dir_type_error(self):
        with self.assertRaises(TypeError):
            NMWorkspaceManager(config_dir=123)
        with self.assertRaises(TypeError):
            NMWorkspaceManager(config_dir=[])

    def test_load_default_when_missing(self):
        ws = self.manager.load()
        self.assertEqual(ws.name, "default")
        self.assertEqual(ws.tools, ["main"])

    def test_load_named_when_missing(self):
        ws = self.manager.load("nonexistent")
        self.assertEqual(ws.name, "nonexistent")
        self.assertEqual(ws.tools, ["main"])

    def test_current_property(self):
        self.assertIsNone(self.manager.current)
        ws = self.manager.load()
        self.assertIs(self.manager.current, ws)

    def test_save_and_load_default(self):
        ws = NMWorkspace(name="default", tools=["stats", "spike"])
        self.manager.save(ws)

        loaded = self.manager.load()
        self.assertEqual(loaded.name, "default")
        self.assertEqual(loaded.tools, ["stats", "spike"])

    def test_save_and_load_named(self):
        ws = NMWorkspace(name="custom", tools=["stats", "event"])
        self.manager.save(ws, "custom")

        loaded = self.manager.load("custom")
        self.assertEqual(loaded.name, "custom")
        self.assertEqual(loaded.tools, ["stats", "event"])

    def test_save_current_workspace(self):
        self.manager.load()  # Sets current
        self.manager.current.tools.append("spike")
        path = self.manager.save()  # Save current
        self.assertTrue(path.exists())

    def test_save_no_workspace_error(self):
        manager = NMWorkspaceManager(config_dir=self.temp_dir)
        with self.assertRaises(ValueError):
            manager.save()  # No workspace loaded

    def test_save_with_tool_configs(self):
        ws = NMWorkspace(
            name="configured",
            tools=["stats"],
            tool_configs={"stats": {"xclip": True, "value": 42}}
        )
        self.manager.save(ws, "configured")

        loaded = self.manager.load("configured")
        self.assertEqual(loaded.tool_configs["stats"]["xclip"], True)
        self.assertEqual(loaded.tool_configs["stats"]["value"], 42)

    def test_available_workspaces_empty(self):
        workspaces = self.manager.available_workspaces()
        self.assertEqual(workspaces, [])

    def test_available_workspaces_with_default(self):
        self.manager.save(NMWorkspace())
        workspaces = self.manager.available_workspaces()
        self.assertIn("default", workspaces)

    def test_available_workspaces_with_named(self):
        self.manager.save(NMWorkspace(), "default")
        self.manager.save(NMWorkspace(name="custom"), "custom")
        self.manager.save(NMWorkspace(name="another"), "another")

        workspaces = self.manager.available_workspaces()
        self.assertIn("default", workspaces)
        self.assertIn("custom", workspaces)
        self.assertIn("another", workspaces)
        self.assertEqual(len(workspaces), 3)

    def test_workspace_exists(self):
        self.assertFalse(self.manager.workspace_exists())
        self.assertFalse(self.manager.workspace_exists("custom"))

        self.manager.save(NMWorkspace())
        self.assertTrue(self.manager.workspace_exists())
        self.assertFalse(self.manager.workspace_exists("custom"))

        self.manager.save(NMWorkspace(name="custom"), "custom")
        self.assertTrue(self.manager.workspace_exists("custom"))

    def test_delete_workspace(self):
        self.manager.save(NMWorkspace(name="temp"), "temp")
        self.assertTrue(self.manager.workspace_exists("temp"))

        result = self.manager.delete_workspace("temp")
        self.assertTrue(result)
        self.assertFalse(self.manager.workspace_exists("temp"))

        # Delete again returns False
        result = self.manager.delete_workspace("temp")
        self.assertFalse(result)

    def test_delete_default_workspace(self):
        self.manager.save(NMWorkspace())
        self.assertTrue(self.manager.workspace_exists())

        result = self.manager.delete_workspace("default")
        self.assertTrue(result)
        self.assertFalse(self.manager.workspace_exists())


class TestNMWorkspaceRoundTrip(unittest.TestCase):
    """Integration tests for workspace save/load round trips."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = NMWorkspaceManager(config_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complex_workspace_roundtrip(self):
        original = NMWorkspace(
            name="complex",
            description="A complex workspace for testing",
            tools=["stats", "spike", "event"],
            tool_configs={
                "stats": {"xclip": True, "ignore_nans": False},
                "spike": {"threshold": -20.0, "refractory_ms": 2.0},
                "event": {"detection_method": "threshold"},
            }
        )

        self.manager.save(original, "complex")
        loaded = self.manager.load("complex")

        self.assertEqual(loaded.name, original.name)
        self.assertEqual(loaded.description, original.description)
        self.assertEqual(loaded.tools, original.tools)
        self.assertEqual(loaded.tool_configs, original.tool_configs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
