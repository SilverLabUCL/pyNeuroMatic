# -*- coding: utf-8 -*-
"""
NM Tool Registry - Central registry for lazy tool loading.

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

import importlib
from typing import TYPE_CHECKING

import pyneuromatic.core.nm_utilities as nmu

if TYPE_CHECKING:
    from pyneuromatic.analysis.nm_tool import NMTool


# Default registry: maps tool names to (module_path, class_name)
DEFAULT_TOOL_REGISTRY: dict[str, tuple[str, str]] = {
    "main": ("pyneuromatic.analysis.nm_tool_main", "NMToolMain"),
    "stats": ("pyneuromatic.analysis.nm_tool_stats", "NMToolStats"),
    # Future tools (uncomment when implemented):
    # "spike": ("pyneuromatic.analysis.nm_tool_spike", "NMToolSpike"),
    # "event": ("pyneuromatic.analysis.nm_tool_event", "NMToolEvent"),
    # "fit": ("pyneuromatic.analysis.nm_tool_fit", "NMToolFit"),
    # "pulse": ("pyneuromatic.analysis.nm_tool_pulse", "NMToolPulse"),
    # "model": ("pyneuromatic.analysis.nm_tool_model", "NMToolModel"),
    # "clamp": ("pyneuromatic.acquisition.nm_tool_clamp", "NMToolClamp"),
}


class NMToolRegistry:
    """
    Registry for NM tools with lazy loading support.

    Tools are registered with their module path and class name, but are only
    imported when explicitly loaded. This allows heavy tools (like acquisition)
    to be available but not loaded unless needed.

    Example:
        registry = NMToolRegistry()
        registry.register("stats", "pyneuromatic.analysis.nm_tool_stats", "NMToolStats")
        tool = registry.load("stats")  # Import happens here
    """

    def __init__(self) -> None:
        self._registry: dict[str, tuple[str, str]] = {}
        self._loaded_classes: dict[str, type] = {}  # Cache loaded classes

    def register(
        self,
        name: str,
        module_path: str,
        class_name: str
    ) -> None:
        """
        Register a tool for lazy loading.

        Args:
            name: Tool name (case-insensitive key, e.g., "stats")
            module_path: Full module path (e.g., "pyneuromatic.analysis.nm_tool_stats")
            class_name: Class name within module (e.g., "NMToolStats")

        Raises:
            TypeError: If any argument is not a string
        """
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        if not isinstance(module_path, str):
            raise TypeError(nmu.type_error_str(module_path, "module_path", "string"))
        if not isinstance(class_name, str):
            raise TypeError(nmu.type_error_str(class_name, "class_name", "string"))
        self._registry[name.lower()] = (module_path, class_name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        key = name.lower()
        if key in self._registry:
            del self._registry[key]
            if key in self._loaded_classes:
                del self._loaded_classes[key]
            return True
        return False

    def load(self, name: str) -> "NMTool":
        """
        Load and instantiate a tool by name.

        Args:
            name: Tool name (case-insensitive)

        Returns:
            Instantiated NMTool object

        Raises:
            KeyError: If tool not in registry
            ImportError: If module cannot be imported
            AttributeError: If class not found in module
        """
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")

        module_path, class_name = self._registry[key]

        # Check class cache first
        if key not in self._loaded_classes:
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            self._loaded_classes[key] = tool_class

        # Instantiate and return
        return self._loaded_classes[key]()

    def get_class(self, name: str) -> type:
        """
        Get the tool class without instantiating.

        Args:
            name: Tool name (case-insensitive)

        Returns:
            The tool class

        Raises:
            KeyError: If tool not in registry
        """
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")

        if key not in self._loaded_classes:
            module_path, class_name = self._registry[key]
            module = importlib.import_module(module_path)
            self._loaded_classes[key] = getattr(module, class_name)

        return self._loaded_classes[key]

    def get_info(self, name: str) -> dict[str, str]:
        """
        Get tool registration info without loading it.

        Args:
            name: Tool name

        Returns:
            Dictionary with 'name', 'module', 'class' keys

        Raises:
            KeyError: If tool not in registry
        """
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")
        module_path, class_name = self._registry[key]
        return {
            "name": key,
            "module": module_path,
            "class": class_name,
        }

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered (case-insensitive)."""
        if not isinstance(name, str):
            return False
        return name.lower() in self._registry

    def __iter__(self):
        """Iterate over registered tool names."""
        return iter(self._registry.keys())

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._registry)

    def keys(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._registry.keys())

    def is_loaded(self, name: str) -> bool:
        """
        Check if a tool class has been loaded (imported).

        Args:
            name: Tool name

        Returns:
            True if the tool's class has been imported
        """
        return name.lower() in self._loaded_classes


# Global registry instance
_global_registry: NMToolRegistry | None = None


def get_global_registry() -> NMToolRegistry:
    """
    Get the global tool registry, initializing with defaults if needed.

    Returns:
        The global NMToolRegistry instance with default tools registered
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = NMToolRegistry()
        for name, (module_path, class_name) in DEFAULT_TOOL_REGISTRY.items():
            _global_registry.register(name, module_path, class_name)
    return _global_registry


def reset_global_registry() -> None:
    """
    Reset the global registry to None.

    Useful for testing to ensure a fresh registry.
    """
    global _global_registry
    _global_registry = None
