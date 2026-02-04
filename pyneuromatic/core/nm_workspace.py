# -*- coding: utf-8 -*-
"""
NM Workspace - User workspace configuration management.

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

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# TOML compatibility layer
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore

import pyneuromatic.core.nm_utilities as nmu


def get_default_config_dir() -> Path:
    """
    Get the default configuration directory for pyNeuroMatic.

    Returns:
        Path to configuration directory:
        - Windows: %APPDATA%/pyneuromatic
        - macOS: ~/Library/Application Support/pyneuromatic
        - Linux: ~/.config/pyneuromatic (or XDG_CONFIG_HOME)
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and other Unix-like systems
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "pyneuromatic"


DEFAULT_CONFIG_DIR = get_default_config_dir()
DEFAULT_WORKSPACE_FILE = "workspace.toml"
WORKSPACES_SUBDIR = "workspaces"


@dataclass
class NMWorkspace:
    """
    Represents a workspace configuration.

    A workspace specifies which tools should be enabled and their configurations.

    Attributes:
        name: Workspace name (used for identification)
        description: Human-readable description
        tools: List of tool names to enable (e.g., ["stats", "spike"])
        tool_configs: Per-tool configuration dictionaries
    """

    name: str = "default"
    description: str = ""
    tools: list[str] = field(default_factory=lambda: ["main"])
    tool_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert workspace to dictionary for TOML serialization.

        Returns:
            Dictionary suitable for writing to TOML file
        """
        result: dict[str, Any] = {
            "workspace": {
                "name": self.name,
                "description": self.description,
            },
            "tools": {
                "enabled": self.tools,
            },
        }
        if self.tool_configs:
            result["tool_config"] = self.tool_configs
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NMWorkspace":
        """
        Create workspace from TOML dictionary.

        Args:
            data: Dictionary loaded from TOML file

        Returns:
            NMWorkspace instance
        """
        workspace_section = data.get("workspace", {})
        tools_section = data.get("tools", {})

        return cls(
            name=workspace_section.get("name", "default"),
            description=workspace_section.get("description", ""),
            tools=tools_section.get("enabled", ["main"]),
            tool_configs=data.get("tool_config", {}),
        )


class NMWorkspaceManager:
    """
    Manages workspace configuration files.

    Workspaces are stored as TOML files:
    - Default workspace: <config_dir>/workspace.toml
    - Named workspaces: <config_dir>/workspaces/<name>.toml

    Example:
        manager = NMWorkspaceManager()
        ws = manager.load("spike_analysis")
        print(ws.tools)  # ["stats", "spike"]

        # Save current config
        ws = NMWorkspace(name="custom", tools=["stats", "event"])
        manager.save(ws)
    """

    def __init__(
        self,
        config_dir: Path | str | None = None
    ) -> None:
        """
        Initialize workspace manager.

        Args:
            config_dir: Configuration directory path. Defaults to platform-specific
                       location (~/.config/pyneuromatic on Linux).

        Raises:
            TypeError: If config_dir is not a Path, string, or None
        """
        if config_dir is None:
            self._config_dir = DEFAULT_CONFIG_DIR
        elif isinstance(config_dir, str):
            self._config_dir = Path(config_dir)
        elif isinstance(config_dir, Path):
            self._config_dir = config_dir
        else:
            raise TypeError(nmu.type_error_str(config_dir, "config_dir", "Path or string"))

        self._current_workspace: NMWorkspace | None = None

    @property
    def config_dir(self) -> Path:
        """Get the configuration directory path."""
        return self._config_dir

    @property
    def current(self) -> NMWorkspace | None:
        """Get the currently loaded workspace."""
        return self._current_workspace

    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory and subdirectories exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        workspaces_dir = self._config_dir / WORKSPACES_SUBDIR
        workspaces_dir.mkdir(parents=True, exist_ok=True)

    def _get_workspace_path(self, name: str | None = None) -> Path:
        """
        Get path to workspace file.

        Args:
            name: Workspace name. None or "default" for default workspace.

        Returns:
            Path to the workspace TOML file
        """
        if name is None or name == "default":
            return self._config_dir / DEFAULT_WORKSPACE_FILE
        else:
            return self._config_dir / WORKSPACES_SUBDIR / f"{name}.toml"

    def load(self, name: str | None = None) -> NMWorkspace:
        """
        Load a workspace configuration.

        Args:
            name: Workspace name. None loads the default workspace.
                  If the file doesn't exist, returns a default workspace.

        Returns:
            Loaded NMWorkspace object

        Raises:
            RuntimeError: If tomllib/tomli is not available
        """
        if tomllib is None:
            raise RuntimeError(
                "TOML support not available. Install 'tomli' package for Python < 3.11"
            )

        workspace_path = self._get_workspace_path(name)

        if workspace_path.exists():
            with open(workspace_path, "rb") as f:
                data = tomllib.load(f)
            workspace = NMWorkspace.from_dict(data)
        else:
            # Return default workspace
            workspace = NMWorkspace(name=name or "default")

        self._current_workspace = workspace
        return workspace

    def save(
        self,
        workspace: NMWorkspace | None = None,
        name: str | None = None
    ) -> Path:
        """
        Save a workspace configuration.

        Args:
            workspace: Workspace to save. Uses current if None.
            name: Override workspace name for file location.

        Returns:
            Path to saved file

        Raises:
            ValueError: If no workspace to save
            RuntimeError: If tomli_w is not available
        """
        if tomli_w is None:
            raise RuntimeError(
                "TOML writing not available. Install 'tomli-w' package."
            )

        if workspace is None:
            workspace = self._current_workspace
        if workspace is None:
            raise ValueError("No workspace to save")

        self._ensure_config_dir()

        save_name = name or workspace.name
        workspace_path = self._get_workspace_path(save_name)

        # Ensure parent directory exists for named workspaces
        workspace_path.parent.mkdir(parents=True, exist_ok=True)

        with open(workspace_path, "wb") as f:
            tomli_w.dump(workspace.to_dict(), f)

        return workspace_path

    def available_workspaces(self) -> list[str]:
        """
        List all available workspace names.

        Returns:
            List of workspace names (includes "default" if exists)
        """
        workspaces = []

        # Check default workspace
        default_path = self._get_workspace_path(None)
        if default_path.exists():
            workspaces.append("default")

        # Check named workspaces
        workspaces_dir = self._config_dir / WORKSPACES_SUBDIR
        if workspaces_dir.exists():
            for f in workspaces_dir.glob("*.toml"):
                workspaces.append(f.stem)

        return sorted(workspaces)

    def delete_workspace(self, name: str) -> bool:
        """
        Delete a workspace configuration file.

        Args:
            name: Workspace name to delete

        Returns:
            True if deleted, False if not found
        """
        workspace_path = self._get_workspace_path(name)
        if workspace_path.exists():
            workspace_path.unlink()
            return True
        return False

    def workspace_exists(self, name: str | None = None) -> bool:
        """
        Check if a workspace configuration file exists.

        Args:
            name: Workspace name to check

        Returns:
            True if workspace file exists
        """
        return self._get_workspace_path(name).exists()
