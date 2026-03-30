# -*- coding: utf-8 -*-
"""
[Module description].

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
import datetime
# import math
# import matplotlib
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_folder import NMFolder, NMFolderContainer
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_command_history as nmch
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_configurations as nmc
from pyneuromatic.core.nm_tool_registry import get_global_registry, NMToolRegistry
from pyneuromatic.core.nm_workspace import NMWorkspace, NMWorkspaceManager
import pyneuromatic.core.nm_utilities as nmu

if TYPE_CHECKING:
    from pyneuromatic.analysis.nm_tool import NMTool

nm = None  # holds Manager, accessed via console

# Hierarchy tier selection keys (folder down to epoch)
HIERARCHY_SELECT_KEYS = ("folder", "data", "dataseries", "channel", "epoch")

# Run target constants (consistent with RunMode in NMObjectContainer)
RUN_SELECTED = "selected"
RUN_ALL = "all"

"""
NM class tree:

NMManager (NMObject, root)
    NMFolderContainer
        NMFolder (folder0, folder1...)
            NMDataContainer
                NMData (recordA0, recordA1... avgA0, avgB0)
            NMDataSeriesContainer
                NMDataSeries (record, avg...)
                    NMChannelContainer
                        NMChannel (A, B, C...)
                    NMEpochContainer
                        NMEpoch (E0, E1, E2...)
"""


class NMManager(NMObject):
    """
    NM Manager - Central manager for pyNeuroMatic.

    Manages folders, tools, and workspaces. Tools are loaded lazily from the
    registry based on the active workspace configuration.

    NMManager is the root of the NM object hierarchy (name = nm_name).

    Args:
        quiet: Suppress history messages
        workspace: Workspace name to load (None for default)
        config_dir: Custom configuration directory (None for platform default)
        nm_name: Variable name used in command history (default "nm")

    Example:
        nm = NMManager(workspace="spike_analysis")
        nm.stats.windows.get("w0").func = "mean"
        nm.run_tool()
    """

    def __init__(
        self,
        quiet: bool = False,
        workspace: str | None = None,
        config_dir: str | Path | None = None,
        nm_name: str = "nm",
        command_history: bool = False,
    ) -> None:
        # Initialize centralized history logging
        self.__history = nmh.NMHistory(quiet=quiet)
        nmh.set_history(self.__history)

        # Initialize centralized command history (disabled by default; GUI enables explicitly)
        self.__command_history = nmch.NMCommandHistory(
            enabled=command_history, quiet=quiet, nm_name=nm_name
        )
        nmch.set_command_history(self.__command_history)
        nmch.add_command(nm_name + " = NMManager()")

        # NMManager is the root NMObject (name = nm_name)
        super().__init__(parent=None, name=nm_name)

        # Create folder container directly (NMManager is the root)
        self.__folders = NMFolderContainer(parent=self)

        # Initialize tool registry and workspace manager
        self._tool_registry: NMToolRegistry = get_global_registry()
        self._workspace_manager: NMWorkspaceManager = NMWorkspaceManager(
            config_dir=config_dir
        )

        self.__toolkit: dict[str, "NMTool"] = {}
        self.__toolselect: str | None = None
        self.__run_config: dict | None = None

        # Load workspace (or default)
        self._load_workspace_internal(workspace, quiet=quiet)

        tstr = str(datetime.datetime.now())
        nmh.history(f"created NM manager {tstr}", quiet=quiet)
        nmh.history("current NM tool is '%s'" % self.__toolselect, quiet=quiet)

    def _load_workspace_internal(
        self,
        name: str | None = None,
        quiet: bool = False
    ) -> None:
        """Internal: Load workspace and enable its tools."""
        workspace = self._workspace_manager.load(name)

        # Clear existing tools
        self.__toolkit.clear()
        self.__toolselect = None

        # Enable tools from workspace
        for tool_name in workspace.tools:
            if tool_name in self._tool_registry:
                try:
                    self.tool_add(tool_name)
                except Exception as e:
                    nmh.history(
                        f"Warning: Failed to load tool '{tool_name}': {e}",
                        quiet=quiet
                    )
            else:
                nmh.history(
                    f"Warning: Tool '{tool_name}' not found in registry, skipping",
                    quiet=quiet
                )

        # Select first tool if any, prefer "main"
        if "main" in self.__toolkit:
            self.__toolselect = "main"
        elif self.__toolkit:
            self.__toolselect = next(iter(self.__toolkit.keys()))

    def tool_add(
        self,
        toolname: str,
        tool: "NMTool | None" = None,
        select: bool = False
    ) -> None:
        """
        Add a tool to the toolkit.

        Args:
            toolname: Name of the tool
            tool: Optional NMTool instance. If None, loads from registry.
            select: Whether to select this tool as active

        Raises:
            TypeError: If toolname is not a string or tool is wrong type
            KeyError: If tool not found in registry (when tool is None)
        """
        from pyneuromatic.analysis.nm_tool import NMTool

        if not isinstance(toolname, str):
            e = nmu.type_error_str(toolname, "toolname", "string")
            raise TypeError(e)

        tname = toolname.lower()

        if tool is None:
            # Lazy load from registry
            if tname not in self._tool_registry:
                raise KeyError(f"Tool '{toolname}' not found in registry")
            tool = self._tool_registry.load(tname)
        elif not isinstance(tool, NMTool):
            e = f"tool '{toolname}' is not an instance of NMTool"
            raise TypeError(e)

        self.__toolkit[tname] = tool
        if select:
            self.__toolselect = tname
        if select:
            nmch.add_nm_command("tool_add(%r, select=True)" % tname)
        else:
            nmch.add_nm_command("tool_add(%r)" % tname)

    def tool_remove(self, toolname: str) -> bool:
        """
        Remove a tool from the toolkit.

        Args:
            toolname: Name of tool to remove

        Returns:
            True if removed, False if not found
        """
        tname = toolname.lower()
        if tname in self.__toolkit:
            del self.__toolkit[tname]
            if self.__toolselect == tname:
                # Select next available tool or None
                if self.__toolkit:
                    self.__toolselect = next(iter(self.__toolkit.keys()))
                else:
                    self.__toolselect = None
            nmch.add_nm_command("tool_remove(%r)" % tname)
            return True
        return False

    @property
    def tool_select(self) -> str | None:
        return self.__toolselect

    @tool_select.setter
    def tool_select(self, toolname: str) -> None:
        if toolname is None:
            self.__toolselect = None
            return None
        tname = toolname.lower()
        if tname in self.__toolkit:
            self.__toolselect = tname
            nmch.add_nm_command("tool_select = %r" % tname)
        else:
            raise KeyError("NM tool key '%s' does not exist" % tname)

    @property
    def history(self) -> nmh.NMHistory:
        """Access the centralized NM history log."""
        return self.__history

    @property
    def command_history(self) -> nmch.NMCommandHistory:
        """Access the centralized NM command history."""
        return self.__command_history

    # override: deep copy not supported (NMManager holds global singletons
    # like history and command history that cannot be meaningfully duplicated)
    def __deepcopy__(self, memo: dict) -> None:
        raise NotImplementedError("NMManager cannot be deep-copied")

    # override: identity-based equality (NMManager is a root object)
    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def folders(self) -> NMFolderContainer:
        return self.__folders

    # override
    @property
    def content(self) -> dict:
        k = super().content
        if self.__folders is not None:
            k.update(self.__folders.content)
        return k

    def _iter_select_hierarchy(self):
        """
        Generator yielding (tier_name, container, selected_value) tuples.

        Traverses the selection hierarchy from folder down to epoch.
        Stops when a required parent container has no selection.

        Yields:
            Tuples of (tier_name, container, selected_value) where:
            - tier_name: one of HIERARCHY_SELECT_KEYS
            - container: the NMObjectContainer at this tier
            - selected_value: the currently selected object (or None)
        """
        folders = self.__folders
        if folders is None:
            return

        f = folders.selected_value
        yield ("folder", folders, f)

        if not isinstance(f, NMFolder):
            return

        # Data and DataSeries are siblings at folder tier
        yield ("data", f.data, f.data.selected_value)

        ds = f.dataseries.selected_value
        yield ("dataseries", f.dataseries, ds)

        if not isinstance(ds, NMDataSeries):
            return

        # Channel and Epoch are siblings at dataseries tier
        yield ("channel", ds.channels, ds.channels.selected_value)
        yield ("epoch", ds.epochs, ds.epochs.selected_value)

    @property
    def select_values(self) -> dict[str, NMObject | None]:
        """Get the currently selected object at each hierarchy tier."""
        result: dict[str, NMObject | None] = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        for tier, container, value in self._iter_select_hierarchy():
            result[tier] = value
        return result

    @property
    def select_keys(self) -> dict[str, str | None]:
        """Get the names of currently selected objects at each hierarchy tier."""
        result: dict[str, str | None] = {tier: None for tier in HIERARCHY_SELECT_KEYS}
        for tier, container, value in self._iter_select_hierarchy():
            if isinstance(value, NMObject):
                result[tier] = value.name
        return result

    @select_keys.setter
    def select_keys(self, select: dict[str, str]) -> None:
        self._select_keys_set(select)

    def _select_keys_set(self, select: dict[str, str]) -> None:
        """
        Set selection at each hierarchy tier by name.

        Args:
            select: Dictionary mapping tier names to object names.
                    Keys must be from HIERARCHY_SELECT_KEYS (folder, data,
                    dataseries, channel, epoch).

        Raises:
            TypeError: If select is not a dict or values aren't strings
            KeyError: If a key is not a valid selection tier
        """
        if not isinstance(select, dict):
            raise TypeError(nmu.type_error_str(select, "select", "dictionary"))

        for tier, value in select.items():
            if not isinstance(tier, str):
                raise TypeError(nmu.type_error_str(tier, "key", "string"))

        # Normalize keys to lowercase for case-insensitive matching
        select = {k.lower(): v for k, v in select.items()}

        for tier, value in select.items():
            if value is not None and not isinstance(value, str):
                raise TypeError(nmu.type_error_str(value, "value", "string"))
            if tier not in HIERARCHY_SELECT_KEYS:
                raise KeyError(f"'{tier}' is not a valid selection tier. "
                               f"Valid tiers: {HIERARCHY_SELECT_KEYS}")

        nmch.add_nm_command("select_keys = %r" % (select,))

        # Traverse hierarchy, setting values as we go so subsequent tiers
        # use the newly selected parent
        folders = self.__folders
        if folders is None:
            return

        if "folder" in select:
            folders._selected_name_set(select["folder"])
        f = folders.selected_value
        if not isinstance(f, NMFolder):
            return

        if "data" in select:
            f.data._selected_name_set(select["data"])
        if "dataseries" in select:
            f.dataseries._selected_name_set(select["dataseries"])

        ds = f.dataseries.selected_value
        if not isinstance(ds, NMDataSeries):
            return

        if "channel" in select:
            ds.channels._selected_name_set(select["channel"])
        if "epoch" in select:
            ds.epochs._selected_name_set(select["epoch"])

    def select_value_set(self, obj: NMObject) -> None:
        """
        Set selection by object reference, auto-populating parent hierarchy.

        Traverses up the parent chain from the given object to set selection
        at each hierarchy tier. Items below the specified tier retain their
        current selection within the newly selected parent.

        Args:
            obj: An NMObject from the hierarchy (NMFolder, NMData, NMDataSeries,
                 NMChannel, or NMEpoch).

        Raises:
            TypeError: If obj is not an NMObject
            ValueError: If obj is not part of this manager's hierarchy or
                       is not a recognized hierarchy type

        Examples:
            >>> nm.select_value_set(some_epoch)  # Sets folder, dataseries, channel, epoch
            >>> nm.select_value_set(some_dataseries)  # Sets folder, dataseries
        """
        if not isinstance(obj, NMObject):
            raise TypeError(nmu.type_error_str(obj, "obj", "NMObject"))

        # Verify object belongs to this manager's hierarchy
        obj_manager = obj._manager
        if obj_manager is not self:
            raise ValueError(
                f"object '{obj.name}' is not part of this manager's hierarchy"
            )

        # Build selection dict by traversing up the parent chain
        # Note: All objects skip their container in the parent chain:
        # - NMEpoch._parent = NMDataSeries (not NMEpochContainer)
        # - NMChannel._parent = NMDataSeries (not NMChannelContainer)
        # - NMDataSeries._parent = NMFolder (not NMDataSeriesContainer)
        # - NMData._parent = NMFolder (not NMDataContainer)
        # - NMFolder._parent = NMManager (not NMFolderContainer)
        select: dict[str, str] = {}
        current = obj

        if isinstance(current, NMEpoch):
            select["epoch"] = current.name
            # Parent is NMDataSeries directly (skips container)
            current = current._parent

        if isinstance(current, NMChannel):
            select["channel"] = current.name
            # Parent is NMDataSeries directly (skips container)
            current = current._parent

        if isinstance(current, NMDataSeries):
            select["dataseries"] = current.name
            # Parent is NMFolder directly (skips container)
            current = current._parent

        if isinstance(current, NMData):
            select["data"] = current.name
            # Parent is NMFolder directly (skips container)
            current = current._parent

        if isinstance(current, NMFolder):
            select["folder"] = current.name

        if not select:
            raise ValueError(
                f"object type '{type(obj).__name__}' is not a recognized "
                f"hierarchy type (NMFolder, NMData, NMDataSeries, NMChannel, NMEpoch)"
            )

        # Use existing _select_keys_set to apply the selection
        self._select_keys_set(select)

    def run_values(
        self,
        dataseries_priority: bool = True
    ) -> list[dict[str, NMObject]]:
        elist: list[dict[str, NMObject]] = []
        folders = self.__folders
        if folders is None:
            return elist
        flist = folders.run_targets
        for f in flist:
            if not isinstance(f, NMFolder):
                continue
            dslist = f.dataseries.run_targets
            if dataseries_priority and dslist:
                for ds in dslist:
                    if not isinstance(ds, NMDataSeries):
                        continue
                    for c in ds.channels.run_targets:
                        for e in ds.epochs.run_targets:
                            x: dict[str, NMObject] = {}
                            x["folder"] = f
                            x["dataseries"] = ds
                            x["channel"] = c
                            x["epoch"] = e
                            elist.append(x)
            else:
                dlist = f.data.run_targets
                for d in dlist:
                    x2: dict[str, NMObject] = {}
                    x2["folder"] = f
                    x2["data"] = d
                    elist.append(x2)
        return elist

    def run_keys(
        self,
        dataseries_priority: bool = True
    ) -> list[dict[str, str]]:
        elist = []
        elist2 = self.run_values(dataseries_priority)
        for e in elist2:
            e2 = {}
            for k, o in e.items():
                e2[k] = o.name
            elist.append(e2)
        return elist

    def run_count(self, dataseries_priority: bool = True) -> int:
        """
        Return count of run targets without executing.

        Use this to preview how many items will be targeted before
        calling run_tool().

        Args:
            dataseries_priority: If True, count dataseries mode targets.
                                If False, count data mode targets.

        Returns:
            Number of run targets
        """
        return len(self.run_keys(dataseries_priority))

    def run_keys_set(
        self,
        run: dict[str, str],
        max_targets: int | None = 1000
    ) -> list[dict[str, str]]:
        """
        Set run targets at each hierarchy tier.

        Args:
            run: Dictionary mapping tier names to target values.
                    Values can be: "select"/"selected", "all", a specific name,
                    or a set name.
                    Must include "folder" and either "data" or "dataseries".
                    If "dataseries", must also include "channel" and "epoch".
            max_targets: Maximum number of resulting targets allowed.
                        Set to None for unlimited. Default is 1000.

        Returns:
            List of run target dictionaries (same as run_keys())

        Raises:
            TypeError: If run is not a dict or values aren't strings
            KeyError: If a key is not a valid selection tier
            ValueError: If target count exceeds max_targets
        """
        if not isinstance(run, dict):
            raise TypeError(nmu.type_error_str(run, "run", "dictionary"))

        for tier, value in run.items():
            if not isinstance(tier, str):
                raise TypeError(nmu.type_error_str(tier, "key", "string"))

        # Normalize keys to lowercase for case-insensitive matching
        run = {k.lower(): v for k, v in run.items()}

        for tier, value in run.items():
            if tier not in HIERARCHY_SELECT_KEYS:
                raise KeyError(f"unknown selection tier '{tier}'")
            if not isinstance(value, str):
                raise TypeError(nmu.type_error_str(value, "value", "string"))

        # Validate mutually exclusive keys
        if "data" in run and "dataseries" in run:
            raise KeyError(
                "encountered both 'data' and 'dataseries' keys "
                "but only one should be defined"
            )

        if "data" not in run and "dataseries" not in run:
            raise KeyError("missing run 'data' or 'dataseries' key")

        if "data" in run and "channel" in run:
            raise KeyError(
                "run 'channel' key should be used with 'dataseries', not 'data'"
            )

        if "data" in run and "epoch" in run:
            raise KeyError(
                "run 'epoch' key should be used with 'dataseries', not 'data'"
            )

        # Folder is required
        if "folder" not in run:
            raise KeyError("missing run 'folder' key")

        folders = self.__folders
        if folders is None:
            raise ValueError("no folder container")

        # Set folder run target (allows select/all/name/set)
        folders.run_target = run["folder"]

        # Data mode - simpler path
        if "data" in run:
            # Set data run target for each folder in run_targets
            for f in folders.run_targets:
                if isinstance(f, NMFolder):
                    f.data.run_target = run["data"]

            result = self.run_keys(dataseries_priority=False)
            self._run_check_max_targets(result, max_targets)
            self.__run_config = dict(run)
            nmch.add_nm_command("run_keys_set(%r)" % (run,))
            return result

        # Dataseries mode - requires dataseries, channel, epoch
        if "dataseries" not in run:
            raise KeyError("missing run 'dataseries' key")
        if "channel" not in run:
            raise KeyError("missing run 'channel' key")
        if "epoch" not in run:
            raise KeyError("missing run 'epoch' key")

        # Set run targets for each folder and dataseries
        for f in folders.run_targets:
            if not isinstance(f, NMFolder):
                continue
            f.dataseries.run_target = run["dataseries"]
            for ds in f.dataseries.run_targets:
                if isinstance(ds, NMDataSeries):
                    ds.channels.run_target = run["channel"]
                    ds.epochs.run_target = run["epoch"]

        result = self.run_keys(dataseries_priority=True)
        self._run_check_max_targets(result, max_targets)
        self.__run_config = dict(run)
        nmch.add_nm_command("run_keys_set(%r)" % (run,))
        return result

    def _run_check_max_targets(
        self,
        result: list,
        max_targets: int | None
    ) -> None:
        """Check if result exceeds max_targets limit."""
        if max_targets is not None and len(result) > max_targets:
            raise ValueError(
                f"Run would target {len(result)} items, exceeding limit "
                f"of {max_targets}. Use max_targets=None to override, or use "
                f"run_count() to preview target count."
            )

    def run_reset_all(self) -> None:
        """Reset all run targets to use the selected item at each tier."""
        folders = self.__folders
        if folders is None:
            return None
        folders.run_target = RUN_SELECTED
        for f in folders.values():
            if not isinstance(f, NMFolder):
                continue
            f.data.run_target = RUN_SELECTED
            f.dataseries.run_target = RUN_SELECTED
            for ds in f.dataseries.values():
                if not isinstance(ds, NMDataSeries):
                    continue
                ds.channels.run_target = RUN_SELECTED
                ds.epochs.run_target = RUN_SELECTED
        self.__run_config = None
        nmch.add_nm_command("run_reset_all()")
        return None

    def run_tool(
        self,
        toolname: str = "selected"
    ) -> bool:
        """Run the selected (or named) tool over all current run targets.

        Resolves the tool by name, fetches run targets from the project
        hierarchy via ``run_values()``, and delegates the execution loop
        to ``tool.run_all(targets)``.

        Args:
            toolname: Name of the tool to run, or ``"selected"`` (default)
                to use the currently selected tool.

        Returns:
            Return value of ``tool.run_finish()``.

        Raises:
            TypeError: If toolname is not a string.
            ValueError: If no tool is selected.
            KeyError: If the named tool is not in the toolkit.
        """
        from pyneuromatic.analysis.nm_tool import NMTool

        tname: str | None = toolname
        if isinstance(tname, str):
            if tname.lower() in ("select", "selected"):
                tname = self.__toolselect
        else:
            raise TypeError(nmu.type_error_str(tname, "toolname", "string"))
        if tname is None:
            raise ValueError("no tool selected")
        tname = tname.lower()
        if tname not in self.__toolkit:
            raise KeyError("NM tool key '%s' does not exist" % toolname)
        tool = self.__toolkit[tname]
        if not isinstance(tool, NMTool):
            raise TypeError("tool '%s' is not an instance of NMTool" % toolname)

        targets = self.run_values()
        if not targets:
            print("nothing to run")
        result = tool.run_all(targets, run_keys=self.__run_config)
        nmch.add_nm_command("run_tool(%r)" % tname)
        return result

    # Workspace methods

    @property
    def workspace(self) -> NMWorkspace | None:
        """Get the current workspace configuration."""
        return self._workspace_manager.current

    def available_workspaces(self) -> list[str]:
        """List available workspace names."""
        return self._workspace_manager.available_workspaces()

    def load_workspace(
        self,
        name: str,
        preserve_state: bool = False
    ) -> None:
        """
        Load a workspace configuration.

        Args:
            name: Workspace name to load
            preserve_state: If True, preserves existing tool state
        """
        if not preserve_state:
            self.__toolkit.clear()
            self.__toolselect = None
        self._load_workspace_internal(name)
        nmh.history(f"Loaded workspace '{name}'")

    def save_workspace(
        self,
        name: str | None = None,
        description: str = ""
    ) -> str:
        """
        Save current toolkit configuration as a workspace.

        Args:
            name: Workspace name (None for default)
            description: Optional description

        Returns:
            Path to saved workspace file as string
        """
        workspace = NMWorkspace(
            name=name or "default",
            description=description,
            tools=list(self.__toolkit.keys()),
        )
        path = self._workspace_manager.save(workspace, name)
        nmh.history(f"Saved workspace to '{path}'")
        return str(path)

    def available_tools(self) -> list[str]:
        """List all tools available in the registry."""
        return self._tool_registry.keys()

    def enabled_tools(self) -> list[str]:
        """List currently enabled (loaded) tools."""
        return list(self.__toolkit.keys())

    # Tool access properties (backward compatibility)

    @property
    def main(self) -> "NMTool | None":
        """Access the Main tool."""
        return self.__toolkit.get("main")

    @property
    def stats(self) -> "NMTool | None":
        """Access the Stats tool."""
        return self.__toolkit.get("stats")

    def __getattr__(self, name: str) -> "NMTool":
        """
        Dynamic attribute access for tools.

        Allows accessing tools as attributes (e.g., nm.spike, nm.event).
        If the tool is in the registry but not loaded, it will be loaded.

        Args:
            name: Attribute name (tool name)

        Returns:
            The requested NMTool instance

        Raises:
            AttributeError: If not a valid tool name
        """
        # Avoid infinite recursion for private/internal attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )

        tool_name = name.lower()

        # Check if already loaded
        if hasattr(self, "_NMManager__toolkit") and tool_name in self.__toolkit:
            return self.__toolkit[tool_name]

        # Attempt lazy load from registry
        if hasattr(self, "_tool_registry") and tool_name in self._tool_registry:
            self.tool_add(tool_name)
            return self.__toolkit[tool_name]

        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )


if __name__ == "__main__":
    # Basic usage example
    nm = NMManager(command_history=True)

    # Create a folder and add data
    folder = nm.folders.new("folder0")
    folder.data.new("recordA0", nparray=np.array([1.0, 2.0, 3.0]))
    folder.data.new("recordA1", nparray=np.array([4.0, 5.0, 6.0]))

    # Set run targets and run a tool
    nm.run_keys_set({"folder": "folder0", "data": "all"})
    nm.run_tool(toolname="stats")

    print(nm.command_history.buffer)

