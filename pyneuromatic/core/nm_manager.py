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

from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_dimension import NMDimension, NMDimensionX
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_history as nmh
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_preferences as nmp
from pyneuromatic.core.nm_project import NMProject
from pyneuromatic.core.nm_tool_registry import get_global_registry, NMToolRegistry
from pyneuromatic.core.nm_workspace import NMWorkspace, NMWorkspaceManager
import pyneuromatic.core.nm_utilities as nmu

if TYPE_CHECKING:
    from pyneuromatic.analysis.nm_tool import NMTool

nm = None  # holds Manager, accessed via console

# Hierarchy levels for selection (folder down to epoch)
SELECT_LEVELS = ("folder", "data", "dataseries", "channel", "epoch")

"""
NM class tree:

NMManager
    NMProject (root)
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


class NMManager:
    """
    NM Manager - Central manager for pyNeuroMatic.

    Manages the project, tools, and workspaces. Tools are loaded lazily from the
    registry based on the active workspace configuration.

    Note: NMManager is not an NMObject - it sits outside the object hierarchy.
    The hierarchy starts at NMProject (root).

    Args:
        quiet: Suppress history messages
        workspace: Workspace name to load (None for default)
        config_dir: Custom configuration directory (None for platform default)

    Example:
        nm = NMManager(workspace="spike_analysis")
        nm.stats.windows.get("w0").func = "mean"
        nm.execute_tool()
    """

    def __init__(
        self,
        quiet: bool = False,
        workspace: str | None = None,
        config_dir: str | Path | None = None,
    ) -> None:
        # Initialize centralized history logging
        self.__history = nmh.NMHistory(quiet=quiet)
        nmh.set_history(self.__history)

        # Create project (root)
        self.__project = NMProject(parent=self, name="root")

        # Initialize tool registry and workspace manager
        self._tool_registry: NMToolRegistry = get_global_registry()
        self._workspace_manager: NMWorkspaceManager = NMWorkspaceManager(
            config_dir=config_dir
        )

        self.__toolkit: dict[str, "NMTool"] = {}
        self.__toolselect: str | None = None

        # Load workspace (or default)
        self._load_workspace_internal(workspace, quiet=quiet)

        tstr = str(datetime.datetime.now())
        nmh.history(f"created NM manager {tstr}", quiet=quiet)
        nmh.history("current NM project is '%s'" % self.__project.name, quiet=quiet)
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
        else:
            raise KeyError("NM tool key '%s' does not exist" % tname)

    @property
    def history(self) -> nmh.NMHistory:
        """Access the centralized NM history log."""
        return self.__history

    @property
    def project(self) -> NMProject:
        return self.__project

    def _iter_select_hierarchy(self):
        """
        Generator yielding (level_name, container, selected_value) tuples.

        Traverses the selection hierarchy from folder down to epoch.
        Stops when a required parent container has no selection.

        Yields:
            Tuples of (level_name, container, selected_value) where:
            - level_name: one of SELECT_LEVELS
            - container: the NMObjectContainer at this level
            - selected_value: the currently selected object (or None)
        """
        folders = self.__project.folders
        if folders is None:
            return

        f = folders.selected_value
        yield ("folder", folders, f)

        if not isinstance(f, NMFolder):
            return

        # Data and DataSeries are siblings at folder level
        yield ("data", f.data, f.data.selected_value)

        ds = f.dataseries.selected_value
        yield ("dataseries", f.dataseries, ds)

        if not isinstance(ds, NMDataSeries):
            return

        # Channel and Epoch are siblings at dataseries level
        yield ("channel", ds.channels, ds.channels.selected_value)
        yield ("epoch", ds.epochs, ds.epochs.selected_value)

    @property
    def select_values(self) -> dict[str, NMObject | None]:
        """Get the currently selected object at each hierarchy level."""
        result: dict[str, NMObject | None] = {level: None for level in SELECT_LEVELS}
        for level, container, value in self._iter_select_hierarchy():
            result[level] = value
        return result

    @property
    def select_keys(self) -> dict[str, str | None]:
        """Get the names of currently selected objects at each hierarchy level."""
        result: dict[str, str | None] = {level: None for level in SELECT_LEVELS}
        for level, container, value in self._iter_select_hierarchy():
            if isinstance(value, NMObject):
                result[level] = value.name
        return result

    @select_keys.setter
    def select_keys(self, select: dict[str, str]) -> None:
        self._select_keys_set(select)

    def _select_keys_set(self, select: dict[str, str]) -> None:
        """
        Set selection at each hierarchy level by name.

        Args:
            select: Dictionary mapping level names to object names.
                    Keys must be from SELECT_LEVELS (folder, data, dataseries,
                    channel, epoch).

        Raises:
            TypeError: If select is not a dict or values aren't strings
            KeyError: If a key is not a valid selection level
        """
        if not isinstance(select, dict):
            raise TypeError(nmu.type_error_str(select, "select", "dictionary"))

        for key, value in select.items():
            if not isinstance(key, str):
                raise TypeError(nmu.type_error_str(key, "key", "string"))
            if value is not None and not isinstance(value, str):
                raise TypeError(nmu.type_error_str(value, "value", "string"))
            if key not in SELECT_LEVELS:
                raise KeyError(f"'{key}' is not a valid selection level. "
                               f"Valid levels: {SELECT_LEVELS}")

        # Traverse hierarchy, setting values as we go so subsequent levels
        # use the newly selected parent
        folders = self.__project.folders
        if folders is None:
            return

        if "folder" in select:
            folders.selected_name = select["folder"]
        f = folders.selected_value
        if not isinstance(f, NMFolder):
            return

        if "data" in select:
            f.data.selected_name = select["data"]
        if "dataseries" in select:
            f.dataseries.selected_name = select["dataseries"]

        ds = f.dataseries.selected_value
        if not isinstance(ds, NMDataSeries):
            return

        if "channel" in select:
            ds.channels.selected_name = select["channel"]
        if "epoch" in select:
            ds.epochs.selected_name = select["epoch"]

    def execute_values(
        self,
        dataseries_priority: bool = True
    ) -> list[dict[str, NMObject]]:
        elist: list[dict[str, NMObject]] = []
        # for p in self.__project_container.execute_values:
        p = self.__project
        folders = p.folders
        if folders is None:
            return elist
        flist = folders.execute_targets
        for f in flist:
            if not isinstance(f, NMFolder):
                continue
            dslist = f.dataseries.execute_targets
            if dataseries_priority and dslist:
                for ds in dslist:
                    if not isinstance(ds, NMDataSeries):
                        continue
                    for c in ds.channels.execute_targets:
                        for e in ds.epochs.execute_targets:
                            x: dict[str, NMObject] = {}
                            x["folder"] = f
                            x["dataseries"] = ds
                            x["channel"] = c
                            x["epoch"] = e
                            elist.append(x)
            else:
                dlist = f.data.execute_targets
                for d in dlist:
                    x2: dict[str, NMObject] = {}
                    x2["folder"] = f
                    x2["data"] = d
                    elist.append(x2)
        return elist

    def execute_keys(
        self,
        dataseries_priority: bool = True
    ) -> list[dict[str, str]]:
        elist = []
        elist2 = self.execute_values(dataseries_priority)
        for e in elist2:
            e2 = {}
            for k, o in e.items():
                e2[k] = o.name
            elist.append(e2)
        return elist

    def execute_keys_set(
        self,
        execute: dict[str, str]
    ) -> list[dict[str, str]]:
        # sets are not allowed with folder, dataseries - too complex
        # can specify 'data' or 'dataseries' but not both

        if not isinstance(execute, dict):
            e = nmu.type_error_str(execute, "execute", "dictionary")
            raise TypeError(e)
        for key, value in execute.items():
            if not isinstance(key, str):
                e = nmu.type_error_str(key, "key", "string")
                raise TypeError(e)
            if key not in SELECT_LEVELS:
                e = "unknown execute key '%s'" % key
                raise KeyError(e)
            if not isinstance(value, str):
                e = nmu.type_error_str(value, "value", "string")
                raise TypeError(e)

        if "data" in execute and "dataseries" in execute:
            e = (
                "encounted both 'data' and 'dataseries' keys "
                "but only one should be defined"
            )
            raise KeyError(e)

        if "data" not in execute and "dataseries" not in execute:
            e = "missing execute 'data' or 'dataseries' key"
            raise KeyError(e)

        if "data" in execute and "channel" in execute:
            e = "execute 'channel' key should be used with 'dataseries', "
            "not 'data'"
            raise KeyError(e)

        if "data" in execute and "epoch" in execute:
            e = "execute 'epoch' key should be used with 'dataseries', "
            "not 'data'"
            raise KeyError(e)

        """
        if "project" not in execute:
            e = "missing execute 'project' key"
            raise KeyError(e)
        value = execute["project"]
        if value.lower() == "select":
            p = self.__project_container.select_key
            if p in self.__project_container:
                self.__project_container.execute_key = "select"
            else:
                e = "bad project select: %s" % p
                raise ValueError(e)
        elif value.lower() == "all":
            e = "'all' projects is not allowed in this function"
            raise ValueError(e)
        elif value in self.__project_container:
            self.__project_container.select_key = value
            self.__project_container.execute_key = "select"
        elif value in self.__project_container.sets:
            e = "project sets are not allowed in this function"
            raise ValueError(e)
        else:
            e = "unknown project execute value: %s" % value
            raise ValueError(e)
        p = self.__project_container.select_value
        if p not in self.__project_container:
            e = "bad project select: %s" % p
            raise ValueError(e)
        """

        p = self.__project
        if "project" in execute:
            value = execute["project"]
            if value.lower() in ("select", "selected", "all"):
                pass  # ok, only one project
            elif value.lower() == p.name.lower():
                pass  # ok
            else:
                e = ("NM supports only one project. " +
                     "The current project is '%s'." % p.name)
                raise ValueError(e)

        if "folder" not in execute:
            e = "missing execute 'folder' key"
            raise KeyError(e)
        folders = p.folders
        if folders is None:
            raise ValueError("project has no folder container")
        value = execute["folder"]
        if value.lower() in ("select", "selected"):
            fkey = folders.selected_name
            if fkey in folders:
                folders.execute_target = "select"
            else:
                e = "bad folder select: %s" % fkey
                raise ValueError(e)
        elif value.lower() == "all":
            e = "'all' folders is not allowed in this function"
            raise ValueError(e)
        elif value in folders:
            folders.selected_name = value
            folders.execute_target = "select"
        elif value in folders.sets:
            e = "folder sets are not allowed in this function"
            raise ValueError(e)
        else:
            e = "unknown folder execute value: %s" % value
            raise ValueError(e)
        f = folders.selected_value
        if f not in folders:
            e = "bad folder select: %s" % f
            raise ValueError(e)
        if not isinstance(f, NMFolder):
            raise ValueError("bad folder select")

        if "data" in execute:
            f.data.execute_target = execute["data"]
            return self.execute_keys(dataseries_priority=False)  # finished

        if "dataseries" not in execute:
            e = "missing execute 'dataseries' key"
            raise KeyError(e)
        value = execute["dataseries"]
        if value.lower() in ("select", "selected"):
            dskey = f.dataseries.selected_name
            if dskey in f.dataseries:
                f.dataseries.execute_target = "select"
            else:
                e = "bad dataseries select: %s" % dskey
                raise ValueError(e)
        elif value.lower() == "all":
            e = "'all' dataseries is not allowed in this function"
            raise ValueError(e)
        elif value in f.dataseries:
            f.dataseries.selected_name = value
            f.dataseries.execute_target = "select"
        elif value in f.dataseries.sets:
            e = "dataseries sets are not allowed in this function"
            raise ValueError(e)
        else:
            e = "unknown dataseries execute value: %s" % value
            raise ValueError(e)
        ds = f.dataseries.selected_value
        if ds not in f.dataseries:
            e = "bad dataseries select: %s" % ds
            raise ValueError(e)
        if not isinstance(ds, NMDataSeries):
            raise ValueError("bad dataseries select")

        if "channel" not in execute:
            e = "missing execute 'channel' key"
            raise KeyError(e)
        ds.channels.execute_target = execute["channel"]

        if "epoch" not in execute:
            e = "missing execute 'epoch' key"
            raise KeyError(e)
        ds.epochs.execute_target = execute["epoch"]

        return self.execute_keys(dataseries_priority=True)

    def execute_reset_all(self) -> None:
        # self.__project_container.execute_key = "select"
        # for p in self.__project_container.values():
        p = self.__project
        folders = p.folders
        if folders is None:
            return None
        folders.execute_target = "select"
        for f in folders.values():
            if not isinstance(f, NMFolder):
                continue
            f.data.execute_target = "select"
            f.dataseries.execute_target = "select"
            for ds in f.dataseries.values():
                if not isinstance(ds, NMDataSeries):
                    continue
                ds.channels.execute_target = "select"
                ds.epochs.execute_target = "select"
        return None

    def execute_tool(
        self,
        toolname: str = "select"
    ) -> bool:

        tname: str | None = toolname
        if tname is None:
            tname = self.__toolselect
        if isinstance(tname, str):
            if tname.lower() == "select":
                tname = self.__toolselect
        else:
            e = nmu.type_error_str(tname, "toolname", "string")
            raise TypeError(e)
        if tname is None:
            raise ValueError("no tool selected")
        tname = tname.lower()
        if tname not in self.__toolkit:
            raise KeyError("NM tool key '%s' does not exist" % toolname)
        tool = self.__toolkit[tname]
        if not isinstance(tool, NMTool):
            e = "tool '%s' is not an instance of NMTool" % toolname
            raise TypeError(e)

        execute_list = self.execute_keys()
        if not execute_list:
            print("nothing to execute")
        if not tool.execute_init():
            return False
        for ex in execute_list:
            self.select_keys = ex
            tool.select_values = self.select_values
            if not tool.execute():
                break
        return tool.execute_finish()

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

    pnts = 30

    nm = NMManager()
    # p = nm.projects.select_value
    # p = nm.project
    assert nm.project.folders is not None
    f0 = nm.project.folders.new("myfolder0")
    f1 = nm.project.folders.new("myfolder1")
    nm.project.folders.selected_name = "myfolder1"
    assert isinstance(f1, NMFolder)

    ydata = np.random.normal(loc=0, scale=1, size=pnts)
    ydim = NMDimension(nm, "y", nparray=ydata)
    f1.data.new("recordA0", ydim=ydim)

    ydata = np.random.normal(loc=0, scale=1, size=pnts)
    dx = 1
    xdata = np.arange(0, pnts * dx, dx)
    ydata = 3.2 * xdata + 5.7
    # print(xdata)
    # print(ydata)
    xdim = NMDimensionX(nm, "x", nparray=xdata, ypair=ydata)
    ydim = NMDimension(nm, "y", nparray=ydata)
    # f1.data.new("recordA1", xdim=xdim, ydim=ydim)
    f1.data.new("recordA1", ydim=ydim)

    ydata = np.random.normal(loc=0, scale=1, size=pnts)
    ydim = NMDimension(nm, "y", nparray=ydata)
    f1.data.new("recordA2", ydim=ydim)

    ydata = np.random.normal(loc=0, scale=1, size=pnts)
    ydim = NMDimension(nm, "y", nparray=ydata)
    f1.data.new("recordA3", ydim=ydim)

    f1.data.sets.new("set1")
    f1.data.sets.add("set1", ["recordA0", "recordA2", "recordA3"])

    e = {
         "folder": "myfolder1",
         "data": "set1"
         }

    nm.execute_keys_set(e)

    stats = nm.stats

    if stats:
        w0 = nm.stats.windows.get("w0")
        w0.func = "mean"
        w1 = nm.stats.windows.new()
        w1.func = "mean"
        # w1.x0 = 3 * dx
        # w1.x1 = 50 * dx
        w1.bsln_on = True
        w1.bsln_x0 = 0
        w1.bsln_x1 = 10
        w1.bsln_func = "mean"
        w1.bsln_subtract = True
        w2 = nm.stats.windows.new()
        w2.func = "mean+sem"
        w3 = nm.stats.windows.new()
        w3.func = "median"
        nm.stats.windows.sets.new("set1")
        nm.stats.windows.sets.add("set1", ["w1"])
        nm.stats.windows.execute_key = "set1"

    for i in range(1):
        nm.execute_tool()

    # f1.toolresults.clear()
    # print(len(f1.toolresults))

    # print(nm.projects.execute_key)
    # print(nm.execute_keys())

    # p = nm.projects.new("project0")
    # p.folders.new("folder0")
