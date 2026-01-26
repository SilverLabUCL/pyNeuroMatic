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

from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_dimension import NMDimension, NMDimensionX
from pyneuromatic.core.nm_folder import NMFolder, NMFolderContainer
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_preferences as nmp
from pyneuromatic.core.nm_project import NMProject, NMProjectContainer
from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.analysis.nm_tool_stats import NMToolStats
import pyneuromatic.core.nm_utilities as nmu

nm = None  # holds Manager, accessed via console

"""
NM class tree:

NMManager
    NMProjectContainer
        NMProject (project0, project1...)
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

# TOOL_NAMES = ("Main", "Stats", "Spike", "Event")
TOOL_NAMES = ("Stats",)


class NMManager(NMObject):
    def __init__(
        self,
        # parent: Union[object, None] = None,
        name: str = "nm",
        project_name: str = "project0",
        quiet: bool = False,
        # copy: NMObject | None = None,  # see copy()
    ) -> None:
        super().__init__(parent=None, name=name)  # NMObject

        # self.__configs = nmp.Configs()
        # self.__configs.quiet = quiet

        # for now, only one project
        # self.__project_container = NMProjectContainer(parent=self)
        if not isinstance(project_name, str):
            e = nmu.typeerror(project_name, "project_name", "string")
            raise TypeError(e)
        self.__project = NMProject(parent=self, name=project_name)

        self.__toolkit: dict[str, NMTool] = {}
        self.__toolselect: str | None = None

        for tool in TOOL_NAMES:
            self.tool_add(tool)

        if "main" in self.__toolkit:
            self.__toolselect = "main"
        else:
            for k in self.__toolkit.keys():
                self.__toolselect = k
                break  # first tool will be selected

        """
        if project_name is None:
            pass
        elif isinstance(project_name, str):
            p = self.__project_container.new(project_name)
            if p:
                self.__project_container.select_key = project_name
        else:
            e = nmu.typeerror(project_name, "project_name", "string")
            raise TypeError(e)
        """

        tstr = str(datetime.datetime.now())
        h = ("created NM manager '%s' %s"
             % (self.name, tstr))
        # self._history(h, quiet=quiet)
        print(h)
        print("current NM project is '%s'" % self.__project.name)
        print("current NM tool is '%s'" % self.__toolselect)

    def tool_add(
        self,
        toolname: str,
        tool: NMTool | None = None,
        select: bool = False
    ) -> None:
        if not isinstance(toolname, str):
            e = nmu.typeerror(toolname, "toolname", "string")
            raise TypeError(e)
        tname = toolname.lower()
        if tool is None:  # look for NM tool
            if tname == "mainTODO":
                # TODO tool = NMToolMain()
                pass
            elif tname == "stats":
                tool = NMToolStats()
            elif tname == "spikeTODO":
                # TODO tool = NMToolSpike()
                pass
            elif tname == "eventTODO":
                # TODO tool = NMToolEvent()
                pass
            else:
                raise KeyError("NM tool key '%s' does not exist" % tname)
        elif isinstance(tool, NMTool):
            pass
        else:
            e = "tool '%s' is not an instance of NMTool" % toolname
            raise TypeError(e)
        if tool:
            self.__toolkit.update({tname: tool})
            if select:
                self.__toolselect = tname

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

    # @property
    # def projects(self) -> NMProjectContainer:
    #     return self.__project_container

    @property
    def project(self) -> NMProject:
        return self.__project

    @property
    def select_values(self) -> dict[str, NMObject | None]:
        s: dict[str, NMObject | None] = {}
        s["project"] = None
        s["folder"] = None
        s["data"] = None
        s["dataseries"] = None
        s["channel"] = None
        s["epoch"] = None
        # p = self.__project_container.select_value
        p = self.__project
        s["project"] = p
        # if p:
        #     s["project"] = p
        # else:
        #     return s
        folders = p.folders
        if folders is None:
            return s
        f = folders.select_value
        if not isinstance(f, NMFolder):
            return s
        s["folder"] = f
        d = f.data.select_value
        if d:
            s["data"] = d
        ds = f.dataseries.select_value
        if not isinstance(ds, NMDataSeries):
            return s
        s["dataseries"] = ds
        c = ds.channels.select_value
        if c:
            s["channel"] = c
        e = ds.epochs.select_value
        if e:
            s["epoch"] = e
        return s

    @property
    def select_keys(self) -> dict[str, str | None]:
        s1 = self.select_values
        s2: dict[str, str | None] = {}
        for k, v in s1.items():
            if isinstance(v, NMObject):
                s2.update({k: v.name})
            else:
                s2.update({k: None})
        return s2

    @select_keys.setter
    def select_keys(self, select: dict[str, str]) -> None:
        return self._select_keys_set(select)

    def _select_keys_set(
        self,
        select: dict[str, str]
        # quiet
    ) -> None:
        if not isinstance(select, dict):
            e = nmu.typeerror(select, "select", "dictionary")
            raise TypeError(e)
        for key, value in select.items():
            if not isinstance(key, str):
                e = nmu.typeerror(key, "key", "string")
                raise TypeError(e)
            if value is None or isinstance(value, str):
                pass  # ok
            else:
                e = nmu.typeerror(value, "value", "string")
                raise TypeError(e)
        p = self.__project
        if "project" in select:
            k = select["project"]
            if k.lower() != p.name.lower():
                e = ("NM supports only one project. " +
                     "The current project is '%s'." % p.name)
                raise KeyError(e)
        #     self.__project_container.select_key = select["project"]
        # p = self.__project_container.select_value
        folders = p.folders
        if folders is None:
            return None
        if "folder" in select:
            folders.select_key = select["folder"]
        f = folders.select_value
        if not isinstance(f, NMFolder):
            return None
        if "data" in select:
            f.data.select_key = select["data"]
        if "dataseries" in select:
            f.dataseries.select_key = select["dataseries"]
        ds = f.dataseries.select_value
        if not isinstance(ds, NMDataSeries):
            return None
        if "channel" in select:
            ds.channels.select_key = select["channel"]
        if "epoch" in select:
            ds.epochs.select_key = select["epoch"]
        return None

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
        flist = folders.execute_values
        for f in flist:
            if not isinstance(f, NMFolder):
                continue
            dslist = f.dataseries.execute_values
            if dataseries_priority and dslist:
                for ds in dslist:
                    if not isinstance(ds, NMDataSeries):
                        continue
                    for c in ds.channels.execute_values:
                        for e in ds.epochs.execute_values:
                            x: dict[str, NMObject] = {}
                            x["project"] = p
                            x["folder"] = f
                            x["dataseries"] = ds
                            x["channel"] = c
                            x["epoch"] = e
                            elist.append(x)
            else:
                dlist = f.data.execute_values
                for d in dlist:
                    x2: dict[str, NMObject] = {}
                    x2["project"] = p
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
        # sets are not allowed with project, folder, dataseries - too complex
        # can specify 'data' or 'dataseries' but not both

        if not isinstance(execute, dict):
            e = nmu.typeerror(execute, "execute", "dictionary")
            raise TypeError(e)
        for key, value in execute.items():
            if not isinstance(key, str):
                e = nmu.typeerror(key, "key", "string")
                raise TypeError(e)
            if key not in [
                "project",
                "folder",
                "data",
                "dataseries",
                "channel",
                "epoch",
            ]:
                e = "unknown execute key '%s'" % key
                raise KeyError(e)
            if not isinstance(value, str):
                e = nmu.typeerror(value, "value", "string")
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
            if value.lower() == "select" or value.lower() == "all":
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
        if value.lower() == "select":
            fkey = folders.select_key
            if fkey in folders:
                folders.execute_key = "select"
            else:
                e = "bad folder select: %s" % fkey
                raise ValueError(e)
        elif value.lower() == "all":
            e = "'all' folders is not allowed in this function"
            raise ValueError(e)
        elif value in folders:
            folders.select_key = value
            folders.execute_key = "select"
        elif value in folders.sets:
            e = "folder sets are not allowed in this function"
            raise ValueError(e)
        else:
            e = "unknown folder execute value: %s" % value
            raise ValueError(e)
        f = folders.select_value
        if f not in folders:
            e = "bad folder select: %s" % f
            raise ValueError(e)
        if not isinstance(f, NMFolder):
            raise ValueError("bad folder select")

        if "data" in execute:
            f.data.execute_key = execute["data"]
            return self.execute_keys(dataseries_priority=False)  # finished

        if "dataseries" not in execute:
            e = "missing execute 'dataseries' key"
            raise KeyError(e)
        value = execute["dataseries"]
        if value.lower() == "select":
            dskey = f.dataseries.select_key
            if dskey in f.dataseries:
                f.dataseries.execute_key = "select"
            else:
                e = "bad dataseries select: %s" % dskey
                raise ValueError(e)
        elif value.lower() == "all":
            e = "'all' dataseries is not allowed in this function"
            raise ValueError(e)
        elif value in f.dataseries:
            f.dataseries.select_key = value
            f.dataseries.execute_key = "select"
        elif value in f.dataseries.sets:
            e = "dataseries sets are not allowed in this function"
            raise ValueError(e)
        else:
            e = "unknown dataseries execute value: %s" % value
            raise ValueError(e)
        ds = f.dataseries.select_value
        if ds not in f.dataseries:
            e = "bad dataseries select: %s" % ds
            raise ValueError(e)
        if not isinstance(ds, NMDataSeries):
            raise ValueError("bad dataseries select")

        if "channel" not in execute:
            e = "missing execute 'channel' key"
            raise KeyError(e)
        ds.channels.execute_key = execute["channel"]

        if "epoch" not in execute:
            e = "missing execute 'epoch' key"
            raise KeyError(e)
        ds.epochs.execute_key = execute["epoch"]

        return self.execute_keys(dataseries_priority=True)

    def execute_reset_all(self) -> None:
        # self.__project_container.execute_key = "select"
        # for p in self.__project_container.values():
        p = self.__project
        folders = p.folders
        if folders is None:
            return None
        folders.execute_key = "select"
        for f in folders.values():
            if not isinstance(f, NMFolder):
                continue
            f.data.execute_key = "select"
            f.dataseries.execute_key = "select"
            for ds in f.dataseries.values():
                if not isinstance(ds, NMDataSeries):
                    continue
                ds.channels.execute_key = "select"
                ds.epochs.execute_key = "select"
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
            e = nmu.typeerror(tname, "toolname", "string")
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

    # @property
    # def configs(self):
    #     return self.__configs

    @property
    def main(self):
        if "main" in self.__toolkit:
            return self.__toolkit["main"]
        return None

    @property
    def stats(self):
        if "stats" in self.__toolkit:
            return self.__toolkit["stats"]
        return None

    """
    def _quiet(self, quiet=nmp.QUIET):
        if self.configs.quiet:  # config quiet overrides
            return True
        return quiet
    """


if __name__ == "__main__":

    pnts = 30

    nm = NMManager(project_name="myproject")
    # p = nm.projects.select_value
    # p = nm.project
    assert nm.project.folders is not None
    f0 = nm.project.folders.new("myfolder0")
    f1 = nm.project.folders.new("myfolder1")
    nm.project.folders.select_key = "myfolder1"
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
