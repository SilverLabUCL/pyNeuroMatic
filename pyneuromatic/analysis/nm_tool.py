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

from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_project import NMProject
from pyneuromatic.core.nm_folder import NMFolder, NMFolderContainer
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_epoch import NMEpoch
import pyneuromatic.core.nm_utilities as nmu

class NMTool(object):
    """
    NM Tool class.
    Used to create NM tools (analysis modules).
    Executed by NMManager.
    See NMToolMain, NMToolStats, NMToolSpike, etc.
    """

    def __init__(self) -> None:
        self.__project: NMProject | None = None
        self.__folder: NMFolder | None = None
        self.__data: NMData | None = None
        self.__dataseries: NMDataSeries | None = None
        self.__channel: NMChannel | None = None
        self.__epoch: NMEpoch | None = None

    @property
    def project(self) -> NMProject | None:
        return self.__project

    @project.setter
    def project(self, project: NMProject | None) -> None:
        if project is None:
            self.__project = None
        elif isinstance(project, NMProject):
            self.__project = project
        else:
            e = nmu.typeerror(project, "project", "NMProject")
            raise TypeError(e)
        return None

    @property
    def folder(self) -> NMFolder | None:
        return self.__folder

    @folder.setter
    def folder(self, folder: NMFolder | None) -> None:
        if folder is None:
            self.__folder = None
        elif isinstance(folder, NMFolder):
            if not isinstance(self.__project, NMProject):
                raise RuntimeError("no project has been selected")
            if isinstance(self.__project.folders, NMFolderContainer):
                if folder in self.__project.folders:
                    self.__folder = folder
                else:
                    e = ("folder '%s' is not in project '%s'" %
                         folder.name, self.__project.name)
                    raise ValueError(e)
        else:
            e = nmu.typeerror(folder, "folder", "NMFolder")
            raise TypeError(e)
        return None

    @property
    def data(self) -> NMData | None:
        return self.__data

    @data.setter
    def data(self, data: NMData | None) -> None:
        if data is None:
            self.__data = None
        elif isinstance(data, NMData):
            if not isinstance(self.__folder, NMFolder):
                raise RuntimeError("no folder has been selected")
            if data in self.__folder.data:
                self.__data = data
            else:
                e = ("data '%s' is not in folder '%s'" %
                     data.name, self.__folder.name)
                raise ValueError(e)
        else:
            e = nmu.typeerror(data, "data", "NMData")
            raise TypeError(e)
        return None

    @property
    def dataseries(self) -> NMDataSeries | None:
        return self.__dataseries

    @dataseries.setter
    def dataseries(self, dataseries: NMDataSeries | None) -> None:
        if dataseries is None:
            self.__dataseries = None
        elif isinstance(dataseries, NMDataSeries):
            if not isinstance(self.__folder, NMFolder):
                raise RuntimeError("no folder has been selected")
            if dataseries in self.__folder.dataseries:
                self.__dataseries = dataseries
            else:
                e = ("dataseries '%s' is not in folder '%s'" %
                     dataseries.name, self.__folder.name)
                raise ValueError(e)
        else:
            e = nmu.typeerror(dataseries, "dataseries", "NMDataSeries")
            raise TypeError(e)
        return None

    @property
    def channel(self) -> NMChannel | None:
        return self.__channel

    @channel.setter
    def channel(self, channel: NMChannel | None) -> None:
        if channel is None:
            self.__channel = None
        elif isinstance(channel, NMChannel):
            if not isinstance(self.__dataseries, NMDataSeries):
                raise RuntimeError("no dataseries has been selected")
            if channel in self.__dataseries.channels:
                self.__channel = channel
            else:
                e = ("channel '%s' is not in dataseries '%s'" %
                     channel.name, self.__dataseries.name)
                raise ValueError(e)
        else:
            e = nmu.typeerror(channel, "channel", "NMChannel")
            raise TypeError(e)
        return None

    @property
    def epoch(self) -> NMEpoch | None:
        return self.__epoch

    @epoch.setter
    def epoch(self, epoch: NMEpoch | None) -> None:
        if epoch is None:
            self.__epoch = None
        elif isinstance(epoch, NMEpoch):
            if not isinstance(self.__dataseries, NMDataSeries):
                raise RuntimeError("no dataseries has been selected")
            if epoch in self.__dataseries.epochs:
                self.__epoch = epoch
            else:
                e = ("epoch '%s' is not in dataseries '%s'" %
                     epoch.name, self.__dataseries.name)
                raise ValueError(e)
        else:
            e = nmu.typeerror(epoch, "epoch", "NMEpoch")
            raise TypeError(e)
        return None

    @property
    def select_values(self) -> dict[str, NMObject | None]:
        s: dict[str, NMObject | None] = {}
        s["project"] = self.__project
        s["folder"] = self.__folder
        s["data"] = self.__data
        s["dataseries"] = self.__dataseries
        s["channel"] = self.__channel
        s["epoch"] = self.__epoch
        return s

    @select_values.setter
    def select_values(
        self,
        select_values: dict[str, NMObject | None]
        # see NMManager.select_values
    ) -> None:
        if "project" in select_values:
            v = select_values["project"]
            if v is None or isinstance(v, NMProject):
                self.project = v
            else:
                e = nmu.typeerror(v, "project", "NMProject")
                raise TypeError(e)
        if "folder" in select_values:
            v = select_values["folder"]
            if v is None or isinstance(v, NMFolder):
                self.folder = v
            else:
                e = nmu.typeerror(v, "folder", "NMFolder")
                raise TypeError(e)
        if "data" in select_values:
            v = select_values["data"]
            if v is None or isinstance(v, NMData):
                self.data = v
            else:
                e = nmu.typeerror(v, "data", "NMData")
                raise TypeError(e)
        if "dataseries" in select_values:
            v = select_values["dataseries"]
            if v is None or isinstance(v, NMDataSeries):
                self.dataseries = v
            else:
                e = nmu.typeerror(v, "dataseries", "NMDataSeries")
                raise TypeError(e)
        if "channel" in select_values:
            v = select_values["channel"]
            if v is None or isinstance(v, NMChannel):
                self.channel = v
            else:
                e = nmu.typeerror(v, "channel", "NMChannel")
                raise TypeError(e)
        if "epoch" in select_values:
            v = select_values["epoch"]
            if v is None or isinstance(v, NMEpoch):
                self.epoch = v
            else:
                e = nmu.typeerror(v, "epoch", "NMEpoch")
                raise TypeError(e)

    @property
    def select_keys(self) -> dict[str, str | None]:
        s: dict[str, str | None] = {}
        if self.__project:
            s["project"] = self.__project.name
        else:
            s["project"] = None
        if self.__folder:
            s["folder"] = self.__folder.name
        else:
            s["folder"] = None
        if self.__data:
            s["data"] = self.__data.name
        else:
            s["data"] = None
        if self.__dataseries:
            s["dataseries"] = self.__dataseries.name
        else:
            s["dataseries"] = None
        if self.__channel:
            s["channel"] = self.__channel.name
        else:
            s["channel"] = None
        if self.__epoch:
            s["epoch"] = self.__epoch.name
        else:
            s["epoch"] = None
        return s

    def execute_init(self) -> bool:
        return True  # ok

    def execute(self) -> bool:
        print(self.select_keys)
        return True  # ok

    def execute_finish(self) -> bool:
        return True  # ok
