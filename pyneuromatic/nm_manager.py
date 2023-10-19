# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import matplotlib
from typing import Dict, List, Union

from pyneuromatic.nm_object import NMObject
import pyneuromatic.nm_preferences as nmp
from pyneuromatic.nm_project import NMProjectContainer
from pyneuromatic.nm_stats import Stats
import pyneuromatic.nm_utilities as nmu

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


class NMManager(NMObject):
    def __init__(
        self,
        # parent: Union[object, None] = None,
        name: str = "nm",
        project_name: str = "project0",
        quiet: bool = False,
        copy: Union[nmu.NMProjectType, None] = None,  # see copy()
    ) -> None:
        super().__init__(parent=None, name=name, copy=copy)  # NMObject

        # self.__configs = nmp.Configs()
        # self.__configs.quiet = quiet
        self.__project_container = NMProjectContainer(parent=self)
        self.__stats = Stats()

        h = "created Neuromatic manager '%s'" % self.name
        # self._history(h, quiet=quiet)

        if project_name is None:
            pass
        elif isinstance(project_name, str):
            p = self.__project_container.new(project_name)
            if p:
                self.__project_container.select_key = project_name
        else:
            e = nmu.typeerror(project_name, "project_name", "string")
            raise TypeError(e)

    @property
    def projects(self) -> nmu.NMProjectContainerType:
        return self.__project_container

    @property
    def select_values(self) -> Dict[str, nmu.NMObjectType]:
        s = {}
        s["project"] = None
        s["folder"] = None
        s["data"] = None
        s["dataseries"] = None
        s["channel"] = None
        s["epoch"] = None
        p = self.__project_container.select_value
        if p:
            s["project"] = p
        else:
            return s
        f = p.folders.select_value
        if f:
            s["folder"] = f
        else:
            return s
        d = f.data.select_value
        if d:
            s["data"] = d
        ds = f.dataseries.select_value
        if ds:
            s["dataseries"] = ds
        else:
            return s
        c = ds.channels.select_value
        if c:
            s["channel"] = c
        e = ds.epochs.select_value
        if e:
            s["epoch"] = e
        return s

    @property
    def select_keys(self) -> Dict[str, str]:
        s1 = self.select_values
        s2 = {}
        for k, v in s1.items():
            if isinstance(v, NMObject):
                s2.update({k: v.name})
            else:
                s2.update({k: ""})
        return s2

    @select_keys.setter
    def select_keys(self, select: Dict[str, str]) -> None:
        return self._select_keys_set(select)

    def _select_keys_set(
        self,
        select: Dict[str, str]
        # quiet
    ) -> None:
        if not isinstance(select, dict):
            e = nmu.typeerror(select, "select", "dict")
            raise TypeError(e)
        for key, value in select.items():
            if not isinstance(key, str):
                e = nmu.typeerror(key, "key", "string")
                raise TypeError(e)
            if not isinstance(value, str):
                e = nmu.typeerror(value, "value", "string")
                raise TypeError(e)
        if "project" in select:
            self.__project_container.select_key = select["project"]
        p = self.__project_container.select_value
        if "folder" in select:
            p.folders.select_key = select["folder"]
        f = p.folders.select_value
        if "data" in select:
            f.data.select_key = select["data"]
        if "dataseries" in select:
            f.dataseries.select_key = select["dataseries"]
        ds = f.dataseries.select_value
        if "channel" in select:
            ds.channels.select_key = select["channel"]
        if "epoch" in select:
            ds.epochs.select_key = select["epoch"]
        return None

    def execute_values(
        self, dataseries_priority: bool = True
    ) -> List[Dict[str, nmu.NMObjectType]]:
        elist = []
        for p in self.__project_container.execute_values:
            flist = p.folders.execute_values
            for f in flist:
                dslist = f.dataseries.execute_values
                if dataseries_priority and dslist:
                    for ds in dslist:
                        for c in ds.channels.execute_values:
                            for e in ds.epochs.execute_values:
                                x = {}
                                x["project"] = p
                                x["folder"] = f
                                x["dataseries"] = ds
                                x["channel"] = c
                                x["epoch"] = e
                                elist.append(x)
                else:
                    dlist = f.data.execute_values
                    for d in dlist:
                        x = {}
                        x["project"] = p
                        x["folder"] = f
                        x["data"] = d
                        elist.append(x)
        return elist

    def execute_keys(self, dataseries_priority: bool = True) -> List[Dict[str, str]]:
        elist = []
        elist2 = self.execute_values(dataseries_priority)
        for e in elist2:
            e2 = {}
            for k, o in e.items():
                e2[k] = o.name
            elist.append(e2)
        return elist

    def execute_keys_set(self, execute: Dict[str, str]) -> List[Dict[str, str]]:
        # sets are not allowed with project, folder, dataseries - too complex
        # can specify 'data' or 'dataseries' but not both

        if not isinstance(execute, dict):
            e = nmu.typeerror(execute, "execute", "dict")
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

        if "folder" not in execute:
            e = "missing execute 'folder' key"
            raise KeyError(e)
        value = execute["folder"]
        if value.lower() == "select":
            f = p.folders.select_key
            if f in p.folders:
                p.folders.execute_key = "select"
            else:
                e = "bad folder select: %s" % f
                raise ValueError(e)
        elif value.lower() == "all":
            e = "'all' folders is not allowed in this function"
            raise ValueError(e)
        elif value in p.folders:
            p.folders.select_key = value
            p.folders.execute_key = "select"
        elif value in p.folders.sets:
            e = "folder sets are not allowed in this function"
            raise ValueError(e)
        else:
            e = "unknown folder execute value: %s" % value
            raise ValueError(e)
        f = p.folders.select_value
        if f not in p.folders:
            e = "bad folder select: %s" % f
            raise ValueError(e)

        if "data" in execute:
            f.data.execute_key = execute["data"]
            return self.execute_keys(dataseries_priority=False)  # finished

        if "dataseries" not in execute:
            e = "missing execute 'dataseries' key"
            raise KeyError(e)
        value = execute["dataseries"]
        if value.lower() == "select":
            ds = f.dataseries.select_key
            if ds in f.dataseries:
                f.dataseries.execute_key = "select"
            else:
                e = "bad dataseries select: %s" % ds
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
        self.__project_container.execute_key = "select"
        for p in self.__project_container.values():
            p.folders.execute_key = "select"
            for f in p.folders.values():
                f.data.execute_key = "select"
                f.dataseries.execute_key = "select"
                for ds in f.dataseries.values():
                    ds.channels.execute_key = "select"
                    ds.epochs.execute_key = "select"
        return None

    # @property
    # def configs(self):
    #     return self.__configs

    @property
    def stats(self):
        return self.__stats

    """
    def _quiet(self, quiet=nmp.QUIET):
        if self.configs.quiet:  # config quiet overrides
            return True
        return quiet
    """


if __name__ == "__main__":
    nm = NMManager(name="NeuroMaticManager")
    # p = nm.projects.new("project0")
    # p.folders.new("folder0")
