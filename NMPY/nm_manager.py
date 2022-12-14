# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import datetime
# import matplotlib

from nm_project import NMProject
from nm_folder import NMFolderContainer
from nm_stats import Stats
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, Union

nm = None  # holds Manager, accessed via console


class NMManager(object):  # TODO: can manager be a NMObejct or static?
    """
    NM Manager class

    NM class tree:

    NMManager (root)
        NMProject
            NMFolderContainer
                NMFolder (Folder0, Folder1...)
                    NMDataContainer
                        NMData (Record0, Record1... Avg0, Avg1)
                    NMDataSeriesContainer
                        NMDataSeries (Record, Avg...)
                            NMChannelContainer
                                NMChannel (A, B, C...)
                            NMDataSeriesSetContainer
                                NMDataSeriesSet (All, Set1, Set2...)
    """
    def __init__(
        self,
        name: str = 'NeuroMatic Manager',
        new_project: bool = True,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(name, str):
            self.__name = name
        else:
            self.__name = 'NeuroMatic Manager'
        self.__project = None
        self.__created = str(datetime.datetime.now())
        self.__configs = nmp.Configs()
        self.__configs.quiet = quiet
        self.__stats = Stats()
        h = 'created ' + nmu.quotes(self.__name)
        self._history(h, quiet=quiet)
        if new_project:
            self.project_new(quiet=quiet)

    @property
    def parameters(self) -> Dict[str, str]:
        k = {'name': self.__name}
        k.update({'created': self.__created})
        return k

    def project_new(
        self,
        name: str = 'default',
        new_folder: bool = True,
        quiet: bool = nmp.QUIET
    ) -> nmu.NMProjectType:
        """Create a new project"""
        if not isinstance(name, str):
            # e = nmu.type_error('name', 'string') DOES NOT EXIST
            e = "ERROR: nm.Manager.project_new: bad name: expected string"
            raise TypeError(e)
        if not nmu.name_ok(name) or name.lower() == 'select':
            # e = nmu.value_error('name') DOES NOT EXIST
            e = "ERROR: nm.Manager.project_new: bad name: " + name
            raise ValueError(e)
        if not name or name.lower() == 'default':
            name = 'NMProject'
        if self.__project:
            n = nmu.quotes(self.__project.name)
            q = ('do you want to save ' + n +
                 ' before creating a new project?')
            ync = nmu.input_yesno(q, cancel=True)
            if ync.lower() == 'y':
                path = "NOWHERE"
                if not self.__project.save(path):
                    return None  # cancel
            elif ync.lower() == 'n':
                pass
            else:
                self._history('cancel', quiet=quiet)
                return None  # cancel
        p = NMProject(parent=self, name=name)
        h = 'created ' + nmu.quotes(name)
        self._history(h, quiet=quiet)
        if new_folder and p and p.folder:
            p.folder.new(quiet=quiet)  # create default folder
        self.__project = p
        return p

    @property
    def project(self) -> nmu.NMProjectType:
        return self.__project

    @property
    def folder(self) -> nmu.NMFolderContainerType:
        if isinstance(NMFolderContainer, self.__project):
            return self.__project.folder
        return None

    def _folder_select(self) -> nmu.NMFolderType:
        f = self.folder
        if f:
            return f.select
        return None

    @property
    def data(self) -> nmu.NMDataContainerType:
        f = self._folder_select()
        if f:
            return f.data
        return None

    @property
    def dataseries(self) -> nmu.NMDataSeriesContainerType:
        f = self._folder_select()
        if f:
            return f.dataseries
        return None

    def _dataseries_select(self) -> nmu.NMDataSeriesType:
        ds = self.dataseries
        if ds:
            return ds.select
        return None

    @property
    def channel(self) -> nmu.NMChannelContainerType:
        ds = self._dataseries_select()
        if ds:
            return ds.channel
        return None

    @property
    def channel_select(self) -> List[str]:
        ds = self._dataseries_select()
        if ds:
            return ds.channel_select
        return []  # channel select is a list

    @channel_select.setter
    def channel_select(
        self,
        chan_char_list: Union[str, List[str]]
    ) -> bool:
        # e.g 'A', 'All' or ['A', 'B']
        ds = self._dataseries_select()
        if ds:
            ds.channel_select = chan_char_list
            return ds.channel_select == chan_char_list
        return False

    @property
    def sets(self) -> nmu.NMDataSeriesSetContainerType:  # data-series sets
        ds = self._dataseries_select()
        if ds:
            return ds.sets
        return None

    @property
    def epoch_select(self) -> List[int]:
        ds = self._dataseries_select()
        if ds:
            return ds.epoch_select
        return []

    @epoch_select.setter
    def epoch_select(
        self,
        epoch_list: List[int]
    ) -> bool:
        ds = self._dataseries_select()
        if ds:
            ds.epoch_select = epoch_list
            return ds.epoch_select == epoch_list
        return False

    @property
    def data_select(self) -> Dict[str, object]:
        ds = self._dataseries_select()
        if ds:
            return ds.data_select
        return {}

    @property
    def select(self) -> Dict[str, object]:
        s = {}
        s['project'] = None
        s['folder'] = None
        s['dataseries'] = None
        s['set'] = None
        s['channel'] = []
        s['epoch'] = -1
        if not self.__project:
            return s
        s['project'] = self.__project
        fs = self.folder.select
        if not fs:
            return s
        s['folder'] = fs
        ps = self.dataseries.select
        if not ps:
            return s
        s['dataseries'] = ps
        ss = self.sets.select
        if ss:
            s['set'] = ss
        s['channel'] = ps.channel_select
        s['epoch'] = ps.epoch_select
        return s

    @property
    def select_names(
        self,
        names: bool = True
    ) -> Dict[str, object]:
        s = {}
        s['project'] = ''
        s['folder'] = ''
        s['dataseries'] = ''
        s['set'] = ''
        s['channel'] = []
        s['epoch'] = -1
        if not self.__project:
            return s
        s['project'] = self.__project.name
        fs = self.folder.select
        if not fs:
            return s
        s['folder'] = fs.name
        ps = self.dataseries.select
        if not ps:
            return s
        s['dataseries'] = ps.name
        ss = self.sets.select
        if ss:
            s['set'] = ss.name
        s['channel'] = ps.channel_select
        s['epoch'] = ps.epoch_select
        return s

    @property
    def configs(self):
        return self.__configs

    @property
    def stats(self):
        return self.__stats

    def _alert(
        self,
        message,
        tp='',
        quiet=False,
        frame=2
    ):
        return nmu.history(message, title='ALERT', tp=tp, frame=frame,
                           red=True, quiet=self._quiet(quiet))

    def _error(
        self,
        message,
        tp='',
        quiet=False,
        frame=2
    ):
        return nmu.history(message, title='ERROR', tp=tp, frame=frame,
                           red=True, quiet=self._quiet(quiet))

    def _history(
        self,
        message,
        tp='',
        quiet=False,
        frame=2
    ):
        return nmu.history(message, tp=tp, frame=frame,
                           quiet=self._quiet(quiet))

    def _quiet(self, quiet=nmp.QUIET):
        if self.configs.quiet:  # manager config quiet overrides
            return True
        return quiet


if __name__ == '__main__':
    nm = NMManager()
