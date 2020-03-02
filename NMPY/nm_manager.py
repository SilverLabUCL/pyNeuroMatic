# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import datetime

from nm_project import Project
from nm_stats import Stats
import nm_preferences as nmp
import nm_utilities as nmu

nm = None  # holds Manager, accessed via console


class Manager(object):
    """
    NM Manager class

    NM class tree:
        Manager (root)
        Project
            FolderContainer
            Folder (Folder0, Folder1...)
                DataContainer
                Data (Record0, Record1...)
                    NoteContainer
                    Note (Note0, Note1, Note2...)
                DataSeriesContainer
                DataSeries (Record, Wave...)
                    ChannelContainer
                    Channel (A, B, C...)
                    DataSeriesSetContainer
                    DataSeriesSet (All, Set1, Set2...)
    """
    def __init__(self, name='NeuroMatic Manager', new_project=True,
                 quiet=nmp.QUIET):
        new_project = nmu.bool_check(new_project, True)
        if isinstance(name, str):
            self.__name = name
        else:
            self.__name = 'NeuroMatic Manager'
        self.__project = None
        self.__date = str(datetime.datetime.now())
        self.__configs = nmp.Configs()
        self.__configs.quiet = quiet
        self.__stats = Stats()
        self._history('created ' + nmu.quotes(self.__name), quiet=quiet)
        if new_project:
            self.project_new(quiet=quiet)

    @property
    def parameters(self):
        k = {'name': self.__name}
        k.update({'date': self.__date})
        return k

    def project_new(self, name='default', new_folder=True, quiet=nmp.QUIET):
        """Create a new project"""
        new_folder = nmu.bool_check(new_folder, True)
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not nmu.name_ok(name) or name.lower() == 'select':
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name or name.lower() == 'default':
            name = nmp.PROJECT_NAME
        if self.__project:
            n = nmu.quotes(self.__project.name)
            q = ('do you want to save ' + n +
                 ' before creating a new project?')
            ync = nmu.input_yesno(q, cancel=True)
            if ync == 'y':
                path = "NOWHERE"
                if not self.__project.save(path):
                    return None  # cancel
            elif ync == 'n':
                pass
            else:
                self._history('cancel', quiet=quiet)
                return None  # cancel
        p = Project(self, name)
        self._history('created ' + nmu.quotes(name), quiet=quiet)
        if new_folder and p and p.folder:
            p.folder.new(quiet=quiet)  # create default folder
        self.__project = p
        return p

    @property
    def project(self):
        return self.__project

    @property
    def folder(self):
        if self.__project:
            return self.__project.folder
        return None

    def _folder_select(self):
        fc = self.folder
        if fc:
            return fc.select
        return None

    @property
    def data(self):
        fs = self._folder_select()
        if fs:
            return fs.data
        return None

    @property
    def dataseries(self):
        fs = self._folder_select()
        if fs:
            return fs.dataseries
        return None

    def _dataseries_select(self):
        pc = self.dataseries
        if pc:
            return pc.select
        return None

    @property
    def channel(self):
        ps = self._dataseries_select()
        if ps:
            return ps.channel
        return None

    @property
    def channel_select(self):
        ps = self._dataseries_select()
        if ps:
            return ps.channel_select
        return []  # channel select is a list

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'All' or ['A', 'B']
        ps = self._dataseries_select()
        if ps:
            ps.channel_select = chan_char_list
            return ps.channel_select == chan_char_list
        return None

    @property
    def eset(self):  # epoch set
        ps = self._dataseries_select()
        if ps:
            return ps.eset
        return None

    @property
    def epoch_select(self):
        ps = self._dataseries_select()
        if ps:
            return ps.epoch_select
        return [-1]

    @epoch_select.setter
    def epoch_select(self, epoch_list):
        ps = self._dataseries_select()
        if ps:
            ps.epoch_select = epoch_list
            return ps.epoch_select == epoch_list
        return False

    @property
    def data_select(self):
        ps = self._dataseries_select()
        if ps:
            return ps.data_select
        return False

    @property
    def select(self):
        s = {}
        s['project'] = None
        s['folder'] = None
        s['dataseries'] = None
        s['eset'] = None
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
        ss = self.eset.select
        if ss:
            s['eset'] = ss
        s['channel'] = ps.channel_select
        s['epoch'] = ps.epoch_select
        return s

    @property
    def select_names(self, names=True):
        names = nmu.bool_check(names, True)
        s = {}
        s['project'] = ''
        s['folder'] = ''
        s['dataseries'] = ''
        s['eset'] = ''
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
        ss = self.eset.select
        if ss:
            s['eset'] = ss.name
        s['channel'] = ps.channel_select
        s['epoch'] = ps.epoch_select
        return s

    @property
    def configs(self):
        return self.__configs

    @property
    def stats(self):
        return self.__stats

    def _alert(self, message, tp='', quiet=False, frame=2):
        return nmu.history(message, title='ALERT', tp=tp, frame=frame,
                           red=True, quiet=self._quiet(quiet))

    def _error(self, message, tp='', quiet=False, frame=2):
        return nmu.history(message, title='ERROR', tp=tp, frame=frame,
                           red=True, quiet=self._quiet(quiet))

    def _history(self, message, tp='', quiet=False, frame=2):
        return nmu.history(message, tp=tp, frame=frame,
                           quiet=self._quiet(quiet))

    def _quiet(self, quiet=nmp.QUIET):
        quiet = nmu.bool_check(quiet, nmp.QUIET)
        if self.configs.quiet:  # manager config quiet overrides
            return True
        return quiet


if __name__ == '__main__':
    nm = Manager()
