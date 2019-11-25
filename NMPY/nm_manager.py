# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_project import Project
from nm_stats import Stats
import nm_utilities as nmu

nm = None  # holds Manager, accessed via console


class Manager(object):
    """
    NM Manager class
    Main outer class that manages everything

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
                    EpochSetContainer
                    EpochSet (All, Set1, Set2...)
    """
    def __init__(self):
        self.__project = None
        self.project_new('NMProject')
        self.__gui = nmc.GUI
        self.data_test()
        self.__stats = Stats(self)
        # nm.exp.folder_open_hdf5()

    def data_test(self):
        noise = [0, 0.1]
        self.dataseries.make(prefix='Data', channels=2, epochs=3, samples=5,
                             noise=noise)
        self.dataseries.make(prefix='Data', channels=2, epochs=3, samples=5,
                             noise=noise)
        self.dataseries.make(prefix='Wave', channels=3, epochs=8, samples=5,
                             noise=noise)
        # self.dataseries.new('Test')
        # self.dataseries.kill('Test')
        self.folder.new()
        # self.folder.duplicate('NMFolder0', 'NMFolder1')
        self.folder.duplicate('NMFolder0', 'NMFolder2')
        self.folder.select = 'NMFolder2'
        # self.eset.add_epoch(['Set1', 'Set2', 'Set3'], [0,3,4,9,11])
        # self.eset.add_epoch('Set1', [0])
        # self.eset.remove_epoch('Set1', [3,4])
        # self.eset.remove_epoch('Set1', [2])
        # s1 = self.eset.get('Set1')
        # s2 = self.eset.get('Set2')
        # self.eset.add_epoch('Set1', [0, 1, 2])
        # self.eset.add_epoch('Set2', [3])
        # print(s1.names)
        # print(s2.names)
        # s1.union(s2)
        # print(s1.names)
        # self.eset.equation('Set3', ['Set1', '|', 'Set2'])
        # self.eset.select = 'Set1'
        # self.eset.add('Set1', range(0, 8, 2))
        # rdic = self.eset.add('SetX', [4])
        # print(rdic)
        # self.eset.select="Set1"
        # clist = self.dataseries.select.data_names
        # print(clist)

    @property
    def gui(self):
        return self.__gui

    @gui.setter
    def gui(self, on):
        self.__gui = on

    def project_new(self, name, quiet=False):
        """Create a new project"""
        if self.__project:
            q = ('do you want to save ' + nmu.quotes(self.__project.name) +
                 ' before creating a new project?')
            ync = nmu.input_yesno(q, cancel=True)
            if ync == 'y':
                path = "NOWHERE"
                if not self.__project.save(path):
                    return None  # cancel
            elif ync == 'n':
                pass
            else:
                return None  # cancel
        if not name or name.casefold() == 'default':
            name = 'NMProject'
        elif not nmu.name_ok(name):
            nmu.error('bad name ' + nmu.quotes(name))
            return None
        p = Project(self, name)
        nmu.history('created' + nmc.S0 + name)
        if p:
            p.folder.new()  # create default folder
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

    def __folder_select(self):
        fc = self.folder
        if fc:
            return fc.select
        return None

    @property
    def data(self):
        fs = self.__folder_select()
        if fs:
            return fs.data
        return None

    @property
    def dataseries(self):
        fs = self.__folder_select()
        if fs:
            return fs.dataseries
        return None

    def __dataseries_select(self):
        pc = self.dataseries
        if pc:
            return pc.select
        return None

    @property
    def channel(self):
        ps = self.__dataseries_select()
        if ps:
            return ps.channel
        return None

    @property
    def channel_select(self):
        ps = self.__dataseries_select()
        if ps:
            return ps.channel_select
        return []  # channel select is a list

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'All' or ['A', 'B']
        ps = self.__dataseries_select()
        if ps:
            ps.channel_select = chan_char_list
            return ps.channel_select == chan_char_list
        return None

    @property
    def eset(self):  # epoch set
        ps = self.__dataseries_select()
        if ps:
            return ps.eset
        return None

    @property
    def epoch_select(self):
        ps = self.__dataseries_select()
        if ps:
            return ps.epoch_select
        return [-1]

    @epoch_select.setter
    def epoch_select(self, epoch_list):
        ps = self.__dataseries_select()
        if ps:
            ps.epoch_select = epoch_list
            return ps.epoch_select == epoch_list
        return False

    @property
    def data_select(self):
        ps = self.__dataseries_select()
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
    def stats(self):
        return self.__stats


if __name__ == '__main__':
    nm = Manager()
