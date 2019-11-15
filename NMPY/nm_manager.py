# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_folder import FolderContainer
from nm_stats import Stats
import nm_utilities as nmu

nm = None  # holds Manager, accessed via console


class Manager(object):
    """
    NM Manager class
    Main outer class that manages everything

    NM object tree:
        Manager
        Project
            FolderContainer
            Folder (Folder0, Folder1...)
                DataContainer
                Data (Record0, Record1...)
                DataPrefixContainer
                DataPrefix ('Record', 'Wave')
                    ChannelContainer
                    Channel (ChanA, ChanB...)
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
        noise = True
        self.data.make(prefix='Data', channels=2, epochs=3, samples=5,
                       noise=noise)
        self.data.make(prefix='Data', channels=2, epochs=3, samples=5,
                       noise=noise)
        self.data.make(prefix='Wave', channels=3, epochs=8, samples=5,
                       noise=noise)
        # self.dataprefix.new('Test')
        # self.dataprefix.kill('Test')
        self.folder.new()
        # self.folder.duplicate('NMFolder0', 'NMFolder1')
        self.folder.duplicate('NMFolder0', 'NMFolder2')
        self.folder.select = 'NMFolder2'
        # self.eset.add_epoch(['Set1', 'Set2', 'Set3'], [0,3,4,9,11])
        # self.eset.add_epoch('Set1', [0])
        # self.eset.remove_epoch('Set1', [3,4])
        # self.eset.remove_epoch('Set1', [2])
        s1 = self.eset.get('Set1')
        s2 = self.eset.get('Set2')
        self.eset.add_epoch('Set1', [0, 1, 2])
        self.eset.add_epoch('Set2', [3])
        # print(s1.name_list)
        # print(s2.name_list)
        # s1.union(s2)
        # print(s1.name_list)
        self.eset.equation('Set3', ['Set1', '|', 'Set2'])
        # self.eset.add('Set1', range(0, 8, 2))
        # rdic = self.eset.add('SetX', [4])
        # print(rdic)
        # self.eset.select="Set1"
        # clist = self.dataprefix.select.data_select
        # print(clist)

    @property
    def stats(self):
        return self.__stats

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
            p.folder_container.new()  # create default folder
        self.__project = p
        return p

    @property
    def project(self):
        if not self.__project:
            nmu.alert('there is no project')
            return None
        return self.__project

    @property
    def folder(self):
        p = self.project
        if not p:
            return None
        return p.folder_container

    @property
    def folder_list(self):
        fc = self.folder
        if not fc:
            return []
        return fc.name_list

    @property
    def folder_select(self):
        fc = self.folder
        if not fc:
            return None
        if not fc.select:
            nmu.alert('there is no selected folder in project ' +
                      nmu.quotes(self.project.name))
            return None
        return fc.select

    @property
    def data(self):
        fs = self.folder_select
        if not fs:
            return None
        return fs.data_container

    @property
    def data_list(self):
        r = {}
        fs = self.folder_select
        if fs:
            r['folder'] = fs.name
            r['data'] = fs.data_list
        else:
            r['folder'] = ''
            r['data'] = []
        return r

    @property
    def dataprefix(self):
        fs = self.folder_select
        if not fs:
            return None
        return fs.dataprefix_container

    @property
    def dataprefix_select(self):
        pc = self.dataprefix
        if not pc:
            return None
        if not pc.select:
            nmu.alert('there is no selected data prefix in folder ' +
                      nmu.quotes(self.folder_select.name))
            return None
        return pc.select

    @property
    def dataprefix_list(self):
        r = {}
        fs = self.folder_select
        if fs:
            r['folder'] = fs.name
            r['dataprefix'] = fs.dataprefix_list
        else:
            r['folder'] = ''
            r['dataprefix'] = []
        return r

    @property
    def channel_select(self):
        ps = self.dataprefix_select
        if not ps:
            return []  # channel select is a list
        return ps.channel_select

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'All' or ['A', 'B']
        ps = self.dataprefix_select
        if not ps:
            return False
        ps.channel_select = chan_char_list
        return ps.channel_select == chan_char_list

    @property
    def channel_list(self):
        r = {}
        r['folder'] = ''
        r['dataprefix'] = ''
        r['channel'] = []
        fs = self.folder_select
        if fs:
            r['folder'] = fs.name
            ps = fs.dataprefix_container.select
            if ps:
                r['dataprefix'] = ps.name
                r['channel'] = ps.channel_list()
        return r

    @property
    def eset(self):  # epoch set
        ps = self.dataprefix_select
        if not ps:
            return None
        return ps.eset_container

    @property
    def eset_select(self):  # epoch set
        sc = self.eset
        if not sc:
            return None
        if not sc.select:
            nmu.alert('there is no selected epoch set for data prefix ' +
                      nmu.quotes(self.dataprefix_select.tree_path))
            return None
        return sc.select

    @property
    def eset_list(self):  # epoch set
        r = {}
        r['folder'] = ''
        r['dataprefix'] = ''
        r['eset'] = []
        fs = self.folder_select
        if fs:
            r['folder'] = fs.name
            ps = fs.dataprefix_container.select
            if ps:
                r['dataprefix'] = ps.name
                r['eset'] = ps.eset_list
        return r

    @property
    def epoch_select(self):
        ps = self.dataprefix_select
        if not ps:
            return -1
        return ps.epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch):
        ps = self.dataprefix_select
        if not ps:
            return False
        ps.epoch_select = epoch
        return ps.epoch_select == epoch

    @property
    def data_select(self):
        ps = self.dataprefix_select
        if not ps:
            return False
        return ps.data_select

    @property
    def select(self):
        s = {}
        s['project'] = None
        s['folder'] = None
        s['dataprefix'] = None
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
        ps = self.dataprefix.select
        if not ps:
            return s
        s['dataprefix'] = ps
        ss = self.eset.select
        if ss:
            s['eset'] = ss
        s['channel'] = ps.channel_select
        s['epoch'] = ps.epoch_select
        return s

    @property
    def select_tree(self):
        s = {}
        s['project'] = ''
        s['folder'] = ''
        s['dataprefix'] = ''
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
        ps = self.dataprefix.select
        if not ps:
            return s
        s['dataprefix'] = ps.name
        ss = self.eset.select
        if ss:
            s['eset'] = ss.name
        s['channel'] = ps.channel_select
        s['epoch'] = ps.epoch_select
        return s


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__folder_container = FolderContainer(self, "NMFolders")

    def rename(self, name, quiet=False):
        if not nmu.name_ok(name):
            return nmu.error('bad name ' + nmu.quotes(name), quiet=quiet)
        self.__name = name
        return True

    @property
    def folder_container(self):
        return self.__folder_container


if __name__ == '__main__':
    nm = Manager()
