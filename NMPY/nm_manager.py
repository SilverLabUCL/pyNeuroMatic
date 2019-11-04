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
        self.dataprefix.new('Test')
        #self.dataprefix.kill('Test')
        self.folder.new()
        #self.folder.duplicate('NMFolder0', 'NMFolder1')
        self.folder.duplicate('NMFolder0', 'NMFolder2')
        #self.eset.add('Set1', range(0, 8, 2))
        #rdic = self.eset.add('SetX', [4])
        # print(rdic)
        # self.eset.select="Set1"
        #clist = self.dataprefix.select.data_select
        #print(clist)

    @property
    def stats(self):
        return self.__stats

    @property
    def gui(self):
        return self.__gui

    @gui.setter
    def gui(self, on):
        self.__gui = on

    def project_new(self, name):
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
        if not p.folder_container:
            nmu.alert('there are no folders in project ' + p.name)
            return None
        return p.folder_container

    @property
    def folder_select(self):
        fc = self.folder
        if not fc or not fc.select:
            return ''
        return fc.select.tree_path

    @property
    def data(self):
        fc = self.folder
        if not fc:
            return None
        fs = fc.select
        if not fs:
            nmu.alert('there is no selected folder')
            return None
        dc = fs.data_container
        if not dc:
            nmu.alert('there is no data in folder ' + fs.name)
            return None
        return dc  # Data list in selected folder

    @property
    def dataprefix(self):
        fc = self.folder
        if not fc:
            return None
        fs = fc.select
        if not fs:
            nmu.alert('there is no selected folder')
            return None
        pc = fs.dataprefix_container
        if not pc:
            nmu.alert('there are no data prefixes in folder ' + fs.name)
            return None
        return pc  # DataPrefix list in selected folder

    @property
    def dataprefix_select(self):
        pc = self.dataprefix
        if not pc or not pc.select:
            return ''
        return pc.select.tree_path

    @property
    def channel(self):
        pc = self.dataprefix
        if not pc:
            return None
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return None
        cc = ps.channel_container
        if not cc:
            nmu.alert('there are no channels for data prefix ' + ps.name)
            return None
        return cc  # Channel list in selected dataprefix

    @property
    def channel_select(self):
        pc = self.dataprefix
        if not pc:
            return 0
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return 0
        return ps.channel_select

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'All' or ['A', 'B']
        pc = self.dataprefix
        if not pc:
            return False
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return False
        ps.channel_select = chan_char_list
        return ps.channel_select == chan_char_list

    @property
    def eset(self):  # epoch set
        pc = self.dataprefix
        if not pc:
            return None
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return None
        sc = ps.eset_container
        if not sc:
            nmu.alert('there are no epoch sets for data prefix ' + ps.name)
            return None
        return sc  # EpochSet list in selected dataprefix

    @property
    def eset_select(self):  # epoch set
        sc = self.eset
        if not sc or not sc.select:
            return ''
        return sc.select.tree_path

    @property
    def epoch_select(self):
        pc = self.dataprefix
        if not pc:
            return 0
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return 0
        return ps.epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch):
        pc = self.dataprefix
        if not pc:
            return False
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return False
        ps.epoch_select = epoch
        return ps.epoch_select == epoch

    @property
    def data_select(self):
        pc = self.dataprefix
        if not pc:
            return False
        ps = pc.select
        if not ps:
            fs = self.folder_select
            nmu.alert('there is no selected data prefix in folder ' + fs)
            return False
        return ps.data_select

    @property
    def select(self):
        s = {}
        project = None
        folder = None
        prefix = None
        chan = ''
        eset = None
        epoch = -1
        if self.project:
            project = self.project
            if self.folder.select:
                folder = self.folder.select
                if self.dataprefix.select:
                    prefix = self.dataprefix.select
                    chan = self.dataprefix.select.channel_select
                    if self.eset.select:
                        eset = self.eset.select
                    epoch = self.dataprefix.select.epoch_select
        s['project'] = project
        s['folder'] = folder
        s['dataprefix'] = prefix
        s['channel'] = chan
        s['eset'] = eset
        s['epoch'] = epoch
        return s

    @property
    def select_names(self):
        s = {}
        project = 'None'
        folder = 'None'
        prefix = 'None'
        chan = 'None'
        eset = 'None'
        epoch = -1
        if self.project:
            project = self.project.name
            if self.folder.select:
                folder = self.folder.select.name
                if self.dataprefix.select:
                    prefix = self.dataprefix.select.name
                    chan = self.dataprefix.select.channel_select
                    if self.eset.select:
                        eset = self.eset.select.name
                    epoch = self.dataprefix.select.epoch_select
        s['project'] = project
        s['folder'] = folder
        s['dataprefix'] = prefix
        s['channel'] = chan
        s['eset'] = eset
        s['epoch'] = epoch
        return s

    def directory(self):
        if not self.project:
            print('no project')
            return None
        return self.project.directory()


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__folder_container = FolderContainer(self, "NMFolders")

    def rename(self, name):
        if not nmu.name_ok(name):
            return nmu.error('bad name ' + nmu.quotes(name))
        self.__name = name
        return True

    @property
    def folder_container(self):
        return self.__folder_container

    def directory(self):
        if not self.folder_container:
            print('no folder container')
            return []
        flist = self.folder_container.name_list
        if len(flist) == 0:
            print('no folders')
            return []
        for f in flist:
            print(f)
        return flist

if __name__ == '__main__':
    nm = Manager()
