# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import datetime

import nm_configs as nmconfig
from nm_container import NMObject
from nm_experiment import ExperimentContainer
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error
from nm_utilities import history

nm = None  # holds Manager, accessed via console


class Manager(object):
    """
    NM Manager class
    Main outer class that manages everything

    NM object tree:
        Project
            ExperimentContainer
            Experiment (Exp0, Exp1...)
                    Folder Container
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
        self.project_new('NMProject')

    def project_new(self, name):
        """Create new project"""
        self.__project = Project(None, name)
        self.experiment.new()  # create default experiment
        #self.experiment.new()  # create default experiment
        self.folder.new()  # create default folder
        #self.folder.new()  # create default folder
        self.data_test()

    def data_test(self):
        n = True
        self.data.make(prefix='Record', channels=2, epochs=3, samples=5, 
                       noise=n)
        if True:
            return False
        self.data.make(prefix='Record', channels=2, epochs=3, samples=5, 
                       noise=n)
        self.data.make(prefix='Wave', channels=3, epochs=8, samples=5, noise=n)
        self.dataprefix.new('Test')
        for i in range(0, 8, 2):
            self.eset.add('Set1', i, quiet=False)
        self.eset.add('SetX', 4, quiet=False)
        # self.eset.select="Set1"
        clist = self.eset.get_selected()
        print(clist)

    @property
    def project(self):
        return self.__project

    @property
    def experiment(self):
        return self.__project.exp_container

    @property
    def folder(self):
        ec = self.__project.exp_container
        if ec.count == 0:
            history('no experiments')
            return None
        exp = ec.get('selected')
        if not exp:
            history('no selected experiment')
            return None
        return exp.folder_container  # Folder objs in selected experiment

    @property
    def data(self):
        f = self.folder
        if not f:
            return None
        if f.count == 0:
            history('no folders')
            return None
        s = f.get('selected')
        if not s:
            history('no selected folder')
            return None
        return s.data_container  # Data objs in seleted folder

    @property
    def dataprefix(self):
        f = self.folder
        if not f:
            return None
        if f.count == 0:
            history('no folders')
            return None
        s = f.get('selected')
        if not s:
            history('no selected folder')
            return None
        return s.dataprefix_container  # DataPrefix objs in seleted folder

    @property
    def channel(self):
        p = self.dataprefix
        if not p:
            return None
        if p.count == 0:
            history('no data prefixes')
            return None
        s = p.get('selected')
        if not s:
            history('no selected data prefix')
            return None
        return s.channel_container  # Channel objs in selected dataprefix

    @property
    def eset(self):  # epoch set
        p = self.dataprefix
        if not p:
            return None
        if p.count == 0:
            history('no data prefixes')
            return None
        s = p.get('selected')
        if not s:
            history('no selected data prefix')
            return None
        return s.eset_container  # EpochSet objs in selected dataprefix

    @property
    def select(self):
        s = {}
        exp = 'None'
        folder = 'None'
        prefix = 'None'
        chan = 'None'
        epoch = 0
        eset = 'All'
        if self.experiment.select:
            exp = self.experiment.select
            if self.folder.select:
                folder = self.folder.select
                if self.dataprefix.select:
                    prefix = self.dataprefix.select
                    chan = self.dataprefix.select.channel_select
                    epoch = self.dataprefix.select.epoch_select
                    if self.eset.select:
                        eset = self.eset.select
        s['project'] = self.project
        s['experiment'] = exp
        s['folder'] = folder
        s['dataprefix'] = prefix
        s['channel'] = chan
        s['epoch'] = epoch
        s['eset'] = eset
        return s

    @property
    def select_names(self):
        s = {}
        exp = 'None'
        folder = 'None'
        prefix = 'None'
        chan = 'None'
        epoch = 0
        eset = 'All'
        if self.experiment.select:
            exp = self.experiment.select.name
            if self.folder.select:
                folder = self.folder.select.name
                if self.dataprefix.select:
                    prefix = self.dataprefix.select.name
                    chan = self.dataprefix.select.channel_select
                    epoch = self.dataprefix.select.epoch_select
                    if self.eset.select:
                        eset = self.eset.select.name
        s['project'] = self.project.name
        s['experiment'] = exp
        s['folder'] = folder
        s['dataprefix'] = prefix
        s['channel'] = chan
        s['epoch'] = epoch
        s['eset'] = eset
        return s

    def stats(self, project, experiment, folder, dataprefix, channel, eset):
        print(project, experiment, folder, dataprefix, channel, eset)


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__exp_container = ExperimentContainer(self, 'NMExps')
        self.__date = str(datetime.datetime.now())

    def rename(self, name):
        if not name_ok(name):
            return error('bad name ' + quotes(name))
        self.__name = name
        return True

    @property
    def exp_container(self):
        return self.__exp_container


if __name__ == '__main__':
    nm = Manager()
    # s = nm.selected_names
    # nm.stats(**s)
    # nm.exp.folder_new(name="NMFolder0")
    # nm.exp.folder.dataprefix_new(prefix="Record")
    # nm.exp.folder_open_hdf5()
