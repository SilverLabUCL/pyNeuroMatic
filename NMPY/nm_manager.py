# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_project import Project
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error
from nm_utilities import history
from nm_utilities import alert

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
        if not name or name.casefold() == 'default':
            name = 'NMProject'
        elif not name_ok(name):
            error('bad name ' + quotes(name))
            return None
        self.__project = Project(parent=None, name=name)
        history('created -> ' + name)
        self.experiment.new()  # create default experiment
        #self.experiment.new()  # create default experiment
        self.folder.new()  # create default folder
        #self.folder.new()  # create default folder
        self.data_test()

    def data_test(self):
        n = True
        self.data.make(prefix='Record', channels=2, epochs=3, samples=5, 
                       noise=n)
        self.data.make(prefix='Record', channels=2, epochs=3, samples=5, 
                       noise=n)
        self.data.make(prefix='Wave', channels=3, epochs=8, samples=5, noise=n)
        # self.dataprefix.new('Test')
        self.eset.add('Set1', range(0, 8, 2))
        rdic = self.eset.add('SetX', [4])
        # print(rdic)
        # self.eset.select="Set1"
        clist = self.eset.get_selected(names=True)
        print(clist)
    
    @property
    def project(self):
        if not self.__project:
            alert('there is no project')
            return None
        return self.__project

    @property
    def experiment(self):
        p = self.project
        if not p:
            return None
        if not p.exp_container:
            alert('there are no experiments in project ' + p.name)
            return None
        return p.exp_container

    @property
    def experiment_select(self):
        ec = self.experiment
        if not ec or not ec.select:
            return ''
        return ec.select.tree_path

    @property
    def folder(self):
        ec = self.experiment
        if not ec:
            return None
        es = ec.select
        if not es:
            alert('there is no selected experiment in project ' +
                  self.project.name)
            return None
        fc = es.folder_container
        if not fc:
            alert('there are no folders in experiment ' + es.name)
            return None
        return fc  # Folder list in selected experiment

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
            alert('there is no selected folder in experiment ' +
                  self.experiment_select)
            return None
        dc = fs.data_container
        if not dc:
            alert('there is no data in folder ' + fs.name)
            return None
        return dc  # Data list in selected folder

    @property
    def dataprefix(self):
        fc = self.folder
        if not fc:
            return None
        fs = fc.select
        if not fs:
            alert('there is no selected folder in experiment ' +
                  self.experiment_select)
            return None
        pc = fs.dataprefix_container
        if not pc:
            alert('there are no data prefixes in folder ' + fs.name)
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
            alert('there is no selected data prefix in folder ' +
                  self.folder_select)
            return None
        cc = ps.channel_container
        if not cc:
            alert('there are no channels for data prefix ' + ps.name)
            return None
        return cc  # Channel list in selected dataprefix

    @property
    def channel_select(self):
        pc = self.dataprefix
        if not pc:
            return 0
        ps = pc.select
        if not ps:
            alert('there is no selected data prefix in folder ' +
                  self.folder_select)
            return 0
        return ps.channel_select

    @channel_select.setter
    def channel_select(self, chan_char_list):  # e.g 'A', 'All' or ['A', 'B']
        pc = self.dataprefix
        if not pc:
            return False
        ps = pc.select
        if not ps:
            alert('there is no selected data prefix in folder ' +
                  self.folder_select)
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
            alert('there is no selected data prefix in folder ' +
                  self.folder_select)
            return None
        sc = ps.eset_container
        if not sc:
            alert('there are no epoch sets for data prefix ' + ps.name)
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
            alert('there is no selected data prefix in folder ' +
                  self.folder_select)
            return 0
        return ps.epoch_select

    @epoch_select.setter
    def epoch_select(self, epoch):
        pc = self.dataprefix
        if not pc:
            return False
        ps = pc.select
        if not ps:
            alert('there is no selected data prefix in folder ' +
                  self.folder_select)
            return False
        ps.epoch_select = epoch
        return ps.epoch_select == epoch

    @property
    def select(self):
        s = {}
        project = None
        exp = None
        folder = None
        prefix = None
        eset = None
        epoch = -1
        chan = ''
        if self.project:
            project = self.project
            if self.experiment.select:
                exp = self.experiment.select
                if self.folder.select:
                    folder = self.folder.select
                    if self.dataprefix.select:
                        prefix = self.dataprefix.select
                        if self.eset.select:
                            eset = self.eset.select
                        epoch = self.dataprefix.select.epoch_select
                        chan = self.dataprefix.select.channel_select
        s['project'] = project
        s['experiment'] = exp
        s['folder'] = folder
        s['dataprefix'] = prefix
        s['eset'] = eset
        s['epoch'] = epoch
        s['channel'] = chan
        return s

    @property
    def select_names(self):
        s = {}
        project = 'None'
        exp = 'None'
        folder = 'None'
        prefix = 'None'
        eset = 'None'
        epoch = -1
        chan = 'None'
        if self.project:
            project = self.project.name
            if self.experiment.select:
                exp = self.experiment.select.name
                if self.folder.select:
                    folder = self.folder.select.name
                    if self.dataprefix.select:
                        prefix = self.dataprefix.select.name
                        if self.eset.select:
                            eset = self.eset.select.name
                        epoch = self.dataprefix.select.epoch_select
                        chan = self.dataprefix.select.channel_select
        s['project'] = self.project.name
        s['experiment'] = exp
        s['folder'] = folder
        s['dataprefix'] = prefix
        s['eset'] = eset
        s['epoch'] = epoch
        s['channel'] = chan
        return s

    def stats(self, project, experiment, folder, dataprefix, channel, eset):
        print(project, experiment, folder, dataprefix, channel, eset)


if __name__ == '__main__':
    nm = Manager()
    # s = nm.selected_names
    # nm.stats(**s)
    # nm.exp.folder_new(name="NMFolder0")
    # nm.exp.folder.dataprefix_new(prefix="Record")
    # nm.exp.folder_open_hdf5()
