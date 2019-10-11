# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import datetime
import nm_configs as nmconfig
from nm_experiment import ExperimentContainer
from nm_utilities import name_ok
from nm_utilities import error
from nm_utilities import history

nm = None  # holds NM Manager, accessed via console


class Manager(object):
    """
    NM Manager class
    Main outer class that manages everything
    """
    def __init__(self):
        self.project_new("NMProject")

    def project_new(self, name):
        """Create new project"""
        self.__project = Project(name)
        self.experiment.new()  # create default experiment
        self.folder.new()  # create default folder
        self.waveprefix_test()

    def waveprefix_test(self):
        self.wave.make(prefix="Record", nchan=2, nwaves=3, points=5)
        self.wave.make(prefix="Wave", nchan=1, nwaves=4, points=5)
        self.waveprefix.new("Record")
        self.waveprefix.new("Wave")
        self.waveprefix.new("Test")
        # p = self.waveprefix.new(waveprefix)
        # p.wave_names_mock(channels=2, waves=5)

    @property
    def project(self):
        return self.__project

    @property
    def experiment(self):
        return self.__project.experiment  # container of experiments

    @property
    def folder(self):
        e = self.__project.experiment
        if e.items == 0:
            history("no experiments")
            return None
        s = e.get("SELECTED")
        if not s:
            history("no selected experiment")
            return None
        return s.folder  # container of folders in selected experiment

    @property
    def wave(self):
        f = self.folder
        if not f:
            return None
        if f.items == 0:
            history("no folders")
            return None
        s = f.get("SELECTED")
        if not s:
            history("no selected folder")
            return None
        return s.wave  # container of waves in seleted folder

    @property
    def waveprefix(self):
        f = self.folder
        if not f:
            return None
        if f.items == 0:
            history("no folders")
            return None
        s = f.get("SELECTED")
        if not s:
            history("no selected folder")
            return None
        return s.waveprefix  # container of wave prefixes in seleted folder

    @property
    def channel(self):
        p = self.waveprefix
        if not p:
            return None
        if p.items == 0:
            history("no wave prefixes")
            return None
        s = p.get("SELECTED")
        if not s:
            history("no selected wave prefix")
            return None
        return s.channel  # container of channels in selected wave prefix

    @property
    def selected(self):
        s = {}
        s['project'] = self.project
        s['experiment'] = self.experiment  # container
        s['folder'] = self.folder  # container
        s['waveprefix'] = self.waveprefix  # container
        s['channel'] = self.channel  # container
        s['waveset'] = None  # container
        return s

    @property
    def selected_names(self):
        s = {}
        if self.experiment:
            e = self.experiment.name
        else:
            e = "None"
        if self.folder:
            f = self.folder.name
        else:
            f = "None"
        if self.waveprefix:
            p = self.waveprefix.name
        else:
            p = "None"
        if self.channel:
            c = self.channel.name
        else:
            c = "None"
        s['project'] = self.project.name
        s['experiment'] = e
        s['folder'] = f
        s['waveprefix'] = p
        s['channel'] = c
        s['waveset'] = "All"
        return s

    def stats(self, project, experiment, folder, waveprefix, channel, waveset):
        print(project, experiment, folder, waveprefix, channel, waveset)


class Project(object):
    """
    NM Project class
    """

    def __init__(self, name):
        self.__name = name
        self.__experiment = ExperimentContainer()
        self.__date = str(datetime.datetime.now())

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name

    @property
    def experiment(self):
        return self.__experiment

    @property
    def date(self):
        return self.__date

if __name__ == '__main__':
    nm = Manager()
    # s = nm.selected_names
    # nm.stats(**s)
    # nm.exp.folder_new(name="NMFolder0")
    # nm.exp.folder.waveprefix_new(prefix="Record")
    # nm.exp.folder_open_hdf5()
