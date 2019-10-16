# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import datetime
import nm_configs as nmconfig
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
    """
    def __init__(self):
        self.project_new("NMProject")

    def project_new(self, name):
        """Create new project"""
        self.__project = Project(name)
        self.experiment.new()  # create default experiment
        self.folder.new()  # create default folder
        self.wave_test()

    def wave_test(self):
        noise = True
        self.wave.make(prefix="Record", nchan=1, nwaves=3, points=5, noise=noise)
        self.wave.make(prefix="Wave", nchan=3, nwaves=4, points=5, noise=noise)
        self.waveprefix.new("Record")
        self.waveprefix.new("Wave")
        self.waveprefix.new("Test")

    @property
    def project(self):
        return self.__project

    @property
    def experiment(self):
        return self.__project.experiment_container

    @property
    def folder(self):
        ec = self.__project.experiment_container
        if ec.count == 0:
            history("no experiments")
            return None
        exp = ec.get('selected')
        if not exp:
            history("no selected experiment")
            return None
        return exp.folder_container  # Folder objs in selected experiment

    @property
    def wave(self):
        f = self.folder
        if not f:
            return None
        if f.count == 0:
            history("no folders")
            return None
        s = f.get('selected')
        if not s:
            history("no selected folder")
            return None
        return s.wave_container  # Wave objs in seleted folder

    @property
    def waveprefix(self):
        f = self.folder
        if not f:
            return None
        if f.count == 0:
            history("no folders")
            return None
        s = f.get('selected')
        if not s:
            history("no selected folder")
            return None
        return s.waveprefix_container  # WavePrefix objs in seleted folder

    @property
    def channel(self):
        p = self.waveprefix
        if not p:
            return None
        if p.count == 0:
            history("no wave prefixes")
            return None
        s = p.get('selected')
        if not s:
            history("no selected wave prefix")
            return None
        return s.channel_container  # Channel objs in selected waveprefix

    @property
    def waveset(self):
        p = self.waveprefix
        if not p:
            return None
        if p.count == 0:
            history("no wave prefixes")
            return None
        s = p.get('selected')
        if not s:
            history("no selected wave prefix")
            return None
        return s.waveset_container  # WaveSet objs in selected waveprefix

    @property
    def select(self):
        s = {}
        exp = 'None'
        folder = 'None'
        prefix = 'None'
        chan = 'None'
        wave = 0
        wset = 'All'
        if self.experiment.select:
            exp = self.experiment.select
            if self.folder.select:
                folder = self.folder.select
                if self.waveprefix.select:
                    prefix = self.waveprefix.select
                    chan = self.waveprefix.select.channel_select
                    wave = self.waveprefix.select.wave_select
                    if self.waveset.select:
                        wset = self.waveset.select
        s['project'] = self.project
        s['experiment'] = exp
        s['folder'] = folder
        s['waveprefix'] = prefix
        s['channel'] = chan
        s['wave'] = wave
        s['waveset'] = wset
        return s

    @property
    def select_names(self):
        s = {}
        exp = 'None'
        folder = 'None'
        prefix = 'None'
        chan = 'None'
        wave = 0
        wset = 'All'
        if self.experiment.select:
            exp = self.experiment.select.name
            if self.folder.select:
                folder = self.folder.select.name
                if self.waveprefix.select:
                    prefix = self.waveprefix.select.name
                    chan = self.waveprefix.select.channel_select
                    wave = self.waveprefix.select.wave_select
                    if self.waveset.select:
                        wset = self.waveset.select.name
        s['project'] = self.project.name
        s['experiment'] = exp
        s['folder'] = folder
        s['waveprefix'] = prefix
        s['channel'] = chan
        s['wave'] = wave
        s['waveset'] = wset
        return s

    def stats(self, project, experiment, folder, waveprefix, channel, waveset):
        print(project, experiment, folder, waveprefix, channel, waveset)


class Project(object):
    """
    NM Project class
    """

    def __init__(self, name):
        self.__name = name
        self.__experiment_container = ExperimentContainer()
        self.__date = str(datetime.datetime.now())

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not name_ok(name):
            return error("bad name " + quotes(name))
        self.__name = name
        return True

    @property
    def experiment_container(self):
        return self.__experiment_container

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
