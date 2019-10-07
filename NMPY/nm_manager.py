# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_experiment import Experiment
from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import name_list
from nm_utilities import exists
from nm_utilities import error
from nm_utilities import history

NM_EXP_PREFIX = "NMExp"

nm = None  # holds NM Manager, accessed via console


class Manager(object):
    """
    NM Manager class
    Main outer class that does everything
    """
    def __init__(self):
        self.project_new("NMProject")

    def project_new(self, name):
        """Create a new project"""
        self.__project = Project(name)

    def getProject(self):
        return self.__project

    def getExperiment(self):
        if self.__project:
            exp = self.__project.get("")
            return exp
        # error("There is no project.")
        return None

    def getFolder(self):
        exp = self.getExperiment()
        if exp:
            folder = exp.get("")
            return folder
        # error("There is no selected experiment.")
        return None

    def getWavePrefix(self):
        folder = self.getFolder()
        if folder:
            prefix = folder.get("")
            return prefix
        # error("There is no selected folder")
        return None

    def getChannel(self):
        prefix = self.getWavePrefix()
        if prefix:
            chan = prefix.get("")
            return chan
        # error("There is no selected wave prefix")
        return None

    @property
    def project(self):
        """Get project name"""
        if self.__project:
            return self.__project.name
        return "None"

    @project.setter
    def project(self, name):
        """Set project name"""
        if self.__project:
            self.__project.name = name
            return True
        return False

    @property
    def experiments(self):
        """Get list of experiment names"""
        if self.__project:
            return self.__project.name_list()
        return []

    @property
    def experiment(self):
        """Get name of selected experiment"""
        exp = self.getExperiment()
        if exp:
            return exp.name
        return "None"

    @experiment.setter
    def experiment(self, name):
        """Select experiment"""
        if self.__project:
            return self.__project.select(name)
        return False

    @property
    def folders(self):
        """Get list of folder names
        (in selected experiment)"""
        exp = self.getExperiment()
        if exp:
            return exp.name_list()
        return []

    @property
    def folder(self):
        """Get name of selected folder
        (in selected experiment)"""
        f = self.getFolder()
        if f:
            return f.name
        return "None"

    @folder.setter
    def folder(self, name):
        """Select folder"""
        exp = self.getExperiment()
        if exp:
            return exp.select(name)
        return False

    @property
    def waveprefixes(self):
        """Get list of wave prefixes
        (in selected folder in selected experiment)"""
        folder = self.getFolder()
        if folder:
            return folder.name_list()
        return []

    @property
    def waveprefix(self):
        """Get selected wave prefix
        (in selected folder in selected experiment)"""
        prefix = self.getWavePrefix()
        if prefix:
            return prefix.name
        return "None"

    @waveprefix.setter
    def waveprefix(self, name):
        """Set wave prefix
        (in selected folder in selected experiment)"""
        folder = self.getFolder()
        if folder:
            return folder.select(name)
        return False

    @property
    def channels(self):
        """Get list of channels
        (in selected waveprefix in selected folder in selected experiment)"""
        prefix = self.getWavePrefix()
        if prefix:
            return prefix.name_list()
        return []

    @property
    def channel(self):
        """Get selected channel
        (in selected waveprefix in selected folder in selected experiment)"""
        channel = self.getChannel()
        if channel:
            return channel.name
        return "None"

    @channel.setter
    def channel(self, name):
        """Set channel
        (in selected waveprefix in selected folder in selected experiment)"""
        prefix = self.getWavePrefix()
        if prefix:
            return prefix.select(name)
        return False

    @property
    def select(self):
        s = {}
        s['project'] = self.project
        s['experiment'] = self.experiment
        s['folder'] = self.folder
        s['waveprefix'] = self.waveprefix
        s['channel'] = self.channel
        s['waveset'] = 'None'
        return s

    def stats(self, project, experiment, folder, waveprefix, channel, waveset):
        print(project, experiment, folder, waveprefix, channel, waveset)


class Project(Container):
    """
    NM Project class
    Container for NM Experimnents
    """

    def __init__(self, name):
        super().__init__(name)
        self.OBJECT_NAME_PREFIX = "NMExp"
        self.new("")
        # self.new("")
        # self.select("NMExp0")

    def object_new(self, name):
        return Experiment(name)

    def instance_ok(self, obj):
        return isinstance(obj, Experiment)


if __name__ == '__main__':
    nm = Manager()
    s = nm.select
    nm.stats(**s)
    # nm.exp.folder_new(name="NMFolder0")
    # nm.exp.folder.waveprefix_new(prefix="Record")
    # nm.exp.folder_open_hdf5()
