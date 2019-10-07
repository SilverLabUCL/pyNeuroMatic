# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Project
from nm_utilities import quotes
from nm_utilities import name_ok
from nm_utilities import name_list
from nm_utilities import exists
from nm_utilities import error
from nm_utilities import history

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

    @property
    def project(self):
        return self.__project
    
    @property
    def experiment(self):
        if self.__project:
            return self.__project.experiment  # container of experiments
        history("no experiments")
        return None
    
    @property
    def folder(self):
        if self.__project:
            e = self.__project.experiment.get("SELECTED")  # selected exp
            if e:
                return e.folder  # container of folders
        history("no folders")
        return None

    @property
    def waveprefix(self):
        if self.__project:
            e = self.__project.experiment.get("SELECTED")  # selected exp
            if e:
                f = e.folder.get("SELECTED")  # selected folder
                if f:
                    return f.waveprefix  # container of wave prefixes
        history("no wave prefixes")
        return None
    
    @property
    def channel(self):
        if self.__project:
            e = self.__project.experiment.get("SELECTED")  # selected exp
            if e:
                f = e.folder.get("SELECTED")  # selected folder
                if f:
                    p = f.waveprefix.get("SELECTED")  # selected wave prefixe
                    if p:
                        return p.channel  # container of channels
        history("no channels")
        return None
    
    @property
    def selected(self):
        s = {}
        s['project'] = self.project
        s['experiment'] = self.experiment
        s['folder'] = self.folder
        s['waveprefix'] = self.waveprefix
        s['channel'] = self.channel
        return s
    
    @property
    def selected_names(self):
        s = {}
        if self.project:
            p = self.project.name
        else:
            p = "None"
        if self.experiment:
            e = self.experiment.name
        else:
            e = "None"
        if self.folder:
            f = self.folder.name
        else:
            f = "None"
        if self.waveprefix:
            wp = self.waveprefix.name
        else:
            wp = "None"
        if self.channel:
            c = self.channel.name
        else:
            c = "None"
        s['project'] = p
        s['experiment'] = e
        s['folder'] = f
        s['waveprefix'] = wp
        s['channel'] = c
        return s

class ManagerOLD(object):
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


if __name__ == '__main__':
    nm = Manager()
    # s = nm.select
    # nm.stats(**s)
    # nm.exp.folder_new(name="NMFolder0")
    # nm.exp.folder.waveprefix_new(prefix="Record")
    # nm.exp.folder_open_hdf5()
