# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmconfig
from nm_container import Container
from nm_folder import FolderContainer
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error


class Experiment(object):
    """
    NM Experiment class
    """

    def __init__(self, name):
        self.__name = name
        self.__folder_container = FolderContainer()

    @property
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, name):
        error("use experiment rename function")

    @property
    def folder_container(self):
        return self.__folder_container


class ExperimentContainer(Container):
    """
    Container for NM Experimnents
    """
    def __init__(self):
        super().__init__()
        self.prefix = nmconfig.EXP_PREFIX

    def object_new(self, name):  # override, do not call super
        return Experiment(name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Experiment)
