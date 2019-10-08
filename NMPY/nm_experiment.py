# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_folder import FolderContainer

FOLDER_PREFIX = "NMFolder"


class Experiment(object):
    """
    NM Experiment class
    """

    def __init__(self, name):
        self.__name = name
        self.__folder = FolderContainer(FOLDER_PREFIX)

    @property
    def name(self):
        return self.__name

    @property
    def folder(self):
        return self.__folder


class ExperimentContainer(Container):
    """
    Container for NM Experimnents
    """

    def object_new(self, name):
        return Experiment(name)

    def instance_ok(self, obj):
        return isinstance(obj, Experiment)
