# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_experiment import ExperimentContainer
from nm_utilities import name_ok
from nm_utilities import quotes
from nm_utilities import error


class Project(NMObject):
    """
    NM Project class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__exp_container = ExperimentContainer(self, 'NMExps')

    def rename(self, name):
        if not name_ok(name):
            return error('bad name ' + quotes(name))
        self.__name = name
        return True

    @property
    def exp_container(self):
        return self.__exp_container
