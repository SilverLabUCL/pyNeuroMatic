# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_experiment
from nm_utilities import name_ok

nm = None  # for Manager object


class Manager(object):
    """
    NM Manager class
    """

    def __init__(self):
        self.__experiments = []
        self.__exp = nm_experiment.Experiment()

    @property
    def exp(self):
        return self.__exp

    @exp.setter
    def exp(self, exp):
        if exp is None:
            return False
        self.__exp = exp
        return True

    @property
    def folder(self):
        return self.__exp.folder.name

    @folder.setter
    def folder(self, name):
        return self.__exp.folder_set(name=name)

    def experiment_new(self, name="Untitled"):
        if name_ok(name):
            self.__exp = nm_experiment.Experiment(name=name)

    def experiment_open(self, path=None):
        pass

    def experiment_save(self):
        pass


if __name__ == '__main__':
    nm = Manager()
    # nm.exp.folder_new(name="NMFolder0")
    nm.exp.folder.wave_prefix_new(prefix="Record")
    # nm.exp.folder_open_hdf5()
