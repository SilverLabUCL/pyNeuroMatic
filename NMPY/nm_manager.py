# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_experiment

nm = None

class Manager(object):
    """
    NM Manager class
    """
    
    def __init__(self):
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
    
    def experiment_new(self, name=""):
        if name.isalnum():
            self.__exp = nm_experiment.Experiment(name=name)
            return True
        else:
            print("bad experiment name: found special characters")
            return False

    def experiment_open(self, path=None):
        pass
    
    def experiment_save(self):
        pass

if __name__ == '__main__':
    nm = Manager()
    nm.exp.folder_new(name="NMFolder0")
    #nm.exp.folder_select.wave_prefix_new(prefix="Record")
    nm.exp.folder_open_hdf5()
        
