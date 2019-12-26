#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:22:44 2019

@author: jason
"""
import nm_configs as nmc
import nm_utilities as nmu


class Stats(object):
    """
    NM Stats class
    """

    def __init__(self, fxns):
        if list(fxns.keys()) == ['quiet', 'alert', 'error', 'history']:
            self.__fxns = fxns
        else:
            e = 'bad fxn arg:  ' + str(fxns)
            raise ValueError(e)
        self.b0 = 0
        self.b1 = 5
        self.x0 = 0
        self.x1 = 5

    @property
    def __quiet(self):
        return self.__fxns['quiet']

    @property
    def __alert(self):
        return self.__fxns['alert']

    @property
    def __error(self):
        return self.__fxns['error']

    @property
    def __history(self):
        return self.__fxns['history']

    def max_(self, select='default'):
        print(select)
