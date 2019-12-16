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

    def __init__(self, manager, fxns):
        self.__manager = manager
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.b0 = 0
        self.b1 = 5
        self.x0 = 0
        self.x1 = 5

    def max_(self, select='default'):
        print(select)
