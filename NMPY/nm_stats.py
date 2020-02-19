#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:22:44 2019

@author: jason
"""
import nm_preferences as nmp
import nm_utilities as nmu


class Stats(object):
    """
    NM Stats class
    """

    def __init__(self, fxns):
        self.b0 = 0
        self.b1 = 5
        self.x0 = 0
        self.x1 = 5

    @property
    def _quiet(self):
        return self._fxns['quiet']

    @property
    def _alert(self):
        return self._fxns['alert']

    @property
    def _error(self):
        return self._fxns['error']

    @property
    def _history(self):
        return self._fxns['history']

    def max_(self, select='default'):
        print(select)
