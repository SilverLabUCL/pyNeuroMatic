#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:22:44 2019

@author: jason
"""
import pyneuromatic.nm_preferences as nmp
import pyneuromatic.nm_utilities as nmu


class Stats(object):
    """
    NM Stats class
    """

    def __init__(self):
        self.b0 = 0
        self.b1 = 5
        self.x0 = 0
        self.x1 = 5

    def max_(self, select="default"):
        print(select)
