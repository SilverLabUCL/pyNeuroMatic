#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:14:20 2019

@author: jason
"""
import math


class Dimensions(object):

    def __init__(self, xstart=0, xdelta=1, xlabel='', xunits='', ylabel='',
                 yunits=''):
        self.__xstart = xstart
        self.__xdelta = xdelta
        self.__xlabel = xlabel
        self.__xunits = xunits
        self.__ylabel = ylabel
        self.__yunits = yunits

    def get(self):
        return {'xstart': self.__xstart, 'xdelta': self.__xdelta,
                'xlabel': self.__xlabel, 'xunits': self.__xunits,
                'ylabel': self.__ylabel, 'yunits': self.__yunits}

    @property
    def xstart(self):
        return self.__xstart

    @xstart.setter
    def xstart(self, xstart):
        if math.isinf(xstart) or math.isnan(xstart):
            return False
        self.__xstart = xstart
        return True

    @property
    def xdelta(self):
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, xdelta):
        if math.isinf(xdelta) or math.isnan(xdelta):
            return False
        self.__xdelta = xdelta
        return True

    @property
    def xlabel(self):
        return self.__xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        if isinstance(xlabel, str):
            self.__xlabel = xlabel
            return True
        return False

    @property
    def xunits(self):
        return self.__xunits

    @xunits.setter
    def xunits(self, xunits):
        if isinstance(xunits, str):
            self.__xunits = xunits
            return True
        return False

    @property
    def ylabel(self):
        return self.__ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        if isinstance(ylabel, str):
            self.__ylabel = ylabel
            return True
        return False

    @property
    def yunits(self):
        return self.__yunits

    @yunits.setter
    def yunits(self, yunits):
        if isinstance(yunits, str):
            self.__yunits = yunits
            return True
        return False
