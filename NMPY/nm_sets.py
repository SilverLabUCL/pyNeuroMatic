#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import Container
from nm_utilities import error


class Sets(object):
    """
    NM Sets class
    """

    def __init__(self, name):
        self.__name = name
        self.__set_ = set()

    @property
    def name(self):
        return self.__name

    @property
    def set_(self):
        return self.__set_

class SetsContainer(Container):
    """
    Container for NM Sets
    """

    def object_new(self, name):
        return Sets(name)

    def instance_ok(self, obj):
        return isinstance(obj, Sets)
