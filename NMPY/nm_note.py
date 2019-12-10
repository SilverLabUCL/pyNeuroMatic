# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import nm_configs as nmc
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


class Note(NMObject):
    """
    NM Note class
    """

    def __init__(self, manager, parent, name, fxns):
        super().__init__(manager, parent, name, fxns, rename=False)
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']
        self.__thenote = ''

    @property  # override, no super
    def content(self):
        return {'note': self.name}

    @property
    def thenote(self):
        return self.__thenote

    @thenote.setter
    def thenote(self, note):
        self.__thenote = note


class NoteContainer(Container):
    """
    Container for NM Note objects
    """

    def __init__(self, manager, parent, name, fxns):
        o = Note(manager, parent, 'temp', fxns)
        super().__init__(manager, parent, name, fxns, nmobj=o, prefix='Note',
                         rename=False, duplicate=False)
        self.__parent = parent
        self.__fxns = fxns
        self.__quiet = fxns['quiet']
        self.__alert = fxns['alert']
        self.__error = fxns['error']
        self.__history = fxns['history']

    @property  # override, no super
    def content(self):
        return {'notes': self.names}

    # override
    def new(self, note='', select=True, quiet=nmc.QUIET):
        if not note:
            return None
        o = Note(self.__parent, 'temp', fxns)  # will be renamed
        n = super().new(name='default', nmobj=o, select=select, quiet=quiet)
        if n:
            n.thenote = note
            return n
        return None
