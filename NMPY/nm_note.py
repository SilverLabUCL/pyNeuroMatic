# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


class Note(NMObject):
    """
    NM Note class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__thenote = ''
        self._NMObject__rename = False

    @property
    def content(self):  # override, no super
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

    def __init__(self, parent, name='NMNoteContainer'):
        o = Note(parent, 'temp')
        super().__init__(parent, name=name, nmobj=o, prefix='Note')
        self.__parent = parent
        self._Container__rename = False
        self._Container__duplicate = False

    @property
    def content(self):  # override, no super
        return {'note': self.names}

    def new(self, note='', select=True, quiet=False):  # override
        if not note:
            return None
        o = Note(self.__parent, 'temp')  # will be renamed
        n = super().new(name='default', nmobj=o, select=select, quiet=quiet)
        if n:
            n.thenote = note
            return n
        return None

    def rename(self, name, newname, quiet=False):  # override, no super
        nmu.error('Notes cannot be renamed')
        return False

    def duplicate(self, name, newname, select=False, quiet=False):
        # override, no super
        nmu.error('Notes cannot be duplicated')
        return None
