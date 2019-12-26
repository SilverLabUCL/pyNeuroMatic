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

    def __init__(self, parent, name, fxns):
        super().__init__(parent, name, fxns, rename=False)
        self.__thenote = ''

    @property
    def __history(self):
        return self._NMObject__history

    @property
    def __tp(self):
        return self.tree_path(history=True)

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'thenote': self.__thenote})
        return k

    # override, no super
    @property
    def content(self):
        return {'note': self.name}

    # override
    def copy(self, note, copy_name=True, quiet=nmc.QUIET):
        name = self.name
        if not super().copy(note, copy_name=copy_name, quiet=True):
            return False
        self.__thenote = note._Note__thenote
        h = 'copied Note ' + nmu.quotes(note.name) + ' to ' + nmu.quotes(name)
        self.__history(h, tp=self.__tp, quiet=quiet)
        return True

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

    def __init__(self, parent, name, fxns):
        t = Note(parent, 'empty', fxns).__class__.__name__
        super().__init__(parent, name, fxns, type_=t, prefix='Note',
                         rename=False, duplicate=False)

    @property
    def __parent(self):
        return self._NMObject__parent

    @property
    def __fxns(self):
        return self._NMObject__fxns

    # override, no super
    @property
    def content(self):
        return {'notes': self.names}

    # override
    def new(self, note='', select=True, quiet=nmc.QUIET):
        if not isinstance(note, str):
            return None
        name = self.name_next(quiet=quiet)
        o = Note(self.__parent, name, self.__fxns)
        n = super().new(name=name, nmobj=o, select=select, quiet=quiet)
        if n:
            n.thenote = note
            return n
        return None

    def notes_all(self):
        notes = []
        for i in range(0, self.count):
            n = self.get(item_num=i)
            notes.append(n.thenote)
        return notes
