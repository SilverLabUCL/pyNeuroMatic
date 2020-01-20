# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
import nm_preferences as nmp
import nm_utilities as nmu


class Note(NMObject):
    """
    NM Note class
    """

    def __init__(self, parent, name, fxns={}):
        super().__init__(parent, name, fxns=fxns, rename=False)
        self.__thenote = ''

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
    def _copy(self, note, copy_name=True, quiet=nmp.QUIET):
        name = self.name
        if not isinstance(note, Note):
            raise TypeError(nmu.type_error(note, 'Note'))
        if not super()._copy(note, copy_name=copy_name, quiet=True):
            return False
        self.__thenote = note._Note__thenote
        h = 'copied Note ' + nmu.quotes(note.name) + ' to ' + nmu.quotes(name)
        self._history(h, tp=self._tp, quiet=quiet)
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

    def __init__(self, parent, name, fxns={}):
        t = Note(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t, prefix='Note',
                         rename=False, duplicate=False)

    # override, no super
    @property
    def content(self):
        return {'notes': self.names}

    # override
    def new(self, note='', select=True, quiet=nmp.QUIET):
        if not isinstance(note, str):
            return None
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        name = self.name_next(quiet=quiet)
        o = Note(self._parent, name, self._fxns)
        n = super().new(name=name, nmobj=o, select=select, quiet=quiet)
        if n:
            n.thenote = note
            return n
        return None

    def notes_all(self):
        return self.getitems(names='all')
