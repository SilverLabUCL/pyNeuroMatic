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

    def __init__(self, parent, name, fxns={}, thenote=''):
        super().__init__(parent, name, fxns=fxns, rename=False)
        if isinstance(thenote, str):
            self.__thenote = thenote
        else:
            raise TypeError(nmu.type_error(thenote, 'string'))
        self._param_list += ['thenote']

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

    # override, no super
    def copy(self):
        return Note(self._parent, self.name, fxns=self._fxns,
                    thenote=self.__thenote)

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
        self.__off = False

    @property
    def off(self):
        return self.__off

    @off.setter
    def off(self, off):
        self.__off = nmu.check_bool(off, False)
        return self.__off

    # override, no super
    @property
    def content(self):
        return {'notes': self.names}

    # override
    def copy(self):
        c = NoteContainer(self._parent, self.name, fxns=self._fxns)
        super().copy(container=c)
        return c

    # override
    def new(self, note='', select=True, quiet=True):
        # notes should be quiet
        if self.__off:
            return None
        if not isinstance(note, str):
            return None
        quiet = nmu.check_bool(quiet, True)
        name = self.name_next(quiet=quiet)
        o = Note(self._parent, name, self._fxns)
        n = super().new(name=name, nmobj=o, select=select, quiet=quiet)
        if n:
            n.thenote = note
            return n
        return None

    def thenotes(self, quiet=nmp.QUIET):
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        notes = []
        self._history('', tp=self._tp, quiet=quiet)
        for n in self.getitems(names='all'):
            notes.append(n.thenote)
            self._history(n.thenote, tp='none', quiet=quiet)
        return notes
