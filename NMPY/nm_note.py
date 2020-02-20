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

    def __init__(self, parent, name, fxns={}, thenote='', **copy):
        super().__init__(parent, name, fxns=fxns)
        self._content_name = 'note'
        self.__thenote = ''
        self._thenote_set(thenote)
        self._param_list += ['thenote']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'thenote': self.__thenote})
        return k

    # override, no super
    def copy(self):
        return Note(self._parent, self.name, fxns=self._fxns,
                    thenote=self.__thenote)

    @property
    def thenote(self):
        return self.__thenote

    @thenote.setter
    def thenote(self, thenote):
        return self._thenote_set(thenote)

    def _thenote_set(self, thenote, quiet=True):
        # notes should be quiet
        old = self.__thenote
        if not thenote:
            self.__thenote = ''
        elif isinstance(thenote, str):
            self.__thenote = thenote
        else:
            self.__thenote = str(thenote)
        self._modified()
        h = nmu.history_change('thenote', old, self.__thenote)
        self._history(h, tp=self._tp, quiet=quiet)
        return True


class NoteContainer(Container):
    """
    Container for NM Note objects
    """

    def __init__(self, parent, name, fxns={}, prefix='Note', rename=False,
                 **copy):
        t = Note(parent, 'empty').__class__.__name__
        super().__init__(parent, name, fxns=fxns, type_=t, prefix=prefix,
                         rename=rename, **copy)
        self._content_name = 'notes'
        self.__off = False

    # override, no super
    def copy(self):
        return NoteContainer(self._parent, self.name, fxns=self._fxns,
                             prefix=self.prefix, rename=self._rename_,
                             thecontainer=self._thecontainer_copy())

    # override
    def new(self, thenote='', select=True, quiet=True):
        # notes should be quiet
        if self.__off:
            return None
        o = Note(self._parent, name='temp', fxns=self._fxns, thenote=thenote)
        return super().new(name='default', nmobject=o, select=select,
                           quiet=quiet)

    def thenotes(self, quiet=True):
        # notes should be quiet
        notes = []
        self._history('', tp=self._tp, quiet=quiet)
        for n in self.getitems(names='all'):
            notes.append(n.thenote)
            self._history(n.thenote, tp='none', quiet=quiet)
        return notes

    @property
    def off(self):
        return self.__off

    @off.setter
    def off(self, off):
        self.__off = off
        self._modified()
        return off

    # override, no super
    def duplicate(self):
        raise RuntimeError('dataseries cannot be duplicated')
