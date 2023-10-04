# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from pyneuromatic.nm_object import NMObject, NMObjectType
from pyneuromatic.nm_object_container import NMObjectContainer
import pyneuromatic.nm_utilities as nmu
from typing import Dict, List, NewType

NOTE_PREFIX = "Note"


class NMNote(NMObject):
    """
    NM Note class
    TODO: change notes to Dictionary within NMObject
    """

    def __init__(self, parent: object, name: str, thenote: str = "", **copy) -> None:
        super().__init__(parent, name)
        if isinstance(thenote, str):
            self.__thenote = thenote
        else:
            self.__thenote = str(thenote)

    # override, no super
    def copy(self) -> nmu.NMNoteType:
        return NMNote(self._parent, self.name, thenote=self.__thenote)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update({"thenote": self.__thenote})
        return k

    @property
    def thenote(self) -> str:
        return self.__thenote

    @thenote.setter
    def thenote(self, thenote: str) -> None:
        self._thenote_set(thenote)

    def _thenote_set(self, thenote: str, quiet: bool = True) -> bool:
        # notes should be quiet
        if thenote == self.__thenote:
            return True
        old = self.__thenote
        if isinstance(thenote, str):
            self.__thenote = thenote
        else:
            self.__thenote = str(thenote)
        self._modified()
        h = nmu.history_change("thenote", old, self.__thenote)
        self._history(h, quiet=quiet)
        return True


class NMNoteContainer(NMObjectContainer):
    """
    Container for NM Note objects
    """

    def __init__(self, parent: object, name: str, **copy) -> None:
        n = NMNote(parent=parent, name="ContainerUtility")
        super().__init__(
            parent, name, nmobject=n, prefix=NOTE_PREFIX, rename=False, **copy
        )
        self.__off = False

    # override, no super
    def copy(self) -> NMNoteContainerType:
        return NMNoteContainer(
            self._parent,
            self.name,
            c_prefix=self.prefix,
            c_rename=self.parameters["rename"],
            c_container=self._container_copy(),
        )

    # override
    def new(
        self, thenote: str = "", select: bool = True, quiet: bool = True
    ) -> nmu.NMNoteType:
        # notes should be quiet
        if self.__off:
            return None
        name = self.name_next()
        o = NMNote(None, name=name, thenote=thenote)
        if super().append(nmobject=o, select=select, quiet=quiet):
            return o
        else:
            return None

    @property
    def all_(self) -> List[str]:
        notes = []
        for p in self.content_parameters:
            notes.append(p["thenote"])
        return notes

    def print_all(self):
        for n in self.all_:
            self._history(n, quiet=False)

    @property
    def off(self) -> bool:
        return self.__off

    @off.setter
    def off(self, off: bool) -> None:
        if isinstance(off, bool):
            self.__off = off
            self._modified()

    # override, no super
    def duplicate(self) -> None:
        e = self._error("notes cannot be duplicated")
        raise RuntimeError(e)
