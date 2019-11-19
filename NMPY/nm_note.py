# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container


class Note(NMObject):
    """
    NM Note class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name)
        self.__thenote = ''

    @property
    def key(self):
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
    __select_alert = ('NOT USED.')

    def __init__(self, parent, name='NMNoteContainer'):
        super().__init__(parent, name, prefix='Note',
                         select_alert=self.__select_alert, rename=False,
                         duplicate=False)

    @property
    def key(self):  # override
        return {'note': self.names}

    def object_new(self, name):  # override, do not call super
        return Note(self.parent, name)

    def new(self, note='', select=True, quiet=False):  # override
        if not note:
            return False
        n = super().new()
        n.thenote = note
        return n
    

