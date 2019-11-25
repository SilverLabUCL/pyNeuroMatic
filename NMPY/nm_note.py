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
    __select_alert = ('NOT USED.')

    def __init__(self, parent, name='NMNoteContainer'):
        super().__init__(parent, name, prefix='Note',
                         select_alert=self.__select_alert, rename=False,
                         duplicate=False)
        self.__parent = parent

    @property
    def content(self):  # override, no super
        return {'note': self.names}

    def new(self, note='', select=True, quiet=False, nmobj=None):
        # override
        if not note:
            return False
        o = Note(self.__parent, 'temp')
        n = super().new(name='default', nmobj=o)
        n.thenote = note
        return n
    

