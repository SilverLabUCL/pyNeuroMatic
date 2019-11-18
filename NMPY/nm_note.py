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

    def __init__(self, parent, name):
        super().__init__(parent, name, {'note': name})
        self.__thenote = ''

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

    def __init__(self, parent, name):
        super().__init__(parent, name, {'note': name}, prefix='Note',
                         select_alert=self.__select_alert, rename=False,
                         duplicate=False)

    def object_new(self, name):  # override, do not call super
        return Note(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Note)

    def new(self, note='', select=True, quiet=False):  # override
        if not note:
            return False
        n = super().new()
        n.thenote = note
        return n
    

