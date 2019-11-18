#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
from nm_container import NMObject
from nm_container import Container
import nm_utilities as nmu


class Channel(NMObject):
    """
    NM Channel class
    """

    def __init__(self, parent, name):
        super().__init__(parent, name, {'chan': name}, rename=False)
        self.__graphXY = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
        self.__transform = []


class ChannelContainer(Container):
    """
    Container for NM Channel objects
    """
    __select_alert = 'NOT USED. See nm.channel_select.'

    def __init__(self, parent, name):
        super().__init__(parent, name, {'chan': name}, prefix='Chan',
                         select_alert=self.__select_alert, rename=False,
                         duplicate=False, kill=False)

    def object_new(self, name):  # override, do not call super
        return Channel(self.parent, name)

    def instance_ok(self, obj):  # override, do not call super
        return isinstance(obj, Channel)

    def name_default(self, quiet=False):  # override, do not call super
        """Get next default channel name."""
        if self.prefix:
            prefix = self.prefix
        else:
            prefix = 'Chan'
        n = 10 + len(self.get_all())
        for i in range(0, n):
            name = prefix + nmu.channel_char(i)
            if not self.exists(name):
                return name
        return ''
