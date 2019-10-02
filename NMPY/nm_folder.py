# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np
from nm_wave_prefix import WavePrefix
from nm_utilities import quotes
from nm_utilities import name_ok


class Folder(object):
    """
    NM File class

    Class information here...

    Attributes:
        name:
        wave_prefixes: list of WavePrefix objects
        wave_prefix: selected WavePrefix
    """

    def __init__(self, name="NMFolder0"):
        self.name = name
        self.__wave_prefixes = []
        self.__wave_prefix_select = None
        # self.d1 = np.random.random(size=10)
        # print(str(self.d1))

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if name_ok(name):
            self.__name = name
            return True
        print("bad folder name")
        return False

    def wave_prefix_new(self,
                        prefix: str,
                        select: bool = True) -> WavePrefix:
        """
        Create new WavePrefix and add to wave_prefixes list.

        Args:
            prefix: new wave prefix
            select: select this WavePrefix

        Returns:
            new WavePrefix if successful, None otherwise
        """
        if prefix is None or not prefix:
            return None
        p = WavePrefix(prefix=prefix)
        self.__wave_prefixes.append(p)
        print("created wave prefix " + quotes(prefix))
        if select or self.__wave_prefix_select is None:
            self.__wave_prefix_select = p
            print("selected wave prefix " + quotes(prefix))
        return p

    def wave_prefix_kill(self, prefix: str) -> bool:
        """
        Kill a wave prefix (i.e. remove from wave_prefixes list).

        Args:
            prefix: wave prefix to kill

        Returns:
            True for success, False otherwise
        """
        if prefix is None or not prefix:
            return False
        kill = None
        for p in self.__wave_prefixes:
            if prefix.casefold() == p.prefix.casefold():
                kill = p
                break
        if kill is not None:
            selected = kill is self.__wave_prefix_select
            self.__wave_prefixes.remove(kill)
            if selected:
                if not self.__wave_prefixes:
                    self.__wave_prefix_select = None
                else:
                    self.__wave_prefix_select = self.__wave_prefixes[0]
            print("killed WavePrefix " + quotes(prefix))
            return True
        return False

    def wave_prefix_select(self, prefix: str) -> bool:
        """
        Select a WavePrefix.

        Args:
            prefix: wave prefix to select

        Returns:
            True for success, False otherwise
        """
        if prefix is None or not prefix:
            return False
        for p in self.__wave_prefixes:
            if prefix.casefold() == p.prefix.casefold():
                self.__wave_prefix_select = p
