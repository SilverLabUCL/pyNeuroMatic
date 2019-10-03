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
        waveprefixes: list of waveprefix objects
        waveprefix: selected waveprefix
    """

    def __init__(self, name="NMFolder0"):
        self.name = name
        self.__waveprefixes = []
        self.__waveprefix = None
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

    @property
    def waveprefix(self):
        return self.__waveprefix

    @waveprefix.setter
    def waveprefix(self, waveprefix):
        self.__waveprefix = waveprefix
        return True
    
    def waveprefix_set(self, prefix: str) -> bool:
        """
        Select a waveprefix.

        Args:
            prefix: wave prefix to select

        Returns:
            True for success, False otherwise
        """
        if prefix is None or not prefix:
            return False
        for p in self.__waveprefixes:
            if prefix.casefold() == p.prefix.casefold():
                self.__waveprefix = p

    def waveprefix_new(self,
                        prefix: str,
                        select: bool = True) -> WavePrefix:
        """
        Create new waveprefix and add to waveprefixes list.

        Args:
            prefix: new wave prefix
            select: select this waveprefix

        Returns:
            new waveprefix if successful, None otherwise
        """
        if prefix is None or not prefix:
            return None
        p = WavePrefix(prefix=prefix)
        self.__waveprefixes.append(p)
        print("created wave prefix " + quotes(prefix))
        if select or self.__waveprefix is None:
            self.__waveprefix = p
            print("selected wave prefix " + quotes(prefix))
        return p

    def waveprefix_kill(self, prefix: str) -> bool:
        """
        Kill a wave prefix (i.e. remove from waveprefixes list).

        Args:
            prefix: wave prefix to kill

        Returns:
            True for success, False otherwise
        """
        if prefix is None or not prefix:
            return False
        kill = None
        for p in self.__waveprefixes:
            if prefix.casefold() == p.prefix.casefold():
                kill = p
                break
        if kill is not None:
            selected = kill is self.__waveprefix
            self.__waveprefixes.remove(kill)
            if selected:
                if not self.__waveprefixes:
                    self.__waveprefix = None
                else:
                    self.__waveprefix = self.__waveprefixes[0]
            print("killed waveprefix " + quotes(prefix))
            return True
        return False
