# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import numpy as np
from nm_wave_prefix import WavePrefix
from nm_utilities import quotes


class Folder(object):
    """
    NM File class

    Class information here...

    Attributes:
        name:
        wave_prefixes: list of WavePrefix objects
        wave_prefix: selected WavePrefix
    """

    def __init__(self, name=None):
        self.name = name
        self.wave_prefixes = []
        self.wave_prefix = None
        #self.d1 = np.random.random(size=10)
        #print(str(self.d1))

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
        self.wave_prefixes.append(p)
        # print("created WavePrefix " + quotes(prefix))
        if self.wave_prefix is None or select:
            self.wave_prefix = p
            print("selected WavePrefix " + quotes(prefix))
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
        for p in self.wave_prefixes:
            if prefix.casefold() == p.prefix.casefold():
                kill = p
                break
        if kill is not None:
            selected = kill is self.wave_prefix
            self.wave_prefixes.remove(kill)
            if selected:
                if not self.wave_prefixes:
                    self.wave_prefix = None
                else:
                    self.wave_prefix = self.wave_prefixes[0]
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
        for p in self.wave_prefixes:
            if prefix.casefold() == p.prefix.casefold():
                self.wave_prefix = p
