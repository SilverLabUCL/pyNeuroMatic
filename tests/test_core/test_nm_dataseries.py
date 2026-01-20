#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:36:01 2023

@author: jason
"""
import unittest

from pyneuromatic.core.nm_channel import NMChannel, NMChannelContainer
from pyneuromatic.core.nm_data import NMData, NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.core.nm_epoch import NMEpoch, NMEpochContainer
from pyneuromatic.core.nm_manager import NMManager
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu

NM = NMManager(quiet=True)

YSCALE = [
    {"label": "current", "units": "pA"},
    {"label": "voltage", "units": "mV"},
    {"label": "TTL1", "units": "V"},
    {"label": "TTL2", "units": "V"},
]
XSCALE = {"label": "sample", "units": "#", "start": 0, "delta": 0.01}


class NMDataSeriesTest(unittest.TestCase):
    def setUp(self):  # executed before each test
        self.dataprefix = "data"

        num_epochs = 10
        num_chans = len(YSCALE)

        self.ds0 = NMDataSeries(parent=NM, name=self.dataprefix)

        chanlist = []
        for j in range(num_chans):
            c = self.ds0.channels.new(yscale=YSCALE[j], xscale=XSCALE)
            chanlist.append(c)

        for i in range(num_epochs):
            e = self.ds0.epochs.new()
            for j in range(num_chans):
                c = chanlist[j]
                n = self.dataprefix + nmu.CHANNEL_CHARS[j] + str(i)
                d = NMData(parent=NM, name=n, yscale=YSCALE[j], xscale=XSCALE)
                e.data.append(d)
                c.data.append(d)

        self.ds0.channels.select_key = "B"
        self.ds0.epochs.select_key = "E2"
        print(self.ds0.get_select(get_keys=True))

        # create sets

        for i in range(0, num_epochs, 2):
            self.ds0.epochs.sets.add("set0", "E" + str(i))
        for i in range(1, num_epochs, 2):
            self.ds0.epochs.sets.add("set1", "E" + str(i))
        self.ds0.epochs.sets.add("set2", ["set0", "|", "set1"])

        self.ds0.channels.sets.add("set0", "A")
        self.ds0.channels.sets.add("set0", "B")
        self.ds0.channels.sets.add("set1", "C")
        self.ds0.channels.sets.add("set1", "D")
        self.ds0.channels.sets.add("set2", ["set0", "|", "set1"])

    def test00_init(self):
        with self.assertRaises(TypeError):
            NMDataSeries(copy=NM)

    def test01_eq(self):
        pass

    def test02_copy(self):
        pass

    def test03_parameters(self):
        pass

    def test04_content(self):
        pass

    def test05_channels(self):
        pass

    def test06_epochs(self):
        pass

    def test07_get_select(self):
        pass
