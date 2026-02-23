# -*- coding: utf-8 -*-
"""
NM Tool - Base class for analysis tools.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source
Software Toolkit for Acquisition, Analysis and Simulation of
Electrophysiological Data. Front. Neuroinform. 12:14.
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations

from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_manager import SELECT_LEVELS


class NMTool:
    """
    NM Tool - Base class for analysis tools.

    Tools receive selection state from NMManager via select_values and
    perform analysis during run(). Subclasses override run() to implement
    specific analysis.

    Selection is set by NMManager - tools have read-only access to
    individual levels (folder, data, dataseries, channel, epoch).

    Example:
        class MyTool(NMTool):
            def run(self) -> bool:
                data = self.dataseries.get_data(self.channel, self.epoch)
                # analyze data...
                return True
    """

    def __init__(self) -> None:
        self._select: dict[str, NMObject | None] = {
            level: None for level in SELECT_LEVELS
        }

    # Read-only convenience properties for accessing selection
    @property
    def folder(self) -> NMFolder | None:
        """Get the currently selected folder."""
        v = self._select.get("folder")
        return v if isinstance(v, NMFolder) else None

    @property
    def data(self) -> NMData | None:
        """Get the currently selected data."""
        v = self._select.get("data")
        return v if isinstance(v, NMData) else None

    @property
    def dataseries(self) -> NMDataSeries | None:
        """Get the currently selected dataseries."""
        v = self._select.get("dataseries")
        return v if isinstance(v, NMDataSeries) else None

    @property
    def channel(self) -> NMChannel | None:
        """Get the currently selected channel."""
        v = self._select.get("channel")
        return v if isinstance(v, NMChannel) else None

    @property
    def epoch(self) -> NMEpoch | None:
        """Get the currently selected epoch."""
        v = self._select.get("epoch")
        return v if isinstance(v, NMEpoch) else None

    @property
    def select_values(self) -> dict[str, NMObject | None]:
        """Get the current selection as a dictionary of NMObjects."""
        return self._select.copy()

    @select_values.setter
    def select_values(self, values: dict[str, NMObject | None]) -> None:
        """
        Set selection from a dictionary of NMObjects.

        Called by NMManager.run_tool() to set the context for run execution.

        Args:
            values: Dictionary mapping level names to NMObjects.
                    Keys should be from SELECT_LEVELS.
        """
        for level in SELECT_LEVELS:
            if level in values:
                self._select[level] = values[level]

    @property
    def select_keys(self) -> dict[str, str | None]:
        """Get the current selection as a dictionary of names."""
        return {
            level: (obj.name if isinstance(obj, NMObject) else None)
            for level, obj in self._select.items()
        }

    def run_init(self) -> bool:
        """Called once before run loop. Override in subclass."""
        return True

    def run(self) -> bool:
        """Called for each run target. Override in subclass."""
        print(self.select_keys)
        return True

    def run_finish(self) -> bool:
        """Called once after run loop. Override in subclass."""
        return True
