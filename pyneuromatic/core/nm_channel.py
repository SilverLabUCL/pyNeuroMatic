# -*- coding: utf-8 -*-
"""
NMChannel module.

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
import copy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_data import NMData

from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_scale import NMScaleX, NMScaleY, _xscale_from_dict, _yscale_from_dict
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager
    NMProject (root)
        NMFolderContainer
            NMFolder (folder0, folder1...)
                NMDataContainer
                    NMData (recordA0, recordA1... avgA0, avgB0)
                NMDataSeriesContainer
                    NMDataSeries (record, avg...)
                        NMChannelContainer
                            NMChannel (A, B, C...)
                        NMEpochContainer
                            NMEpoch (E0, E1, E2...)
DataSeries:
      E0  E1  E2... (epochs)
Ch A  A0  A1  A2...
Ch B  B0  B1  B2...
.
.
.
"""


class NMChannel(NMObject):
    """
    NM Channel class.

    Stores channel-level x/y scale metadata as simple dicts
    and a list of NMData references belonging to this channel.
    """

    # Extend NMObject's special attrs with NMChannel's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMChannel__thedata",
        "_NMChannel__xscale",
        "_NMChannel__yscale",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMChannel0",
        xscale: dict | None = None,
        yscale: dict | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__thedata: list[NMData] = []  # list of NMData refs for this channel
        self.__xscale: NMScaleX = _xscale_from_dict(xscale, parent=self)
        self.__yscale: NMScaleY = _yscale_from_dict(yscale, parent=self)

    # override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMChannel):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__xscale != other.xscale:
            return False
        if self.__yscale != other.yscale:
            return False
        if len(self.__thedata) != len(other.data):
            return False
        if not all(a == b for a, b in zip(self.__thedata, other.data)):
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMChannel:
        """Support Python's copy.deepcopy() protocol."""
        import datetime

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes NMObject's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._container = None
        result._NMObject__copy_of = self

        # __xscale/__yscale: deep copy and reset parent
        result._NMChannel__xscale = copy.deepcopy(self._NMChannel__xscale, memo)
        result._NMChannel__xscale._parent = result
        result._NMChannel__yscale = copy.deepcopy(self._NMChannel__yscale, memo)
        result._NMChannel__yscale._parent = result

        # __thedata: copy list of references
        # If copying within a folder context, try to resolve to copied NMData
        if result._folder is not None:
            from pyneuromatic.core.nm_data import NMData
            result._NMChannel__thedata = []
            data_container = result._folder.data
            for d in self._NMChannel__thedata:
                # Check if this NMData was already copied (in memo)
                if id(d) in memo:
                    result._NMChannel__thedata.append(memo[id(d)])
                else:
                    # Try to find by name in the folder's data container
                    o = data_container.get(d.name)
                    if isinstance(o, NMData):
                        result._NMChannel__thedata.append(o)
        else:
            # Direct copy: just copy the list of references
            result._NMChannel__thedata = list(self._NMChannel__thedata)

        return result

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        k.update({"xscale": self.__xscale.to_dict()})
        k.update({"yscale": self.__yscale.to_dict()})
        return k

    @property
    def data(self) -> list[NMData]:
        return self.__thedata

    @property
    def xscale(self) -> NMScaleX:
        return self.__xscale

    @property
    def yscale(self) -> NMScaleY:
        return self.__yscale


class NMChannelContainer(NMObjectContainer):
    """
    Container of NMChannels
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMChannelContainer0",
        rename_on: bool = False,
        name_prefix: str = "",  # default is no prefix
        name_seq_format: str = "A",
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            auto_name_prefix=name_prefix,
            auto_name_seq_format=name_seq_format,
        )

    # override, no super
    def content_type(self) -> str:
        return NMChannel.__name__

    # override
    def new(
        self,
        name: str | None = None,  # not used, instead name = name_next()
        select: bool = False,
        xscale: dict | None = None,
        yscale: dict | None = None,
        quiet: bool = nmp.QUIET,
    ) -> NMChannel | None:
        actual_name = self.auto_name_next()
        # Use self._parent (NMDataSeries) to skip container in parent chain,
        # consistent with NMFolder, NMData, NMDataSeries, NMEpoch
        c = NMChannel(
            parent=self._parent,
            name=actual_name,
            xscale=xscale,
            yscale=yscale
        )
        if super()._new(c, select=select, quiet=quiet):
            return c
        return None
