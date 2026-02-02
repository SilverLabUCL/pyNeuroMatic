# -*- coding: utf-8 -*-
"""
[Module description].

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
import datetime
from typing import Any
import h5py

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeriesContainer
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.analysis.nm_tool_folder import NMToolFolderContainer
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager
    NMProject (project0)
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
"""


class NMFolder(NMObject):
    """
    NM Data Folder class
    """

    # Extend NMObject's special attrs with NMFolder's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMFolder__data_container",
        "_NMFolder__dataseries_container",
        "_NMFolder__toolfolder_container",
        "_NMFolder__toolresults",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMFolder0",
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__data_container: NMDataContainer = NMDataContainer(parent=self)
        self.__dataseries_container: NMDataSeriesContainer = NMDataSeriesContainer(parent=self)
        self.__toolfolder_container: NMToolFolderContainer = NMToolFolderContainer(parent=self)
        self.__toolresults: dict[str, object] = {}  # tool results saved to dict

    # override
    def __eq__(
        self, 
        other: object
    ) -> bool:
        if not isinstance(other, NMFolder):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__data_container != other.__data_container:
            return False
        if self.__dataseries_container != other.__dataseries_container:
            return False
        if self.__toolfolder_container != other.__toolfolder_container:
            return False
        if self.__toolresults != other.__toolresults:
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMFolder:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMFolder by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMFolder
        """
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
        result._NMObject__notes_on = self._NMObject__notes_on
        result._NMObject__notes = copy.deepcopy(self._NMObject__notes, memo)
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMFolder's special attributes

        # __data_container: deep copy and update parent
        result._NMFolder__data_container = copy.deepcopy(
            self._NMFolder__data_container, memo
        )
        result._NMFolder__data_container._parent = result

        # __dataseries_container: deep copy and update parent
        result._NMFolder__dataseries_container = copy.deepcopy(
            self._NMFolder__dataseries_container, memo
        )
        result._NMFolder__dataseries_container._parent = result

        # __toolfolder_container: deep copy and update parent
        result._NMFolder__toolfolder_container = copy.deepcopy(
            self._NMFolder__toolfolder_container, memo
        )
        result._NMFolder__toolfolder_container._parent = result

        # __toolresults: deep copy the dict
        result._NMFolder__toolresults = copy.deepcopy(
            self._NMFolder__toolresults, memo
        )

        return result

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        k.update(self.__data_container.content)
        k.update(self.__dataseries_container.content)
        k.update(self.__toolfolder_container.content)
        return k

    @property
    def data(self) -> NMDataContainer:
        return self.__data_container

    @property
    def dataseries(self) -> NMDataSeriesContainer:
        return self.__dataseries_container

    @property
    def toolfolder(self) -> NMToolFolderContainer:
        return self.__toolfolder_container

    @property
    def toolresults(self) -> dict[str, object]:
        return self.__toolresults

    def toolresults_save(self, tool: str, results: Any) -> str:
        imax_keys = 99
        if not isinstance(tool, str):
            e = nmu.type_error_str(tool, "tool", "string")
            raise TypeError(e)

        tp = self.path_str
        foundkey = False
        for i in range(imax_keys):
            newkey = tool + str(i)
            if newkey not in self.__toolresults:
                foundkey = True
                break
        if not foundkey:
            e = "failed to find unused key for %s results in %s" % (tool, tp)
            raise KeyError(e)

        t = str(datetime.datetime.now())
        r = {}
        r["tool"] = tool
        r["date"] = t
        r["results"] = results
        self.__toolresults[newkey] = r
        print("saved %s results to %s via key '%s' (%s)" %
              (tool, tp, newkey, t))
        return newkey


class NMFolderContainer(NMObjectContainer):
    """
    Container of NMFolders
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMFolderContainer0",
        rename_on: bool = True,
        name_prefix: str = "folder",
        name_seq_format: str = "0",
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
        return NMFolder.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMFolder | None:
        actual_name = self._newkey(name)
        f = NMFolder(parent=self, name=actual_name)
        if super()._new(f, select=select):
            return f
        return None

    def open_hdf5(self):
        dataseries = "Record"
        with h5py.File("nmFolder0.hdf5", "r") as f:
            # print(f.keys())
            data = []
            for k in f.keys():
                if k[0 : len(dataseries)] == dataseries:
                    print(k)
            # for name in f:
            # print(name)
            d = f["RecordA0"]

            for i in d.attrs.keys():
                print(i)
            # cannot get access to attribute values for keys:
            # probably need to update h5py to v 2.10
            # IGORWaveNote
            # IGORWaveType
            # print(d.attrs.__getitem__('IGORWaveNote'))
            # for a in d.attrs:
            # print(item + ":", d.attrs[item])
            # print(item + ":", d.attrs.get(item))
            # print(a.shape)
            # for k in a.keys():
            # print(k)
            # print(a)
            # pf = f['NMPrefix_Record']
            # print(pf)
