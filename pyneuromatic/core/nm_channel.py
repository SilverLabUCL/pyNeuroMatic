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
from pyneuromatic.core.nm_transform import NMTransform, _transform_from_dict, apply_transforms
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager (NMObject, root)
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

    A channel can have a list of filters and transforms (``transform``),
    applied in order. They never change the arrays: :meth:`transformed`
    returns a transformed copy, which is what the tools analyse (as the
    channel graphs of NeuroMatic).
    """

    # Extend NMObject's special attrs with NMChannel's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMChannel__thedata",
        "_NMChannel__xscale",
        "_NMChannel__yscale",
        "_NMChannel__transform",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMChannel0",
        xscale: dict | None = None,
        yscale: dict | None = None,
        transform: list[NMTransform] | list[dict] | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__thedata: list[NMData] = []  # list of NMData refs for this channel
        self.__xscale: NMScaleX = _xscale_from_dict(xscale, parent=self)
        self.__yscale: NMScaleY = _yscale_from_dict(yscale, parent=self)
        self.__transform: list[NMTransform] = []
        if transform is not None:
            self._transform_set(transform, quiet=True)

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
        if [t.to_dict() for t in self.__transform] != [
            t.to_dict() for t in other.transform
        ]:
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

        # __transform: deep copy each step and reset parent
        result._NMChannel__transform = []
        for t in self._NMChannel__transform:
            t_copy = copy.deepcopy(t, memo)
            t_copy._parent = result
            result._NMChannel__transform.append(t_copy)

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
        k.update({"transform": [t.to_dict() for t in self.__transform]})
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

    @property
    def transform(self) -> list[NMTransform]:
        """Filters and transforms, applied in order (empty if none).

        Set with a list of NMTransform objects or their dicts (e.g.
        ``{"type": "NMTransformFilter", ...}``); None or [] clears it.
        """
        return self.__transform

    @transform.setter
    def transform(self, transform_list: list[NMTransform] | list[dict] | None) -> None:
        self._transform_set(transform_list)

    def _transform_set(
        self,
        transform_list: list[NMTransform] | list[dict] | None,
        quiet: bool = nmc.QUIET,
    ) -> None:
        if transform_list is None:
            transform_list = []
        if not isinstance(transform_list, list):
            e = nmu.type_error_str(transform_list, "transform", "list")
            raise TypeError(e)
        result = []
        for item in transform_list:
            if isinstance(item, NMTransform):
                t = item
            elif isinstance(item, dict):
                t = _transform_from_dict(item)
            else:
                e = nmu.type_error_str(item, "transform item", "NMTransform or dict")
                raise TypeError(e)
            t._parent = self
            result.append(t)
        self.__transform = result
        nmh.history(
            "set transform=[%s]" % ", ".join(t.type_str for t in result),
            path=self.path_str,
            quiet=quiet,
        )

    @property
    def transform_on(self) -> bool:
        """True if any filter or transform is turned on."""
        return any(t.on for t in self.__transform)

    def transform_str(self) -> str:
        """The steps that are on, for notes, e.g. "Filter(...) > Baseline(...)";
        empty if none."""
        prefix = NMTransform.__name__
        steps = []
        for t in self.__transform:
            if t.on:
                text = repr(t)
                steps.append(text[len(prefix):] if text.startswith(prefix) else text)
        return " > ".join(steps)

    def transformed(self, data: NMData) -> NMData:
        """*data* as the tools analyse it, with the channel's filters and
        transforms.

        Returns a copy with the same name, path and scales (the array is
        never changed), or *data* itself when no step is on or it has no
        values.
        """
        from pyneuromatic.core.nm_data import NMData

        if not isinstance(data, NMData):
            raise TypeError(nmu.type_error_str(data, "data", "NMData"))
        if not self.transform_on or data.nparray is None:
            return data
        ydata = apply_transforms(data.nparray, self.__transform, xscale=data.xscale)
        return NMData(
            parent=data._parent,  # same path, e.g. in results
            name=data.name,
            nparray=ydata,
            xscale=data.xscale.to_dict(),
            yscale=data.yscale.to_dict(),
            xarray=data.xarray,
        )


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
        name: str | None = None,  # not used, instead name = auto_name_next()
        select: bool = False,
        xscale: dict | None = None,
        yscale: dict | None = None,
        quiet: bool = nmc.QUIET,
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
        if super()._add(c, select=select, quiet=quiet):
            return c
        return None
