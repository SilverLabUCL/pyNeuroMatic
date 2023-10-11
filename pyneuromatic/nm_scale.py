#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:24:16 2020

@author: jason
"""
from typing import Dict, Union

from nm_object import NMObject
import nm_preferences as nmp
import nm_utilities as nmu


class NMScale(NMObject):
    """
    NM Scale class

    Contains data y-scale parameters
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMScale",
        scale: Dict[str, object] = {},
        copy: nmu.NMScaleType = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__label = ""
        self.__units = ""

        if copy is None:
            pass  # ok
        elif isinstance(copy, NMScale):
            self.__label = copy.label
            self.__units = copy.units
        else:
            e = nmu.typeerror(copy, "copy", "NMScale")
            raise TypeError(e)

        if scale is None:
            pass  # ok
        elif isinstance(scale, dict):
            if not copy:
                self._scale_set(scale, quiet=True)
        else:
            e = nmu.typeerror(scale, "scale", "dictionary")
            raise TypeError(e)

    # override
    def __eq__(self, other: nmu.NMScaleType) -> bool:
        if not super().__eq__(other):
            return False
        if self.__label.lower() != other.label.lower():
            return False
        if self.__units.lower() != other.units.lower():
            return False
        return True

    # override, no super
    def copy(self) -> nmu.NMScaleType:
        return NMScale(copy=self)

    # override
    @property
    def parameters(self) -> Dict[str, object]:
        k = super().parameters
        k.update(self.scale)
        return k

    @property
    def scale(self) -> Dict[str, object]:
        return {"label": self.__label, "units": self.__units}

    @scale.setter
    def scale(self, scale: Dict[str, object]) -> None:
        self._scale_set(scale)

    def _scale_set(self, scale: Dict[str, object], quiet: bool = nmp.QUIET) -> None:
        if not isinstance(scale, dict):
            e = nmu.typeerror(scale, "scale", "dictionary")
            raise TypeError(e)
        for k, v in scale.items():
            k = k.lower()
            if k == "label":
                self._label_set(v, quiet=quiet)
            elif k == "units":
                self._units_set(v, quiet=quiet)
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        return self._label_set(label)

    def _label_set(self, label: str, quiet: bool = nmp.QUIET) -> None:
        if label is None:
            label = ""
        elif not isinstance(label, str):
            e = nmu.typeerror(label, "label", "string")
            raise TypeError(e)
        old = self.__label
        if label == old:
            return None
        self.__label = label
        self.modified()
        h = nmu.history_change("label", old, label)
        # self.note = h
        # self._history(h, quiet=quiet)
        return None

    @property
    def units(self) -> str:
        return self.__units

    @units.setter
    def units(self, units: str) -> None:
        return self._units_set(units)

    def _units_set(self, units: str, quiet: bool = nmp.QUIET) -> None:
        if units is None:
            units = ""
        elif not isinstance(units, str):
            e = nmu.typeerror(units, "units", "string")
            raise TypeError(e)
        old = self.__units
        if units == old:
            return None
        self.__units = units
        self.modified()
        h = nmu.history_change("units", old, units)
        # self.note = h
        # self._history(h, quiet=quiet)
        return None


class NMScaleX(NMScale):
    """
    NM X-Scale class

    Contains data x-scale parameters
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMScaleX",
        scale: Dict[str, object] = {},
        copy: nmu.NMScaleXType = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        self.__start = 0.0
        self.__delta = 1.0

        if copy is None:
            pass  # ok
        elif isinstance(copy, NMScaleX):
            self.__start = copy.start
            self.__delta = copy.delta
        else:
            e = nmu.typeerror(copy, "copy", "NMScaleX")
            raise TypeError(e)

        if scale is None:
            pass  # ok
        elif isinstance(scale, dict):
            if not copy:
                self._scale_set(scale, quiet=True)
        else:
            e = nmu.typeerror(scale, "scale", "dictionary")
            raise TypeError(e)

    # override
    def __eq__(self, other: nmu.NMScaleType) -> bool:
        if not super().__eq__(other):
            return False
        if self.__start != other.start:
            return False
        if self.__delta != other.delta:
            return False
        return True

    # override, no super
    def copy(self) -> nmu.NMScaleXType:
        return NMScaleX(copy=self)

    # override
    @property
    def scale(self) -> Dict[str, object]:
        d = super().scale
        d.update({"start": self.__start, "delta": self.__delta})
        return d

    @scale.setter
    def scale(self, scale: Dict[str, object]) -> None:
        self._scale_set(scale)

    # override, no super
    def _scale_set(self, scale: Dict[str, object], quiet: bool = nmp.QUIET) -> None:
        if not isinstance(scale, dict):
            e = nmu.typeerror(scale, "scale", "dictionary")
            raise TypeError(e)
        for k, v in scale.items():
            k = k.lower()
            if k == "label":
                self._label_set(v, quiet=quiet)
            elif k == "units":
                self._units_set(v, quiet=quiet)
            elif k == "start":
                self._start_set(v, quiet=quiet)
            elif k == "delta":
                self._delta_set(v, quiet=quiet)
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def start(self) -> float:
        return self.__start

    @start.setter
    def start(self, start: Union[float, int]) -> None:
        return self._start_set(start)

    def _start_set(self, start: Union[float, int], quiet: bool = nmp.QUIET) -> None:
        if isinstance(start, float):
            pass
        elif isinstance(start, int) and not isinstance(start, bool):
            pass
        else:
            e = nmu.typeerror(start, "start", "number")
            raise TypeError(e)
        if not nmu.number_ok(start):
            raise ValueError("start: %s" % start)
        old = self.__start
        if start == old:
            return None
        self.__start = start
        self.modified()
        h = nmu.history_change("start", old, start)
        # self.note = h
        # self._history(h, quiet=quiet)
        return None

    @property
    def delta(self) -> float:
        return self.__delta

    @delta.setter
    def delta(self, delta: Union[float, int]) -> None:
        return self._delta_set(delta)

    def _delta_set(self, delta: Union[float, int], quiet: bool = nmp.QUIET) -> None:
        if isinstance(delta, float):
            pass
        elif isinstance(delta, int) and not isinstance(delta, bool):
            pass
        else:
            e = nmu.typeerror(delta, "delta", "number")
            raise TypeError(e)
        if not nmu.number_ok(delta):
            raise ValueError("delta: %s" % delta)
        old = self.__delta
        if delta == old:
            return None
        self.__delta = delta
        self.modified()
        h = nmu.history_change("delta", old, delta)
        # self.note = h
        # self._history(h, quiet=quiet)
        return None
