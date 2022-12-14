#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:24:16 2020

@author: jason
"""
from nm_object import NMObject
import nm_preferences as nmp
import nm_utilities as nmu
from typing import Dict, List, Union

SCALE_DEFAULT = {'offset': 0, 'label': '', 'units': ''}
SCALEX_DEFAULT = {'offset': 0, 'start': 0, 'delta': 1, 'label': '',
                  'units': ''}


class NMScale(NMObject):
    """
    NM Scale class

    Contains data y-scale parameters
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMScale',
        scale: Dict[str, str] = {},
        copy: nmu.NMScaleType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__offset = 0
        self.__label = ''
        self.__units = ''

        if isinstance(copy, NMScale):
            self.offset = copy.offset
            self.label = copy.label
            self.units = copy.units
        else:
            if isinstance(scale, dict):
                self._scale_set(scale, quiet=True)
            elif scale is None:
                pass  # OK
            else:
                e = self._type_error('scale', 'dictionary', tp='')  # no tp yet
                raise TypeError(e)

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
    def scale(self) -> Dict[str, str]:
        d = {'offset': self.__offset}
        d.update({'label': self.__label, 'units': self.__units})
        return d

    @scale.setter
    def scale(self, scale: Dict[str, str]) -> None:
        self._scale_set(scale)

    def _scale_set(
        self,
        scale: Dict[str, str],
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not isinstance(scale, dict):
            e = self._type_error('scale', 'dictionary')
            raise TypeError(e)
        for k in scale.keys():
            if k.lower() not in SCALE_DEFAULT.keys():
                raise KeyError('unknown scale key ' + nmu.quotes(k))
            if k.lower() == 'offset':
                if not self._offset_set(scale['offset'], quiet=quiet):
                    return False
            if k.lower() == 'label':
                if not self._label_set(scale['label'], quiet=quiet):
                    return False
            if k.lower() == 'units':
                if not self._units_set(scale['units'], quiet=quiet):
                    return False
        return True

    @property
    def offset(self) -> Union[float, int]:
        return self.__offset

    @offset.setter
    def offset(self, offset: Union[float, int]) -> None:
        self._offset_set(offset)

    def _offset_set(
        self,
        offset: Union[float, int],
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not isinstance(offset, int) and not isinstance(offset, float):
            e = self._type_error('offset', 'number')
            raise TypeError(e)
        if not nmu.number_ok(offset):
            e = self._value_error('offset')
            raise ValueError(e)
        if offset == self.__offset:
            return True  # no change
        old = self.__offset
        self.__offset = offset
        self._modified()
        h = nmu.history_change('offset', old, offset)
        self.note = h
        self._history(h, quiet=quiet)
        return True

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        self._label_set(label)

    def _label_set(
        self,
        label: str,
        quiet: bool = nmp.QUIET
    ) -> bool:
        if label is None:
            label = ''
        elif not isinstance(label, str):
            e = self._type_error('label', 'string')
            raise TypeError(e)
        if label == self.__label:
            return True  # no change
        old = self.__label
        self.__label = label
        self._modified()
        h = nmu.history_change('label', old, label)
        self.note = h
        self._history(h, quiet=quiet)
        return True

    @property
    def units(self) -> str:
        return self.__units

    @units.setter
    def units(self, units: str) -> None:
        self._units_set(units)

    def _units_set(
        self,
        units: str,
        quiet: bool = nmp.QUIET
    ) -> bool:
        if units is None:
            units = ''
        elif not isinstance(units, str):
            e = self._type_error('units', 'string')
            raise TypeError(e)
        if units == self.__units:
            return True  # no change
        old = self.__units
        self.__units = units
        self._modified()
        h = nmu.history_change('units', old, units)
        self.note = h
        self._history(h, quiet=quiet)
        return True


class NMScaleX(NMScale):
    """
    NM X-Scale class

    Contains data x-scale parameters
    """

    def __init__(
        self,
        parent: object = None,
        name: str = 'NMScaleX',
        scale: Dict[str, str] = {},
        copy: nmu.NMScaleXType = None  # see copy()
    ) -> None:

        super().__init__(parent=parent, name=name, copy=copy)

        self.__start = 0
        self.__delta = 1

        if isinstance(copy, NMScaleX):
            self.start = copy.start
            self.delta = copy.delta
        elif isinstance(scale, dict):
            self._scale_set(scale, quiet=True)

    # override, no super
    def copy(self) -> nmu.NMScaleXType:
        return NMScaleX(copy=self)

    # override, no super
    @property
    def scale(self) -> Dict[str, str]:
        d = {'offset': self.offset}
        d.update({'start': self.start, 'delta': self.delta})
        d.update({'label': self.label, 'units': self.units})
        return d

    # override, no super
    def _scale_set(
        self,
        scale: Dict[str, str],
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not isinstance(scale, dict):
            e = self._type_error('scale', 'dictionary')
            raise TypeError(e)
        for k in scale.keys():
            if k.lower() not in SCALEX_DEFAULT.keys():
                raise KeyError('unknown scale key ' + nmu.quotes(k))
            if k.lower() == 'offset':
                if not self._offset_set(scale['offset'], quiet=quiet):
                    # TODO: self.__offset_set is INCORRECT
                    return False
            if k.lower() == 'start':
                if not self._start_set(scale['start'], quiet=quiet):
                    return False
            if k.lower() == 'delta':
                if not self._delta_set(scale['delta'], quiet=quiet):
                    return False
            if k.lower() == 'label':
                if not self._label_set(scale['label'], quiet=quiet):
                    # TODO: self.__label_set is INCORRECT
                    return False
            if k.lower() == 'units':
                if not self._units_set(scale['units'], quiet=quiet):
                    # TODO: self.__units_set is INCORRECT
                    return False
        return True

    @property
    def start(self) -> Union[float, int]:
        return self.__start

    @start.setter
    def start(self, start: Union[float, int]) -> None:
        self._start_set(start)

    def _start_set(
        self,
        start: Union[float, int],
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not isinstance(start, int) and not isinstance(start, float):
            e = self._type_error('start', 'number')
            raise TypeError(e)
        if not nmu.number_ok(start):
            e = self._value_error('start')
            raise ValueError(e)
        if start == self.__start:
            return True  # no change
        old = self.__start
        self.__start = start
        self._modified()
        h = nmu.history_change('start', old, start)
        self.note = h
        self._history(h, quiet=quiet)
        return True

    @property
    def delta(self) -> Union[float, int]:
        return self.__delta

    @delta.setter
    def delta(self, delta: Union[float, int]) -> None:
        self._delta_set(delta)

    def _delta_set(
        self,
        delta: Union[float, int],
        quiet: bool = nmp.QUIET
    ) -> bool:
        if not isinstance(delta, int) and not isinstance(delta, float):
            e = self._type_error('delta', 'number')
            raise TypeError(e)
        if not nmu.number_ok(delta):
            e = self._value_error('delta')
            raise ValueError(e)
        if delta == self.__delta:
            return True  # no change
        old = self.__delta
        self.__delta = delta
        self._modified()
        h = nmu.history_change('delta', old, delta)
        self.note = h
        self._history(h, quiet=quiet)
        return True
