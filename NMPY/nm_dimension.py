#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:24:16 2020

@author: jason
"""
from nm_object import NMObject
from nm_note import NoteContainer
import nm_preferences as nmp
import nm_utilities as nmu
# cannot import Data class


class Dimension(NMObject):
    """
    NM Dimension class
    """

    def __init__(self, parent, name, dim={}, notes=None, **copy):
        super().__init__(parent, name)
        if dim is None:
            dim = {}
        elif not isinstance(dim, dict):
            e = self._type_error(dim, 'dimension dictionary')
            raise TypeError(e)
        if notes is None:
            self._note_container = None
        elif isinstance(notes, NoteContainer):
            self._note_container = notes  # reference to notes container
        else:
            e = self._type_error(notes, 'NoteContainer')
            raise TypeError(e)
        self._offset = 0  # not to be controlled by master
        self._label = ''
        self._units = ''
        self._master = None  # e.g. dimension of a DataSeries
        self._dim_list = ['offset', 'label', 'units', 'master']
        self._param_list += self._dim_list
        if dim:
            self._dim_set(dim, quiet=True)

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update(self.dim)
        return k

    # override, no super
    def copy(self):
        return Dimension(self._parent, self.name, dim=self.dim,
                         notes=self._note_container)

    def _note_new(self, thenote, quiet=True):
        # notes should be quiet
        if isinstance(self._note_container, NoteContainer):
            return self._note_container.new(thenote, quiet=quiet)
        return None

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, dimension):
        return self._master_set(dimension)

    def _master_set(self, dimension, quiet=nmp.QUIET):
        if dimension is None:
            pass  # ok, master off
        elif dimension.__class__.__name__ != self.__class__.__name__:
            e = self._type_error(dimension, self.__class__.__name__)
            raise TypeError(e)
        elif dimension == self:
            e = self._error('got ' + nmu.quotes('self') + ' for master')
            raise ValueError(e)
        elif dimension._master_lock:
            e = self._error('dimension has master and therefore cannot act ' +
                            'as master: ' + dimension.treepath())
            raise RuntimeError(e)
        old = self._master
        if old == dimension:
            return True
        self._master = dimension
        self._modified()
        h = nmu.history_change('master', old, dimension)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True

    @property
    def _master_lock(self):
        return self._master.__class__.__name__ == self.__class__.__name__

    @property
    def _master_error(self):
        if self._master is None:
            return ''
        e = ('this dimension is locked to master ' +
             nmu.quotes(self._master.treepath()))
        return self._error(e)

    @property
    def dim(self):
        if self._master_lock:
            d = self._master.dim
            d.update({'master': self._master})
        else:
            d = {'offset': self._offset}
            d.update({'label': self._label, 'units': self._units})
            d.update({'master': None})
        return d

    @dim.setter
    def dim(self, dim):
        return self._dim_set(dim)

    def _dim_set(self, dim, quiet=nmp.QUIET):
        if not isinstance(dim, dict):
            e = self._type_error(dim, 'dimension dictionary')
            raise TypeError(e)
        keys = dim.keys()
        for k in keys:
            if k not in self._dim_list:
                raise KeyError('unknown dimension key ' + nmu.quotes(k))
        if 'offset' in keys:
            if not self._offset_set(dim['offset'], quiet=quiet):
                return False
        if 'master' in keys:
            if not self._master_set(dim['master'], quiet=quiet):
                return False
        if 'label' in keys:
            if not self._label_set(dim['label'], quiet=quiet):
                return False
        if 'units' in keys:
            if not self._units_set(dim['units'], quiet=quiet):
                return False
        return True

    @property
    def offset(self):
        # no master
        return self._offset

    @offset.setter
    def offset(self, offset):
        return self._offset_set(offset)

    def _offset_set(self, offset, quiet=nmp.QUIET):
        # no master
        if isinstance(offset, bool):
            e = self._type_error(offset, 'number')
            raise TypeError(e)
        if not isinstance(offset, float) and not isinstance(offset, int):
            e = self._type_error(offset, 'number')
            raise TypeError(e)
        if not nmu.number_ok(offset):
            e = self._value_error(offset)
            raise ValueError(e)
        old = self._offset
        if offset == old:
            return True
        self._offset = offset
        self._modified()
        h = nmu.history_change('offset', old, offset)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True

    @property
    def label(self):
        if self._master_lock:
            return self._master.label
        return self._label

    @label.setter
    def label(self, label):
        return self._label_set(label)

    def _label_set(self, label, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if label is None:
            label = ''
        elif not isinstance(label, str):
            e = self._type_error(label, 'string')
            raise TypeError(e)
        old = self._label
        if label == old:
            return True
        self._label = label
        self._modified()
        h = nmu.history_change('label', old, label)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True

    @property
    def units(self):
        if self._master_lock:
            return self._master.units
        return self._units

    @units.setter
    def units(self, units):
        return self._units_set(units)

    def _units_set(self, units, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if units is None:
            units = ''
        elif not isinstance(units, str):
            e = self._type_error(units, 'string')
            raise TypeError(e)
        old = self._units
        if units == old:
            return True
        self._units = units
        self._modified()
        h = nmu.history_change('units', old, units)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True


class XDimension(Dimension):
    """
    NM XDimension class
    """

    def __init__(self, parent, name, dim={}, notes=None, **copy):
        super().__init__(parent, name, notes=notes, **copy)
        if dim is None:
            dim = {}
        elif not isinstance(dim, dict):
            e = self._type_error(dim, 'dimension dictionary')
            raise TypeError(e)
        self._start = 0
        self._delta = 1
        self._xdata = None
        self._dim_list += ['start', 'delta', 'xdata']
        self._param_list += ['start', 'delta', 'xdata']
        if dim and isinstance(dim, dict):
            self._dim_set(dim, quiet=True)

    # override, no super
    def copy(self):
        return XDimension(self._parent, self.name, dim=self.dim,
                          notes=self._note_container)

    @property
    def xdata(self):
        if self._master_lock:
            return self._master.xdata
        return self._xdata

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if xdata is None:
            pass  # ok, unlock
        elif xdata.__class__.__name__ != 'Data':
            e = self._type_error(xdata, 'Data')
            raise TypeError(e)
        old = self._xdata
        if xdata == old:
            return True
        self._xdata = xdata
        self._modified()
        if old:
            oldname = old.name
        else:
            oldname = 'None'
        if xdata:
            newname = xdata.name
        else:
            newname = 'None'
        h = nmu.history_change('xdata', oldname, newname)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True

    @property
    def _xdata_lock(self):
        return self._xdata.__class__.__name__ == 'Data'

    @property
    def _xdata_error(self):
        if self._xdata is None:
            return ''
        e = ('this x-dimension is locked to xdata ' +
             nmu.quotes(self._xdata.treepath()))
        return self._error(e)

    # override, no super
    @property
    def dim(self):
        if self._master_lock:
            d = self._master.dim
            d.update({'master': self._master})
        else:
            if self._xdata_lock:
                d = {'offset': self._offset, 'xdata': self._xdata}
                d.update({'start': self.start, 'delta': self.delta})
                d.update({'label': self.label, 'units': self.units})
                d.update({'master': None})
            else:
                d = {'offset': self._offset}
                d.update({'start': self.start, 'delta': self.delta})
                d.update({'label': self.label, 'units': self.units})
                d.update({'xdata': None, 'master': None})
        return d

    # override, no super
    def _dim_set(self, dim, quiet=nmp.QUIET):
        if not isinstance(dim, dict):
            e = self._type_error(dim, 'dimension dictionary')
            raise TypeError(e)
        keys = dim.keys()
        for k in keys:
            if k not in self._dim_list:
                raise KeyError('unknown dimension key ' + nmu.quotes(k))
        if 'offset' in keys:
            if not self._offset_set(dim['offset'], quiet=quiet):
                return False
        if 'master' in keys:
            if not self._master_set(dim['master'], quiet=quiet):
                return False
        if 'xdata' in keys:
            if not self._xdata_set(dim['xdata'], quiet=quiet):
                return False
        if 'label' in keys:
            if not self._label_set(dim['label'], quiet=quiet):
                return False
        if 'units' in keys:
            if not self._units_set(dim['units'], quiet=quiet):
                return False
        if 'start' in keys:
            if not self._start_set(dim['start'], quiet=quiet):
                return False
        if 'delta' in keys:
            if not self._delta_set(dim['delta'], quiet=quiet):
                return False
        return True

    @property
    def start(self):
        if self._master_lock:
            return self._master.start
        if self._xdata_lock:
            return 0
        return self._start

    @start.setter
    def start(self, start):
        return self._start_set(start)

    def _start_set(self, start, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if self._xdata_lock:
            raise RuntimeError(self._xdata_error)
        if isinstance(start, bool):
            e = self._type_error(start, 'number')
            raise TypeError(e)
        if not isinstance(start, float) and not isinstance(start, int):
            e = self._type_error(start, 'number')
            raise TypeError(e)
        old = self._start
        if start == old:
            return True
        self._start = start
        self._modified()
        h = nmu.history_change('start', old, start)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True

    @property
    def delta(self):
        if self._master_lock:
            return self._master.delta
        if self._xdata_lock:
            return 1
        return self._delta

    @delta.setter
    def delta(self, delta):
        return self._delta_set(delta)

    def _delta_set(self, delta, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if self._xdata_lock:
            raise RuntimeError(self._xdata_error)
        if isinstance(delta, bool):
            e = self._type_error(delta, 'number')
            raise TypeError(e)
        if not isinstance(delta, float) and not isinstance(delta, int):
            e = self._type_error(delta, 'number')
            raise TypeError(e)
        old = self._delta
        if delta == old:
            return True
        self._delta = delta
        self._modified()
        h = nmu.history_change('delta', old, delta)
        self._note_new(h)
        self._history(h, quiet=quiet)
        return True

    # override, no super
    @property
    def label(self):
        if self._master_lock:
            return self._master.label
        if self._xdata_lock:
            return 'sample'
        return self._label

    # override
    def _label_set(self, label, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if self._xdata_lock:
            raise RuntimeError(self._xdata_error)
        return super()._label_set(label=label, quiet=quiet)

    # override, no super
    @property
    def units(self):
        if self._master_lock:
            return self._master.units
        if self._xdata_lock:
            return '#'
        return self._units

    # override
    def _units_set(self, units, quiet=nmp.QUIET):
        if self._master_lock:
            raise RuntimeError(self._master_error)
        if self._xdata_lock:
            raise RuntimeError(self._xdata_error)
        return super()._units_set(units=units, quiet=quiet)
