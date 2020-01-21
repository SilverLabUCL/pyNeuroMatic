#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:24:16 2020

@author: jason
"""
from nm_container import NMObject
from nm_note import NoteContainer
import nm_preferences as nmp
import nm_utilities as nmu
# cannot import Data class

DIM_LIST = ['label', 'units', 'master', 'offset'] + ['start', 'delta', 'xdata']


class Dimensions(NMObject):
    """
    NM Dimensions class
    """

    def __init__(self, parent, name, fxns={}, notes=None):
        super().__init__(parent, name, fxns=fxns)
        if isinstance(notes, NoteContainer):
            self._note_container = notes
        else:
            self._note_container = None
        self._offset = 0  # not controlled by master
        self._label = ''
        self._units = ''
        self._master = None  # e.g. DataSeries

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update(self.dims)
        return k

    # override, no super
    @property
    def content(self):
        return {'dimension': self.name}

    def _note_new(self, note, quiet=True):
        if isinstance(self._note_container, NoteContainer):
            return self._note_container.new(note=note, quiet=quiet)
        return None

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, dimensions):
        return self._master_set(dimensions)

    def _master_set(self, dimensions, quiet=nmp.QUIET):
        if dimensions is None:
            pass  # ok
        elif not isinstance(dimensions, Dimensions):
            raise TypeError(nmu.type_error(dimensions, 'Dimensions'))
        old = self._master
        if old == dimensions:
            return True
        self._master = dimensions
        self._modified()
        h = nmu.history_change('master', old, dimensions)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def _master_lock(self, tp='', quiet=nmp.QUIET):
        if isinstance(self._master, Dimensions):
            if not quiet:
                e = ('dims are locked to master ' +
                     nmu.quotes(self._master.tree_path()))
                self._error(e, tp=tp, frame=3)
            return True
        return False

    @property
    def dims(self):
        if isinstance(self._master, Dimensions):
            d = self._master.dims
            d.update({'master': self._master})
        else:
            d = {'offset': self._offset, 'label': self._label}
            d.update({'units': self._units, 'master': None})
        return d

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dimensions dictionary')
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in DIM_LIST:
                raise KeyError('unknown dimensions key: ' + k)
        if 'master' in keys:
            self._master_set(dims['master'], quiet=quiet)
        if isinstance(self._master, Dimensions):
            return True  # master is on, skip anything else
        if 'offset' in keys:
            self._offset_set(dims['offset'], quiet=quiet)
        if 'label' in keys:
            self._label_set(dims['label'], quiet=quiet)
        if 'units' in keys:
            self._units_set(dims['units'], quiet=quiet)
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
        if not isinstance(offset, float) and not isinstance(offset, int):
            raise TypeError(nmu.type_error(offset, 'number'))
        if not nmu.number_ok(offset):
            raise ValueError('bad offset: ' + str(offset))
        old = self._offset
        if offset == old:
            return True
        self._offset = offset
        self._modified()
        h = nmu.history_change('offset', old, offset)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def label(self):
        if isinstance(self._master, Dimensions):
            return self._master.label
        return self._label

    @label.setter
    def label(self, label):
        return self._label_set(label)

    def _label_set(self, label, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if not isinstance(label, str):
            raise TypeError(nmu.type_error(label, 'string'))
        old = self._label
        if label == old:
            return True
        self._label = label
        self._modified()
        h = nmu.history_change('label', old, label)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def units(self):
        if isinstance(self._master, Dimensions):
            return self._master.units
        return self._units

    @units.setter
    def units(self, units):
        return self._units_set(units)

    def _units_set(self, units, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if not isinstance(units, str):
            raise TypeError(nmu.type_error(units, 'string'))
        old = self._units
        if units == old:
            return True
        self._units = units
        self._modified()
        h = nmu.history_change('units', old, units)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True


class XDimensions(Dimensions):
    """
    NM XDimensions class
    """

    def __init__(self, parent, name, fxns={}, notes=None):
        super().__init__(parent, name, fxns=fxns, notes=notes)
        self._xdata = None
        self._start = 0
        self._delta = 1

    # override
    def _master_set(self, xdimensions, quiet=nmp.QUIET):
        if xdimensions is None:
            pass  # ok
        elif not isinstance(xdimensions, XDimensions):
            raise TypeError(nmu.type_error(xdimensions, 'XDimensions'))
        return super()._master_set(xdimensions, quiet=quiet)

    # override, no super
    @property
    def dims(self):
        if isinstance(self._master, XDimensions):
            d = self._master.dims
            d.update({'master': self._master})
        else:
            d = {'xdata': self._xdata}
            d.update({'start': self._start, 'delta': self._delta})
            d.update({'label': self._label, 'units': self._units})
            d.update({'master': None})
        return d

    # override, no super
    def _dims_set(self, dims, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if self._xdata_lock(tp=self._tp, quiet=quiet):
            return False
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dimensions dictionary')
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in DIM_LIST:
                raise KeyError('unknown dimensions key: ' + k)
        if 'master' in keys:
            self._master_set(dims['master'], quiet=quiet)
        if isinstance(self._master, XDimensions):
            return True  # master is on, skip the rest
        if 'xdata' in keys:
            self._xdata_set(dims['xdata'], quiet=quiet)
        if self.xdata.__class__.__name__ == 'Data':
            return True  # xdata is on, skip the rest
        if 'label' in keys:
            self._label_set(dims['label'], quiet=quiet)
        if 'units' in keys:
            self._units_set(dims['units'], quiet=quiet)
        if 'start' in keys:
            self._start_set(dims['start'], quiet=quiet)
        if 'delta' in keys:
            self._delta_set(dims['delta'], quiet=quiet)
        return True

    @property
    def xdata(self):
        if isinstance(self._master, XDimensions):
            return self._master.xdata
        return self._xdata

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if xdata is None:
            pass  # ok
        elif xdata.__class__.__name__ != 'Data':
            raise TypeError(nmu.type_error(xdata, 'Data'))
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
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def _xdata_lock(self, tp='', quiet=nmp.QUIET):
        if self.xdata.__class__.__name__ == 'Data':
            if not quiet:
                e = ('x-dims are locked to xdata ' +
                     nmu.quotes(self._master.tree_path()))
                self._error(e, tp=tp, frame=3)
            return True
        return False

    @property
    def start(self):
        if isinstance(self._master, XDimensions):
            return self._master.start
        if self.xdata.__class__.__name__ == 'Data':
            return 0
        return self._start

    @start.setter
    def start(self, start):
        return self._start_set(start)

    def _start_set(self, start, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if self._xdata_lock(tp=self._tp, quiet=quiet):
            return False
        if not isinstance(start, float) and not isinstance(start, int):
            raise TypeError(nmu.type_error(start, 'number'))
        if not nmu.number_ok(start):
            raise ValueError('bad start: ' + str(start))
        old = self._start
        if start == old:
            return True
        self._start = start
        self._modified()
        h = nmu.history_change('start', old, start)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def delta(self):
        if isinstance(self._master, XDimensions):
            return self._master.delta
        if self.xdata.__class__.__name__ == 'Data':
            return 1
        return self._delta

    @delta.setter
    def delta(self, delta):
        return self._delta_set(delta)

    def _delta_set(self, delta, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if self._xdata_lock(tp=self._tp, quiet=quiet):
            return False
        if not isinstance(delta, float) and not isinstance(delta, int):
            raise TypeError(nmu.type_error(delta, 'number'))
        if not nmu.number_ok(delta):
            raise ValueError('bad delta: ' + str(delta))
        old = self._delta
        if delta == old:
            return True
        self._delta = delta
        self._modified()
        h = nmu.history_change('delta', old, delta)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    # override, no super
    @property
    def label(self):
        if isinstance(self._master, Dimensions):
            return self._master.label
        if self.xdata.__class__.__name__ == 'Data':
            return 'sample'
        return self._label

    # override
    def _label_set(self, label, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if self._xdata_lock(tp=self._tp, quiet=quiet):
            return False
        return super()._label_set(label=label, quiet=quiet)

    # override, no super
    @property
    def units(self):
        if isinstance(self._master, Dimensions):
            return self._master.units
        if self.xdata.__class__.__name__ == 'Data':
            return '#'
        return self._units

    # override
    def _units_set(self, units, quiet=nmp.QUIET):
        if self._master_lock(tp=self._tp, quiet=quiet):
            return False
        if self._xdata_lock(tp=self._tp, quiet=quiet):
            return False
        return super()._units_set(units=units, quiet=quiet)
