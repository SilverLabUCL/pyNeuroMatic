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

DIM_LIST = ['label', 'units', 'master'] + ['xdata', 'start', 'delta']

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

    def _master_lock(self, alert=True, tp=''):
        if isinstance(self._master, Dimensions):
            if alert:
                a = ('dims are locked to master ' +
                     nmu.quotes(self._master.tree_path()))
                self._alert(a, tp=tp, frame=3)
            return True
        return False

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

    @property
    def dims(self):
        if isinstance(self._master, Dimensions):
            d = self._master.dims
            d.update({'master': self._master})
        else:
            d = {'label': self._label, 'units': self._units, 'master': None}
        return d

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dimensions dictionary')
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in DIM_LIST:
                raise KeyError('unknown dimensions key: ' + k)
        if 'label' in keys:
            self._label_set(dims['label'], alert=alert, quiet=quiet)
        if 'units' in keys:
            self._units_set(dims['units'], alert=alert, quiet=quiet)
        if 'master' in keys:
            self._master_set(dims['master'], quiet=quiet)
        return True

    @property
    def label(self):
        if isinstance(self._master, Dimensions):
            return self._master.label
        return self._label

    @label.setter
    def label(self, label):
        return self._label_set(label)

    def _label_set(self, label, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
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

    def _units_set(self, units, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
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

    # override
    def _dims_set(self, dims, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dimensions dictionary')
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in DIM_LIST:
                raise KeyError('unknown dimensions key: ' + k)
        if 'xdata' in keys:
            self._xdata_set(dims['xdata'], alert=alert, quiet=quiet)
        if 'start' in keys:
            self._start_set(dims['start'], alert=alert, quiet=quiet)
        if 'delta' in keys:
            self._delta_set(dims['delta'], alert=alert, quiet=quiet)
        return super()._dims_set( dims=dims, alert=alert, quiet=quiet)

    @property
    def xdata(self):
        if isinstance(self._master, XDimensions):
            return self._master.xdata
        return self._xdata

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if xdata is None:
            pass  # ok
        elif xdata.__class__.__name__ != 'Data':  # cannot import Data class
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

    def _xdata_lock(self, alert=True, tp=''):
        if self.xdata.__class__.__name__ == 'Data':
            if alert:
                a = ('x-dims are locked to xdata ' +
                     nmu.quotes(self._master.tree_path()))
                self._alert(a, tp=tp, frame=3)
            return True
        return False
    
    def _xdata_alert(self):
        if self.xdata.__class__.__name__ == 'Data':
            xn = nmu.quotes(self.xdata.name)
            return ('x-dims are superceded by xdata ' + xn + '.' + '\n' +
                    'do you want to continue?')
        return ''

    @property
    def start(self):
        if isinstance(self._master, XDimensions):
            return self._master.start
        return self._start

    @start.setter
    def start(self, start):
        return self._start_set(start)

    def _start_set(self, start, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if self._xdata_lock(alert=alert, tp=self._tp):
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
        return self._delta

    @delta.setter
    def delta(self, delta):
        return self._delta_set(delta)

    def _delta_set(self, delta, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if self._xdata_lock(alert=alert, tp=self._tp):
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

    # override
    def _label_set(self, label, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if self._xdata_lock(alert=alert, tp=self._tp):
            return False
        return super()._label_set(label=label, alert=alert, quiet=quiet)

    # override
    def _units_set(self, units, alert=True, quiet=nmp.QUIET):
        if self._master_lock(alert=alert, tp=self._tp):
            return False
        if self._xdata_lock(alert=alert, tp=self._tp):
            return False
        return super()._units_set(units=units, alert=alert, quiet=quiet)
