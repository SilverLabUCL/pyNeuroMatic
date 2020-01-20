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

DIMS_LIST = ['xdata', 'xstart', 'xdelta', 'xlabel', 'xunits', 'ylabel',
             'yunits']


class Dimensions(NMObject):
    """
    NM Dimensions class
    """

    def __init__(self, parent, name, fxns={}, rename=True):
        super().__init__(parent, name, fxns=fxns, rename=rename)
        self._note_container = None
        self.__dims_master = None  # e.g. DataSeries
        self.__xdata = None
        self.__xstart = 0
        self.__xdelta = 1
        self.__xlabel = ''
        self.__xunits = ''
        self.__ylabel = ''
        self.__yunits = ''

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update(self.dims)
        return k

    @property
    def dims_master(self):
        return self.__dims_master

    @dims_master.setter
    def dims_master(self, dimensions):
        return self._dims_master_set(dimensions)

    def __dims_master_alert(self):
        if not isinstance(self.__dims_master, Dimensions):
            return 'dims are unlocked'
        a = 'dims are locked to master ' + nmu.quotes(self.__dims_master.name)
        return a

    def _dims_master_set(self, dimensions, quiet=nmp.QUIET):
        if dimensions is None:
            pass  # ok
        elif not isinstance(dimensions, Dimensions):
            raise TypeError(nmu.type_error(dimensions, 'Dimensions'))
        old = self.__dims_master
        if old == dimensions:
            return True
        self.__dims_master = dimensions
        self._modified()
        if old:
            oldname = old.name
        else:
            oldname = 'None'
        if dimensions:
            newname = dimensions.name
        else:
            newname = 'None'
        h = nmu.history_change('dims_master', oldname, newname)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def dims(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.dims
        d = {'xdata': self.__xdata}
        d.update({'xstart': self.__xstart, 'xdelta': self.__xdelta})
        d.update({'xlabel': self.__xlabel, 'xunits': self.__xunits})
        d.update({'ylabel': self.__ylabel, 'yunits': self.__yunits})
        return d

    @dims.setter
    def dims(self, dims):
        return self._dims_set(dims)

    def _dims_set(self, dims, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(dims, dict):
            e = nmu.type_error(dims, 'dimensions dictionary')
            raise TypeError(e)
        keys = dims.keys()
        for k in keys:
            if k not in DIMS_LIST:
                raise KeyError('unknown dimensions key: ' + k)
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        if 'xdata' in keys:
            self._xdata_set(dims['xdata'], quiet=quiet)
        if 'xstart' in keys:
            self._xstart_set(dims['xstart'], quiet=quiet)
        if 'xdelta' in keys:
            self._xdelta_set(dims['xdelta'], quiet=quiet)
        if 'xlabel' in keys:
            self._xlabel_set(dims['xlabel'], quiet=quiet)
        if 'xunits' in keys:
            self._xunits_set(dims['xunits'], quiet=quiet)
        if 'ylabel' in keys:
            self._ylabel_set(dims['ylabel'], quiet=quiet)
        if 'yunits' in keys:
            self._yunits_set(dims['yunits'], quiet=quiet)
        return True

    def _note_new(self, note, quiet=True):
        if isinstance(self._note_container, NoteContainer):
            return self._note_container.new(note=note, quiet=quiet)
        return None

    @property
    def xdata(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.xdata
        return self.__xdata

    @xdata.setter
    def xdata(self, xdata):
        return self._xdata_set(xdata)

    def _xdata_set(self, xdata, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if xdata is None:
            pass  # ok
        elif xdata.__class__.__name__ != 'Data':  # cannot import Data class
            raise TypeError(nmu.type_error(xdata, 'Data'))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__xdata
        if xdata == old:
            return True
        self.__xdata = xdata
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

    def _xdata_alert(self):
        if self.xdata.__class__.__name__ == 'Data':
            xn = nmu.quotes(self.xdata.name)
            return ('x-dims are superceded by xdata ' + xn + '.' + '\n' +
                    'do you want to continue?')
        return ''

    @property
    def xstart(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.xstart
        return self.__xstart

    @xstart.setter
    def xstart(self, xstart):
        return self._xstart_set(xstart)

    def _xstart_set(self, xstart, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xstart, float) and not isinstance(xstart, int):
            raise TypeError(nmu.type_error(xstart, 'number'))
        if not nmu.number_ok(xstart):
            raise ValueError('bad xstart: ' + str(xstart))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__xstart
        if xstart == old:
            return True
        self.__xstart = xstart
        self._modified()
        h = nmu.history_change('xstart', old, xstart)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xdelta(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.xdelta
        return self.__xdelta

    @xdelta.setter
    def xdelta(self, xdelta):
        return self._xdelta_set(xdelta)

    def _xdelta_set(self, xdelta, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xdelta, float) and not isinstance(xdelta, int):
            raise TypeError(nmu.type_error(xdelta, 'number'))
        if not nmu.number_ok(xdelta):
            raise ValueError('bad xdelta: ' + str(xdelta))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__xdelta
        if xdelta == old:
            return True
        self.__xdelta = xdelta
        self._modified()
        h = nmu.history_change('xdelta', old, xdelta)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xlabel(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.xlabel
        return self.__xlabel

    @xlabel.setter
    def xlabel(self, xlabel):
        return self._xlabel_set(xlabel)

    def _xlabel_set(self, xlabel, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xlabel, str):
            raise TypeError(nmu.type_error(xlabel, 'string'))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__xlabel
        if xlabel == old:
            return True
        self.__xlabel = xlabel
        self._modified()
        h = nmu.history_change('xlabel', old, xlabel)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def xunits(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.xunits
        return self.__xunits

    @xunits.setter
    def xunits(self, xunits):
        return self._xunits_set(xunits)

    def _xunits_set(self, xunits, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if self.xdata and alert:
            if nmu.input_yesno(self._xdata_alert(), tp=self._tp) == 'n':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return False
        if not isinstance(xunits, str):
            raise TypeError(nmu.type_error(xunits, 'string'))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__xunits
        if xunits == old:
            return True
        self.__xunits = xunits
        self._modified()
        h = nmu.history_change('xunits', old, xunits)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def ylabel(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.ylabel
        return self.__ylabel

    @ylabel.setter
    def ylabel(self, ylabel):
        return self._ylabel_set(ylabel)

    def _ylabel_set(self, ylabel, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(ylabel, str):
            raise TypeError(nmu.type_error(ylabel, 'string'))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__ylabel
        if ylabel == old:
            return True
        self.__ylabel = ylabel
        self._modified()
        h = nmu.history_change('ylabel', old, ylabel)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def yunits(self):
        if isinstance(self.__dims_master, Dimensions):
            return self.__dims_master.yunits
        return self.__yunits

    @yunits.setter
    def yunits(self, yunits):
        return self._yunits_set(yunits)

    def _yunits_set(self, yunits, alert=True, quiet=nmp.QUIET):
        alert = nmu.check_bool(alert, True)
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(yunits, str):
            raise TypeError(nmu.type_error(yunits, 'string'))
        if isinstance(self.__dims_master, Dimensions):
            if alert:
                self._alert(self.__dims_master_alert, tp=self._tp)
            return False
        old = self.__yunits
        if yunits == old:
            return True
        self.__yunits = yunits
        self._modified()
        h = nmu.history_change('yunits', old, yunits)
        self._note_new(h)
        self._history(h, tp=self._tp, quiet=quiet)
        return True
