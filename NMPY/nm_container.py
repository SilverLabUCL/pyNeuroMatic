#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import copy
import datetime
import math

import nm_preferences as nmp
import nm_utilities as nmu


class NMObject(object):
    """
    NM objects to be stored in a 'Container' list (see below).

    Known children:
        Channel, Data, DataSeries, Dimension, EpochSet, Folder, Note, Project

    Attributes:
        parent (NMObject):
        name (str):
        fxns ({}):
        rename (bool)
        date (str):
        modified (str):
    """

    def __init__(self, parent, name, fxns={}, rename=True):
        self._parent = parent
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not name or not self.name_ok(name):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        self.__name = name
        if isinstance(fxns, dict):
            self._fxns = fxns
        else:
            self._fxns = {}
        self._rename = nmu.check_bool(rename, True)
        self.__date = str(datetime.datetime.now())
        self.__modified = self.__date
        self._param_list = ['name', 'rename', 'date', 'modified']
        # param_list should match those listed in parameters()
        # see param_test()

    def name_ok(self, name, ok=[]):
        if not nmu.name_ok(name):
            return False
        if not isinstance(ok, list):
            ok = [ok]
        bad = [n.lower() for n in self._bad_names]
        for n in ok:
            if n.lower() in bad:
                bad.remove(n.lower())
        return name.lower() not in bad

    @property
    def _bad_names(self):  # names not allowed
        return ['select', 'default', 'all']

    @property
    def _cname(self):  # class name
        return self.__class__.__name__

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        return self._name_set(name)

    def _name_set(self, name, quiet=nmp.QUIET):
        if not self._rename:
            raise RuntimeError(nmu.quotes(self.__name) + ' cannot be renamed')
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not name or not self.name_ok(name):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        old = self.__name
        self.__name = name
        h = nmu.history_change('name', old, self.__name)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def parameters(self):  # child class should override
        # and add class parameters
        p = {'name': self.__name}
        p.update({'rename': self._rename})
        p.update({'date': self.__date})
        p.update({'modified': self.__modified})
        return p

    def _param_test(self, quiet=nmp.QUIET):
        pkeys = self.parameters.keys()
        for k in pkeys:
            if k not in self._param_list:
                e = 'missing parameter ' + nmu.quotes(k)
                self._error(e, tp=self._tp, quiet=quiet)
                return False
        for k in self._param_list:
            if k not in pkeys:
                e = 'missing parameter ' + nmu.quotes(k)
                self._error(e, tp=self._tp, quiet=quiet)
                return False
        return True

    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', 'data', etc.
        return {'nmobject': self.__name}

    @property
    def content_tree(self):
        if self._parent and isinstance(self._parent, NMObject):
            k = {}
            k.update(self._parent.content_tree)
            k.update(self.content)
            return k
        return self.content

    def _modified(self):
        self.__modified = str(datetime.datetime.now())
        if self._parent and isinstance(self._parent, NMObject):  # tree-path
            self._parent._modified()

    @property
    def _tp(self):  # used with history()
        return self.treepath(for_history=True)

    def treepath(self, for_history=False):
        for_history = nmu.check_bool(for_history, False)
        if for_history:  # create tree path for history
            skip = nmp.HISTORY_TREEPATH_SKIP
        else:
            skip = []
        plist = self.treepath_list(skip=skip)
        if len(plist) > 0:
            tp = '.'.join(plist)
        else:
            tp = self.__name
        return tp

    def treepath_list(self, names=True, skip=[]):
        names = nmu.check_bool(names, True)
        if not isinstance(skip, list):
            skip = []
        if self._cname in skip:
            return []
        p = self._parent
        if p and isinstance(p, NMObject) and p.__class__.__name__ not in skip:
            t = p.treepath_list(names=names, skip=skip)
            if names:
                t.append(self.__name)
            else:
                t.append(self)
            return t
        if names:
            return [self.__name]
        return [self]

    def _equal(self, nmobj, alert=False):
        if nmobj.__class__.__name__ != self._cname:
            if alert:
                a = ('unequal object types: ' + self._cname + ' vs ' +
                     nmobj.__class__.__name__)
                self._alert(a, tp=self._tp)
            return False
        if self == nmobj:
            return True
        # if self._parent != nmobj._parent:  # problematic for containers
        #    a = ('unequal parents: ' +
        #         str(self._parent) + ' vs ' + str(nmobj._parent))
        #    self._alert(a, tp=self._tp)
        #    return False
        # if self._fxns != nmobj._fxns:  # ignore?
        #    a = ('unequal fxns: ' + str(self._fxns) + ' vs ' +
        #         str(nmobj._fxns))
        #    self._alert(a, tp=self._tp)
        #    return False
        sp = self.parameters
        op = nmobj.parameters
        if op.keys() != sp.keys():
            return False
        for k in sp.keys():
            if k == 'date':
                continue  # ignore, will be different
            if k == 'modified':
                continue  # ignore, will be different
            if op[k] != sp[k]:
                if nmp.NAN_EQ_NAN:
                    op_nan = isinstance(op[k], float) and math.isnan(op[k])
                    sp_nan = isinstance(sp[k], float) and math.isnan(sp[k])
                    if op_nan and sp_nan:
                        continue  # ok (nan=nan)
                if alert:
                    a = ('unequal ' + nmu.quotes(k) + ': ' + str(sp[k]) +
                         ' vs ' + str(op[k]))
                    self._alert(a, tp=self._tp)
                return False
        return True

    def copy(self):
        return NMObject(self._parent, self.__name, fxns=self._fxns,
                        rename=self._rename)

    def save(self, path='', quiet=nmp.QUIET):
        self._alert('under construction')
        return False

    @property
    def _quiet(self):
        if 'quiet' in self._fxns.keys():
            return self._fxns['quiet']
        return self.__quiet

    @property
    def _alert(self):
        if 'alert' in self._fxns.keys():
            return self._fxns['alert']
        return self.__alert

    @property
    def _error(self):
        if 'error' in self._fxns.keys():
            return self._fxns['error']
        return self.__error

    @property
    def _history(self):
        if 'history' in self._fxns.keys():
            return self._fxns['history']
        return self.__history

    def __quiet(self, quiet=False):
        if nmp.QUIET:  # this quiet overrides
            return True
        return quiet

    def __alert(self, message, tp='', quiet=False, frame=2):
        quiet = self._quiet(quiet)
        return nmu.history(message, title='ALERT', tp=tp, frame=frame,
                           red=True, quiet=quiet)

    def __error(self, message, tp='', quiet=False, frame=2):
        quiet = self._quiet(quiet)
        return nmu.history(message, title='ERROR', tp=tp, frame=frame,
                           red=True, quiet=quiet)

    def __history(self, message, tp='', quiet=False, frame=2):
        quiet = self._quiet(quiet)
        return nmu.history(message, tp=tp, frame=frame, quiet=quiet)


class Container(NMObject):
    """
    class collections.abc.MutableSequence
    class collections.abc.MutableMapping (implement this?)

    A list container for NMObject items (above), one of which is assigned to
    'select'.

    Each NMObject item must have a unique name. The name can start with the
    same prefix (e.g. "NMExp") but this is optional. Use name_next() to
    create unique names in a sequence (e.g. "NMExp0", "NMExp1", etc.).
    One NMObject is selected/activated at a given time. This NMObject can be
    accessed via 'select' property.

    Known Children:
        FolderContainer, DataContainer, NoteContainer, DataSeriesContainer,
        ChannelContainer, EpochSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_next(),
        name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject
    """

    def __init__(self, parent, name, fxns={}, type_='NMObject',
                 prefix='NMObject', rename=True, duplicate=True):
        super().__init__(parent, name, fxns=fxns, rename=rename)
        if not isinstance(type_, str):
            raise TypeError(nmu.type_error(type_, 'string'))
        if not type_ or not nmu.name_ok(type_):
            raise ValueError('bad type_: ' + nmu.quotes(type_))
        self._type = type_
        if prefix is None:
            prefix = ''
        elif not isinstance(prefix, str):
            raise TypeError(nmu.type_error(prefix, 'string'))
        elif not self.name_ok(prefix):
            raise ValueError('bad prefix:  ' + nmu.quotes(prefix))
        self.__prefix = prefix
        self._duplicate = duplicate
        self.__thecontainer = []  # container of NMObject items
        self.__select = None  # selected NMObject
        self._param_list += ['type', 'prefix', 'duplicate', 'select']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'type': self._type})
        k.update({'prefix': self.__prefix})
        k.update({'duplicate': self._duplicate})
        if self.__select:
            k.update({'select': self.__select.name})
            # need name for equal() to work
        else:
            k.update({'select': ''})
        return k

    # override
    @property
    def content(self):  # child class should override
        # and change 'nmobjects' to 'folders', etc
        return {'nmobjects': self.names}

    # override
    def _equal(self, container, alert=False):
        if not super()._equal(container, alert=alert):
            return False
        if container.count != self.count:
            if alert:
                a = ('unequal container count: ' + str(self.count) + ' vs ' +
                     str(container.count))
                self._alert(a, tp=self._tp)
            return False
        for i, s in enumerate(self.__thecontainer):
            o = container._Container__getitem(index=i, quiet=True)  # mangled
            if not s._equal(o, alert=alert):
                return False
        return True

    def copy(self, container=None):
        if container is None:
            c = Container(self._parent, self.name, self._fxns,
                          type_=self._type, prefix=self.__prefix,
                          rename=self._rename, duplicate=self._duplicate)
        elif isinstance(container, Container):
            c = container
        else:
            raise TypeError(nmu.type_error(container, 'Container'))
        for o in self.__thecontainer:
            if o:
                c._Container__thecontainer.append(o.copy())
        if self.__select and self.__select.name:
            name = self.__select.name
            if c._Container__exists(name):
                c._Container__select = c._Container__getitem(name)
        return c

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        return self._prefix_set(prefix)

    def _prefix_set(self, prefix, quiet=nmp.QUIET):
        if not self._rename:
            raise RuntimeError(self._type + ' items cannot be renamed')
        if not isinstance(prefix, str):
            raise TypeError(nmu.type_error(prefix, 'string'))
        if not self.name_ok(prefix):
            raise ValueError('bad prefix: ' + nmu.quotes(prefix))
        old = self.__prefix
        self.__prefix = prefix
        self._modified()
        h = nmu.history_change('prefix', old, self.__prefix)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def names(self):
        """Get list of names of NMObject items in container"""
        return [o.name for o in self.__thecontainer]

    @property
    def count(self):
        """Number of NMObject items stored in container"""
        return len(self.__thecontainer)

    def index(self, name):
        """Find item # of NMObject in container"""
        if not isinstance(name, str):
            return -1
        if not name or not nmu.name_ok(name):
            return -1
        if name.lower() == 'select':
            if self.__select and self.__select.name:
                name = self.__select.name
            else:
                return -1
        for i, o in enumerate(self.__thecontainer):
            if name.lower() == o.name.lower():
                return i
        return -1

    __index = index

    def exists(self, name):
        """Check if NMObject exists within container"""
        return self.__index(name) >= 0

    __exists = exists

    def getitem(self, name='', index=-1, quiet=nmp.QUIET):
        """Get NMObject from Container"""
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not isinstance(index, int):
            raise TypeError(nmu.type_error(index, 'integer'))
        if index < 0:
            pass  # ok, do not use index
        elif index >= 0 and index < len(self.__thecontainer):
            return self.__thecontainer[index]
        else:
            e = 'bad index:  expected 0-' + str(len(self.__thecontainer)-1)
            raise IndexError(e + ', got  ' + str(index))
        if not name:
            return None
        if name.lower() == 'select':
            return self.__select
        for o in self.__thecontainer:
            if name.lower() == o.name.lower():
                return o
        e = self._exists_error(name)
        self._error(e, tp=self._tp, quiet=quiet)
        return None

    __getitem = getitem

    def getitems(self, names=[], indexes=[], quiet=nmp.QUIET):
        """Get a list of NMObjects from Container"""
        if not isinstance(names, list):
            names = [names]
        if not isinstance(indexes, list):
            indexes = [indexes]
        olist = []
        if len(names) == 1 and names[0].lower() == 'all':
            get_all = True
            ok = 'all'
        else:
            get_all = False
            ok = 'select'
        for name in names:
            if not isinstance(name, str):
                raise TypeError(nmu.type_error(name, 'string'))
            if not self.name_ok(name, ok=ok):
                raise ValueError('bad name:  ' + nmu.quotes(name))
            if name.lower() == 'select':
                if self.__select not in olist:
                    olist.append(self.__select)
            else:
                for o in self.__thecontainer:
                    if get_all:
                        olist.append(o)
                    elif name.lower() == o.name.lower() and o not in olist:
                        olist.append(o)
        if get_all:
            return olist
        for index in indexes:
            if not isinstance(index, int):
                raise TypeError(nmu.type_error(index, 'integer'))
            if index >= 0 and index < len(self.__thecontainer):
                o = self.__thecontainer[index]
                if o not in olist:
                    olist.append(o)
            else:
                raise IndexError('index out of range:  ' + str(index))
        return olist

    def _exists_error(self, name):
        e = 'failed to find ' + nmu.quotes(name)
        e += '\n' + 'acceptable names: ' + str(self.names)
        return e

    @property
    def select(self):
        return self.__select

    @select.setter
    def select(self, name):
        return self._select_set(name)

    def _select_set(self, name, failure_alert=True, quiet=nmp.QUIET):
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name:
            return None
        # if self.__select and name.lower() == self.__select.name.lower():
            # return self.__select  # already selected
        if self.__exists(name):
            old = self.__select
            o = self.__getitem(name)
            self.__select = o
            self._modified()
            h = Container.__select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
            return o
        if failure_alert:
            q = ('failed to find ' + nmu.quotes(name) + '.' + '\n' +
                 'do you want to create a new ' + self._type +
                 ' named ' + nmu.quotes(name) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if yn == 'y':
                return self.new(name=name, select=True)
            self._history('cancel', tp=self._tp, quiet=quiet)
            return None
        e = self._exists_error(name)
        self._error(e, tp=self._tp, quiet=quiet)
        return None

    @staticmethod
    def __select_history(old, new):
        if new:
            n = new.name
        else:
            n = 'None'
        if old:
            return nmu.history_change('select', old.name, n)
        return 'selected ' + nmu.quotes(n)

    def new(self, name='default', nmobj=None, select=True, quiet=nmp.QUIET):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        select = nmu.check_bool(select, True)
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='default'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name or name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        if self.__exists(name):
            e = self._type + ' ' + nmu.quotes(name) + ' already exists'
            raise RuntimeError(e)
        if not nmobj:
            o = NMObject(self._parent, name, self._fxns, rename=False)
            # rename = False, enforce using Container.rename()
        elif isinstance(nmobj, NMObject):  # child 'new' should pass nmobj
            if nmobj.__class__.__name__ == self._type:
                o = nmobj
                # mangled...
                o._NMObject__name = name  # this.name overrides o.name
                o._parent = self._parent  # reset parent reference
                o._rename = False  # enforce using Container.rename()
            else:
                raise TypeError(nmu.type_error(nmobj, self._type))
        else:
            raise TypeError(nmu.type_error(nmobj, 'NMObject'))
        if not o:
            return None
        self.__thecontainer.append(o)
        if select or not self.__select:
            old = self.__select
            self.__select = o
            if old:
                h = 'created ' + nmu.quotes(name)
                self._history(h, tp=self._tp, quiet=quiet)
                h = Container.__select_history(old, self.__select)
                self._history(h, tp=self._tp, quiet=quiet)
            else:
                h = 'created/selected ' + nmu.quotes(name)
                self._history(h, tp=self._tp, quiet=quiet)
        else:
            h = 'created ' + nmu.quotes(name)
            self._history(h, tp=self._tp, quiet=quiet)
        self._modified()
        return o

    def rename(self, name, newname, quiet=nmp.QUIET):
        if not self._rename:
            raise RuntimeError(self._type + ' items cannot be renamed')
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name:
            return ''
        if not self.__exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return ''
        o = self.__getitem(name, quiet=quiet)
        if not o:
            return ''
        if not isinstance(newname, str):
            raise TypeError(nmu.type_error(newname, 'string'))
        if not newname or not self.name_ok(newname, ok='default'):
            raise ValueError('bad newname: ' + nmu.quotes(newname))
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.__exists(newname):
            e = self._type + ' ' + nmu.quotes(newname) + ' already exists'
            raise RuntimeError(e)
        old = nmu.quotes(o.name)
        # o.name = newname  # double history
        o._NMObject__name = newname  # mangled
        self._modified()
        new = nmu.quotes(o.name)
        h = 'renamed ' + old + ' to ' + new
        self._history(h, tp=self._tp, quiet=quiet)
        return newname

    def duplicate(self, name, newname, select=True, quiet=nmp.QUIET):
        """
        Copy NMObject.

        Args:
            name: name of NMObject to copy
            newname: name of new NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not self._duplicate:
            raise RuntimeError(self._type + ' items cannot be duplicated')
        select = nmu.check_bool(select, True)
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name:
            return None
        if not self.__exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return None
        o = self.__getitem(name, quiet=quiet)
        if not o:
            return None
        if not isinstance(newname, str):
            raise TypeError(nmu.type_error(newname, 'string'))
        if not newname or not self.name_ok(newname, ok='default'):
            raise ValueError('bad newname: ' + nmu.quotes(newname))
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.__exists(newname):
            e = self._type + ' ' + nmu.quotes(newname) + ' already exists'
            raise RuntimeError(e)
        # c = copy.deepcopy(o) NOT WORKING
        # TypeError: can't pickle Context objects
        # c = copy.copy(o)
        # c = NMObject(self._parent, newname, self._fxns, rename=self._rename)
        c = o.copy()
        if not c:
            return None
        c._NMObject__name = newname
        # c._parent = self._parent  # reset parent reference
        self.__thecontainer.append(c)
        old = nmu.quotes(o.name)
        new = nmu.quotes(c.name)
        h = 'copied ' + old + ' to ' + new
        self._history(h, tp=self._tp, quiet=quiet)
        if select or not self.__select:
            old = self.__select
            self.__select = c
            h = Container.__select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
        self._modified()
        return c

    def kill(self, name='', all_=False, confirm=True, quiet=nmp.QUIET):
        """
        Kill NMObject.

        Args:
            name: name of NMObject to kill

        Returns:
            True for success, False otherwise
        """
        all_ = nmu.check_bool(all_, False)
        confirm = nmu.check_bool(confirm, True)
        klist = []
        if all_:
            n = ', '.join(self.names)
            if confirm and not self._quiet(quiet):
                q = ('are you sure you want to kill all ' + self._type +
                     ' items?' + '\n' + 'This will kill ' + n)
                yn = nmu.input_yesno(q, tp=self._tp)
                if not yn == 'y':
                    self._history('cancel', tp=self._tp, quiet=quiet)
                    return []
            klist = [o for o in self.__thecontainer]
            self.__thecontainer.clear()
            self.__select = None
            self._modified()
            self._history('killed ' + n, tp=self._tp, quiet=quiet)
            return klist
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name:
            return []
        if not self.__exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return []
        o = self.__getitem(name, quiet=quiet)
        if not o:
            return []
        if confirm and not self._quiet(quiet):
            q = 'are you sure you want to kill ' + nmu.quotes(o.name) + '?'
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn == 'y':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return []
        select_next = o is self.__select  # killing select, so need new one
        if select_next:
            i = self.__index(o.name)
        klist.append(o)
        self.__thecontainer.remove(o)
        h = 'killed ' + nmu.quotes(o.name)
        self._history(h, tp=self._tp, quiet=quiet)
        items = len(self.__thecontainer)
        if select_next and items > 0:
            i = max(i, 0)
            i = min(i, items - 1)
            o = self.__thecontainer[i]
            old = self.__select
            self.__select = o
            h = Container.__select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
        self._modified()
        return klist

    def name_next(self, first=0, quiet=nmp.QUIET):
        """Get next default NMObject name based on prefix and sequence #."""
        if not isinstance(first, int):
            raise TypeError(nmu.type_error(first, 'integer'))
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not self.__prefix or first < 0:
            raise RuntimeError('cannot generate default names')
        i = self.name_next_seq(prefix=self.__prefix, first=first, quiet=quiet)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(self, prefix='default', first=0, quiet=nmp.QUIET):
        """Get next seq num of default NMObject name based on prefix."""
        if not isinstance(first, int):
            raise TypeError(nmu.type_error(first, 'integer'))
        quiet = nmu.check_bool(quiet, nmp.QUIET)
        if not isinstance(prefix, str):
            raise TypeError(nmu.type_error(prefix, 'string'))
        if not self.name_ok(prefix, ok='default'):
            raise ValueError('bad prefix: ' + nmu.quotes(prefix))
        if prefix.lower() == 'default':
            prefix = self.__prefix
        if not prefix or first < 0:
            raise RuntimeError('cannot generate default names')
        elist = []
        for o in self.__thecontainer:
            name = o.name.lower()
            istr = name.replace(prefix.lower(), '')
            if str.isdigit(istr):
                elist.append(int(istr))
        if len(elist) == 0:
            return 0
        i = max(elist)
        return i + 1
