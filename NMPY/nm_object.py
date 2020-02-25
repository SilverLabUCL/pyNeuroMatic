#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import copy
import datetime
import math
import types

import nm_preferences as nmp
import nm_utilities as nmu


class NMObject(object):
    """
    NM objects to be stored in a Container class (see below).

    Known children:
        Channel, Data, DataSeries, Dimension, DataSet, Folder, Note, Project

    Attributes:
        parent (NMObject or any object):
        name (str):
        date (str):
        modified (str):
    """

    def __init__(self, parent, name):
        self._parent = parent
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not name or not self.name_ok(name):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        self.__name = name
        self.__rename_fr = self._name_set
        self.__date = str(datetime.datetime.now())
        self.__modified = self.__date
        self._param_list = ['name', 'date', 'modified']
        # param_list should match keys in parameters()
        # see param_test()

    @property
    def parameters(self):  # child class should override
        # and add class parameters
        p = {'name': self.__name}
        p.update({'date': self.__date})
        p.update({'modified': self.__modified})
        return p

    def _param_test(self, quiet=False):
        pkeys = self.parameters.keys()
        for k in pkeys:
            if k not in self._param_list:
                e = 'missing param_list item ' + nmu.quotes(k)
                self._error(e, tp=self._tp, quiet=quiet)
                return False
        for k in self._param_list:
            if k not in pkeys:
                e = 'missing parameters key ' + nmu.quotes(k)
                self._error(e, tp=self._tp, quiet=quiet)
                return False
        return True

    @property
    def _content_name(self):
        return self.__class__.__name__.lower()

    @property
    def content(self):
        return {self._content_name: self.__name}

    @property
    def content_tree(self):
        if self._parent and isinstance(self._parent, NMObject):
            k = {}
            k.update(self._parent.content_tree)
            k.update(self.content)
            return k
        return self.content

    @property
    def _tp(self):  # use with history()
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
        cname = self.__class__.__name__
        if cname in skip:
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

    @property
    def _manager(self):
        p = self._parent
        if p is None:
            return None
        if p.__class__.__name__ == 'Manager':
            return p
        if isinstance(p, NMObject):
            return p._manager
        return None

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
        return ['select', 'default', 'all']  # use lower-case

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, newname):
        # calls name_set() or NMObjectContainer.rename()
        return self.__rename_fr(self.__name, newname)

    def _name_set(self, name_notused, newname, quiet=nmp.QUIET):
        # name_notused, to be consistent with
        # NMObjectContainer.rename(name, newname)
        if not isinstance(newname, str):
            raise TypeError(nmu.type_error(newname, 'string'))
        if not newname or not self.name_ok(newname):
            raise ValueError('bad name:  ' + nmu.quotes(newname))
        old = self.__name
        self.__name = newname
        self._modified()
        h = nmu.history_change('name', old, self.__name)
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def _rename_fr_set(self, rename_fr):
        if not isinstance(rename_fr, types.MethodType):
            raise TypeError(nmu.type_error(rename_fr, 'MethodType'))
        self.__rename_fr = rename_fr
        return True

    def _equal(self, nmobject, alert=False):
        if nmobject.__class__.__name__ != self.__class__.__name__:
            if alert:
                a = ('unequal object types: ' + self.__class__.__name__ +
                     ' vs ' + nmobject.__class__.__name__)
                self._alert(a, tp=self._tp)
            return False
        if self == nmobject:
            return True
        # if nmobject._parent != self._parent:
            # problematic for copying containers
            # compare parent name?
            # a = ('unequal parents: ' + str(self._parent) + ' vs ' +
            #      str(nmobject._parent))
            # self._alert(a, tp=self._tp)
            # return False
        # if nmobject._NMObject__rename_fr != self.__rename_fr:
        #     different, unless in same container
        #     a = ('unequal rename() refs: ' + str(self.__rename_fr) +
        #     ' vs ' + str(nmobject._NMObject__rename_fr))
        #     self._alert(a, tp=self._tp)
        #     return False
        sp = self.parameters
        op = nmobject.parameters
        if op.keys() != sp.keys():
            return False
        for k in sp.keys():
            if k == 'date' or k == 'modified':
                continue  # ignore, will be different
            if op[k] != sp[k]:
                if nmp.NAN_EQ_NAN:
                    op_nan = isinstance(op[k], float) and math.isnan(op[k])
                    sp_nan = isinstance(sp[k], float) and math.isnan(sp[k])
                    if op_nan and sp_nan:
                        continue  # ok (nan=nan)
                if alert:
                    a = ('unequal ' + nmu.quotes(k) + ': ' +
                         nmu.quotes(sp[k]) + ' vs ' + nmu.quotes(op[k]))
                    self._alert(a, tp=self._tp)
                return False
        return True

    def copy(self):
        return NMObject(self._parent, self.__name)

    def save(self, path='', quiet=nmp.QUIET):
        # TODO
        raise RuntimeError('save under construction')

    def _modified(self):
        self.__modified = str(datetime.datetime.now())
        if self._parent and isinstance(self._parent, NMObject):  # tree-path
            self._parent._modified()

    def _alert(self, message, tp='', quiet=False, frame=2):
        return nmu.history(message, title='ALERT', tp=tp, frame=frame,
                           red=True, quiet=self._quiet(quiet))

    def _error(self, message, tp='', quiet=False, frame=2):
        return nmu.history(message, title='ERROR', tp=tp, frame=frame,
                           red=True, quiet=self._quiet(quiet))

    def _history(self, message, tp='', quiet=False, frame=2):
        return nmu.history(message, tp=tp, frame=frame,
                           red=False, quiet=self._quiet(quiet))

    def _quiet(self, quiet=False):
        m = self._manager
        if m.__class__.__name__ == 'Manager':
            return m._quiet(quiet=quiet)
        if nmp.QUIET:  # this quiet overrides
            return True
        return nmu.check_bool(quiet, False)


class NMObjectContainer(NMObject):
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
        ChannelContainer, DataSetContainer

    Attributes:
        prefix (str): For creating NMObject name via name_next(),
        name = prefix + seq #
        __objects : list
            List container of NMObject items
        __object_select : NMObject
            The selected NMObject
    """

    def __init__(self, parent, name, type_='NMObject',
                 prefix='NMObject', rename=True, **copy):
        super().__init__(parent, name)
        if not isinstance(type_, str):
            raise TypeError(nmu.type_error(type_, 'string'))
        if not type_ or not nmu.name_ok(type_):
            raise ValueError('bad type_: ' + nmu.quotes(type_))
        self.__type = type_
        if prefix is None:
            prefix = ''
        elif not isinstance(prefix, str):
            raise TypeError(nmu.type_error(prefix, 'string'))
        elif not self.name_ok(prefix):
            raise ValueError('bad prefix:  ' + nmu.quotes(prefix))
        self.__prefix = prefix
        self.__rename = nmu.check_bool(rename, True)
        self.__thecontainer = []  # container of NMObject items
        self.__select = None  # selected NMObject
        for k, v in copy.items():  # see copy() and thecontainer_copy()
            if k.lower() == 'c_type' and isinstance(v, str):
                self.__type = v
            if k.lower() == 'c_prefix' and isinstance(v, str):
                self.__prefix = v
            if k.lower() == 'c_rename' and isinstance(v, bool):
                self.__rename = v
            if k.lower() == 'c_thecontainer' and isinstance(v, dict):
                if 'thecontainer' in v.keys():
                    if isinstance(v['thecontainer'], list):
                        self.__thecontainer = v['thecontainer']
                        self._thecontainer_rename_fr_set()
                        if 'select' in v.keys():
                            if isinstance(v['select'], NMObject):
                                self.__select = v['select']
        self._param_list += ['type', 'prefix', 'rename', 'select']

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'type': self.__type})
        k.update({'prefix': self.__prefix})
        k.update({'rename': self.__rename})
        if self.__select:
            k.update({'select': self.__select.name})
            # need select's name for equal() to work
        else:
            k.update({'select': 'None'})
        return k

    # override, no super
    @property
    def _content_name(self):
        n = self.__class__.__name__.lower()  # e.g. 'foldercontainer'
        n = n.replace('container', 's')  # e.g. 'folders'
        n = n.replace('ss', 's')
        return n

    # override, no super
    @property
    def content(self):
        return {self._content_name: self.names}

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
            o = container.getitem(index=i, quiet=True)
            if not s._equal(o, alert=alert):
                return False
        return True

    # override, no super
    def copy(self):
        return NMObjectContainer(self._parent, self.name, type_=self.__type,
                                 prefix=self.prefix, rename=self.__rename,
                                 c_thecontainer=self._thecontainer_copy())

    def _thecontainer_copy(self):
        c = []
        if self.__select and self.__select.name:
            select_name = self.__select.name
        else:
            select_name = ''
        select = None
        for o in self.__thecontainer:
            if o:
                oc = o.copy()
                c.append(oc)
                if oc.name.lower() == select_name.lower():
                    select = oc
        return {'thecontainer': c, 'select': select}

    def _thecontainer_rename_fr_set(self):
        for o in self.__thecontainer:
            o._rename_fr_set(self.rename)

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        return self._prefix_set(prefix)

    def _prefix_set(self, prefix, quiet=nmp.QUIET):
        if not self.__rename:
            raise RuntimeError(self.__type + ' items cannot be renamed')
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

    def exists(self, name):
        """Check if NMObject exists within container"""
        return self.index(name) >= 0

    def getitem(self, name='', index=None, quiet=nmp.QUIET):
        """Get NMObject from Container"""
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if index is None:
            pass  # ok
        elif not isinstance(index, int):
            raise TypeError(nmu.type_error(index, 'integer'))
        elif index < 0 and index >= -1 * len(self.__thecontainer):
            return self.__thecontainer[index]
        elif index >= 0 and index < len(self.__thecontainer):
            return self.__thecontainer[index]
        else:
            raise IndexError('bad index: ' + str(index))
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

    def getitems(self, names=[], indexes=[]):
        """Get a list of NMObjects from Container"""
        if isinstance(names, list) or isinstance(names, tuple):
            pass  # ok
        elif isinstance(names, str):
            names = [names]
        else:
            raise TypeError(nmu.type_error(names, 'string'))
        if isinstance(indexes, list) or isinstance(indexes, tuple):
            pass  # ok
        elif isinstance(indexes, int):
            indexes = [indexes]
        else:
            raise TypeError(nmu.type_error(indexes, 'integer'))
        olist = []
        all_ = False
        for n in names:
            if not isinstance(n, str):
                raise TypeError(nmu.type_error(n, 'string'))
            if n.lower() == 'all':
                all_ = True
            elif not self.name_ok(n, ok='select'):
                raise ValueError('bad name:  ' + nmu.quotes(n))
            if n.lower() == 'select':
                if self.__select not in olist:
                    olist.append(self.__select)
            else:
                for o in self.__thecontainer:
                    if all_ and o not in olist:
                        olist.append(o)
                    elif n.lower() == o.name.lower() and o not in olist:
                        olist.append(o)
        if all_:
            return olist
        for i in indexes:
            if not isinstance(i, int):
                raise TypeError(nmu.type_error(i, 'integer'))
            o = None
            if i < 0 and i >= -1 * len(self.__thecontainer):
                o = self.__thecontainer[i]
            elif i >= 0 and i < len(self.__thecontainer):
                o = self.__thecontainer[i]
            else:
                raise IndexError('index out of range:  ' + str(i))
            if o and o not in olist:
                olist.append(o)
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
        if self.exists(name):
            old = self.__select
            o = self.getitem(name)
            self.__select = o
            self._modified()
            h = NMObjectContainer.__select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
            return o
        if failure_alert:
            q = ('failed to find ' + nmu.quotes(name) + '.' + '\n' +
                 'do you want to create a new ' + self.__type +
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

    def new(self, name='default', nmobject=None, select=True, quiet=nmp.QUIET):
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
        if not name or not self.name_ok(name, ok='default'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        if self.exists(name):
            e = self.__type + ' ' + nmu.quotes(name) + ' already exists'
            raise RuntimeError(e)
        if nmobject is None:
            o = NMObject(self._parent, name)
        elif isinstance(nmobject, NMObject):
            # child 'new' should pass nmobject
            if nmobject.__class__.__name__ == self.__type:
                o = nmobject
                o._NMObject__name = name  # rename nmobject
                o._parent = self._parent  # reset parent reference
            else:
                raise TypeError(nmu.type_error(nmobject, self.__type))
        else:
            raise TypeError(nmu.type_error(nmobject, 'NMObject'))
        if not o:
            return None
        self.__thecontainer.append(o)
        self._thecontainer_rename_fr_set()
        if select or not self.__select:
            old = self.__select
            self.__select = o
            if old:
                h = 'created ' + nmu.quotes(name)
                self._history(h, tp=self._tp, quiet=quiet)
                h = NMObjectContainer.__select_history(old, self.__select)
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
        if not self.__rename:
            raise RuntimeError(self.__type + ' items cannot be renamed')
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name:
            return ''
        if not self.exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return ''
        o = self.getitem(name, quiet=quiet)
        if not o:
            return ''
        if not isinstance(newname, str):
            raise TypeError(nmu.type_error(newname, 'string'))
        if not newname or not self.name_ok(newname, ok='default'):
            raise ValueError('bad newname: ' + nmu.quotes(newname))
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.exists(newname):
            e = self.__type + ' ' + nmu.quotes(newname) + ' already exists'
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
        select = nmu.check_bool(select, True)
        if not isinstance(name, str):
            raise TypeError(nmu.type_error(name, 'string'))
        if not self.name_ok(name, ok='select'):
            raise ValueError('bad name:  ' + nmu.quotes(name))
        if not name:
            return None
        if not self.exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return None
        o = self.getitem(name, quiet=quiet)
        if not o:
            return None
        if not isinstance(newname, str):
            raise TypeError(nmu.type_error(newname, 'string'))
        if not newname or not self.name_ok(newname, ok='default'):
            raise ValueError('bad newname: ' + nmu.quotes(newname))
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.exists(newname):
            e = self.__type + ' ' + nmu.quotes(newname) + ' already exists'
            raise RuntimeError(e)
        c = o.copy()
        if not c:
            return None
        c._NMObject__name = newname
        # c._parent = self._parent  # reset parent reference
        self.__thecontainer.append(c)
        self._thecontainer_rename_fr_set()
        old = nmu.quotes(o.name)
        new = nmu.quotes(c.name)
        h = 'copied ' + old + ' to ' + new
        self._history(h, tp=self._tp, quiet=quiet)
        if select or not self.__select:
            old = self.__select
            self.__select = c
            h = NMObjectContainer.__select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
        self._modified()
        return c

    def kill(self, names=[], indexes=[], confirm=True, quiet=nmp.QUIET):
        olist = self.getitems(names=names, indexes=indexes)
        if len(olist) == 0:
            return []
        if confirm:
            nlist = []
            for o in olist:
                nlist.append(o.name)
            q = ('are you sure you want to kill the following items?' + '\n' +
                 ', '.join(nlist))
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn == 'y':
                self._history('cancel', tp=self._tp, quiet=quiet)
                return []
        klist = []
        nlist = []
        for o in olist:
            klist.append(o)
            self.__thecontainer.remove(o)
            nlist.append(o.name)
            if self.__select is o:
                select_old = self.__select
                self.__select = None
        h = 'killed ' + ', '.join(nlist)
        self._history(h, tp=self._tp, quiet=quiet)
        if self.__select is None and len(self.__thecontainer) > 0:
            self.__select = self.__thecontainer[0]
            h = NMObjectContainer.__select_history(select_old, self.__select)
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
