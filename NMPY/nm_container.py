#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import copy
import datetime

import nm_preferences as nmp
import nm_utilities as nmu


class NMObject(object):
    """
    NM objects to be stored in a 'Container' list (see below).

    Known Children:
        Project, Folder, Data, DataSeries, Channel, EpochSet, Note

    Attributes:
        parent (NMObject):
        name (str):
        date (str):
    """

    def __init__(self, parent, name, fxns={}, rename=True):
        self._parent = parent
        if nmu.name_ok(name):
            self.__name = name
        else:
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        if isinstance(fxns, dict):
            self._fxns = fxns
        else:
            self._fxns = {}
        if isinstance(rename, bool):
            self._rename = rename
        else:
            self._rename = True
        self.__date = str(datetime.datetime.now())
        self.__modified = self.__date
        self.__source = self.tree_path()

    @property
    def _cname(self):  # class name
        return self.__class__.__name__

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not self._rename:
            raise RuntimeError(nmu.quotes(self.__name) + ' cannot be renamed')
        if not name or not nmu.name_ok(name):
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        if name.lower() == 'select' or name.lower() == 'default':
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        old = self.__name
        self.__name = name
        h = ('changed name from ' + nmu.quotes(old) + ' to ' +
             nmu.quotes(self.__name))
        self._history(h, tp=self._tp)
        return True

    @property
    def parameters(self):  # child class should override
        # and add class parameters
        k = {'name': self.__name}
        k.update({'rename': self._rename})
        k.update({'date': self.__date})
        k.update({'modified': self.__modified})
        k.update({'source': self.__source})
        return k

    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', 'data', etc.
        return {'nmobject': self.__name}

    @property
    def content_tree(self):
        p = self._parent
        if p and isinstance(p, NMObject):
            k = {}
            k.update(p.content_tree)
            k.update(self.content)
            return k
        return self.content

    def _modified(self):
        self.__modified = str(datetime.datetime.now())

    @property
    def _tp(self):  # used with history()
        return self.tree_path(for_history=True)

    def tree_path(self, for_history=False):
        if not isinstance(for_history, bool):
            for_history = False
        if for_history:  # create tree path for history
            skip = nmp.HISTORY_TREE_PATH_SKIP
        else:
            skip = []
        plist = self.tree_path_list(skip=skip)
        if len(plist) > 0:
            tp = '.'.join(plist)
        else:
            tp = self.__name
        return tp

    def tree_path_list(self, names=True, skip=[]):
        if not isinstance(names, bool):
            names = True
        if not isinstance(skip, list):
            skip = []
        if self._cname in skip:
            return []
        p = self._parent
        if p and isinstance(p, NMObject) and p.__class__.__name__ not in skip:
            t = p.tree_path_list(names=names, skip=skip)
            if names:
                t.append(self.__name)
            else:
                t.append(self)
            return t
        if names:
            return [self.__name]
        return [self]

    def _equal(self, nmobj, ignore_name=False, alert=False):
        if nmobj.__class__.__name__ != self._cname:
            if alert:
                a = ('object type mismatch: ' + self._cname + ' vs ' +
                     nmobj.__class__.__name__)
                self._alert(a, tp=self._tp)
            return False
        sp = self.parameters
        op = nmobj.parameters
        if op.keys() != sp.keys():
            return False
        for k in sp.keys():
            if k == 'date':
                continue  # ignore, will be different
            if k == 'modified':
                continue  # ignore, will be different
            if k == 'source':
                continue  # ignore, can be different
            if k == 'name':
                if not ignore_name and op[k] != sp[k]:
                    if alert:
                        a = (nmu.quotes(k) + ' mismatch: ' + nmu.quotes(sp[k])
                             + ' vs ' + nmu.quotes(op[k]))
                        self._alert(a, tp=self._tp)
                    return False
                continue
            if op[k] != sp[k]:
                if alert:
                    a = (nmu.quotes(k) + ' mismatch: ' + str(sp[k]) + ' vs ' +
                         str(op[k]))
                    self._alert(a, tp=self._tp)
                return False
        return True

    def _copy(self, nmobj, copy_name=True, quiet=nmp.QUIET):
        if not isinstance(nmobj, NMObject):
            raise TypeError('nmobj argument: expected NMObject')
        if not isinstance(copy_name, bool):
            copy_name = True
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        # self._parent = nmobj._parent
        name = self.__name
        if copy_name:
            self.__name = nmobj.name
        self._fxns = nmobj._fxns
        self._rename = nmobj._rename
        # self.__date = nmobj._NMObject__date  # mangled
        self.__source = nmobj._NMObject__source  # mangled
        self._modified()
        h = ('copied ' + self._cname + ' ' + nmu.quotes(nmobj.name) + ' to ' +
             nmu.quotes(name))
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def save(self, path='', quiet=nmp.QUIET):
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
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
    A list container for NMObject items (above), one of which is assigned to
    'select'.

    Each NMObject item must have a unique name. The name can start with the
    same prefix (e.g. "NMExp") but this is optional. Use name_next() to
    create unique names in a sequence (e.g. "NMExp0", "NMExp1", etc.).
    One NMObject is selected/activated at a given time. This NMObject can be
    accessed via 'select' property.

    Known Children:
        ExperimentContainer, FolderContainer, DataContainer,
        DataSeriesContainer, ChannelContainer, EpochSetContainer

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
        if type_ and nmu.name_ok(type_):
            self._type = type_
        else:
            raise ValueError('bad type_ argument: ' + nmu.quotes(type_))
        if nmu.name_ok(prefix):
            self.__prefix = prefix
        else:
            raise ValueError('bad prefix argument: ' + nmu.quotes(prefix))
        if isinstance(duplicate, bool):
            self._duplicate = duplicate
        else:
            self._duplicate = True
        self.__thecontainer = []  # container of NMObject items
        self.__select = None  # selected NMObject

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'type': self._type})
        k.update({'prefix': self.__prefix})
        k.update({'duplicate': self._duplicate})
        return k

    # override
    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', etc
        # and change 'select' to 'folder_select', etc
        k = {'nmobjects': self.names}
        if self.select:
            s = self.select.name
        else:
            s = ''
        k.update({'select': s})
        return k

    # override
    def _equal(self, container, ignore_name=False, alert=False):
        if not super()._equal(container, ignore_name=ignore_name, alert=alert):
            return False
        if container.count != self.count:
            if alert:
                a = ('container count mismatch: ' + str(self.count) + ' vs ' +
                     str(container.count))
                self._alert(a, tp=self._tp)
            return False
        for i in range(0, self.count):
            s = self.__get(item_num=i, quiet=True)
            o = container._Container__get(item_num=i, quiet=True)  # mangled
            if not s._equal(o, alert=alert):
                return False
        return True

    # override
    def _copy(self, container, copy_name=True, clear_before_copy=False,
              quiet=nmp.QUIET):
        if not isinstance(container, Container):
            raise TypeError('container argument: expected Container')
        if not super()._copy(container, copy_name=copy_name, quiet=True):
            return False
        n = self.name
        self._type = container._type
        self.__prefix = container._Container__prefix  # mangled
        self._duplicate = container._duplicate
        if clear_before_copy:
            self.__thecontainer.clear()
        for o0 in container._Container__thecontainer:
            o1 = self.new(o0.name, quiet=True)
            if not o1 or not o1._copy(o0, quiet=True):
                return False
        if container.select and container.select.name:
            select = container.select.name
            self.__select = self.__get(select, quiet=True)
        else:
            self.__select = None
        self._modified()
        h = ('copied Container ' + nmu.quotes(container.name) + ' to ' +
             nmu.quotes(n))
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        if not self._rename:
            raise RuntimeError(self._type + ' items cannot be renamed')
        if len(prefix) == 0:
            pass  # ok
        elif not nmu.name_ok(prefix):
            raise ValueError('bad prefix argument: ' + nmu.quotes(prefix))
        if prefix.lower() == 'select' or prefix.lower() == 'default':
            raise ValueError('bad prefix argument: ' + nmu.quotes(prefix))
        old = self.__prefix
        self.__prefix = prefix
        self._modified()
        h = ('changed prefix from ' + nmu.quotes(old) + ' to ' +
             nmu.quotes(self.__prefix))
        self._history(h, tp=self._tp)
        return True

    @property
    def names(self):
        """Get list of names of NMObject items in Container"""
        nlist = []
        if self.__thecontainer:
            for o in self.__thecontainer:
                nlist.append(o.name)
        return nlist

    @property
    def count(self):
        """Number of NMObject items stored in Container"""
        return len(self.__thecontainer)

    def item_num(self, name='select'):
        """Find item # of NMObject in container"""
        if not self.__thecontainer:
            return -1
        if not nmu.name_ok(name) or name.lower() == 'default':
            return -1
        if not name or name.lower() == 'select':
            if self.__select and self.__select.name:
                name = self.__select.name
            else:
                return -1
        for i in range(0, len(self.__thecontainer)):
            if name.lower() == self.__thecontainer[i].name.lower():
                return i
        return -1

    __item_num = item_num

    def exists(self, name):
        """Check if NMObject exists within container"""
        return self.__item_num(name) >= 0

    __exists = exists

    def get(self, name='select', item_num=-1, quiet=nmp.QUIET):
        """Get NMObject from Container"""
        if not self.__thecontainer:
           raise RuntimeError('container = None')
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not nmu.name_ok(name) or name.lower() == 'default':
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        if isinstance(item_num, int):
            if item_num < 0:
                pass  # ok, do not use
            elif item_num < len(self.__thecontainer):
                return self.__thecontainer[item_num]
            else:
                raise IndexError('item_num out of range:  ' + str(item_num))
        else:
            raise TypeError('item_num argument: expected integer')
        if not name or name.lower() == 'select':
            return self.__select
        for o in self.__thecontainer:
            if name.lower() == o.name.lower():
                return o
        e = self._exists_error(name)
        self._error(e, tp=self._tp, quiet=quiet)
        return None

    __get = get

    def get_items(self, names=[], item_nums=[], quiet=nmp.QUIET):
        """Get a list of NMObjects from Container"""
        if not self.__thecontainer:
            raise RuntimeError('container = None')
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not isinstance(names, list):
            names = [names]
        if not isinstance(item_nums, list):
            item_nums = [item_nums]
        olist = []
        for n in names:
            if not nmu.name_ok(n) or n.lower() == 'default':
                raise ValueError('bad name argument:  ' + nmu.quotes(n))
            elif n.lower() == 'select':
                if self.__select not in olist:
                    olist.append(self.__select)
            else:
                for o in self.__thecontainer:
                    if n.lower() == o.name.lower() and o not in olist:
                        olist.append(o)
        for i in item_nums:
            if isinstance(i, int):
                if i >= 0 and i < len(self.__thecontainer):
                    o = self.__thecontainer[i]
                    if o not in olist:
                        olist.append(o)
                else:
                    raise IndexError('item numnber out of range:  ' + str(i))
            else:
                raise TypeError('item_nums argument: expected integers')
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
        if not nmu.name_ok(name):
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        if name.lower() == 'select' or name.lower() == 'default':
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        # if self.__select and name.lower() == self.__select.name.lower():
            # return self.__select  # already selected
        if self.__exists(name):
            old = self.__select
            o = self.__get(name)
            self.__select = o
            self._modified()
            h = Container._select_history(old, self.__select)
            self._history(h, tp=self._tp)
            return o
        if not self._quiet():
            q = ('failed to find ' + nmu.quotes(name) + '.' + '\n' +
                 'do you want to create a new ' + self._type +
                 ' named ' + nmu.quotes(name) + '?')
            yn = nmu.input_yesno(q, tp=self._tp)
            if yn == 'y':
                return self.new(name=name, select=True)
            self._history('cancel', tp=self._tp)
            return None
        e = self._exists_error(name)
        self._error(e, tp=self._tp)
        return None

    @staticmethod
    def _select_history(old, new):
        if new:
            n = nmu.quotes(new.name)
        else:
            n = 'None'
        if old:
            return 'changed select from ' + nmu.quotes(old.name) + ' to ' + n
        return 'selected ' + n

    def new(self, name='default', nmobj=None, select=True, quiet=nmp.QUIET):
        """
        Create a new NMObject and add to container.

        Args:
            name: unique name of new NMObject, pass 'default' for default
            select: select this NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not isinstance(select, bool):
            select = True
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not nmu.name_ok(name) or name.lower() == 'select':
            raise ValueError('bad name argument:  ' + nmu.quotes(name))
        if not name or name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        if self.__exists(name):
            e = nmu.quotes(name) + ' already exists'
            self._error(e, tp=self._tp, quiet=quiet)
            return None
        if not nmobj:
            o = NMObject(self._parent, name, self._fxns, rename=self._rename)
        elif isinstance(nmobj, NMObject):  # child 'new' should pass nmobj
            if nmobj.__class__.__name__ == self._type:
                o = nmobj
                # mangled...
                o._NMObject__name = name  # in case name='default'
                o._parent = self._parent  # reset parent reference
                o._rename = self._rename
            else:
                raise TypeError('nmobj argument: expected ' + self._type)
        else:
            raise TypeError('nmobj argument: expected NMObject')
        if not o:
            return None
        self.__thecontainer.append(o)
        if select or not self.__select:
            old = self.__select
            self.__select = o
            if old:
                h = 'created ' + nmu.quotes(name)
                self._history(h, tp=self._tp, quiet=quiet)
                h = Container._select_history(old, self.__select)
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
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not self._rename:
            raise RuntimeError(self._type + ' items cannot be renamed')
        if not self.__exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        o = self.__get(name, quiet=quiet)
        if not o:
            return False
        nameok = nmu.name_ok(newname)
        if not newname or not nameok or newname.lower() == 'select':
            raise ValueError('bad newname argument: ' + nmu.quotes(newname))
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.__exists(newname):
            e = nmu.quotes(newname) + ' already exists'
            self._error(e, tp=self._tp, quiet=quiet)
            return False
        old = nmu.quotes(o.name)
        # o.name = newname  # double history
        o._NMObject__name = newname  # mangled
        self._modified()
        new = nmu.quotes(o.name)
        h = 'renamed ' + old + ' to ' + new
        self._history(h, tp=self._tp, quiet=quiet)
        return True

    def duplicate(self, name, newname, select=True, quiet=nmp.QUIET):
        """
        Copy NMObject.

        Args:
            name: name of NMObject to copy
            newname: name of new NMObject

        Returns:
            new NMObject if successful, None otherwise
        """
        if not isinstance(select, bool):
            select = True
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not self._duplicate:
            raise RuntimeError(self._type + ' items cannot be duplicated')
        if not self.__exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return None
        o = self.__get(name, quiet=quiet)
        if not o:
            return None
        nameok = nmu.name_ok(newname)
        if not newname or not nameok or newname.lower() == 'select':
            raise ValueError('bad newname argument: ' + nmu.quotes(newname))
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.__exists(newname):
            e = nmu.quotes(newname) + ' already exists'
            self._error(e, tp=self._tp, quiet=quiet)
            return None
        # c = copy.deepcopy(o) NOT WORKING
        # TypeError: can't pickle Context objects
        # c = copy.copy(o)
        c = NMObject(self._parent, newname, self._fxns, rename=self._rename)
        if not c._copy(o, copy_name=False, quiet=True):
            return None
        # c._parent = self._parent  # reset parent reference
        self.__thecontainer.append(c)
        old = nmu.quotes(o.name)
        new = nmu.quotes(c.name)
        h = 'copied ' + old + ' to ' + new
        self._history(h, tp=self._tp, quiet=quiet)
        if select or not self.__select:
            old = self.__select
            self.__select = c
            h = Container._select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
        self._modified()
        return c

    def kill(self, name='', all_=False, ask=True, quiet=nmp.QUIET):
        """
        Kill NMObject.

        Args:
            name: name of NMObject to kill

        Returns:
            True for success, False otherwise
        """
        if not isinstance(all_, bool):
            all_ = False
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        klist = []
        if all_:
            n = ', '.join(self.names)
            if ask and not self._quiet(quiet):
                q = ('are you sure you want to kill all ' + self._type +
                     ' items?' + '\n' + 'This will kill ' + n)
                yn = nmu.input_yesno(q, tp=self._tp)
                if not yn == 'y':
                    self._history('cancel', tp=self._tp)
                    return []
            for o in self.__thecontainer:
                klist.append(o)
            self.__thecontainer.clear()
            self.__select = None
            self._history('killed ' + n, tp=self._tp, quiet=quiet)
            return klist
        if not name or not self.__exists(name):
            e = self._exists_error(name)
            self._error(e, tp=self._tp, quiet=quiet)
            return []
        o = self.__get(name, quiet=quiet)
        if not o:
            return []
        if ask and not self._quiet(quiet):
            q = 'are you sure you want to kill ' + nmu.quotes(o.name) + '?'
            yn = nmu.input_yesno(q, tp=self._tp)
            if not yn == 'y':
                self._history('cancel', tp=self._tp)
                return []
        select_next = o is self.__select  # killing select, so need new one
        if select_next:
            i = self.__item_num(o.name)
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
            h = Container._select_history(old, self.__select)
            self._history(h, tp=self._tp, quiet=quiet)
        self._modified()
        return klist

    def name_next(self, first=0, quiet=nmp.QUIET):
        """Get next default NMObject name based on prefix and sequence #."""
        if not isinstance(first, int):
            first = 0
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not self.__prefix or first < 0:
            raise RuntimeError(self._type + ' items do not have default names')
        i = self.name_next_seq(prefix=self.__prefix, first=first, quiet=quiet)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(self, prefix='default', first=0, quiet=nmp.QUIET):
        """Get next seq num of default NMObject name based on prefix."""
        if not isinstance(first, int):
            first = 0
        if not isinstance(quiet, bool):
            quiet = nmp.QUIET
        if not nmu.name_ok(prefix) or prefix.lower() == 'select':
            raise ValueError('bad prefix argument: ' + nmu.quotes(prefix))
        if not prefix or prefix.lower() == 'default':
            prefix = self.__prefix
        if not prefix or first < 0:
            raise RuntimeError(self._type + ' items do not have default names')
        elist = []
        for o in self.__thecontainer:
            name = o.name.lower()
            istr = name.replace(prefix.lower(), '')
            if str.isdigit(istr):
                i = int(istr)
                elist.append(i)
        if len(elist) == 0:
            return 0
        i = max(elist)
        return i + 1
