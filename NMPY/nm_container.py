#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmpy - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
# import copy
import datetime
import nm_configs as nmc
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

    def __init__(self, parent, name, fxns, rename=True):
        self.__parent = parent
        if nmu.name_ok(name):
            self.__name = name
        else:
            e = 'bad name arg:  ' + nmu.quotes(name)
            raise ValueError(e)
        if list(fxns.keys()) == ['quiet', 'alert', 'error', 'history']:
            self.__fxns = fxns
        else:
            e = 'bad fxn arg:  ' + str(fxns)
            raise ValueError(e)
        if isinstance(rename, bool):
            self.__rename = rename
        else:
            self.__rename = True
        self.__date = str(datetime.datetime.now())
        self.__source = self.tree_path()

    @property
    def __quiet(self):
        return self.__fxns['quiet']

    @property
    def __alert(self):
        return self.__fxns['alert']

    @property
    def __error(self):
        return self.__fxns['error']

    @property
    def __history(self):
        return self.__fxns['history']

    @property
    def __tp(self):
        return self.tree_path(history=True)

    @property
    def __type(self):
        return self.__class__.__name__

    @property
    def parameters(self):  # child class should override
        # and add parameters
        k = {'name': self.name}
        k.update({'rename': self.__rename})
        k.update({'date': self.__date})
        k.update({'source': self.__source})
        return k

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if not self.__rename:
            e = nmu.quotes(self.__name) + ' cannot be renamed'
            self.__error(e, tp=self.__tp)
            return False
        notok = name.lower() == 'select' or name.lower() == 'default'
        if not nmu.name_ok(name) or notok:
            e = 'bad name arg:  ' + nmu.quotes(name)
            self.__error(e, tp=self.__tp)
            return False
        old = nmu.quotes(self.__name)
        self.__name = name
        new = nmu.quotes(self.__name)
        self.__history('renamed ' + old + ' to ' + new, tp=self.__tp)
        return True

    @property
    def date(self):
        return self.__date

    @property
    def source(self):
        return self.__source

    @property
    def content(self):  # child class should override
        # and change 'nmobject' to 'folder', 'data', etc.
        return {'nmobject': self.name}

    @property
    def content_tree(self):
        p = self.__parent
        if p and isinstance(p, NMObject):
            k = {}
            k.update(p.content_tree)
            k.update(self.content)
            return k
        return self.content

    def tree_path(self, history=False):
        if not isinstance(history, bool):
            history = False
        if history:  # create tree path for history
            skip = nmc.HISTORY_TREE_PATH_SKIP
        else:
            skip = []
        plist = self.tree_path_list(skip=skip)
        if len(plist) > 0:
            tp = '.'.join(plist)
        else:
            tp = self.name
        return tp

    def tree_path_list(self, names=True, skip=[]):
        if not isinstance(names, bool):
            names = True
        if not isinstance(skip, list):
            skip = []
        if self.__type in skip:
            return []
        p = self.__parent
        if p and isinstance(p, NMObject) and p.__class__.__name__ not in skip:
            t = p.tree_path_list(names=names, skip=skip)
            if names:
                t.append(self.name)
            else:
                t.append(self)
            return t
        if names:
            return [self.name]
        return [self]

    def equal(self, nmobj, ignore_name=False, alert=False):
        if nmobj.__class__.__name__ != self.__type:
            if alert:
                a = ('object type mismatch: ' + self.__type + ' vs ' +
                     nmobj.__class__.__name__)
                self.__alert(a, tp=self.__tp)
            return False
        sp = self.parameters
        op = nmobj.parameters
        if op.keys() != sp.keys():
            return False
        for k in sp.keys():
            if k == 'date' or k == 'source':
                continue  # ignore
            if k == 'name':
                if not ignore_name and op[k] != sp[k]:
                    if alert:
                        a = (nmu.quotes(k) + ' mismatch: ' + nmu.quotes(sp[k])
                             + ' vs ' + nmu.quotes(op[k]))
                        self.__alert(a, tp=self.__tp)
                    return False
                continue
            if op[k] != sp[k]:
                if alert:
                    a = (nmu.quotes(k) + ' mismatch: ' + str(sp[k]) + ' vs ' +
                         str(op[k]))
                    self.__alert(a, tp=self.__tp)
                return False
        return True

    def copy(self, nmobj, copy_name=True, quiet=nmc.QUIET):
        if not isinstance(copy_name, bool):
            copy_name = True
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not isinstance(nmobj, NMObject):
            e = 'nmobj arg: expected type NMObject'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return False
        # self.__parent = nmobj._NMObject__parent
        name = self.__name
        if copy_name:
            self.__name = nmobj._NMObject__name
        # self.__fxns = nmobj._NMObject__fxns
        self.__rename = nmobj._NMObject__rename
        # self.__date = nmobj._NMObject__date
        self.__source = nmobj._NMObject__source
        h = ('copied ' + self.__type + ' ' + nmu.quotes(nmobj.name) + ' to ' +
             nmu.quotes(name))
        self.__history(h, tp=self.__tp, quiet=quiet)
        return True

    def save(self, path='', quiet=nmc.QUIET):
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        self.__alert('under construction')
        return False


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

    def __init__(self, parent, name, fxns, type_='NMObject', prefix='NMObject',
                 rename=True, duplicate=True):
        super().__init__(parent, name, fxns, rename=rename)
        if nmu.name_ok(type_):
            self.__type = type_
        else:
            e = 'bad type_ arg: ' + nmu.quotes(type_)
            raise ValueError(e)
        if nmu.name_ok(prefix):
            self.__prefix = prefix
        else:
            e = 'bad prefix arg: ' + nmu.quotes(prefix)
            raise ValueError(e)
        if isinstance(duplicate, bool):
            self.__duplicate = duplicate
        else:
            self.__duplicate = True
        self.__thecontainer = []  # container of NMObject items
        self.__select = None  # selected NMObject

    @property
    def __parent(self):
        return self._NMObject__parent

    @property
    def __fxns(self):
        return self._NMObject__fxns

    @property
    def __quiet(self):
        return self._NMObject__quiet

    @property
    def __alert(self):
        return self._NMObject__alert

    @property
    def __error(self):
        return self._NMObject__error

    @property
    def __history(self):
        return self._NMObject__history

    @property
    def __rename(self):
        return self._NMObject__rename

    @property
    def __tp(self):
        return self.tree_path(history=True)

    # override
    @property
    def parameters(self):
        k = super().parameters
        k.update({'type': self.__type})
        k.update({'prefix': self.__prefix})
        k.update({'duplicate': self.__duplicate})
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
    def equal(self, container, ignore_name=False, alert=False):
        if not super().equal(container, ignore_name=ignore_name, alert=alert):
            return False
        if container.count != self.count:
            if alert:
                a = ('container count mismatch: ' + str(self.count) + ' vs ' +
                     str(container.count))
                self.__alert(a, tp=self.__tp)
            return False
        for i in range(0, self.count):
            s = self.get(item_num=i, quiet=True)
            o = container.get(item_num=i, quiet=True)
            if not s.equal(o, alert=alert):
                return False
        return True

    # override
    def copy(self, container, copy_name=True, clear_before_copy=False,
             quiet=nmc.QUIET):
        n = self.name
        if not super().copy(container, copy_name=copy_name, quiet=True):
            return False
        self.__type = container._Container__type
        self.__prefix = container._Container__prefix
        self.__duplicate = container._Container__duplicate
        if clear_before_copy:
            self.__thecontainer.clear()
        for o0 in container._Container__thecontainer:
            o1 = self.new(o0.name, quiet=True)
            if not o1 or not o1.copy(o0):
                return False
        if container.select and container.select.name:
            select = container.select.name
            self.__select = self.get(select, quiet=True)
        else:
            self.__select = None
        h = ('copied Container ' + nmu.quotes(container.name) + ' to ' +
             nmu.quotes(n))
        self.__history(h, tp=self.__tp, quiet=quiet)
        return True

    @property
    def prefix(self):  # see name_next()
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        if not self.__rename:
            e = self.__type + ' items cannot be renamed'
            self.__error(e, tp=self.__tp)
            return False
        notok = prefix.lower() == 'select' or prefix.lower() == 'default'
        if not nmu.name_ok(prefix) or notok:
            e = 'bad prefix arg: ' + nmu.quotes(prefix)
            self.__error(e, tp=self.__tp)
            return False
        old = nmu.quotes(self.__prefix)
        self.__prefix = prefix
        new = nmu.quotes(self.__prefix)
        h = 'changed prefix ' + old + ' to ' + new
        self.__history(h, tp=self.__tp)
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

    def exists(self, name):
        """Check if NMObject exists within container"""
        return self.item_num(name) >= 0

    def get(self, name='select', item_num=-1, quiet=nmc.QUIET):
        """Get NMObject from Container"""
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not nmu.name_ok(name) or name.lower() == 'default':
            e = 'bad name arg:  ' + nmu.quotes(name)
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        if isinstance(item_num, int) and item_num >= 0:
            if item_num < len(self.__thecontainer):
                return self.__thecontainer[item_num]
            else:
                e = 'item_num out of range:  ' + str(item_num)
                self.__error(e, tp=self.__tp, quiet=quiet)
                return None
        if not name or name.lower() == 'select':
            return self.__select
        if not self.__thecontainer:
            return None
        for o in self.__thecontainer:
            if name.lower() == o.name.lower():
                return o
        self.__error(self.__exists_error(name), tp=self.__tp, quiet=quiet)
        return None

    def get_items(self, names=[], item_nums=[], quiet=nmc.QUIET):
        """Get a list of NMObjects from Container"""
        if not self.__thecontainer:
            return []
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not isinstance(names, list):
            names = [names]
        if not isinstance(item_nums, list):
            item_nums = [item_nums]
        olist = []
        for n in names:
            if not nmu.name_ok(n) or n.lower() == 'default':
                e = 'bad name:  ' + nmu.quotes(n)
                self.__error(e, tp=self.__tp, quiet=quiet)
            elif n.lower() == 'select':
                if self.__select not in olist:
                    olist.append(self.__select)
            else:
                for o in self.__thecontainer:
                    if n.lower() == o.name.lower() and o not in olist:
                        olist.append(o)
        for i in item_nums:
            if not isinstance(i, int) or i < 0:
                e = 'bad item number:  ' + str(i)
                self.__error(e, tp=self.__tp, quiet=quiet)
            elif i >= 0 and i < len(self.__thecontainer):
                o = self.__thecontainer[i]
                if o not in olist:
                    olist.append(o)
            else:
                e = 'item numnber out of range:  ' + str(i)
                self.__error(e, tp=self.__tp, quiet=quiet)
        return olist

    def __exists_error(self, name):
        e = 'failed to find ' + nmu.quotes(name)
        e += '\n' + 'acceptable names: ' + str(self.names)
        return e

    @property
    def select(self):
        return self.__select

    @select.setter
    def select(self, name):
        notok = name.lower() == 'select' or name.lower() == 'default'
        if not nmu.name_ok(name) or notok:
            e = 'bad name arg:  ' + nmu.quotes(name)
            self.__error(e, tp=self.__tp)
            return None
        # if self.__select and name.lower() == self.__select.name.lower():
            # return self.__select  # already selected
        if self.exists(name):
            o = self.get(name)
            self.__select = o
            self.__history('selected ' + nmu.quotes(name), tp=self.__tp)
            return o
        if not self.__quiet():
            q = ('failed to find ' + nmu.quotes(name) + '.' + '\n' +
                 'do you want to create a new ' + self.__type +
                 ' named ' + nmu.quotes(name) + '?')
            yn = nmu.input_yesno(q, tp=self.__tp)
            if yn == 'y':
                return self.new(name=name, select=True)
            self.__history('cancel', tp=self.__tp)
            return None
        self.__error(self.__exists_error(name), tp=self.__tp)
        return None

    def new(self, name='default', nmobj=None, select=True, quiet=nmc.QUIET):
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
            quiet = nmc.QUIET
        if not nmu.name_ok(name) or name.lower() == 'select':
            e = 'bad name arg:  ' + nmu.quotes(name)
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        if not name or name.lower() == 'default':
            name = self.name_next(quiet=quiet)
        if self.exists(name):
            e = nmu.quotes(name) + ' already exists'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        if nmobj is None:
            o = NMObject(self.__parent, name, self.__fxns,
                         rename=self.__rename)
        elif isinstance(nmobj, NMObject):  # child 'new' should pass nmobj
            if nmobj.__class__.__name__ == self.__type:
                o = nmobj
                # mangled...
                o._NMObject__name = name  # in case name='default'
                o._NMObject__parent = self.__parent  # reset parent reference
                o._NMObject__rename = self.__rename
            else:
                e = 'nmobj arg: expected type ' + self.__type
                self.__error(e, tp=self.__tp, quiet=quiet)
                return None
        else:
            e = 'nmobj arg: expected type NMObject'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        if not o:
            return None
        self.__thecontainer.append(o)
        h = 'created'
        if select or not self.__select:
            self.__select = o
            h += '/selected'
        self.__history(h + ' ' + nmu.quotes(name), tp=self.__tp, quiet=quiet)
        return o

    def rename(self, name, newname, quiet=nmc.QUIET):
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not self.__rename:
            e = self.__type + ' items cannot be renamed'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return False
        if not self.exists(name):
            self.__error(self.__exists_error(name), tp=self.__tp, quiet=quiet)
            return False
        o = self.get(name, quiet=quiet)
        if not o:
            return False
        nameok = nmu.name_ok(newname)
        if not newname or not nameok or newname.lower() == 'select':
            e = 'bad newname arg: ' + nmu.quotes(newname)
            self.__error(e, tp=self.__tp, quiet=quiet)
            return False
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.exists(newname):
            e = nmu.quotes(newname) + ' already exists'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return False
        old = nmu.quotes(o.name)
        # o.name = newname  # double history
        o._NMObject__name = newname  # mangled
        new = nmu.quotes(o.name)
        h = 'renamed ' + old + ' to ' + new
        self.__history(h, tp=self.__tp, quiet=quiet)
        return True

    def duplicate(self, name, newname, select=True, quiet=nmc.QUIET):
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
            quiet = nmc.QUIET
        if not self.__duplicate:
            e = self.__type + ' items cannot be duplicated'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        if not self.exists(name):
            self.__error(self.__exists_error(name), tp=self.__tp, quiet=quiet)
            return None
        o = self.get(name, quiet=quiet)
        if not o:
            return None
        nameok = nmu.name_ok(newname)
        if not newname or not nameok or newname.lower() == 'select':
            e = 'bad newname arg: ' + nmu.quotes(newname)
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        if newname.lower() == 'default':
            newname = self.name_next(quiet=quiet)
        if self.exists(newname):
            e = nmu.quotes(newname) + ' already exists'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return None
        # c = copy.deepcopy(o) NOT WORKING
        # TypeError: can't pickle Context objects
        # c = copy.copy(o)
        c = NMObject(self.__parent, newname, self.__fxns, rename=self.__rename)
        if not c.copy(o, copy_name=False):
            return None
        # mangled
        # c._NMObject__parent = self.__parent  # reset parent reference
        self.__thecontainer.append(c)
        old = nmu.quotes(o.name)
        new = nmu.quotes(c.name)
        h = 'copied ' + old + ' to ' + new
        self.__history(h, tp=self.__tp, quiet=quiet)
        if select or not self.__select:
            self.__select = c
            self.__history('selected ' + new, tp=self.__tp, quiet=quiet)
        return c

    def kill(self, name='', all_=False, ask=True, quiet=nmc.QUIET):
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
            quiet = nmc.QUIET
        klist = []
        if all_:
            n = ', '.join(self.names)
            if ask and not self.__quiet(quiet):
                q = ('are you sure you want to kill all ' + self.__type +
                     ' items?' + '\n' + 'This will kill ' + n)
                yn = nmu.input_yesno(q, tp=self.__tp)
                if not yn == 'y':
                    self.__history('cancel', tp=self.__tp)
                    return []
            for o in self.__thecontainer:
                klist.append(o)
            self.__thecontainer.clear()
            self.__select = None
            self.__history('killed ' + n, tp=self.__tp, quiet=quiet)
            return klist
        if not name or not self.exists(name):
            self.__error(self.__exists_error(name), tp=self.__tp, quiet=quiet)
            return []
        o = self.get(name, quiet=quiet)
        if not o:
            return []
        if ask and not self.__quiet(quiet):
            q = 'are you sure you want to kill ' + nmu.quotes(o.name) + '?'
            yn = nmu.input_yesno(q, tp=self.__tp)
            if not yn == 'y':
                self.__history('cancel', tp=self.__tp)
                return []
        select_next = o is self.__select  # killing select, so need new one
        if select_next:
            i = self.item_num(o.name)
        klist.append(o)
        self.__thecontainer.remove(o)
        h = 'killed ' + nmu.quotes(o.name)
        self.__history(h, tp=self.__tp, quiet=quiet)
        items = len(self.__thecontainer)
        if select_next and items > 0:
            i = max(i, 0)
            i = min(i, items - 1)
            o = self.__thecontainer[i]
            self.__select = o
            h = 'selected ' + nmu.quotes(o.name)
            self.__history(h, tp=self.__tp, quiet=quiet)
        return klist

    def name_next(self, first=0, quiet=nmc.QUIET):
        """Get next default NMObject name based on prefix and sequence #."""
        if not isinstance(first, int):
            first = 0
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not self.__prefix or first < 0:
            e = self.__type + ' items do not have default names'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return ''
        i = self.name_next_seq(prefix=self.__prefix, first=first, quiet=quiet)
        if i >= 0:
            return self.__prefix + str(i)
        return ''

    def name_next_seq(self, prefix='default', first=0, quiet=nmc.QUIET):
        """Get next seq num of default NMObject name based on prefix."""
        if not isinstance(first, int):
            first = 0
        if not isinstance(quiet, bool):
            quiet = nmc.QUIET
        if not nmu.name_ok(prefix) or prefix.lower() == 'select':
            e = 'bad prefix ' + nmu.quotes(prefix)
            self.__error(e, tp=self.__tp, quiet=quiet)
            return -1
        if not prefix or prefix.lower() == 'default':
            prefix = self.__prefix
        if not prefix or first < 0:
            e = self.__type + ' items do not have default names'
            self.__error(e, tp=self.__tp, quiet=quiet)
            return -1
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
