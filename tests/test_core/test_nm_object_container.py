#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:43:19 2022

@author: jason
"""

import unittest
from unittest.mock import patch

import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from tests.test_core.test_nm_object import NMObject2

QUIET = True
NM = NMManager(quiet=QUIET)
NM0 = NMManager(quiet=QUIET)
NM1 = NMManager(quiet=QUIET)
CNAME0 = "map0"  # name
CNAME1 = "map1"
OPREFIX0 = "object"
OPREFIX1 = "obj"
OSEQFORMAT0 = "0"
OSEQFORMAT1 = "A"
ONLIST0 = [OPREFIX0 + str(i) for i in range(6)]
ONLIST1 = [OPREFIX1 + nmu.CHANNEL_CHARS[i] for i in range(6)]
SETS_NLIST0 = ["set" + str(i) for i in range(3)]
SETS_NLIST1 = ["s" + str(i) for i in range(3)]


class NMObjectContainerTest(unittest.TestCase):
    def setUp(self):  # executed before each test
        self.olist0 = []  # NM object list
        self.olist1 = []
        for n in ONLIST0:
            self.olist0.append(NMObject(parent=NM0, name=n))
        for n in ONLIST1:
            self.olist1.append(NMObject(parent=NM1, name=n))

        self.map0 = NMObjectContainer(
            parent=NM0,
            name=CNAME0,
            rename_on=True,
            auto_name_prefix=OPREFIX0,
            auto_name_seq_format=OSEQFORMAT0,
        )
        self.map0.update(self.olist0)

        self.sets0 = {SETS_NLIST0[0]: [ONLIST0[0], ONLIST0[2], ONLIST0[3]]}
        self.map0.sets.update(self.sets0)

        self.map1 = NMObjectContainer(
            parent=NM1,
            name=CNAME1,
            rename_on=False,
            auto_name_prefix=OPREFIX1,
            auto_name_seq_format=OSEQFORMAT1,
        )
        self.map1.update(self.olist1)
        self.map1.selected_name = ONLIST1[2]

        self.sets1 = {
            SETS_NLIST1[0]: [ONLIST1[0], ONLIST1[2], ONLIST1[4]],
            SETS_NLIST1[1]: [ONLIST1[1], ONLIST1[3], ONLIST1[5]],
        }
        self.map1.sets.update(self.sets1)

        self.map1_copy = NMObjectContainer(copy=self.map1)

    # __init__, copy (NMObject), parameters (NMObject)
    # content (NMObject), content_type, content_parameters (NMObject)

    def test00_init(self):
        # args: parent, name (NMObject)
        # args: nmobjects, rename_on, auto_name_prefix, name_seq_format, copy
        # nmobjects: see test_update()
        # auto_name_prefix: see test_auto_name_prefix_set()
        # name_seq_format: see test_name_seq_format()

        bad = list(nmu.BADTYPES)
        bad.remove(None)
        bad.remove(True)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectContainer(rename_on=b)

        bad = list(nmu.BADTYPES)
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                NMObjectContainer(copy=b)

        NMObjectContainer()  # no arguments is ok

        self.assertEqual(self.map0._parent, NM0)
        self.assertEqual(self.map0.name, CNAME0)
        self.assertTrue(self.map0._NMObjectContainer__rename_on)
        self.assertEqual(self.map0._NMObjectContainer__auto_name_prefix, OPREFIX0)
        self.assertEqual(self.map0._NMObjectContainer__auto_name_seq_format, OSEQFORMAT0)
        self.assertIsNone(self.map0.selected_name)
        self.assertIsNone(self.map0.selected_value)
        self.assertEqual(self.map0.execute_target, "selected")
        self.assertEqual(self.map0.execute_targets, [])
        self.assertEqual(list(self.map0.sets.keys()), [SETS_NLIST0[0]])
        self.assertEqual(len(self.map0._NMObjectContainer__map), len(ONLIST0))

        self.assertEqual(self.map1._parent, NM1)
        self.assertEqual(self.map1.name, CNAME1)
        self.assertFalse(self.map1._NMObjectContainer__rename_on)
        self.assertEqual(self.map1._NMObjectContainer__auto_name_prefix, OPREFIX1)
        self.assertEqual(self.map1._NMObjectContainer__auto_name_seq_format, OSEQFORMAT1)
        self.assertEqual(self.map1.selected_name, ONLIST1[2])
        self.assertEqual(self.map1.selected_value, self.olist1[2])
        self.assertEqual(self.map1.execute_target, "selected")
        self.assertEqual(self.map1.execute_targets, [self.olist1[2]])
        self.assertEqual(list(self.map1.sets.keys()), [SETS_NLIST1[0], SETS_NLIST1[1]])
        self.assertEqual(len(self.map1._NMObjectContainer__map), len(ONLIST1))

        self.assertEqual(self.map1_copy._parent, NM1)
        self.assertEqual(self.map1_copy.name, CNAME1)
        self.assertFalse(self.map1_copy._NMObjectContainer__rename_on)
        self.assertEqual(self.map1_copy._NMObjectContainer__auto_name_prefix, OPREFIX1)
        self.assertEqual(
            self.map1_copy._NMObjectContainer__auto_name_seq_format, OSEQFORMAT1
        )
        self.assertEqual(self.map1_copy.selected_name, ONLIST1[2])
        self.assertEqual(self.map1_copy.execute_target, "selected")
        self.assertEqual(
            list(self.map1_copy.sets.keys()), [SETS_NLIST1[0], SETS_NLIST1[1]]
        )
        self.assertEqual(len(self.map1_copy._NMObjectContainer__map), len(ONLIST1))

    def test01_copy(self):
        for n in ONLIST1:
            s = self.map1.get(n)
            c = self.map1_copy.get(n)
            self.assertTrue(s == c)
            self.assertFalse(s is c)
        self.assertEqual(self.map1.selected_name, self.map1_copy.selected_name)
        self.assertTrue(self.map1.selected_value == self.map1_copy.selected_value)
        self.assertFalse(self.map1.selected_value is self.map1_copy.selected_value)

    def test02_parameters(self):
        klist = ["name", "created", "copy of"]
        klist += [
            "content_type",
            "rename_on",
            "auto_name_prefix",
            "auto_name_seq_format",
            "selected_name",
            "execute_target",
            "sets",
        ]

        plist = self.map0.parameters
        self.assertEqual(list(plist.keys()), klist)
        self.assertEqual(plist["name"], CNAME0)
        self.assertIsNone(plist["copy of"])
        self.assertEqual(plist["content_type"], "nmobject")
        self.assertTrue(plist["rename_on"])
        self.assertEqual(plist["auto_name_prefix"], OPREFIX0)
        self.assertEqual(plist["auto_name_seq_format"], OSEQFORMAT0)
        self.assertIsNone(plist["selected_name"])
        self.assertEqual(plist["execute_target"], "selected_name")
        self.assertEqual(plist["sets"], [SETS_NLIST0[0]])

        plist = self.map1.parameters
        self.assertEqual(plist["name"], CNAME1)
        self.assertIsNone(plist["copy of"])
        self.assertEqual(plist["content_type"], "nmobject")
        self.assertFalse(plist["rename_on"])
        self.assertEqual(plist["auto_name_prefix"], OPREFIX1)
        self.assertEqual(plist["auto_name_seq_format"], OSEQFORMAT1)
        self.assertEqual(plist["selected_name"], ONLIST1[2])
        self.assertEqual(plist["execute_target"], "selected_name")
        self.assertEqual(plist["sets"], [SETS_NLIST1[0], SETS_NLIST1[1]])

        plist = self.map1_copy.parameters
        self.assertEqual(plist["name"], CNAME1)
        tp = NM1.name + "." + CNAME1
        self.assertEqual(plist["copy of"], tp)
        self.assertEqual(plist["content_type"], "nmobject")
        self.assertFalse(plist["rename_on"])
        self.assertEqual(plist["auto_name_prefix"], OPREFIX1)
        self.assertEqual(plist["auto_name_seq_format"], OSEQFORMAT1)
        self.assertEqual(plist["selected_name"], ONLIST1[2])
        self.assertEqual(plist["execute_target"], "selected_name")
        self.assertEqual(plist["sets"], [SETS_NLIST1[0], SETS_NLIST1[1]])

    def test03_content_type(self):
        self.assertEqual(self.map0.content_type(), "NMObject")

    def test04_content_type_ok(self):  # NMObject type
        self.assertFalse(self.map0.content_type_ok(NM0))
        self.assertFalse(self.map0.content_type_ok(self.map1))
        self.assertTrue(self.map0.content_type_ok(self.olist0[0]))

    def test05_content_parameters(self):
        plist = self.map0.content_parameters
        self.assertTrue(isinstance(plist, list))
        self.assertEqual(len(plist), len(self.map0))
        for p in plist:
            self.assertTrue(isinstance(p, dict))
            # print(p.keys())  see NMObject.parameters

    # MutableMapping Abstract Methods
    # __getitem__, __setitem__, __delitem__, __iter__, __len__

    def test06_getitem(self):
        # args: key (see key_check)
        # get(), items(), values()

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertIsNone(self.map0.get(b))

        for i, n in enumerate(ONLIST0):
            o = self.map0.get(n)
            self.assertEqual(o, self.olist0[i])

        for i, n in enumerate(ONLIST0):
            o = self.map0[n]
            self.assertEqual(o, self.olist0[i])

        with self.assertRaises(KeyError):
            o = self.map0.__getitem__("test")

        with self.assertRaises(KeyError):
            o = self.map0["test"]

        o = self.map0.get("test")  # get() does not throw key error
        self.assertIsNone(o)

        for i, k in enumerate(ONLIST0):
            self.assertEqual(self.map0.get(k), self.olist0[i])
            self.assertEqual(self.map0.get(k.upper()), self.olist0[i])
            # case insensitive

        for i, (k, v) in enumerate(self.map0.items()):
            self.assertEqual(k, ONLIST0[i])
            self.assertEqual(v, self.olist0[i])

        for i, v in enumerate(self.map0.values()):
            self.assertEqual(v, self.olist0[i])

    def test07_setitem(self):
        # '=' symbol calls __setitem__()
        # see update()

        n = ONLIST0[1]

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0[b] = NMObject(parent=NM0, name="test")

        bad = list(nmu.BADTYPES)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0[n] = b

        badname = n + "x"  # name should be the same as key
        with self.assertRaises(KeyError):
            self.map0[n] = NMObject(parent=NM0, name=badname)

        # change existing NMObject
        o1 = self.map0.get(n)
        self.assertTrue(o1 is self.olist0[1])
        o2 = NMObject(parent=NM0, name=n)
        rfr_before = o2._NMObject__rename_fxnref
        self.map0[n.upper()] = o2
        o1 = self.map0.get(n)
        self.assertFalse(o1 is self.olist0[1])
        self.assertTrue(o1 is o2)
        rfr_after = o1._NMObject__rename_fxnref
        self.assertNotEqual(rfr_before, rfr_after)
        self.assertEqual(rfr_after, self.map0.rename)
        self.assertEqual(len(self.map0), len(ONLIST0))  # stays the same

        # add new NMObject
        n = "test"
        o3 = NMObject(parent=NM0, name=n)
        self.assertFalse(n in self.map0)
        self.map0[n.upper()] = o3
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)

    def test08_delitem(self):
        # test 'del' command, which calls pop()
        # see pop(), popitem() and clear()
        with self.assertRaises(KeyError):
            del self.map0["test"]

        print("\nanswer NO")
        with patch('pyneuromatic.core.nm_utilities.input_yesno', return_value='n'):
            del self.map0[ONLIST0[1]]
        self.assertTrue(ONLIST0[1] in self.map0)

        print("\nanswer YES")
        with patch('pyneuromatic.core.nm_utilities.input_yesno', return_value='y'):
            del self.map0[ONLIST0[1]]
        self.assertFalse(ONLIST0[1] in self.map0)
        with self.assertRaises(KeyError):
            del self.map0[ONLIST0[1]]

    def test09_iter(self):
        o_iter = iter(self.map0)
        for n in ONLIST0:
            self.assertEqual(next(o_iter), n)

    def test10_len(self):
        self.assertEqual(len(self.map0), len(ONLIST0))

    # Mapping Mixin Methods
    # __contains__, __eq__, __ne__
    # keys, items, values, get (no override)

    def test11_contains(self):  # 'in' operator
        self.assertFalse("" in self.map0)
        self.assertFalse("test" in self.map0)
        for n in ONLIST0:
            self.assertTrue(n in self.map0)
            self.assertTrue(n.upper() in self.map0)  # case insensitive
            self.assertFalse(n in self.map1)
        for o in self.olist0:
            self.assertTrue(o in self.map0)
            self.assertFalse(o in self.map1)
        for n in ONLIST1:
            self.assertTrue(n in self.map1)
        for o in self.olist1:
            self.assertTrue(o in self.map1)

    def test12_eq(self):  # '==' and '!=' and 'is' operators
        # arg; other

        for b in nmu.BADTYPES:
            self.assertFalse(self.map0 == b)

        self.assertTrue(self.map0 is self.map0)
        self.assertFalse(self.map0 is self.map1)
        self.assertTrue(self.map0 == self.map0)
        self.assertFalse(self.map0 == self.map1)
        self.assertFalse(self.map0 != self.map0)
        self.assertTrue(self.map0 != self.map1)

        # recreate map0 and compare to self.map0

        map0 = NMObjectContainer(
            parent=NM0,
            name=CNAME0,
            rename_on=True,
            auto_name_prefix=OPREFIX0,
        )

        olist0 = []
        for n in ONLIST0:
            olist0.append(NMObject(parent=NM0, name=n))
        map0.update(olist0)

        self.assertFalse(map0 == self.map0)  # sets are not equal
        self.assertFalse(map0 is self.map0)

        map0.sets.update(self.sets0)
        self.assertTrue(map0 == self.map0)

        map0.sets.new(SETS_NLIST0[1])
        self.assertFalse(map0 == self.map0)

        self.map0.sets.new(SETS_NLIST0[1])
        self.assertTrue(map0 == self.map0)

    def test13_keys(self):
        klist = list(self.map0.keys())
        self.assertEqual(klist, ONLIST0)
        klist = list(self.map1.keys())
        self.assertEqual(klist, ONLIST1)

    def test14_items(self):
        # see test06_getitem
        pass

    def test15_values(self):
        # see test06_getitem
        pass

    def test16_get(self):
        # see test06_getitem
        pass

    # MutableMapping Mixin functions
    # pop, popitem, clear, update, setdefault

    def test17_pop(self):
        # see test07_delitem

        with self.assertRaises(KeyError):
            self.map0.pop("test")

        o = self.map0.pop(ONLIST0[1], auto_confirm="n")
        self.assertIsNone(o)
        self.assertTrue(ONLIST0[1] in self.map0)
        o = self.map0.pop(ONLIST0[1], auto_confirm="y")
        self.assertEqual(o, self.olist0[1])
        self.assertFalse(ONLIST0[1] in self.map0)

        with self.assertRaises(KeyError):
            o = self.map0.pop(ONLIST0[1])

        for i, n in enumerate(ONLIST1):
            o = self.map1.pop(n, auto_confirm="y")
            self.assertEqual(o, self.olist1[i])
            self.assertFalse(n in self.map1)
        self.assertEqual(len(self.map1), 0)

    def test18_popitem(self):
        # with self.assertRaises(RuntimeError):
        #    self.map0.popitem()  # NOT ALLOWED
        # popitem returns a tuple
        o = self.map0.popitem(auto_confirm="n")
        # print(o)
        self.assertEqual(o, ())
        self.assertTrue(ONLIST0[-1] in self.map0)
        o = self.map0.popitem(auto_confirm="y")
        self.assertFalse(ONLIST0[-1] in self.map0)
        t = (ONLIST0[-1], self.olist0[-1])
        self.assertEqual(o, t)
        for i, n in reversed(list(enumerate(ONLIST1))):
            o = self.map1.popitem(auto_confirm="y")
            self.assertFalse(n in self.map1)
        self.assertEqual(len(self.map1), 0)

    def test19_clear(self):
        o = self.map0.clear(auto_confirm="n")
        # print(o)
        self.assertIsNone(o)
        self.assertEqual(len(self.map0), len(ONLIST0))
        o = self.map0.clear(auto_confirm="y")
        self.assertIsNone(o)
        self.assertEqual(len(self.map0), 0)

    def test20_update(self):  # add NMObject to map
        n1 = "test"
        o1 = NMObject(parent=NM0, name=n1)
        o2 = NMObject(parent=NM0, name=n1.upper())
        o3 = NMObject2(parent=NM0, name="test3")

        bad = list(nmu.BADTYPES)
        bad.remove(None)
        bad.remove([])
        bad.remove({})
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.update(b)

        bad = list(nmu.BADTYPES)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.update([b, o1])

        rfr_before = o1._NMObject__rename_fxnref
        self.map0.update(o1)
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)
        rfr_after = o1._NMObject__rename_fxnref
        self.assertNotEqual(rfr_before, rfr_after)
        self.assertEqual(rfr_after, self.map0.rename)
        for i, k in enumerate(self.map0.keys()):
            if k.lower() == n1.lower():
                self.assertEqual(i, len(self.map0) - 1)

        self.map0.update(o1)
        self.map0.update(o2)  # o1 and o2 have the same name
        self.assertTrue(self.map0.get(n1) is o2)
        self.map0.update({"test": o2})  # key not used

        with self.assertRaises(KeyError):
            self.map0.update({"test1": o1, "test2": o2})  # key not used

        self.assertTrue(self.map0.get(n1) is o2)

        with self.assertRaises(TypeError):
            self.map0.update(o3)  # wrong type

        new_len = len(self.map0) + len(self.map1)
        self.map0.update(self.map1)  # another NMObjectContainer is ok
        self.assertEqual(len(self.map0), new_len)
        for n in ONLIST1:
            self.assertTrue(n in self.map0)

    def test21_setdefault(self):
        # calls __getitem__()
        # should be called get_value_or_default()
        # does not remove NMObject
        with self.assertRaises(KeyError):
            self.map0.setdefault("test")
        o = self.map0.setdefault("test", default="ok")
        self.assertEqual(o, "ok")
        for i, n in enumerate(ONLIST0):
            self.assertEqual(self.map0.setdefault(n), self.olist0[i])
        self.assertEqual(len(self.map0), len(ONLIST0))

    # NMObjectContainer methods
    # getkey, newkey
    # rename, reorder, duplicate
    # name_prefix (property), name_prefix_set, name_next
    # select (property), select_set, select_get

    def test22_getkey(self):
        # args: key, ok, error

        bad = list(nmu.BADTYPES)
        for b in bad:
            key = self.map0._getkey(b)
            self.assertIsNone(key)

        bad = list(nmu.BADNAMES)
        for b in bad:
            key = self.map0._getkey(b)
            self.assertIsNone(key)

        for n in ONLIST0:
            key = self.map0._getkey(n + "x")
            self.assertIsNone(key)

        # test keys are case insensitive
        key = self.map0._getkey(ONLIST0[1].upper())
        self.assertEqual(key, ONLIST0[1])

        # test key = 'selected'
        key = self.map0._getkey("selected")
        self.assertIsNone(key)
        self.map0.selected_name = ONLIST0[1]
        key = self.map0._getkey("selected")
        self.assertEqual(key, ONLIST0[1])
        # select/key should be case insensitive
        self.map0.selected_name = ONLIST0[1].upper()
        key = self.map0._getkey("selected")
        self.assertEqual(key, ONLIST0[1])

    def test23_newkey(self):
        # args: key, ok, error

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                key = self.map0._newkey(b)

        bad = list(nmu.BADNAMES)
        for b in bad:
            with self.assertRaises(ValueError):
                key = self.map0._newkey(b)

        for n in ONLIST0:
            with self.assertRaises(KeyError):
                key = self.map0._newkey(n)
            with self.assertRaises(KeyError):
                key = self.map0._newkey(n.upper())
            key = self.map0._newkey(n + "x")
            self.assertEqual(key, n + "x")

        # test key = None
        key = self.map0._newkey(None)
        n = len(ONLIST0)
        self.assertEqual(key, OPREFIX0 + str(n))

    def test24_rename(self):
        # args: key, newkey
        bad = list((None, 3, 3.14, True, [], ()))
        for b in bad:  # test key
            print(b)
            with self.assertRaises(KeyError):
                self.map0.rename(b, ONLIST0[3])
        for b in bad:  # test newkey
            with self.assertRaises(KeyError):
                self.map0.rename(ONLIST0[4], b)

        bad = list(nmu.BADNAMES)
        # bad.remove('select')  # ok
        for b in bad:  # test key
            with self.assertRaises(KeyError):
                self.map0.rename(b, ONLIST0[3])

        bad = list(nmu.BADNAMES)
        for b in bad:  # test newkey
            with self.assertRaises(ValueError):
                self.map0.rename(ONLIST0[4], b)
        """
        with self.assertRaises(KeyError):
            self.map0.rename('select', ONLIST0[0])  # name already used
        with self.assertRaises(RuntimeError):
            self.map1.rename('select', 'test')  # rename = False
        """
        self.map0.pop(ONLIST0[3], auto_confirm="y")
        self.assertFalse(ONLIST0[3] in self.map0)
        klist = [OPREFIX0 + str(i) for i in [0, 1, 2, 4, 5]]
        self.assertEqual(list(self.map0.keys()), klist)
        # self.map0.select = ONLIST0[0]
        # s = self.map0.rename('select', ONLIST0[3])
        # self.assertTrue(s)
        s = self.map0.rename(ONLIST0[0], ONLIST0[3])
        self.assertTrue(s)
        klist = [OPREFIX0 + str(i) for i in [3, 1, 2, 4, 5]]
        self.assertEqual(list(self.map0.keys()), klist)
        nnext = self.map0.auto_name_next()
        self.assertEqual(nnext, OPREFIX0 + "0")
        s = self.map0.rename(ONLIST0[4], None)
        self.assertTrue(s)
        klist = [OPREFIX0 + str(i) for i in [3, 1, 2, 0, 5]]
        self.assertEqual(list(self.map0.keys()), klist)
        for i, v in enumerate(self.map0.values()):
            o = self.map0.get(v.name)
            if i == 0:
                o.name = "test"  # executes rename()
            else:
                with self.assertRaises(KeyError):
                    o.name = "test"  # name already exists
        for i, v in enumerate(self.map0.values()):
            o = self.map0.get(v.name)
            o.name = "test" + str(i)
        klist = ["test0", "test1", "test2", "test3", "test4"]
        self.assertEqual(list(self.map0.keys()), klist)

        # map_before = self.map0._NMObjectContainer__map
        # ms = self.map0.rename('test2', 'test2b')
        # mmap_after = self.map0._NMObjectContainer__map
        # mself.assertEqual(map_before, map_after)

    def test25_reorder(self):
        # args: newkeys list
        bad = list(nmu.BADTYPES)
        bad.remove([])  # ok
        for b in bad:  # test key
            with self.assertRaises(TypeError):
                self.map0.reorder(b)

        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:  # test key
            with self.assertRaises(TypeError):
                self.map0.reorder([b])

        klist = [OPREFIX0 + str(i) for i in [0, 1, 2, 3, 4]]
        with self.assertRaises(KeyError):
            self.map0.reorder(klist)  # number mismatch

        klist = [OPREFIX0 + str(i) for i in [0, 1, 2, 3, 4, 6]]
        with self.assertRaises(KeyError):
            self.map0.reorder(klist)

        self.map0.reorder(ONLIST0)  # OK, no change

        klist = ONLIST0
        klist.reverse()
        self.map0.reorder(klist)
        self.assertEqual(list(self.map0.keys()), klist)

    def test26_duplicate(self):
        # args: newkeys list
        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:  # test key
            with self.assertRaises(TypeError):
                self.map0.duplicate(b, None)
            with self.assertRaises(TypeError):
                self.map0.duplicate(OPREFIX0 + "0", b)

        # self.map0.select = ONLIST0[1]

        with self.assertRaises(KeyError):
            self.map0.duplicate(ONLIST0[0], ONLIST0[3])
            # name already used

        nnext = self.map0.auto_name_next()
        self.assertEqual(nnext, OPREFIX0 + "6")
        c = self.map0.duplicate(ONLIST0[1], None)
        self.assertEqual(c.name, nnext)
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)
        o = self.map0.get(ONLIST0[1])
        self.assertFalse(c == o)  # same name
        pc = c.parameters
        tp = NM0.name + "." + ONLIST0[1]
        self.assertEqual(pc["copy of"], tp)

        c = self.map0.duplicate(ONLIST0[0], "test")
        self.assertEqual(c.name, "test")
        self.assertEqual(len(self.map0), len(ONLIST0) + 2)
        pc = c.parameters
        tp = NM0.name + "." + ONLIST0[0]
        self.assertEqual(pc["copy of"], tp)

    def test27_new(self):
        # args: nmobject
        bad = list(nmu.BADTYPES)
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.new(b)

        with self.assertRaises(KeyError):
            self.map0.new(self.olist0[3])  # already in container

        nnext = self.map0.auto_name_next()
        self.assertEqual(nnext, OPREFIX0 + "6")
        o = NMObject(parent=NM0, name=nnext)
        o2 = self.map0.new(o)
        self.assertEqual(o2.name, nnext)
        self.assertEqual(len(self.map0), len(ONLIST0) + 1)
        """
        old_select = self.map0.select
        o = self.map0.new('test', select=False)
        self.assertEqual(o.name, 'test')
        self.assertEqual(len(self.map0), len(ONLIST0)+2)
        self.assertEqual(old_select, self.map0.select)  # does not change
        """

    def test28_auto_name_prefix(self):
        # args: prefix

        bad = list(nmu.BADTYPES)
        bad.remove(None)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.auto_name_prefix = b

        bad = list(nmu.BADNAMES)
        bad.remove("")
        for b in bad:
            with self.assertRaises(ValueError):
                self.map0.auto_name_prefix = b

        self.assertEqual(self.map0.auto_name_prefix, OPREFIX0)
        self.map0.auto_name_prefix = "Test"
        self.assertEqual(self.map0.auto_name_prefix, "Test")

    def test29_name_seq_format(self):
        # args: seq_format
        bad = list(nmu.BADTYPES)
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.auto_name_seq_format = b
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "string"
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "*"
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "5"

        self.assertEqual(self.map0.auto_name_seq_format, OSEQFORMAT0)
        self.assertEqual(len(self.map0.auto_name_seq_format), 1)
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "01"
        with self.assertRaises(ValueError):
            self.map0.auto_name_seq_format = "A0"
        self.map0.auto_name_seq_format = 0  # ok
        self.assertEqual(self.map0.auto_name_seq_format, "0")
        self.map0.auto_name_seq_format = "000"
        self.assertEqual(self.map0.auto_name_seq_format, "000")

    def test30_name_seq_next_str(self):  # and name_seq_counter
        seq_str = self.map0._auto_name_seq_next_str()
        seq_next_str = str(len(ONLIST0))
        self.assertEqual(seq_str, seq_next_str)
        seq_str = self.map0._auto_name_seq_counter()
        self.assertEqual(seq_str, "0")  # first seq using counter
        self.map0.auto_name_seq_format = "000"
        seq_str = self.map0._auto_name_seq_next_str()
        self.assertEqual(seq_str, "00" + seq_next_str)
        seq_str = self.map0._auto_name_seq_counter()
        self.assertEqual(seq_str, "000")

        seq_str = self.map1._auto_name_seq_next_str()
        seq_next = len(ONLIST1)
        seq_next_str = nmu.CHANNEL_CHARS[seq_next]
        self.assertEqual(seq_str, seq_next_str)
        seq_str = self.map1._auto_name_seq_counter()
        self.assertEqual(seq_str, "A")  # first seq using counter
        self.map1.auto_name_seq_format = "AAA"
        self.assertEqual(self.map1._auto_name_seq_next_str(), "AA" + seq_next_str)
        seq_str = self.map1._auto_name_seq_counter()
        self.assertEqual(seq_str, "AAA")

    def test31_name_seq_counter_increment(self):
        for i in range(10):
            self.assertEqual(self.map0._auto_name_seq_counter(), str(i))
            if i == 9:
                with self.assertRaises(RuntimeError):
                    self.map0._auto_name_seq_counter_increment()
            else:
                self.map0._auto_name_seq_counter_increment()

        self.map0.auto_name_seq_format = "000"
        for i in range(1000):
            if i < 10:
                seq_str = self.map0._auto_name_seq_counter()
                self.assertEqual(seq_str, "00" + str(i))
            elif i < 100:
                seq_str = self.map0._auto_name_seq_counter()
                self.assertEqual(seq_str, "0" + str(i))
            elif i < 1000:
                seq_str = self.map0._auto_name_seq_counter()
                self.assertEqual(seq_str, str(i))
            if i == 999:
                with self.assertRaises(RuntimeError):
                    self.map0._auto_name_seq_counter_increment()
            else:
                self.map0._auto_name_seq_counter_increment()
        self.assertEqual(self.map0._auto_name_seq_counter(), "999")

        for s in nmu.CHANNEL_CHARS:
            self.assertEqual(self.map1._auto_name_seq_counter(), s)
            if s == "Z":
                with self.assertRaises(RuntimeError):
                    self.map1._auto_name_seq_counter_increment()
            else:
                self.map1._auto_name_seq_counter_increment()

        self.map1.auto_name_seq_format = "AA"
        for s1 in nmu.CHANNEL_CHARS:
            for s0 in nmu.CHANNEL_CHARS:
                self.assertEqual(self.map1._auto_name_seq_counter(), s1 + s0)
                if s1 == "Z" and s0 == "Z":
                    with self.assertRaises(RuntimeError):
                        self.map1._auto_name_seq_counter_increment()
                else:
                    self.map1._auto_name_seq_counter_increment()
        self.assertEqual(self.map1._auto_name_seq_counter(), "ZZ")

    def test32_name_next(self):
        name = self.map0.auto_name_next()
        seq_str = str(len(ONLIST0))
        self.assertEqual(name, OPREFIX0 + seq_str)
        name = self.map0.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX0 + seq_str)
        self.map0._auto_name_seq_counter_increment()
        self.map0._auto_name_seq_counter_increment()
        name = self.map0.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX0 + "8")

        name = self.map1.auto_name_next()
        i = len(ONLIST1)
        seq_str = nmu.CHANNEL_CHARS[i]
        self.assertEqual(name, OPREFIX1 + seq_str)
        name = self.map1.auto_name_next(use_counter=True)
        n = 6
        self.assertEqual(name, OPREFIX1 + nmu.CHANNEL_CHARS[n])
        self.map1._auto_name_seq_counter_increment()
        name = self.map1.auto_name_next(use_counter=True)
        n += 1
        self.assertEqual(name, OPREFIX1 + nmu.CHANNEL_CHARS[n])
        for i in range(10):
            self.map1._auto_name_seq_counter_increment()
            n += 1
        name = self.map1.auto_name_next(use_counter=True)
        self.assertEqual(name, OPREFIX1 + nmu.CHANNEL_CHARS[n])

    def test33_select(self):
        # args: key

        bad = list(nmu.BADTYPES)
        bad.remove(None)  # ok
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.selected_name = b

        self.map0.selected_name = ONLIST0[3]
        self.assertEqual(self.map0.selected_name, ONLIST0[3])
        self.assertEqual(self.map0.selected_value, self.olist0[3])
        self.assertEqual(self.map0.get("selected"), self.olist0[3])
        self.assertTrue(self.map0.is_selected(ONLIST0[3]))
        self.assertFalse(self.map0.is_selected(ONLIST0[0]))
        self.assertFalse(self.map0.is_selected(1))

        with self.assertRaises(KeyError):
            self.map0.selected_name = "test"

        self.map0.selected_name = None
        self.assertIsNone(self.map0.selected_name)

        self.map0.selected_name = ONLIST0[3]
        self.map0.pop(ONLIST0[3], auto_confirm="y")
        self.assertIsNone(self.map0.selected_name)

        # print((nmu.quotes('test') + ' this' '1'))
        # namez = 'test'
        # print(f'Encountered duplicate field name: {namez!r}')

    def test34_execute(self):
        # args: key

        bad = list(nmu.BADTYPES)
        bad.remove(None)  # ok
        bad.remove("string")  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.map0.execute_target = b

        bad = list(nmu.BADTYPES)
        for b in bad:
            self.assertFalse(self.map0.is_execute_target(b))

        with self.assertRaises(KeyError):
            self.map0.execute_target = "test"
        self.assertFalse(self.map0.is_execute_target("test"))

        self.map0.execute_target = "selected"

        self.map0.selected_name = ONLIST0[3]
        self.assertEqual(self.map0.selected_value, self.olist0[3])
        self.assertTrue(self.map0.is_execute_target("selected"))
        self.assertTrue(self.map0.is_execute_target(ONLIST0[3]))
        self.assertEqual(self.map0.execute_target, "selected")
        self.assertEqual(self.map0.execute_targets, [self.olist0[3]])

        self.map0.execute_target = ONLIST0[4]
        self.assertEqual(self.map0.execute_targets, [self.olist0[4]])
        self.assertFalse(self.map0.is_execute_target("selected"))
        self.assertFalse(self.map0.is_execute_target(ONLIST0[3]))
        self.assertTrue(self.map0.is_execute_target(ONLIST0[4]))

        self.map0.execute_target = SETS_NLIST0[0]
        self.assertEqual(
            self.map0.execute_targets, [self.olist0[0], self.olist0[2], self.olist0[3]]
        )

        self.map0.execute_target = "all"
        self.assertEqual(self.map0.execute_targets, self.olist0)
        self.assertFalse(self.map0.is_execute_target("selected"))
        self.assertTrue(self.map0.is_execute_target("all"))
        for n in ONLIST0:
            self.assertTrue(self.map0.is_execute_target(n))

    def test35_sets(self):
        # TODO
        pass

if __name__ == '__main__':
    unittest.main()