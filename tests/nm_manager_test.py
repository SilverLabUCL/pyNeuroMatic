#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:30:25 2023

@author: jason
"""
import unittest

# from pyneuromatic.nm_channel import NMChannelContainer
# from pyneuromatic.nm_data import NMDataContainer
# from pyneuromatic.nm_dataseries import NMDataSeriesContainer
# from pyneuromatic.nm_epoch import NMEpochContainer
# from pyneuromatic.nm_folder import NMFolderContainer
from pyneuromatic.nm_manager import NMManager
# from pyneuromatic.nm_project import NMProjectContainer
from pyneuromatic.nm_project import NMProject
import pyneuromatic.nm_utilities as nmu

QUIET = True
NUMPROJECTS = 1  # 3 # for now, only one project
PROJECTNAME = "ManagerTest"
NUMFOLDERS = 5
DATASERIES = ["data", "avg", "stim"]
NUMDATA = [8, 9, 10]
NUMCHANNELS = [2, 3, 4]
NUMEPOCHS = [5, 6, 7]
ISELECT = 0  # 0, 1, -1


class NMManagerTest(unittest.TestCase):

    def setUp(self):
        self.nm = NMManager(name="NM", project_name=PROJECTNAME, quiet=QUIET)
        ilast = ISELECT == -1
        self.select_values = {}
        self.select_keys = {}

        p = self.nm.project
        pselect = True
        self.select_values["project"] = p
        self.select_keys["project"] = p.name

        for iproject in range(NUMPROJECTS):
            """
            if ilast or iproject == ISELECT:
                p = self.nm.projects.new(select=True)
                self.select_values["project"] = p
                self.select_keys["project"] = p.name
                pselect = True
            else:
                p = self.nm.projects.new(select=False)
                pselect = False
            """
            for ifolder in range(NUMFOLDERS):
                fselect = False
                if ilast or ifolder == ISELECT:
                    f = p.folders.new(select=True)
                    if pselect:
                        self.select_values["folder"] = f
                        self.select_keys["folder"] = f.name
                        fselect = True
                else:
                    f = p.folders.new(select=False)
                jdata = 0  # only one data container
                for idataseries in range(len(DATASERIES)):
                    prefix = DATASERIES[idataseries]
                    # data
                    for idata in range(NUMDATA[idataseries]):
                        n = prefix + str(idata)
                        if ilast or jdata == ISELECT:
                            d = f.data.new(n, select=True)
                            if fselect:
                                self.select_values["data"] = d
                                self.select_keys["data"] = d.name
                        else:
                            d = f.data.new(n, select=False)
                        jdata += 1
                    # dataseries
                    dsselect = False
                    if ilast or idataseries == ISELECT:
                        ds = f.dataseries.new(prefix, select=True)
                        if fselect:
                            self.select_values["dataseries"] = ds
                            self.select_keys["dataseries"] = ds.name
                            dsselect = True
                    else:
                        ds = f.dataseries.new(prefix, select=False)
                    for ichannel in range(NUMCHANNELS[idataseries]):
                        if ilast or ichannel == ISELECT:
                            c = ds.channels.new(select=True)
                            if dsselect:
                                self.select_values["channel"] = c
                                self.select_keys["channel"] = c.name
                        else:
                            c = ds.channels.new(select=False)
                    for iepoch in range(NUMEPOCHS[idataseries]):
                        if ilast or iepoch == ISELECT:
                            e = ds.epochs.new(select=True)
                            if dsselect:
                                self.select_values["epoch"] = e
                                self.select_keys["epoch"] = e.name
                        else:
                            e = ds.epochs.new(select=False)
                    ds.channels.sets.add("set0", ["A", "B"])
                    ds.epochs.sets.add("set0", ["E0", "E1"])
                self.data_set0 = ["data0", "avg0", "stim0"]
                f.data.sets.add("set0", self.data_set0)
                f.dataseries.sets.add("set0", ["data", "avg"])
            p.folders.sets.add("set0", ["folder0", "folder1"])
        # self.nm.projects.sets.add("set0", ["project0", "project1"])

    def test00_init(self):
        # args: name, project_name, quiet

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                NMManager(name=b, quiet=True)

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        bad.remove(None)
        for b in bad:
            with self.assertRaises(TypeError):
                NMManager(project_name=b, quiet=True)

        bad = list(nmu.BADNAMES)
        bad.remove("default")
        for b in bad:
            with self.assertRaises(ValueError):
                NMManager(name=b, quiet=True)
            with self.assertRaises(ValueError):
                NMManager(project_name=b, quiet=True)

        # self.assertTrue(isinstance(self.nm.projects, NMProjectContainer))
        self.assertTrue(isinstance(self.nm.project, NMProject))

    def test01_parameters(self):
        d = self.nm.parameters
        self.assertEqual(d["name"], "NM")
        keys = ["name", "created", "modified", "copy of"]
        self.assertEqual(list(d.keys()), keys)

    def test02_select(self):
        # select_values
        # select_keys
        self.assertEqual(self.nm.select_values, self.select_values)
        self.assertEqual(self.nm.select_keys, self.select_keys)

        with self.assertRaises(AttributeError):
            self.nm.select_values = {}

        bad = list(nmu.BADTYPES)
        bad.remove({})  # ok
        for b in bad:
            with self.assertRaises(TypeError):
                self.nm.select_keys = b

        s1 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "data": "data3",
            "dataseries": "data",
            "channel": "A",
            "epoch": "E3",
        }
        self.nm.select_keys = s1
        self.assertEqual(self.nm.select_keys, s1)

        s2 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "data": "data3",
            "dataseries": "avg",
            "channel": "B",
            "epoch": "E1",
        }
        self.nm.select_keys = s2
        self.assertEqual(self.nm.select_keys, s2)

        self.nm.select_keys = {"dataseries": "data"}
        self.assertEqual(self.nm.select_keys, s1)

        self.nm.select_keys = {"dataseries": "avg"}
        self.assertEqual(self.nm.select_keys, s2)

        with self.assertRaises(KeyError):
            self.nm.select_keys = {"project": "test"}
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"folder": "test"}
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"data": "test"}
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"dataseries": "test"}
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"channel": "test"}
        with self.assertRaises(KeyError):
            self.nm.select_keys = {"epoch": "test"}

    def test03_execute(self):
        # execute_values()
        # execute_keys()
        s = self.nm.select_keys
        s.pop("data")
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, [s])
        if False:
            for e in elist:
                print(e)

        s = self.nm.select_keys
        s.pop("dataseries")
        s.pop("channel")
        s.pop("epoch")
        elist = self.nm.execute_keys(dataseries_priority=False)
        self.assertEqual(elist, [s])
        if False:
            for e in elist:
                print(e)

        """
        self.nm.projects.execute_key = "set0"
        slist = []
        for p in self.nm.projects.sets.get("set0"):
            f = p.folders.select_value
            ds = f.dataseries.select_value
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.projects.execute_key = "all"
        slist = []
        for p in self.nm.projects.values():
            f = p.folders.select_value
            ds = f.dataseries.select_value
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)
        """
        p = self.nm.project

        self.nm.execute_reset_all()
        s = self.nm.select_keys
        s = {
            "project": s["project"],
            "folder": s["folder"],
            "dataseries": s["dataseries"],
            "channel": s["channel"],
            "epoch": s["epoch"],
        }
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, [s])
        if False:
            for e in elist:
                print(e)

        # p = self.nm.projects.select_value
        # p = self.nm.project
        p.folders.execute_key = "set0"
        slist = []
        for f in p.folders.sets.get("set0"):
            ds = f.dataseries.select_value
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        # p = self.nm.projects.select_value
        # p = self.nm.project
        p.folders.execute_key = "all"
        slist = []
        for f in p.folders.values():
            ds = f.dataseries.select_value
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        # p = self.nm.project
        f = p.folders.select_value
        f.data.execute_key = "set0"
        slist = []
        for d in f.data.sets.get("set0"):
            s = {
                "project": p.name,
                "folder": f.name,
                "data": d.name,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=False)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        f.data.execute_key = "all"
        slist = []
        for d in f.data.values():
            s = {
                "project": p.name,
                "folder": f.name,
                "data": d.name,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=False)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        f.dataseries.execute_key = "set0"
        slist = []
        for ds in f.dataseries.sets.get("set0"):
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        f.dataseries.execute_key = "all"
        slist = []
        for ds in f.dataseries.values():
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        ds = f.dataseries.select_value
        ds.channels.execute_key = "set0"
        slist = []
        for c in ds.channels.sets.get("set0"):
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": c.name,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        ds = f.dataseries.select_value
        ds.channels.execute_key = "all"
        slist = []
        for c in ds.channels.values():
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": c.name,
                "epoch": ds.epochs.select_key,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        ds = f.dataseries.select_value
        ds.epochs.execute_key = "set0"
        slist = []
        for e in ds.epochs.sets.get("set0"):
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": e.name,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

        self.nm.execute_reset_all()
        # p = self.nm.projects.select_value
        p = self.nm.project
        f = p.folders.select_value
        ds = f.dataseries.select_value
        ds.epochs.execute_key = "all"
        slist = []
        for e in ds.epochs.values():
            s = {
                "project": p.name,
                "folder": f.name,
                "dataseries": ds.name,
                "channel": ds.channels.select_key,
                "epoch": e.name,
            }
            slist.append(s)
        elist = self.nm.execute_keys(dataseries_priority=True)
        self.assertEqual(elist, slist)
        if False:
            for e in elist:
                print(e)

    def test04_execute_set(self):
        # args: execute
        bad = list(nmu.BADTYPES)
        bad.remove({})
        for b in bad:
            with self.assertRaises(TypeError):
                self.nm.execute_keys_set(b)

        bad = list(nmu.BADTYPES)
        bad.remove("string")
        for b in bad:
            with self.assertRaises(TypeError):
                self.nm.execute_keys_set({b: ""})
            with self.assertRaises(TypeError):
                self.nm.execute_keys_set({"project": b})

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set({"test": ""})
        with self.assertRaises(KeyError):
            self.nm.execute_keys_set({"data": "", "dataseries": ""})

        """
        e0 = {"folder": "folder1",
              "dataseries": "stim",
              "channel": "A",
              "epoch": "E0"
              }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)
        """

        e0 = {
              # "project": "project2",
              "dataseries": "stim",
              "channel": "A",
              "epoch": "E0",
              }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

        e0 = {
              # "project": "project2",
              "folder": "folder1",
              "channel": "A",
              "epoch": "E0"
              }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

        e0 = {
            # "project": "project2",
            "folder": "folder1",
            "data": "stim",
            "channel": "A",
            "epoch": "E0",
            }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

        e0 = {
            # "project": "project2",
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "E0",
            }

        """
        e0.update({"project": "all"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"project": "set0"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"project": "test"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"project": "project2"})
        """

        e0.update({"folder": "all"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"folder": "set0"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"folder": "test"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"folder": "folder1"})

        e0.update({"dataseries": "all"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"dataseries": "set0"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)
        e0.update({"dataseries": "test"})
        with self.assertRaises(ValueError):
            self.nm.execute_keys_set(e0)

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "data": "all",
            # 'dataseries': 'stim',
            "channel": "A",
            "epoch": "E0",
            }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            # 'epoch': 'E0',
            }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "dataseries": "stim",
            # 'channel': 'A',
            "epoch": "E0",
        }

        with self.assertRaises(KeyError):
            self.nm.execute_keys_set(e0)

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "E0",
        }

        elist = self.nm.execute_keys_set(e0)
        self.assertEqual(elist, [e0])
        select = self.nm.select_keys
        select.pop("data")
        self.assertEqual(elist, [select])

        e1 = {
            "project": "select",
            "folder": "select",
            "dataseries": "select",
            "channel": "select",
            "epoch": "select",
        }

        elist = self.nm.execute_keys_set(e1)
        self.assertEqual(elist, [e0])

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "all",
            "epoch": "E0",
        }

        elist = self.nm.execute_keys_set(e0)
        e1 = []
        for i in range(NUMCHANNELS[2]):
            c = nmu.CHANNEL_LIST[i]
            e0c = e0.copy()
            e0c.update({"channel": c})
            e1.append(e0c)
        self.assertEqual(elist, e1)

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "dataseries": "stim",
            "channel": "A",
            "epoch": "all",
        }

        elist = self.nm.execute_keys_set(e0)
        e1 = []
        for i in range(NUMEPOCHS[2]):
            ename = "E" + str(i)
            e0c = e0.copy()
            e0c.update({"epoch": ename})
            e1.append(e0c)
        self.assertEqual(elist, e1)

        e0 = {
            "project": PROJECTNAME,
            "folder": "folder1",
            "data": "set0",
        }

        elist = self.nm.execute_keys_set(e0)

        e1 = []
        for d in self.data_set0:
            e0c = e0.copy()
            e0c.update({"data": d})
            e1.append(e0c)
        self.assertEqual(elist, e1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
