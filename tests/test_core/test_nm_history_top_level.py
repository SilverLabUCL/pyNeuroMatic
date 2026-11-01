"""History: only the top-level command is recorded, not the calls it makes (#437)."""
import logging
import unittest

import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_history as nmh
from pyneuromatic.core.nm_command_history import NMCommandHistory, records_command
from pyneuromatic.core.nm_history import NMHistory
from pyneuromatic.core.nm_manager import NMManager


class _Capture(logging.Handler):
    """Every record, DEBUG included (as the GUI's History panel sees them)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _Thing:
    """Methods that record commands, one calling the other."""

    @records_command
    def inner(self):
        nmh.history("inner note", path="NONE")
        nmch.add_command("thing.inner()")

    @records_command
    def outer(self):
        nmh.history("outer note", path="NONE")
        self.inner()
        self.inner()
        nmch.add_command("thing.outer()")

    @records_command
    def warns(self):
        self.inner_warns()
        nmch.add_command("thing.warns()")

    @records_command
    def inner_warns(self):
        nmh.alert("careful", path="NONE")
        nmch.add_command("thing.inner_warns()")

    @records_command
    def fails(self):
        nmh.history("before failing", path="NONE")
        raise RuntimeError("failed")


class TestTopLevelOnly(unittest.TestCase):

    def setUp(self):
        self._old = (nmh._nm_history, nmch.get_command_history())
        self.history = NMHistory(quiet=True)
        nmh.set_history(self.history)
        self.commands = NMCommandHistory(enabled=True, quiet=True)
        nmch.set_command_history(self.commands)
        self.capture = _Capture()
        logging.getLogger("pyneuromatic").addHandler(self.capture)
        self.thing = _Thing()

    def tearDown(self):
        logging.getLogger("pyneuromatic").removeHandler(self.capture)
        nmh.set_history(self._old[0])
        nmch.set_command_history(self._old[1])

    def _messages(self):
        return [(e["kind"] or e["level"], e["message"]) for e in self.history.buffer]

    def test_only_the_top_level_command_is_recorded(self):
        self.thing.outer()
        self.assertEqual([c["command"] for c in self.commands.buffer], ["thing.outer()"])
        self.thing.inner()  # called directly: it is the top level
        self.assertEqual([c["command"] for c in self.commands.buffer],
                         ["thing.outer()", "thing.inner()"])

    def test_the_command_comes_before_its_notes(self):
        self.thing.outer()
        self.assertEqual(self._messages(), [("command", "thing.outer()"), ("INFO", "outer note")])

    def test_nested_entries_are_debug(self):
        self.thing.outer()
        debug = [(r.getMessage(), getattr(r, "kind", "")) for r in self.capture.records
                 if r.levelno == logging.DEBUG]
        self.assertEqual(debug, [("inner note", ""), ("thing.inner()", "command")] * 2)
        # Not in the buffer, which keeps INFO and up (the session's record)
        self.assertNotIn("inner note", [m for _k, m in self._messages()])

    def test_nested_warnings_keep_their_level(self):
        self.thing.warns()
        self.assertEqual(self._messages(), [("command", "thing.warns()"), ("WARNING", "careful")])

    def test_a_failure_writes_what_was_held(self):
        with self.assertRaises(RuntimeError):
            self.thing.fails()
        self.assertEqual(self._messages(), [("INFO", "before failing")])
        self.assertEqual(nmch.command_depth(), 0)
        self.thing.inner()  # and the next call is the top level again
        self.assertEqual(self.commands.buffer[-1]["command"], "thing.inner()")

    def test_disabled_command_history_still_writes_notes(self):
        self.commands.enabled = False
        self.thing.outer()
        self.assertEqual(self._messages(), [("INFO", "outer note")])


class TestCoreCommands(unittest.TestCase):
    """The core's own commands: one per action, replayable."""

    def setUp(self):
        self.nm = NMManager(quiet=True, command_history=True)
        self.nm.history.clear()
        self.nm.command_history.clear()

    def _commands(self):
        return [c["command"] for c in self.nm.command_history.buffer]

    def test_new_series_is_one_command_and_one_note(self):
        folder = self.nm.folders.new("Demo")
        self.nm.command_history.clear()
        self.nm.history.clear()
        folder.new_dataseries("Long", n_channels=2, n_epochs=200, n_points=3)
        commands = self._commands()
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].startswith(
            "nm.folders[\"Demo\"].new_dataseries('Long', n_channels=2, n_epochs=200"))
        messages = [e["message"] for e in self.nm.history.buffer]
        self.assertEqual(messages, [commands[0],
                                    "new dataseries 'Long': channels A, B; epochs E0..E199"])

    def test_folder_commands_name_the_folder_once(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=2, n_points=3)
        folder.add_dataseries_epochs("Rec", 1)
        folder.remove_dataseries_epoch("Rec", "E0")
        for command in self._commands()[1:]:
            self.assertTrue(command.startswith('nm.folders["Demo"].'), command)
            self.assertNotIn('["Demo"]["Demo"]', command)

    def test_building_objects_writes_no_history(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=3, n_points=3)
        messages = " ".join(e["message"] for e in self.nm.history.buffer)
        for noise in ("changed name from NMObject0", "changed prefix", "changed seq_format",
                      "set 'Set0': added"):
            self.assertNotIn(noise, messages)

    def test_the_commands_replay(self):
        demo = self.nm.folders.new("Demo")
        other = self.nm.folders.new("Other")
        demo.new_dataseries("Rec", n_channels=2, n_epochs=5, n_points=3)
        demo.add_dataseries_channels("Rec", 1)
        demo.add_dataseries_epochs("Rec", 2)
        demo.remove_dataseries_epoch("Rec", "E1", delete_data=True)
        demo.renumber_dataseries_epochs("Rec")
        demo.copy_dataseries("Rec", new_prefix="Copy", folder=other)
        demo.data.new("Extra")
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)

        def shape(manager):
            return {name: (sorted(f.data.keys()), sorted(f.dataseries.keys()))
                    for name, f in manager.folders.items()}

        self.assertEqual(shape(replayed), shape(self.nm))

    def test_fill_and_noise_are_written(self):
        import numpy as np

        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_points=4, fill=1.0, noise=0.5)
        folder.add_dataseries_epochs("Rec", 1, fill=2.0, noise=0.1)
        folder.add_dataseries_channels("Rec", 1, fill=np.nan)
        folder.new_dataseries("Zero", n_points=3, fill=np.zeros)
        folder.new_dataseries("Lam", n_points=3, fill=lambda n: np.ones(n))
        commands = self._commands()[1:]
        self.assertIn("fill=1.0, noise=0.5,", commands[0])
        self.assertIn("fill=2.0, noise=0.1)", commands[1])
        self.assertIn("fill=nan,", commands[2])
        self.assertIn("fill=numpy.zeros,", commands[3])
        self.assertTrue(commands[4].endswith("  # fill was a function, not recorded"))
        self.assertNotIn("<function", " ".join(commands))
        values = folder.data["RecA0"].nparray
        self.assertNotEqual(len(set(values)), 1)  # noise
        self.assertAlmostEqual(float(np.mean(values)), 1.0, delta=1.5)
        # The script export imports numpy for fill=numpy.zeros
        from pyneuromatic.core.nm_command_history import _build_import_block
        self.assertIn("import numpy", _build_import_block(self.nm.command_history.buffer))

    def test_select_keys_command_in_hierarchy_order(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_points=3)
        self.nm.select_keys = {"dataseries": "Rec", "toolfolder": None, "folder": "Demo"}
        self.assertEqual(self._commands()[-1],
                         "nm.select_keys = {'folder': 'Demo', 'toolfolder': None, 'dataseries': 'Rec'}")

    def test_select_keys_command_has_the_path_above(self):
        demo = self.nm.folders.new("Demo")
        demo.new_dataseries("Rec", n_channels=2, n_epochs=3, n_points=3)
        demo.data.new("Extra")
        self.nm.select_keys = {"folder": "Demo", "dataseries": "Rec"}
        last = lambda: self._commands()[-1]
        self.nm.select_keys = {"channel": "B"}
        self.assertEqual(last(), "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Rec', "
                                 "'channel': 'B'}")
        self.nm.select_keys = {"epoch": "E2"}  # not the channel: only the path above
        self.assertEqual(last(), "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Rec', "
                                 "'epoch': 'E2'}")
        self.nm.select_keys = {"data": "Extra"}  # the data branch: not the series
        self.assertEqual(last(), "nm.select_keys = {'folder': 'Demo', 'data': 'Extra'}")
        # A tier given anchors the path: nothing added above it
        self.nm.select_keys = {"folder": "Demo", "channel": "A"}
        self.assertEqual(last(), "nm.select_keys = {'folder': 'Demo', 'channel': 'A'}")
        # With toolfolders, the toolfolder is part of the path (data tiers are its contents)
        demo.toolfolders.new("Stats_0")
        self.nm.select_keys = {"toolfolder": None}
        self.nm.select_keys = {"channel": "A"}
        self.assertEqual(last(), "nm.select_keys = {'folder': 'Demo', 'toolfolder': None, "
                                 "'dataseries': 'Rec', 'channel': 'A'}")
        # The commands replay: the same selection
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)
        self.assertEqual(replayed.select_keys, self.nm.select_keys)

    def test_set_commands(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=4, n_points=3)
        folder.data.new("Extra")
        self.nm.command_history.clear()
        series = folder.dataseries["Rec"]
        epoch_sets = series.epochs.sets
        epoch_sets.add("Set0", ["E0", series.epochs["E1"]])  # names or objects
        epoch_sets.remove("Set0", "E1")
        epoch_sets.define_or("Set2", "Set0", "Set1")
        epoch_sets.rename("Set2", "Set5")
        epoch_sets.duplicate("Set0", "Set7")
        epoch_sets.pop("Set7")
        epoch_sets.empty("Set1")
        folder.data.sets.add("Set0", ["RecA0", "Extra"])
        folder.data.sets["Set1"] = ["Extra"]
        epochs = 'nm.folders["Demo"].dataseries["Rec"].epochs.sets.'
        data = 'nm.folders["Demo"].data.sets'
        self.assertEqual(self._commands(), [
            epochs + "add('Set0', ['E0', 'E1'])",
            epochs + "remove('Set0', ['E1'])",
            epochs + "define_or('Set2', 'Set0', 'Set1')",
            epochs + "rename('Set2', 'Set5')",
            epochs + "duplicate('Set0', 'Set7')",
            epochs + "pop('Set7')",
            epochs + "empty('Set1')",
            data + ".add('Set0', ['RecA0', 'Extra'])",
            data + "['Set1'] = ['Extra']",
        ])

    def test_set_commands_replay(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=4, n_points=3)
        epoch_sets = folder.dataseries["Rec"].epochs.sets
        epoch_sets.add("Set0", ["E0", "E1", "E2"])
        epoch_sets.remove("Set0", "E1")
        epoch_sets.add("Set1", ["E3"])
        epoch_sets.define_and("Set4", "Set0", "Set1")
        folder.data.sets.add("Set0", ["RecA0"])
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)

        def sets_of(sets):
            return {k: (sets.get_items(k, get_equation=True) if sets.is_equation(k)
                        else sets.get_items(k, get_keys=True)) for k in sets.keys()}

        again = replayed.folders["Demo"]
        self.assertEqual(sets_of(again.dataseries["Rec"].epochs.sets), sets_of(epoch_sets))
        self.assertEqual(sets_of(again.data.sets), sets_of(folder.data.sets))

    def test_group_commands(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=6, n_points=3)
        self.nm.command_history.clear()
        self.nm.history.clear()
        groups = folder.dataseries["Rec"].epochs.groups
        groups.add_group([0, 1, 2])
        groups.assign(["E0", "E1", "E2"], 1)  # one command for them all
        groups.assign("E3", 2)
        groups.unassign(["E1", "E2"])
        groups.assign_cyclic(["E0", "E1", "E2", "E3"], 2, block=2)
        groups.clear_group(1)
        groups.delete_group([0, 2])
        groups.clear()
        path = 'nm.folders["Demo"].dataseries["Rec"].epochs.groups.'
        self.assertEqual(self._commands(), [
            path + "add_group([0, 1, 2])",
            path + "assign(['E0', 'E1', 'E2'], 1)",
            path + "assign('E3', 2)",
            path + "unassign(['E1', 'E2'])",
            path + "assign_cyclic(['E0', 'E1', 'E2', 'E3'], 2, block=2)",  # not each assign
            path + "clear_group(1)",
            path + "delete_group([0, 2])",
            path + "clear()",
        ])
        notes = [e["message"] for e in self.nm.history.buffer if e["kind"] == ""]
        self.assertIn("groups: 'E0', 'E1', 'E2' → 1", notes)
        self.assertEqual(sum("→" in n for n in notes), 2)  # cyclic: one note, not each

    def test_group_commands_replay(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=8, n_points=3)
        groups = folder.dataseries["Rec"].epochs.groups
        groups.add_group([0, 1, 2, 3])
        groups.assign_cyclic([f"E{i}" for i in range(8)], 3)
        groups.unassign(["E0", "E5"])
        groups.assign(["E1", "E2"], 3)
        groups.delete_group(2)
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)
        again = replayed.folders["Demo"].dataseries["Rec"].epochs.groups
        self.assertEqual(again, groups)

    def test_long_lists_of_names_are_short(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=1000, n_points=2)
        self.nm.command_history.clear()
        epochs = folder.dataseries["Rec"].epochs
        names = list(epochs.keys())
        epochs.groups.assign_cyclic(names, 3)
        epochs.groups.unassign(names[::2])
        epochs.sets.add("Set0", names[:10] + ["E15"] + names[20:40])
        path = 'nm.folders["Demo"].dataseries["Rec"].epochs.'
        self.assertEqual(self._commands(), [
            path + 'groups.assign_cyclic([f"E{i}" for i in range(1000)], 3, block=1)',
            path + 'groups.unassign([f"E{i}" for i in range(0, 999, 2)])',  # E0, E2... E998
            path + 'sets.add(\'Set0\', [f"E{i}" for i in range(10)] + [\'E15\'] '
                   '+ [f"E{i}" for i in range(20, 40)])',
        ])
        # They replay
        replayed = NMManager(quiet=True)
        replayed.folders.new("Demo").new_dataseries("Rec", n_channels=1, n_epochs=1000,
                                                     n_points=2)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)
        again = replayed.folders["Demo"].dataseries["Rec"].epochs
        self.assertEqual(again.groups, epochs.groups)
        self.assertEqual(again.sets.get_items("Set0", get_keys=True),
                         epochs.sets.get_items("Set0", get_keys=True))

    def test_sets_outside_the_manager_record_nothing(self):
        from pyneuromatic.core.nm_sets import NMSets
        sets = NMSets(nmobjects={})
        sets.add("Set0", [])
        self.assertEqual(self._commands(), [])

    def test_noise_is_checked(self):
        folder = self.nm.folders.new("Demo")
        with self.assertRaises(ValueError):
            folder.new_dataseries("Rec", noise=-1.0)
        with self.assertRaises(TypeError):
            folder.new_dataseries("Rec", noise="lots")


if __name__ == "__main__":
    unittest.main()
