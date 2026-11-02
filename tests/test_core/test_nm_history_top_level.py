"""History: only the top-level command is recorded, not the calls it makes (#437)."""
import logging
import unittest

import numpy as np

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
            "nm.folders['Demo'].new_dataseries('Long', n_channels=2, n_epochs=200"))
        messages = [e["message"] for e in self.nm.history.buffer]
        self.assertEqual(messages, [commands[0],
                                    "new data series Long: channels A, B; epochs E0..E199"])
        self.assertEqual(self.nm.history.buffer[1]["path"], "Demo")

    def test_folder_commands_name_the_folder_once(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=2, n_points=3)
        folder.add_dataseries_epochs("Rec", 1)
        folder.remove_dataseries_epoch("Rec", "E0")
        for command in self._commands()[1:]:
            self.assertTrue(command.startswith("nm.folders['Demo']."), command)
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
        # Counts, points and fill always; other defaults left out
        self.assertEqual(commands[0], "nm.folders['Demo'].new_dataseries('Rec', n_channels=1, "
                                      "n_epochs=1, n_points=4, fill=1.0, noise=0.5)")
        self.assertIn("fill=2.0, noise=0.1)", commands[1])
        self.assertIn("fill=nan)", commands[2])
        self.assertIn("fill=numpy.zeros)", commands[3])
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
        folder.new_dataseries("Two", n_points=3)
        self.nm.select_keys = {"dataseries": "Two", "folder": "Demo"}
        self.assertEqual(self._commands()[-1],
                         "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Two'}")

    def test_select_keys_command_is_short(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_points=3)
        folder.new_dataseries("Two", n_points=3)
        before = len(self._commands())
        self.nm.select_keys = {"folder": "Demo"}  # already selected: no change
        self.assertEqual(len(self._commands()), before)
        # toolfolder None only when one is selected (clearing it changes something)
        self.nm.select_keys = {"folder": "Demo", "toolfolder": None, "dataseries": "Two"}
        self.assertEqual(self._commands()[-1],
                         "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Two'}")

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
        # Toolfolders but none selected: no toolfolder (None would change nothing)
        demo.toolfolders.new("Stats_0")
        self.nm.select_keys = {"channel": "B"}
        self.assertEqual(last(), "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Rec', "
                                 "'channel': 'B'}")
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
        epochs = "nm.folders['Demo'].dataseries['Rec'].epochs.sets."
        data = "nm.folders['Demo'].data.sets"
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
        path = "nm.folders['Demo'].dataseries['Rec'].epochs.groups."
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
        # Notes only say what the commands don't: cyclic gives group sizes (#441)
        notes = [e["message"] for e in self.nm.history.buffer if e["kind"] == ""]
        self.assertEqual(notes, ["2 groups of 2"])

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

    def test_select_commands_in_a_row_keep_the_last(self):
        """Clicking through epochs: one select command, the last; a select
        that leaves an earlier choice in place keeps it (#441)."""
        demo = self.nm.folders.new("Demo")
        demo.new_dataseries("Rec", n_channels=2, n_epochs=5, n_points=2)
        demo.new_dataseries("Two", n_epochs=2, n_points=2)
        self.nm.command_history.clear()
        for epoch in ("E1", "E2", "E3"):
            self.nm.select_keys = {"folder": "Demo", "dataseries": "Rec", "epoch": epoch}
        self.assertEqual(self._commands(), [
            "nm.select_keys = {'folder': 'Demo', 'dataseries': 'Rec', 'epoch': 'E3'}"])
        self.nm.select_keys = {"channel": "B"}  # the epoch choice stays: kept
        self.nm.select_keys = {"dataseries": "Two", "epoch": "E1"}  # another series: kept
        self.assertEqual(len(self._commands()), 3)
        demo.data.new("Extra")  # a command between: nothing replaced
        self.nm.select_keys = {"dataseries": "Two", "epoch": "E0"}
        self.assertEqual(len(self._commands()), 5)

    def test_select_commands_replay_the_same_selection(self):
        demo = self.nm.folders.new("Demo")
        demo.new_dataseries("Rec", n_channels=2, n_epochs=5, n_points=2)
        demo.new_dataseries("Two", n_epochs=2, n_points=2)
        for keys in ({"dataseries": "Rec", "epoch": "E1"}, {"epoch": "E4"},
                     {"channel": "B"}, {"dataseries": "Two", "epoch": "E1"},
                     {"dataseries": "Rec"}, {"epoch": "E2"}):
            self.nm.select_keys = keys
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)
        self.assertEqual(replayed.select_keys, self.nm.select_keys)
        two = replayed.folders["Demo"].dataseries["Two"]
        self.assertEqual(two.epochs.selected_name, "E1")

    def test_new_data_command_has_its_values(self):
        """data.new records points, fill and scales (not just the name), and
        replays; an array it can't write gets a comment (#441)."""
        folder = self.nm.folders.new("TEST")
        folder.data.new(  # as the New Data dialog does: every field
            "data0", n_points=50, fill=2.5,
            xscale={"start": 0, "delta": 0.1, "label": "", "units": "ms"},
            yscale={"label": "Vm", "units": "mV"})
        folder.data.new("Zeros", nparray=np.zeros(3))
        folder.data.new("Sine", nparray=np.sin(np.arange(5)))
        path = "nm.folders['TEST'].data."
        self.assertEqual(self._commands()[-3:], [
            path + "new('data0', n_points=50, fill=2.5, xscale={'delta': 0.1, 'units': 'ms'}, "
                   "yscale={'label': 'Vm', 'units': 'mV'})",
            path + "new('Zeros', n_points=3, fill=0.0)",
            path + "new('Sine')  # nparray not recorded",
        ])
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        for command in self._commands():
            exec(command, namespace)
        again = replayed.folders["TEST"].data["data0"]
        self.assertEqual(list(again.nparray), [2.5] * 50)
        self.assertEqual((again.xscale.delta, again.xscale.units), (0.1, "ms"))
        self.assertEqual((again.yscale.label, again.yscale.units), ("Vm", "mV"))

    def test_scales_are_one_command_each_and_replay(self):
        """Edit Scales: one set_scales command (no message per value); a
        series' x array names its source (#441)."""
        folder = self.nm.folders.new("Demo1")
        data = folder.data.new("Test0", n_points=4)
        series = folder.new_dataseries("Rec", n_channels=2, n_epochs=2, n_points=4)
        times = folder.data.new("Times", n_points=4)
        times.nparray = np.arange(4.0)
        self.nm.command_history.clear()
        self.nm.history.clear()
        data.set_scales(xstart=-1.0, xdelta=0.1, xunits="ms", ylabel="Voltage")
        series.set_scales(xdelta=0.2, xunits="s")
        series.set_scales(xstart=5.0, ylabel="I", channel="B")
        series.set_xarray(times, channel="A")
        series.set_xarray(None, channel="A")
        self.assertEqual(self._commands(), [
            "nm.folders['Demo1'].data['Test0'].set_scales("
            "xstart=-1.0, xdelta=0.1, xunits='ms', ylabel='Voltage')",
            "nm.folders['Demo1'].dataseries['Rec'].set_scales(xdelta=0.2, xunits='s')",
            "nm.folders['Demo1'].dataseries['Rec'].set_scales(xstart=5.0, ylabel='I', channel='B')",
            "nm.folders['Demo1'].dataseries['Rec'].set_xarray("
            "nm.folders['Demo1'].data['Times'], channel='A')",
            "nm.folders['Demo1'].dataseries['Rec'].set_xarray(None, channel='A')",
        ])
        notes = [e for e in self.nm.history.buffer if e["kind"] == ""]
        self.assertEqual(notes, [])  # the commands say it all
        self.assertEqual(series.channels["B"].xscale.start, 5.0)  # the channel's own too
        # Replayed (the commands since the folder), the scales are the same
        replayed = NMManager(quiet=True)
        namespace = {"nm": replayed}
        exec("nm.folders.new('Demo1')", namespace)
        again = replayed.folders["Demo1"]
        again.data.new("Test0", n_points=4)
        again.new_dataseries("Rec", n_channels=2, n_epochs=2, n_points=4)
        again.data.new("Times", nparray=np.arange(4.0))
        for command in self._commands():
            exec(command, namespace)
        self.assertEqual(again.data["Test0"].xscale, data.xscale)
        self.assertEqual(again.data["Test0"].yscale, data.yscale)
        self.assertEqual(again.data["RecB1"].xscale, folder.data["RecB1"].xscale)
        self.assertEqual(again.data["RecB1"].yscale, folder.data["RecB1"].yscale)

    def test_commands_use_single_quotes(self):
        """Paths are written with repr, so quotes are single, like the
        arguments (#441)."""
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_epochs=20, n_points=2)
        folder.dataseries["Rec"].epochs.groups.assign([f"E{i}" for i in range(20)], 0)
        self.assertEqual(self._commands()[-2:], [
            "nm.folders['Demo'].new_dataseries('Rec', n_channels=1, n_epochs=20, n_points=2, fill=0.0)",
            "nm.folders['Demo'].dataseries['Rec'].epochs.groups.assign("
            "[f'E{i}' for i in range(20)], 0)",
        ])

    def test_channel_runs_are_short(self):
        """Channels A..F, like epochs E0..E9; two channels listed (#441)."""
        folder = self.nm.folders.new("Demo")
        notes = lambda: [e["message"] for e in self.nm.history.buffer if e["kind"] == ""]
        folder.new_dataseries("Six", n_channels=6, n_epochs=10, n_points=2)
        self.assertEqual(notes()[-1], "new data series Six: channels A..F; epochs E0..E9")
        folder.new_dataseries("Two", n_channels=2, n_epochs=1, n_points=2)
        self.assertEqual(notes()[-1], "new data series Two: channels A, B; epochs E0..E0")

    def test_set_notes_name_the_arrays(self):
        """An epoch set's note names the epochs' arrays, and the count (#441)."""
        folder = self.nm.folders.new("Demo")
        epochs = folder.new_dataseries("Rec", n_channels=2, n_epochs=20, n_points=2).epochs
        notes = lambda: [e["message"] for e in self.nm.history.buffer if e["kind"] == ""]
        epochs.sets.add("Set0", ["E0"])
        self.assertEqual(notes()[-1], "Set0 added RecA0, RecB0; n=1")
        self.assertEqual(self.nm.history.buffer[-1]["path"], "Demo.Rec.sets")
        epochs.sets.add("Set0", [f"E{i}" for i in range(2, 12)])  # 20 arrays: the first 6
        self.assertEqual(notes()[-1], "Set0 added RecA2, RecB2, RecA3, RecB3, RecA4, RecB4, "
                                      "… (+14); n=11")
        epochs.sets.remove("Set0", "E3")
        self.assertEqual(notes()[-1], "Set0 removed RecA3, RecB3; n=10")

    def test_long_lists_of_names_are_short(self):
        folder = self.nm.folders.new("Demo")
        folder.new_dataseries("Rec", n_channels=1, n_epochs=1000, n_points=2)
        self.nm.command_history.clear()
        epochs = folder.dataseries["Rec"].epochs
        names = list(epochs.keys())
        epochs.groups.assign_cyclic(names, 3)
        epochs.groups.unassign(names[::2])
        epochs.sets.add("Set0", names[:10] + ["E15"] + names[20:40])
        path = "nm.folders['Demo'].dataseries['Rec'].epochs."
        self.assertEqual(self._commands(), [
            path + "groups.assign_cyclic([f'E{i}' for i in range(1000)], 3)",
            path + "groups.unassign([f'E{i}' for i in range(0, 999, 2)])",  # E0, E2... E998
            path + "sets.add('Set0', [f'E{i}' for i in range(10)] + ['E15'] "
                   "+ [f'E{i}' for i in range(20, 40)])",
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
