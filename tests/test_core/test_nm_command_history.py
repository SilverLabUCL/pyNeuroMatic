# -*- coding: utf-8 -*-
"""Tests for NMCommandHistory and the module-level command history API."""

import json
import os
import tempfile

import pytest

from pyneuromatic.core.nm_command_history import (
    NMCommandHistory,
    add_command,
    add_nm_command,
    disable_command_history,
    enable_command_history,
    get_command_history,
    set_command_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> NMCommandHistory:
    """Return a new NMCommandHistory isolated from NMHistory (for unit tests)."""
    return NMCommandHistory(quiet=True, log_to_nm_history=False)


# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestNMCommandHistoryInit:
    def test_defaults(self):
        h = _fresh()
        assert h.enabled is True
        assert h.quiet is True
        assert h.buffer == []
        assert len(h) == 0

    def test_enabled_false(self):
        h = NMCommandHistory(enabled=False, quiet=True)
        assert h.enabled is False

    def test_enabled_rejects_non_bool(self):
        with pytest.raises(TypeError):
            NMCommandHistory(enabled=1)

    def test_quiet_rejects_non_bool(self):
        with pytest.raises(TypeError):
            NMCommandHistory(quiet=0)

    def test_enabled_setter(self):
        h = _fresh()
        h.enabled = False
        assert h.enabled is False

    def test_enabled_setter_rejects_non_bool(self):
        h = _fresh()
        with pytest.raises(TypeError):
            h.enabled = "yes"

    def test_quiet_setter(self):
        h = _fresh()
        h.quiet = False
        assert h.quiet is False

    def test_quiet_setter_rejects_non_bool(self):
        h = _fresh()
        with pytest.raises(TypeError):
            h.quiet = 1


# ---------------------------------------------------------------------------
# add / clear / len
# ---------------------------------------------------------------------------


class TestNMCommandHistoryAdd:
    def test_add_records_entry(self):
        h = _fresh()
        entry = "NMMainOpBaseline(x0=0.0, x1=10.0).run_all()"
        h.add(entry)
        assert len(h) == 1
        assert h.buffer[0]["command"] == entry

    def test_add_timestamps_entry(self):
        h = _fresh()
        h.add("cmd1")
        entry = h.buffer[0]
        assert "date" in entry
        assert isinstance(entry["date"], str)
        assert len(entry["date"]) > 10  # ISO-8601 has some length

    def test_add_multiple(self):
        h = _fresh()
        h.add("cmd1")
        h.add("cmd2")
        h.add("cmd3")
        assert len(h) == 3

    def test_disabled_suppresses_add(self):
        h = NMCommandHistory(enabled=False, quiet=True)
        h.add("cmd1")
        assert len(h) == 0

    def test_enable_disable_toggle(self):
        h = _fresh()
        h.enabled = False
        h.add("not recorded")
        assert len(h) == 0
        h.enabled = True
        h.add("recorded")
        assert len(h) == 1

    def test_add_rejects_non_string(self):
        h = _fresh()
        with pytest.raises(TypeError):
            h.add(123)

    def test_buffer_returns_copy(self):
        h = _fresh()
        h.add("cmd1")
        buf = h.buffer
        buf.append({"date": "x", "command": "injected"})
        assert len(h) == 1  # original unchanged

    def test_clear(self):
        h = _fresh()
        h.add("cmd1")
        h.add("cmd2")
        h.clear()
        assert len(h) == 0
        assert h.buffer == []


# ---------------------------------------------------------------------------
# Console output (quiet flag)
# ---------------------------------------------------------------------------


class TestNMCommandHistoryQuiet:
    def test_quiet_suppresses_print(self, capsys):
        h = NMCommandHistory(quiet=True, log_to_nm_history=False)
        h.add("silent_cmd")
        captured = capsys.readouterr()
        assert "silent_cmd" not in captured.out

    def test_not_quiet_prints(self, capsys):
        # quiet=False + log_to_nm_history=False → direct print() to stdout
        h = NMCommandHistory(quiet=False, log_to_nm_history=False)
        h.add("loud_cmd")
        captured = capsys.readouterr()
        assert "loud_cmd" in captured.out


# ---------------------------------------------------------------------------
# print_all
# ---------------------------------------------------------------------------


class TestNMCommandHistoryPrintAll:
    def test_print_all_outputs_commands(self, capsys):
        h = _fresh()
        h.add("cmd_alpha")
        h.add("cmd_beta")
        h.print_all()
        out = capsys.readouterr().out
        assert "cmd_alpha" in out
        assert "cmd_beta" in out

    def test_print_all_last_n(self, capsys):
        h = _fresh()
        h.add("cmd1")
        h.add("cmd2")
        h.add("cmd3")
        h.print_all(last_n=1)
        out = capsys.readouterr().out
        assert "cmd3" in out
        assert "cmd1" not in out


# ---------------------------------------------------------------------------
# to_notebook
# ---------------------------------------------------------------------------


class TestNMCommandHistoryToNotebook:
    def test_to_notebook_creates_file(self):
        pytest.importorskip("nbformat")
        h = _fresh()
        h.add("NMMainOpBaseline(x0=0.0).run_all(folder='f', prefix='R', channels=['A'], epochs=[0])")
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_notebook(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_to_notebook_is_valid_json(self):
        pytest.importorskip("nbformat")
        h = _fresh()
        h.add("cmd1")
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_notebook(path)
            with open(path, "r") as fh:
                nb_dict = json.load(fh)
            assert "cells" in nb_dict
        finally:
            os.unlink(path)

    def test_to_notebook_has_markdown_cell(self):
        pytest.importorskip("nbformat")
        import nbformat
        h = _fresh()
        h.add("cmd1")
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_notebook(path, title="TestHistory")
            with open(path, "r") as fh:
                nb = nbformat.read(fh, as_version=4)
            assert nb.cells[0].cell_type == "markdown"
            assert "TestHistory" in nb.cells[0].source
        finally:
            os.unlink(path)

    def test_to_notebook_import_cell(self):
        pytest.importorskip("nbformat")
        import nbformat
        h = _fresh()
        # Use a real class name so _build_import_block can locate it
        h.add("NMMainOpBaseline(x0=0.0, x1=1.0).run_all(folder='f', prefix='R', channels=['A'], epochs=[0])")
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_notebook(path)
            with open(path, "r") as fh:
                nb = nbformat.read(fh, as_version=4)
            import_cell = nb.cells[1]
            assert import_cell.cell_type == "code"
            assert "NMMainOpBaseline" in import_cell.source
            assert "pyneuromatic" in import_cell.source
        finally:
            os.unlink(path)

    def test_to_notebook_one_cell_per_command(self):
        pytest.importorskip("nbformat")
        import nbformat
        h = _fresh()
        h.add("cmd1")
        h.add("cmd2")
        h.add("cmd3")
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_notebook(path)
            with open(path, "r") as fh:
                nb = nbformat.read(fh, as_version=4)
            # cells: markdown + import + 3 commands = 5 total
            assert len(nb.cells) == 5
            cmd_cells = nb.cells[2:]
            assert all(c.cell_type == "code" for c in cmd_cells)
            assert "cmd1" in nb.cells[2].source
            assert "cmd2" in nb.cells[3].source
            assert "cmd3" in nb.cells[4].source
        finally:
            os.unlink(path)

    def test_to_notebook_empty_buffer(self):
        pytest.importorskip("nbformat")
        import nbformat
        h = _fresh()
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_notebook(path)
            with open(path, "r") as fh:
                nb = nbformat.read(fh, as_version=4)
            # markdown + import = 2 cells, no command cells
            assert len(nb.cells) == 2
        finally:
            os.unlink(path)

    def test_to_notebook_no_nbformat_raises(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nbformat":
                raise ImportError("mocked: nbformat not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        h = _fresh()
        h.add("cmd1")
        with pytest.raises(ImportError, match="nbformat"):
            h.to_notebook("/tmp/test.ipynb")


# ---------------------------------------------------------------------------
# to_script
# ---------------------------------------------------------------------------


class TestNMCommandHistoryToScript:
    def test_to_script_creates_file(self):
        h = _fresh()
        h.add("NMMainOpBaseline(x0=0.0).run_all()")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_script(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_to_script_contains_commands(self):
        h = _fresh()
        h.add("cmd_alpha")
        h.add("cmd_beta")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_script(path)
            with open(path, "r") as fh:
                content = fh.read()
            assert "cmd_alpha" in content
            assert "cmd_beta" in content
        finally:
            os.unlink(path)

    def test_to_script_has_import(self):
        h = _fresh()
        h.add("NMMainOpBaseline(x0=0.0, x1=1.0).run_all(folder='f', prefix='R', channels=['A'], epochs=[0])")
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            path = tmp.name
        try:
            h.to_script(path)
            with open(path, "r") as fh:
                content = fh.read()
            assert "NMMainOpBaseline" in content
            assert "from pyneuromatic" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


class TestModuleLevelAPI:
    def setup_method(self):
        """Replace global instance with a fresh isolated one before each test."""
        set_command_history(NMCommandHistory(quiet=True, log_to_nm_history=False))

    def test_add_command_logs_to_global(self):
        add_command("global_cmd")
        assert len(get_command_history()) == 1
        assert get_command_history().buffer[0]["command"] == "global_cmd"

    def test_disable_command_history(self):
        disable_command_history()
        add_command("not_recorded")
        assert len(get_command_history()) == 0

    def test_enable_command_history(self):
        disable_command_history()
        enable_command_history()
        add_command("recorded")
        assert len(get_command_history()) == 1

    def test_get_command_history_returns_instance(self):
        h = get_command_history()
        assert isinstance(h, NMCommandHistory)

    def test_set_command_history_replaces_global(self):
        custom = NMCommandHistory(quiet=True)
        set_command_history(custom)
        add_command("in_custom")
        assert len(custom) == 1

    def test_set_command_history_rejects_non_instance(self):
        with pytest.raises(TypeError):
            set_command_history("not_an_instance")


# ---------------------------------------------------------------------------
# log_to_nm_history
# ---------------------------------------------------------------------------


class TestLogToNMHistory:
    def test_default_is_true(self):
        h = NMCommandHistory(quiet=True)
        assert h.log_to_nm_history is True

    def test_init_false(self):
        h = NMCommandHistory(quiet=True, log_to_nm_history=False)
        assert h.log_to_nm_history is False

    def test_init_rejects_non_bool(self):
        with pytest.raises(TypeError):
            NMCommandHistory(quiet=True, log_to_nm_history=1)

    def test_setter(self):
        h = NMCommandHistory(quiet=True, log_to_nm_history=False)
        h.log_to_nm_history = True
        assert h.log_to_nm_history is True

    def test_setter_rejects_non_bool(self):
        h = _fresh()
        with pytest.raises(TypeError):
            h.log_to_nm_history = "yes"

    def test_add_calls_nm_history_when_enabled(self):
        """When log_to_nm_history=True, add() forwards to nm_history.history()."""
        import pyneuromatic.core.nm_history as nmh
        calls = []

        def _fake_history(message, **kwargs):
            calls.append(message)
            return message

        original = nmh.history
        nmh.history = _fake_history
        try:
            h = NMCommandHistory(quiet=True, log_to_nm_history=True)
            h.add("test_cmd_forward")
            assert "test_cmd_forward" in calls
        finally:
            nmh.history = original

    def test_add_skips_nm_history_when_disabled(self):
        """When log_to_nm_history=False, nm_history.history() is not called."""
        import pyneuromatic.core.nm_history as nmh
        calls = []

        def _fake_history(message, **kwargs):
            calls.append(message)
            return message

        original = nmh.history
        nmh.history = _fake_history
        try:
            h = NMCommandHistory(quiet=True, log_to_nm_history=False)
            h.add("should_not_forward")
            assert len(calls) == 0
        finally:
            nmh.history = original

    def test_add_passes_path_none_to_nm_history(self):
        """path='NONE' is passed so no function-path prefix appears."""
        import pyneuromatic.core.nm_history as nmh
        received = {}

        def _fake_history(message, **kwargs):
            received.update(kwargs)
            return message

        original = nmh.history
        nmh.history = _fake_history
        try:
            h = NMCommandHistory(quiet=True, log_to_nm_history=True)
            h.add("cmd_path_check")
            assert received.get("path") == "NONE"
        finally:
            nmh.history = original

    def test_quiet_does_not_print_when_log_to_nm_history_true(self, capsys):
        """When log_to_nm_history=True the direct print() is suppressed."""
        import pyneuromatic.core.nm_history as nmh

        def _noop(message, **kwargs):
            return message

        original = nmh.history
        nmh.history = _noop
        try:
            h = NMCommandHistory(quiet=False, log_to_nm_history=True)
            h.add("no_direct_print")
            captured = capsys.readouterr()
            assert "no_direct_print" not in captured.out
        finally:
            nmh.history = original


# ---------------------------------------------------------------------------
# nm_name and add_nm
# ---------------------------------------------------------------------------


class TestNMName:
    def test_default_nm_name(self):
        h = _fresh()
        assert h.nm_name == "nm"

    def test_custom_nm_name_constructor(self):
        h = NMCommandHistory(quiet=True, log_to_nm_history=False, nm_name="manager")
        assert h.nm_name == "manager"

    def test_nm_name_setter(self):
        h = _fresh()
        h.nm_name = "my_nm"
        assert h.nm_name == "my_nm"

    def test_nm_name_rejects_empty(self):
        with pytest.raises(TypeError):
            NMCommandHistory(quiet=True, log_to_nm_history=False, nm_name="")

    def test_nm_name_rejects_non_string(self):
        with pytest.raises(TypeError):
            NMCommandHistory(quiet=True, log_to_nm_history=False, nm_name=42)

    def test_add_nm_prepends_nm_name_and_dot(self):
        h = _fresh()
        h.add_nm("tool_add('main')")
        assert h.buffer[0]["command"] == "nm.tool_add('main')"

    def test_add_nm_custom_name(self):
        h = NMCommandHistory(quiet=True, log_to_nm_history=False, nm_name="manager")
        h.add_nm("run_tool('main')")
        assert h.buffer[0]["command"] == "manager.run_tool('main')"

    def test_add_nm_command_module_level(self):
        set_command_history(NMCommandHistory(quiet=True, log_to_nm_history=False))
        add_nm_command("run_reset_all()")
        assert get_command_history().buffer[0]["command"] == "nm.run_reset_all()"


