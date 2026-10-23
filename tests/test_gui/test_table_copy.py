"""Tests for selecting and copying in the Table panel."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtGui, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui

Flag = QtCore.QItemSelectionModel.SelectionFlag


@pytest.fixture
def app(qtbot):
    """Series Record: 3 epochs x 3 points, x from 0 in 0.5 ms steps."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    for epoch in range(3):
        folder.data.new(f"RecordA{epoch}",
                        nparray=np.array([epoch, epoch + 0.123456789, np.nan]),
                        xscale={"start": 0.0, "delta": 0.5, "units": "ms"})
    series = folder.sync_dataseries("Record", select=True)
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.selection_model.update(epoch_set="All")
    return window


def _copied(app):
    app.table_panel.copy_action.trigger()
    return QtWidgets.QApplication.clipboard().text().splitlines()


def test_menu(app):
    panel = app.table_panel
    texts = [a.text() for a in panel.table_menu.actions() if a.text()]
    assert texts == ["Copy", "Copy All", "Show in Browser",
                     "Include Column Names", "Include x Values / Row Names"]
    assert panel.copy_action.shortcut() == QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy)
    assert panel.names_action.isChecked() and panel.x_action.isChecked()


def test_copy_columns_with_names_and_x(app):
    view = app.table_panel.channel_table.view
    view.clearSelection()
    view.selectColumn(1)  # a header click: RecordA0
    view.selectionModel().select(view.model().index(0, 3), Flag.Select | Flag.Columns)  # ⌘ RecordA2
    assert _copied(app) == [
        "x (ms)\tRecordA0\tRecordA2",
        "0\t0\t2",
        "0.5\t0.123456789\t2.123456789",  # full precision, not the 6 digits shown
        "1\tNaN\tNaN",  # NaN as NaN, whatever the table shows
    ]


def test_copy_options(app):
    panel = app.table_panel
    view = panel.channel_table.view
    view.clearSelection()
    view.selectColumn(2)
    panel.names_action.setChecked(False)
    panel.x_action.setChecked(False)
    assert _copied(app) == ["1", "1.123456789", "NaN"]


def test_copy_a_block_of_cells(app):
    view = app.table_panel.channel_table.view
    model = view.model()
    selection = QtCore.QItemSelection(model.index(0, 1), model.index(1, 2))
    view.selectionModel().select(selection, Flag.ClearAndSelect)
    assert _copied(app) == ["0\t1", "0.123456789\t1.123456789"]  # no names for a block


def test_copy_all(app):
    app.table_panel.copy_all_action.trigger()
    lines = QtWidgets.QApplication.clipboard().text().splitlines()
    assert lines[0] == "x (ms)\tRecordA0\tRecordA1\tRecordA2"
    assert len(lines) == 4


def test_several_cells_can_be_selected(app):
    view = app.table_panel.channel_table.view
    assert view.selectionMode() == QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection


def test_copy_stats_cells_with_row_names(qtbot):
    from tests.test_gui.test_results_view import _stats_window

    _manager, _folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    panel = window.table_panel
    panel.show_tool("Stats")
    tables = window.stats_tab.stats2.tables
    tables.arrays_table.clearSelection()
    tables.arrays_table.selectColumn(1)  # max_y
    lines = _copied(window)
    assert lines[0] == "\tmax_y"
    assert lines[1] == "RecordA0\t-50"


def test_show_in_browser_shows_the_current_array(app):
    panel = app.table_panel
    app.selection_model.update(epoch=app.selection_model.selection["dataseries"].epochs["E1"])
    panel.browser_action.trigger()
    assert app.tool_rail.current_tool_name == "Browser"
    browser = app.browser_widget
    assert browser.tree.currentIndex().internalPointer().name == "RecordA1"


def test_show_in_browser_shows_the_results_folder(qtbot):
    from tests.test_gui.test_results_view import _stats_window

    _manager, folder, window = _stats_window(qtbot)
    window.stats_tab.run_button.click()
    panel = window.table_panel
    panel.show_tool("Stats")
    panel.browser_action.trigger()
    shown = window.browser_widget.tree.currentIndex().internalPointer()
    assert shown is window.stats_tab.stats_tool.toolfolder


# --- right-click: the ⋯ menu -----------------------------------------------

def _right_click(monkeypatch, view, pos, header=False):
    """Emit a right-click at *pos*; the menu it opens (not shown)."""
    shown = []
    monkeypatch.setattr(QtWidgets.QMenu, "exec", lambda menu, *a, **k: shown.append(menu))
    target = view.horizontalHeader() if header else view
    target.customContextMenuRequested.emit(pos)
    return shown[0] if shown else None


def _cell_pos(view, row, column):
    return view.visualRect(view.model().index(row, column)).center()


def _header_pos(view, column):
    header = view.horizontalHeader()
    return QtCore.QPoint(header.sectionViewportPosition(column) + 5, 5)


def test_right_click_a_cell_selects_it(app, qtbot, monkeypatch):
    app.show()
    panel = app.table_panel
    view = panel.channel_table.view
    view.clearSelection()
    menu = _right_click(monkeypatch, view, _cell_pos(view, 1, 2))
    assert menu is panel.table_menu
    assert [(i.row(), i.column()) for i in view.selectedIndexes()] == [(1, 2)]
    assert _copied(app) == ["1.123456789"]


def test_right_click_keeps_the_selection(app, qtbot, monkeypatch):
    app.show()
    view = app.table_panel.channel_table.view
    view.clearSelection()
    view.selectColumn(1)
    _right_click(monkeypatch, view, _cell_pos(view, 0, 1))  # inside the selection
    assert _copied(app)[0] == "x (ms)\tRecordA0"


def test_right_click_a_header_selects_its_column(app, qtbot, monkeypatch):
    app.show()
    panel = app.table_panel
    view = panel.channel_table.view
    view.clearSelection()
    menu = _right_click(monkeypatch, view, _header_pos(view, 3), header=True)
    assert menu is panel.table_menu
    assert _copied(app)[0] == "x (ms)\tRecordA2"


def test_right_click_the_stats_tables(qtbot, monkeypatch):
    from tests.test_gui.test_results_view import _stats_window

    _manager, _folder, window = _stats_window(qtbot)
    window.show()
    window.stats_tab.run_button.click()
    panel = window.table_panel
    panel.show_tool("Stats")
    tables = window.stats_tab.stats2.tables
    for table in (tables.arrays_table, tables.summary_table):
        assert _right_click(monkeypatch, table, _header_pos(table, 1), header=True) \
            is panel.table_menu
