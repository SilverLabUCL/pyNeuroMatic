"""The shared results panel (#525), as another tool's (e.g. Spike2) would use it."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.results_panel import ResultsPanel

pytestmark = pytest.mark.gui


class SpikePanel(ResultsPanel):
    """A tool's panel: its subfolders and their SP_ arrays."""

    TOOLFOLDER_PREFIX = "Spike_"
    TOOL_NAME = "Spike"
    PANEL_NAME = "Spike2"
    OBJECT_PREFIX = "spike2"
    EXAMPLE_ARRAY = "SP_RX"

    def __init__(self, manager, parent=None):
        super().__init__(manager, parent)
        self.refresh()

    def _array_names(self):
        toolfolder = self.toolfolder()
        if toolfolder is None:
            return []
        return [name for name in toolfolder.data.keys() if name.startswith("SP_")]

    def default_array(self):
        return "SP_count" if "SP_count" in self._array_names() else None

    def selected_plot(self):
        name = self.selected_array()
        if name is None:
            return None
        y = self.toolfolder().data.get(name).nparray
        return (f"{self.toolfolder().name}: {name}", y, None, "index", name, "points")


@pytest.fixture
def manager():
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    for name in ("Spike_Record_A_All_0", "Spike_Record_A_All_1", "Stats_Record_A_All_0"):
        sub = folder.toolfolders.new(name)
        sub.data.new("SP_RX", nparray=np.array([1.0, 2.0]))
        sub.data.new("SP_count", nparray=np.array([2.0]))
        sub.data.new("Other", nparray=np.array([0.0]))
    return manager


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_lists_the_tools_subfolders_and_their_arrays(qtbot, manager):
    panel = SpikePanel(manager)
    qtbot.addWidget(panel)
    assert _items(panel.folder_combo) == ["Spike_Record_A_All_0", "Spike_Record_A_All_1"]
    assert panel.toolfolder().name == "Spike_Record_A_All_1"  # the newest
    assert _items(panel.array_combo) == ["SP_RX", "SP_count"]
    assert panel.selected_array() == "SP_count"  # its default
    assert panel.has_results() and panel.message_label.isHidden()
    assert panel.folder_combo.objectName() == "spike2Folder"
    # The Array ⋯ menu: Plot | Table | Show in Browser, with no tool items
    texts = [a.text() or "---" for a in panel.plot_menu.actions()]
    assert texts == ["Plot", "---", "Table", "---", "Show in Browser"]
    assert "Spike2 tables" in panel.table_action.toolTip()
    assert "keeps Spike_" in panel.rename_folder_action.toolTip()


def test_plot_and_table_requests(qtbot, manager):
    panel = SpikePanel(manager)
    qtbot.addWidget(panel)
    plots, tables = [], []
    panel.plot_requested.connect(lambda *args: plots.append(args))
    panel.table_requested.connect(lambda: tables.append(True))
    panel.select_array("SP_RX")
    panel.plot_action.trigger()
    assert plots[-1][0] == "Spike_Record_A_All_1: SP_RX"
    assert list(plots[-1][1]) == [1.0, 2.0]
    panel.table_action.trigger()
    assert tables == [True]


def test_no_results_says_so(qtbot):
    manager = NMManager(quiet=True)
    manager.folders.new("Empty", select=True)
    panel = SpikePanel(manager)
    qtbot.addWidget(panel)
    assert not panel.has_results()
    assert panel.message_label.text() == "No saved Spike results in Empty."
    assert not panel.plot_action.isEnabled() and not panel.table_action.isEnabled()


# --- the shared results tables (#525) ---------------------------------------

from pyneuromatic.gui.results_tables import ResultsTables  # noqa: E402


class SpikeTables(ResultsTables):
    """A tool's tables: its per-epoch arrays, rows from SP_epoch_num."""

    def _column_names(self, toolfolder, key, **_options):
        return [n for n in ("SP_count", "SP_rate") if n in toolfolder.data]

    def _rows(self, toolfolder, key):
        epochs = toolfolder.data.get("SP_epoch_num")
        return [f"E{int(n)}" for n in epochs.nparray] if epochs is not None else []

    def _header(self, name, key):
        return name[len("SP_"):]


def test_tables_show_a_tools_arrays_with_a_summary(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=True)
    sub = folder.toolfolders.new("Spike_Record_A_All_0")
    sub.data.new("SP_count", nparray=np.array([2.0, 4.0, np.nan]))
    sub.data.new("SP_rate", nparray=np.array([20.0, 40.0, 0.0]))
    sub.data.new("SP_epoch_num", nparray=np.array([0.0, 1.0, 3.0]))
    tables = SpikeTables()
    qtbot.addWidget(tables)
    tables.show_results(sub)
    assert tables.title_label.text() == "Spike_Record_A_All_0"
    assert tables.columns == ["SP_count", "SP_rate"]
    header = [tables.arrays_table.horizontalHeaderItem(c).text() for c in range(2)]
    assert header == ["count", "rate"]
    rows = [tables.arrays_table.verticalHeaderItem(r).text() for r in range(3)]
    assert rows == ["E0", "E1", "E3"]
    # The summary: NaNs left out
    labels = [tables.summary_table.verticalHeaderItem(r).text()
              for r in range(tables.summary_table.rowCount())]
    assert tables.summary_table.item(labels.index("mean"), 0).text() == "3.000"
    assert tables.summary_table.item(labels.index("NaNs"), 0).text() == "1"
    tables.select_array("SP_rate")
    assert tables.selected_array() == "SP_rate"
    # No subfolder: empty, saying why
    tables.show_results(None, message="No saved Spike results")
    assert tables.columns == [] and tables.message_label.text() == "No saved Spike results"
