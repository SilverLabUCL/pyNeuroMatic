"""Tests for shared GUI selection state."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.selection_model import SelectionModel

pytestmark = pytest.mark.gui


def test_selection_updates_manager_and_notifies_subscribers(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Folder", select=False)
    data = folder.data.new("Record", nparray=np.array([1.0, 2.0]))
    selection = SelectionModel(manager)
    changes = []
    selection.selection_changed.connect(changes.append)

    selection.update(
        folder=folder,
        data=data,
        data_set="SetA",
        epoch_set="Epochs",
        group=1,
        group_operator="AND",
    )

    assert manager.select_values["folder"] is folder
    assert manager.select_values["data"] is data
    assert selection.selection["source"] == "data"
    assert selection.selection["data_set"] == "SetA"
    assert selection.selection["epoch_set"] == "Epochs"
    assert selection.selection["group"] == 1
    assert selection.selection["group_operator"] == "AND"
    assert len(changes) == 1
    assert changes[0]["data"] is data
    assert changes[0]["data_set"] == "SetA"


def test_selection_supports_partial_updates_and_clear(qtbot):
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Folder", select=False)
    data = folder.data.new("Record", nparray=np.array([1.0, 2.0]))
    selection = SelectionModel(manager)

    selection.update(folder=folder, data=data, epoch_set="SetA", group=1)
    selection.update(epoch_set=None)

    assert selection.selection["folder"] is folder
    assert selection.selection["data"] is data
    assert selection.selection["epoch_set"] is None
    assert selection.selection["group"] == 1
    assert selection.selection["group_operator"] is None

    selection.clear()

    assert all(value is None for value in selection.selection.values())
    assert all(value is None for value in manager.select_values.values())


def _folder_with_data_and_series(manager):
    folder = manager.folders.new("Folder", select=False)
    for name in ("RecordA0", "RecordA1", "Other0"):
        folder.data.new(name, nparray=np.array([1.0, 2.0]))
    series = folder.sync_dataseries("Record")
    return folder, series


def test_data_and_dataseries_stay_selected_and_source_follows_choice(qtbot):
    manager = NMManager(quiet=True)
    folder, series = _folder_with_data_and_series(manager)
    selection = SelectionModel(manager)

    selection.update(folder=folder, dataseries=series)
    assert selection.selection["source"] == "dataseries"
    selection.update(data=folder.data["Other0"])
    assert selection.selection["source"] == "data"
    # Choosing data no longer clears the data series
    assert selection.selection["dataseries"] is series

    selection.update(epoch_set="Set1")
    assert selection.selection["source"] == "dataseries"
    selection.update(data_set="DataSet")
    assert selection.selection["source"] == "data"

    selection.update(source="dataseries")
    assert selection.selection["source"] == "dataseries"
    assert selection.selection["data"] is folder.data["Other0"]

    # Clearing the source's selection falls back to the other one
    selection.update(dataseries=None)
    assert selection.selection["source"] == "data"
    selection.update(data=None)
    assert selection.selection["source"] is None


def test_source_must_be_known(qtbot):
    selection = SelectionModel(NMManager(quiet=True))
    with pytest.raises(ValueError):
        selection.update(source="epoch")


def test_data_set_and_epoch_scope_are_remembered_separately(qtbot):
    manager = NMManager(quiet=True)
    folder, series = _folder_with_data_and_series(manager)
    other = manager.folders.new("Other", select=False)
    other.data.new("X0", nparray=np.array([1.0]))
    selection = SelectionModel(manager)

    selection.update(folder=folder, dataseries=series)
    selection.update(data_set="DataSet", epoch_set="Set1", group=0, group_operator="OR")

    selection.update(folder=other)
    assert selection.selection["data_set"] is None
    assert selection.selection["epoch_set"] is None

    selection.update(folder=folder)
    selection.update(dataseries=series)
    assert selection.selection["data_set"] == "DataSet"
    assert selection.selection["epoch_set"] == "Set1"
    assert selection.selection["group"] == 0
    assert selection.selection["group_operator"] == "OR"

    # Changing the data array keeps the data set
    selection.update(data=folder.data["RecordA1"])
    assert selection.selection["data_set"] == "DataSet"


def test_selection_summary_lists_the_active_source(qtbot):
    from pyneuromatic.gui.selection_model import selection_summary

    manager = NMManager(quiet=True)
    folder, series = _folder_with_data_and_series(manager)
    selection = SelectionModel(manager)
    selection.update(folder=folder, dataseries=series, epoch_set="Set1", group=0,
                     group_operator="AND")
    selection.update(data=folder.data["Other0"], data_set="DataSet")

    assert selection_summary(selection.selection) == [
        "Folder", "Other0", "Data Set: DataSet",
    ]
    selection.update(source="dataseries")
    assert selection_summary(selection.selection) == [
        "Folder", "Record", "A", "E0", "Epoch Set: Set1 AND Group 0",
    ]
