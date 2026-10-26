"""Tests for Add Channels (Channel ⋯) and Add Epochs (Prefix ⋯)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.app_window import NMAppWindow
from pyneuromatic.gui.tier_actions import AddToSeriesDialog

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot, monkeypatch):
    """Series Record: channels A, B x epochs E0..E2 (4 points). The Add
    dialog is replaced: ``app.answer`` is returned and each call's
    (kind, preview, validate) kept in ``app.asked``."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    folder.new_dataseries("Record", n_channels=2, n_epochs=3, n_points=4)
    manager.select_value_set(folder)
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    series = folder.dataseries["Record"]
    window.selection_model.update(dataseries=series)
    window.folder, window.series = folder, series
    window.answer, window.asked = None, []

    def ask(kind, context, preview, validate, maximum=1_000_000):
        window.asked.append((kind, context, preview, validate, maximum))
        return window.answer

    for actions in (window.channel_actions, window.series_actions):
        monkeypatch.setattr(actions, "_ask_add", ask)
    return window


def _values(**changes):
    return {"count": 1, "value": 0.0, "noise": 0.0, "ylabel": "", "yunits": ""} | changes


def test_add_channels(app):
    app.answer = _values(count=2, value=3.0, ylabel="Im", yunits="pA")
    app.channel_actions.add_action.trigger()
    kind, context, preview, validate, maximum = app.asked[0]
    assert kind == "channels" and context == ("Series", "Demo › Record")
    assert maximum == 24  # C to Z
    assert preview(2) == ("Adds channels C, D: 6 new arrays (RecordC0, RecordC1, ... "
                          "RecordD2), sized as each epoch's.")
    assert list(app.series.channels.keys()) == ["A", "B", "C", "D"]
    assert list(app.folder.data["RecordD1"].nparray) == [3.0] * 4
    assert app.folder.data["RecordD1"].yscale.units == "pA"
    # The first new channel is shown, and offered in the strip
    assert app.selection_model.selection["channel"].name == "C"
    assert app.plot_widget.view_tabs.current_view() == ("channel", "C")
    combo = app.selection_strip.combo("channel")
    assert [combo.itemData(i) for i in range(combo.count())][-2:] == ["C", "D"]


def test_add_channels_validates(app):
    app.folder.data.new("RecordC1", nparray=np.zeros(4))
    app.channel_actions.add_action.trigger()
    validate = app.asked[0][3]
    assert validate(_values()) == "Data 'RecordC1' already exists."
    assert list(app.series.channels.keys()) == ["A", "B"]  # cancelled


def test_add_channels_not_while_locked(app):
    app.series_actions.lock_action.trigger()  # Lock
    assert app.series.locked
    action = app.channel_actions.add_action
    assert not action.isEnabled() and "locked" in action.toolTip()


def test_add_epochs(app):
    app.answer = _values(count=2, value=1.0)
    app.series_actions.add_epochs_action.trigger()
    kind, context, preview, _validate, _maximum = app.asked[0]
    assert kind == "epochs" and context == ("Series", "Demo › Record")
    assert preview(2) == ("Adds E3 to E4: 4 new arrays (RecordA3, RecordB3, ... RecordB4), "
                          "sized as each channel's last.")
    assert list(app.series.epochs.keys()) == ["E0", "E1", "E2", "E3", "E4"]
    assert list(app.folder.data["RecordB4"].nparray) == [1.0] * 4
    assert app.plot_widget.trace_count_label.text() == "of 5"


def test_add_epochs_with_noise(app):
    app.answer = _values(value=10.0, noise=1.0)
    app.series_actions.add_epochs_action.trigger()
    values = app.folder.data["RecordA3"].nparray
    assert values.std() > 0 and abs(values.mean() - 10.0) < 5


def test_add_epochs_not_while_locked(app):
    app.series_actions.lock_action.trigger()  # Lock
    assert app.series.locked
    action = app.series_actions.add_epochs_action
    assert not action.isEnabled() and "locked" in action.toolTip()


def test_dialog(qtbot):
    dialog = AddToSeriesDialog(
        "channels", ("Series", "Demo › Record"), lambda n: f"{n} new",
        validate=lambda values: None)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Add Channels"
    assert dialog.preview_label.text() == "1 new"
    dialog.count_spin.setValue(3)
    assert dialog.preview_label.text() == "3 new"
    dialog.value_edit.setText("NaN")
    dialog.ylabel_edit.setText("Im")
    values = dialog.values()
    assert values["count"] == 3 and np.isnan(values["value"]) and values["ylabel"] == "Im"
    dialog.noise_edit.setText("-1")
    dialog.accept()
    assert dialog.error_label.text() == "Noise SD must not be negative"
    epochs = AddToSeriesDialog("epochs", ("Series", "x"), lambda n: "")
    qtbot.addWidget(epochs)
    assert epochs.ylabel_edit is None  # epochs keep each channel's scales


def test_remove_channel_from_the_channel_menu(app, monkeypatch):
    asked = []

    def choose(title, **options):
        asked.append((title, options))
        return (options["current"], None, "Delete")

    monkeypatch.setattr(app.series_actions, "_choose", choose)
    app.selection_model.update(channel=app.series.channels["B"])
    menu = app.selection_strip.action_button("channel").menu()
    action = next(a for a in menu.actions() if a.text() == "Remove Channel…")
    assert action is app.series_actions.remove_channel_action  # the series' action
    action.trigger()
    title, options = asked[0]
    assert title == "Remove Channel" and options["current"] == "B"  # the strip's
    assert list(app.series.channels.keys()) == ["A"]
    assert "RecordB0" not in app.folder.data
