"""Tests for parameter forms built from descriptions (#555)."""

import math

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets

from pyneuromatic.core.nm_params import (
    ARRAY, BOOLEAN, CHOICE, INTEGER, NUMBER, X_RANGE, XRANGE, NMParam)
from pyneuromatic.gui.param_form import ParamForm, bound_text

PARAMS = (
    X_RANGE,
    NMParam("baseline", XRANGE, "Baseline", default=(0.0, 5.0),
            names=("bbgn", "bend"), color="#16a34a"),
    NMParam("factor", NUMBER, "Factor", default=2.0, units="mV"),
    NMParam("points", INTEGER, "Points", default=3, minimum=1),
    NMParam("ignore", BOOLEAN, "Ignore NaNs", default=True),
    NMParam("mode", CHOICE, "Mode", default="2d",
            choices=(("1d", "End to end"), ("2d", "Rows"))),
    NMParam("ref", ARRAY, "Array"),
)


@pytest.fixture
def form(qtbot):
    form = ParamForm(PARAMS)
    qtbot.addWidget(form)
    return form


def test_a_control_each_starting_at_its_default(form):
    assert isinstance(form.control("ignore"), QtWidgets.QCheckBox)
    assert isinstance(form.control("mode"), QtWidgets.QComboBox)
    assert isinstance(form.control("factor"), QtWidgets.QLineEdit)
    begin, end = form.control("x")
    assert (begin.text(), end.text()) == ("-inf", "inf")
    form.set_array_names("ref", ["RecordA0", "RecordA1"])
    values = form.values()
    assert values["x"] == (-math.inf, math.inf)
    assert values["baseline"] == (0.0, 5.0)
    assert (values["factor"], values["points"], values["ignore"]) == (2.0, 3, True)
    assert (values["mode"], values["ref"]) == ("2d", "RecordA0")


def test_values_are_checked(form):
    form.set_array_names("ref", ["RecordA0"])
    form.control("points").setText("0")
    with pytest.raises(ValueError, match="Points: at least 1"):
        form.values()
    form.control("points").setText("4")
    form.control("factor").setText("big")
    with pytest.raises(ValueError, match="Factor: 'big' is not a number"):
        form.values()


def test_set_values_and_signals(form, qtbot):
    form.set_array_names("ref", ["RecordA0"])
    with qtbot.assertNotEmitted(form.changed):
        form.set_values({"ignore": False, "mode": "1d", "factor": 0.5})
    assert form.values()["ignore"] is False and form.values()["mode"] == "1d"
    assert form.control("factor").text() == "0.5"
    with qtbot.waitSignal(form.changed):
        form.control("ignore").setChecked(True)
    with qtbot.waitSignal(form.ranges_changed):
        form.control("baseline")[0].editingFinished.emit()


def test_x_ranges_for_the_plot(form):
    ranges = form.x_ranges()
    assert [(p.name, r) for p, r in ranges] == [
        ("x", (-math.inf, math.inf)), ("baseline", (0.0, 5.0))]
    assert ranges[1][0].color == "#16a34a"
    form.set_x_range("baseline", 1.0, 2.5)  # dragged
    assert [b.text() for b in form.control("baseline")] == ["1", "2.5"]
    form.control("baseline")[0].setText("9")  # past its end: not drawn
    assert [p.name for p, _r in form.x_ranges()] == ["x"]


def test_no_params(qtbot):
    form = ParamForm(())
    qtbot.addWidget(form)
    assert form.values() == {} and form.x_ranges() == []


def test_bound_text():
    assert [bound_text(v) for v in (-math.inf, math.inf, 0.5, 1e-7)] == [
        "-inf", "inf", "0.5", "1e-07"]


def test_reset_to_the_defaults(form, qtbot):
    form.set_array_names("ref", ["RecordA0", "RecordA1"])
    form.set_values({"ignore": False, "factor": 9.0, "baseline": (1.0, 2.0)})
    form.control("ref").setCurrentIndex(1)
    with qtbot.waitSignal(form.ranges_changed):
        form.reset()
    values = form.values()
    assert values["ignore"] is True and values["factor"] == 2.0
    assert values["baseline"] == (0.0, 5.0)
    assert values["ref"] == "RecordA1"  # no default: kept


def test_rows_at_the_left():
    """A form's rows start at the left, not centred (macOS's default)."""
    from PyQt6 import QtCore
    form = ParamForm((X_RANGE,))
    alignment = form.layout().formAlignment()
    assert alignment & QtCore.Qt.AlignmentFlag.AlignLeft
    assert not alignment & QtCore.Qt.AlignmentFlag.AlignHCenter
    form.deleteLater()
