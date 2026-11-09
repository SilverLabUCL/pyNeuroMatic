"""Tests for the Plot's trace style: Settings › Plot (#465)."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")
from PyQt6 import QtCore, QtGui

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_settings import NMSettings
from pyneuromatic.gui.app_window import NMAppWindow

pytestmark = pytest.mark.gui


@pytest.fixture
def app(qtbot):
    """Channel A x 4 epochs of 10 points."""
    manager = NMManager(quiet=True)
    folder = manager.folders.new("Demo", select=False)
    series = folder.new_dataseries("Record", n_channels=1, n_epochs=4, n_points=10)
    for i, d in enumerate(series.channels["A"].data):
        d.nparray = np.arange(10.0) + i
    manager.select_keys = {"folder": folder.name, "dataseries": series.name,
                           "channel": "A", "epoch": "E0"}
    window = NMAppWindow(manager)
    qtbot.addWidget(window)
    window.settings = manager.settings
    return window


def _curves(app):
    return app.plot_widget.plot_widget.getPlotItem().listDataItems()


def _current(app):
    """The current trace's curve (drawn last)."""
    return _curves(app)[-1]


def _pen(curve):
    return curve.opts["pen"]


def _dialog(app):
    dialog = app.settings_dialog()
    dialog.show_page("Plot")
    return dialog


def test_black_by_default(app):
    assert NMSettings().plot_trace_color == "#000000"
    pen = _pen(_current(app))
    assert pen.color().name() == "#000000"
    assert pen.widthF() == 2.0
    assert pen.style() == QtCore.Qt.PenStyle.SolidLine
    assert _current(app).opts["symbol"] is None


def test_colour_width_and_line(app):
    dialog = _dialog(app)
    dialog._choose_color = lambda initial: "#c62828"
    dialog.plot_controls["plot_trace_color"].click()
    dialog.plot_controls["plot_trace_width"].setValue(3.5)
    line = dialog.plot_controls["plot_trace_line"]
    line.setCurrentIndex(line.findData("dash"))
    pen = _pen(_current(app))
    assert pen.color().name() == "#c62828"
    assert pen.widthF() == 3.5
    assert pen.style() == QtCore.Qt.PenStyle.DashLine
    assert app.settings.plot_trace_color == "#c62828"
    assert dialog.plot_controls["plot_trace_color"].text() == "#c62828"


def test_colour_picker_cancelled(app):
    dialog = _dialog(app)
    dialog._choose_color = lambda initial: None
    dialog.plot_controls["plot_trace_color"].click()
    assert app.settings.plot_trace_color == "#000000"


def test_symbols(app):
    dialog = _dialog(app)
    symbol = dialog.plot_controls["plot_trace_symbol"]
    symbol.setCurrentIndex(symbol.findData("square"))
    dialog.plot_controls["plot_trace_symbol_size"].setValue(9.0)
    curve = _current(app)
    assert curve.opts["symbol"] == "s" and curve.opts["symbolSize"] == 9.0
    assert _pen(curve) is not None  # with its line
    # Symbols only
    line = dialog.plot_controls["plot_trace_line"]
    line.setCurrentIndex(line.findData("none"))
    curve = _current(app)
    assert curve.opts["symbol"] == "s"
    assert _pen(curve) is None or _pen(curve).style() == QtCore.Qt.PenStyle.NoPen


def test_no_line_and_no_symbol_still_shows_the_trace(app):
    app.settings.plot_trace_line = "none"
    app.settings.plot_trace_symbol = "none"
    app._apply_settings()
    assert _current(app).opts["symbol"] == "o"  # small circles


def test_overlay_traces(app):
    app.settings.plot_overlay_color = "#ff9800"
    app.settings.plot_overlay_width = 1.5
    app._apply_settings()
    app.plot_widget.overlay_count.setValue(2)
    curves = _curves(app)
    assert len(curves) > 1
    current = [c for c in curves if _pen(c).color().name() == "#000000"]
    overlays = [c for c in curves if c not in current]
    assert len(current) == 1 and overlays
    for curve in overlays:
        pen = _pen(curve)
        assert pen.color().name() == "#ff9800" and pen.widthF() == 1.5
        assert pen.color().alpha() < 255  # behind the current trace


def test_default_style_button(app):
    dialog = _dialog(app)
    dialog.plot_controls["plot_trace_width"].setValue(5.0)
    dialog._choose_color = lambda initial: "#00ff00"
    dialog.plot_controls["plot_overlay_color"].click()
    dialog.plot_defaults_button.click()
    assert app.settings.plot_trace_width == 2.0
    assert app.settings.plot_overlay_color == "#78909c"
    assert dialog.plot_controls["plot_trace_width"].value() == 2.0
    assert _pen(_current(app)).widthF() == 2.0


def test_trace_style_opens_settings_at_plot(app, monkeypatch):
    opened = []
    monkeypatch.setattr(app, "open_settings", lambda page=None: opened.append(page))
    app.plot_widget.trace_style_action.trigger()
    assert opened == ["Plot"]


def test_saved_and_restored(tmp_path):
    path = tmp_path / "settings.toml"
    settings = NMSettings()
    settings.load_user(path)
    settings.plot_trace_color = "#123456"
    settings.plot_trace_width = 3  # a whole number is fine
    settings.plot_trace_symbol = "triangle"
    settings.save_user()
    restored = NMSettings()
    assert restored.load_user(path)
    assert restored.plot_trace_color == "#123456"
    assert restored.plot_trace_width == 3.0
    assert restored.plot_trace_symbol == "triangle"


def test_a_bad_colour_in_the_settings_file_draws_the_default(app):
    app.settings.plot_trace_color = "not a colour"
    app._apply_settings()
    assert _pen(_current(app)).color().name() == "#000000"


def test_bad_values_are_refused():
    settings = NMSettings()
    with pytest.raises(ValueError):
        settings.plot_trace_line = "wavy"
    with pytest.raises(ValueError):
        settings.plot_trace_width = 50.0


def test_a_settings_file_from_before_has_the_defaults(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('table_nan = "blank"\nconfirm_quit = true\n\n'
                    '[pyneuromatic]\ntype = "nm_settings"\nversion = "1"\n')
    settings = NMSettings()
    assert settings.load_user(path)
    assert settings.table_nan == "blank"
    assert settings.plot_trace_color == "#000000" and settings.plot_trace_width == 2.0


# --- right-click: a trace's style, or the Plot menu ------------------------------

class _Click:
    """A right-click at *scene_pos* (as pyqtgraph passes it)."""

    def __init__(self, scene_pos):
        self._pos = scene_pos

    def scenePos(self):
        return self._pos

    def screenPos(self):
        return QtCore.QPointF(10, 10)


@pytest.fixture
def shown(app, qtbot):
    app.resize(1200, 700)
    app.show()
    qtbot.waitExposed(app)
    app.plot_widget.rescale()
    menus = []
    app.plot_widget._popup = lambda menu, pos: menus.append(menu)
    app.menus = menus
    return app


def _scene(app, x, y):
    view_box = app.plot_widget.plot_widget.getPlotItem().getViewBox()
    return view_box.mapViewToScene(QtCore.QPointF(x, y))


def _right_click(app, x, y):
    view_box = app.plot_widget.plot_widget.getPlotItem().getViewBox()
    view_box.raiseContextMenu(_Click(_scene(app, x, y)))
    return app.menus[-1]


def _texts(menu):
    return [a.text() for a in menu.actions() if a.text()]


def _choose(menu, title, text):
    submenu = next(a.menu() for a in menu.actions() if a.text() == title)
    next(a for a in submenu.actions() if a.text() == text).trigger()


def test_right_click_a_trace_gives_its_style(shown):
    menu = _right_click(shown, 4.5, 4.5)  # on the current trace (y = x)
    assert _texts(menu) == ["Current Trace", "Colour...", "Line Width", "Line", "Symbol",
                            "Symbol Size", "Default Style", "Trace Style..."]
    _choose(menu, "Line Width", "3")
    assert shown.settings.plot_trace_width == 3.0
    assert _pen(_current(shown)).widthF() == 3.0
    menu = _right_click(shown, 4.5, 4.5)
    _choose(menu, "Line", "Dashed")
    _choose(_right_click(shown, 4.5, 4.5), "Symbol", "Diamond")
    assert _pen(_current(shown)).style() == QtCore.Qt.PenStyle.DashLine
    assert _current(shown).opts["symbol"] == "d"
    # The menu shows the style chosen
    submenu = next(a.menu() for a in _right_click(shown, 4.5, 4.5).actions()
                   if a.text() == "Line Width")
    assert [a.text() for a in submenu.actions() if a.isChecked()] == ["3"]


def test_trace_menu_colour_default_and_settings(shown, monkeypatch):
    shown.plot_widget._choose_color = lambda initial: "#c62828"
    next(a for a in _right_click(shown, 4.5, 4.5).actions() if a.text() == "Colour...").trigger()
    assert shown.settings.plot_trace_color == "#c62828"
    assert _pen(_current(shown)).color().name() == "#c62828"
    next(a for a in _right_click(shown, 4.5, 4.5).actions()
         if a.text() == "Default Style").trigger()
    assert shown.settings.plot_trace_color == "#000000"
    opened = []
    monkeypatch.setattr(shown, "open_settings", lambda page=None: opened.append(page))
    next(a for a in _right_click(shown, 4.5, 4.5).actions()
         if a.text() == "Trace Style...").trigger()
    assert opened == ["Plot"]


def test_right_click_an_overlay_trace(shown):
    shown.plot_widget.overlay_count.setValue(2)  # 3 shown, E0..E2: y = x + i
    shown.plot_widget.rescale()
    menu = _right_click(shown, 4.5, 6.5)  # on E2 (y = x + 2), an overlay
    assert _texts(menu) == ["Overlay Traces", "Colour...", "Line Width",
                            "Default Style", "Trace Style..."]
    _choose(menu, "Line Width", "2")
    assert shown.settings.plot_overlay_width == 2.0


def test_right_click_elsewhere_gives_the_plot_menu(shown):
    menu = _right_click(shown, 1.0, 8.0)  # far from the trace
    assert menu is shown.plot_widget.graph_menu
    assert "Axes" in _texts(menu) and "Export..." in _texts(menu)
    axes = next(a.menu() for a in menu.actions() if a.text() == "Axes")
    assert "View All" in [a.text() for a in axes.actions()]


def test_the_axes_give_the_plot_menu(shown):
    """An axis passes its right-click to the view box."""
    plot_item = shown.plot_widget.plot_widget.getPlotItem()
    axis = plot_item.getAxis("bottom")
    assert axis.linkedView() is plot_item.getViewBox()
    axis.linkedView().raiseContextMenu(_Click(axis.mapToScene(axis.boundingRect().center())))
    assert shown.menus[-1] is shown.plot_widget.graph_menu
