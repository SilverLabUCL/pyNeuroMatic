"""Ensure a QApplication exists for every test in this directory.

QAbstractItemModel/QWidget instances are unsafe to construct without one
(PyQt6 does not implicitly create one), and relying on test execution
order to pick up a QApplication created incidentally by some other test
module is fragile. Depending on pytest-qt's session-scoped ``qapp``
fixture here makes every test in ``tests/test_gui/`` safe on its own.

Every app window a test creates is also closed (and deleted) when that test
ends. pytest-qt holds widgets only weakly, so a window could otherwise be
garbage-collected during a later test without its closeEvent; its pyqtgraph
plots were then deleted half-way and crashed on a paint or resize event.
"""
import gc

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets


@pytest.fixture(autouse=True)
def _qapp(qapp):
    pass


@pytest.fixture(autouse=True)
def _close_app_windows(monkeypatch):
    from pyneuromatic.gui.app_window import NMAppWindow

    windows = []
    original_init = NMAppWindow.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        windows.append(self)  # a strong reference, until this test ends

    monkeypatch.setattr(NMAppWindow, "__init__", init)
    yield
    for window in windows:
        window.close()  # closeEvent tears the pyqtgraph views down first
        window.deleteLater()
    windows.clear()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    gc.collect()



@pytest.fixture(autouse=True)
def _reset_table_nan():
    """Each test starts with tables showing NaN as "NaN"."""
    from pyneuromatic.gui import table_format

    yield
    table_format.set_nan_mode("NaN")


@pytest.fixture(autouse=True)
def _no_real_input_dialogs(monkeypatch):
    """A test that opens an input dialog without stubbing it fails at once,
    instead of hanging on a real dialog (its own stubs still apply)."""
    def unstubbed(*_args, **_kwargs):
        raise RuntimeError("QInputDialog opened in a test without being stubbed")

    for name in ("getText", "getInt", "getDouble", "getItem", "getMultiLineText"):
        monkeypatch.setattr(QtWidgets.QInputDialog, name, staticmethod(unstubbed))

    # Any other dialog (e.g. the strip's choose dialogs) shown for real
    def unstubbed_exec(self, *_args, **_kwargs):
        raise RuntimeError(f"{type(self).__name__} opened in a test without being stubbed")

    monkeypatch.setattr(QtWidgets.QDialog, "exec", unstubbed_exec)
