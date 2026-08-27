"""Ensure a QApplication exists for every test in this directory.

QAbstractItemModel/QWidget instances are unsafe to construct without one
(PyQt6 does not implicitly create one), and relying on test execution
order to pick up a QApplication created incidentally by some other test
module is fragile. Depending on pytest-qt's session-scoped ``qapp``
fixture here makes every test in ``tests/test_gui/`` safe on its own.
"""
import pytest

pytest.importorskip("PyQt6")


@pytest.fixture(autouse=True)
def _qapp(qapp):
    pass
