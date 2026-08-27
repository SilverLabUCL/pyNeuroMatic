# -*- coding: utf-8 -*-
"""
FolderBrowserWidget - read-only tree view over an NMManager's folder hierarchy.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source
Software Toolkit for Acquisition, Analysis and Simulation of
Electrophysiological Data. Front. Neuroinform. 12:14.
doi: 10.3389/fninf.2018.00014

Copyright (c) 2026 The Silver Lab, University College London.
Licensed under MIT License - see LICENSE file for details.

Original NeuroMatic: https://github.com/SilverLabUCL/NeuroMatic
Website: https://github.com/SilverLabUCL/pyNeuroMatic
Paper: https://doi.org/10.3389/fninf.2018.00014
"""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.gui.folder_model import FolderTreeModel


class FolderBrowserWidget(QtWidgets.QWidget):
    """Read-only tree view for navigating an NMManager's folder hierarchy.

    Pure navigation for v1: expand/collapse and Qt-level row highlighting
    only. Clicking a row does **not** call ``manager.select_value_set()``
    or otherwise touch ``NMManager``'s selection state — that coupling
    turned out to be confusing (e.g. a stale ``toolfolder`` selection
    silently redirecting where later selections resolve) and is deferred
    until the model settles. Synthetic group rows ("Data", "Data Series",
    "Tool Folders", "Channels", "Epochs") are not selectable, same as
    before. Use ``tree.selectionModel().currentChanged`` directly if you
    need to observe clicks (see ``scripts-for-testing/try_browser.py``).

    Double-click / Enter ("activate") is wired to a placeholder for now
    — reserved for a future "open" action (e.g. quick-plot a leaf) that
    doesn't exist yet.

    No create/rename/delete in this pass — navigation only.
    """

    def __init__(
        self,
        manager: NMManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._nm = manager
        self._model = FolderTreeModel(manager, parent=self)

        self._tree = QtWidgets.QTreeView(self)
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(True)
        self._tree.activated.connect(self._on_activated)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tree)

    @property
    def model(self) -> FolderTreeModel:
        return self._model

    @property
    def tree(self) -> QtWidgets.QTreeView:
        return self._tree

    def refresh(self) -> None:
        """Re-read the hierarchy from the manager (see FolderTreeModel.refresh)."""
        self._model.refresh()

    def _on_activated(self, index: QtCore.QModelIndex) -> None:
        """Placeholder for a future "open" action (e.g. quick-plot a leaf).

        No behavior yet — v1 has no viewer to open a leaf into. Reserved
        so wiring one in later doesn't require new signal plumbing.
        """
        pass
