# -*- coding: utf-8 -*-
"""
FolderTreeModel - Qt tree model over an NMManager's folder hierarchy.

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

from PyQt6 import QtCore

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_tool_folder import NMToolFolder


class _GroupNode:
    """Synthetic, non-selectable row grouping one container's children.

    e.g. the "Data" row under an NMFolder, standing in for
    ``folder.data`` so a folder with thousands of flat NMData items
    doesn't interleave with the (usually much smaller) dataseries
    branch.  Not an NMObject — never passed to ``NMManager`` selection
    methods.

    Instances are cached by the model (one canonical ``_GroupNode`` per
    ``(kind, owner)`` — see ``FolderTreeModel._group_node``) rather than
    constructed fresh per call: ``QAbstractItemModel.createIndex()``
    does not keep an arbitrary Python ``internalPointer()`` object
    alive on its own, so an uncached, ephemeral node can be garbage
    collected while a ``QModelIndex`` still points at it, corrupting
    memory. The cache also lets sibling/parent lookups use plain
    identity (``is``) instead of a custom ``__eq__``.
    """

    __slots__ = ("kind", "label", "owner")

    def __init__(self, kind: str, label: str, owner: NMObject) -> None:
        self.kind = kind
        self.label = label
        self.owner = owner

    def children(self) -> list[NMObject]:
        container = getattr(self.owner, self.kind, None)
        if container is None:
            return []
        return list(container.values())


# (label, container attribute name) candidates, in display order, per owner type
_FOLDER_LIKE_GROUPS = (("Data", "data"), ("Data Series", "dataseries"))
_FOLDER_ONLY_GROUPS = (("Tool Folders", "toolfolders"),)
_DATASERIES_GROUPS = (("Channels", "channels"), ("Epochs", "epochs"))


class FolderTreeModel(QtCore.QAbstractItemModel):
    """Read-only Qt tree model over ``NMManager.folders``.

    Wraps the manager directly — no state is copied.  ``rowCount()`` /
    ``data()`` / ``index()`` read live from the underlying
    ``NMObjectContainer``s on every call, so the model scales to large
    folders the same way ``QTreeView`` already lazily requests only
    expanded/visible rows.

    Tree shape, per folder (and identically per tool folder, since
    ``NMToolFolder`` exposes the same ``.data`` / ``.dataseries``
    containers)::

        NMFolder
        |-- "Data"          (only if non-empty) -> NMData leaves
        |-- "Data Series"   (only if non-empty) -> NMDataSeries
        |   |-- "Channels"  (only if non-empty) -> NMChannel leaves
        |   `-- "Epochs"    (only if non-empty) -> NMEpoch leaves
        `-- "Tool Folders"  (only if non-empty) -> NMToolFolder

    There is no change-notification hookup from the core object model,
    so this model must be refreshed explicitly (:meth:`refresh`) after
    mutating actions elsewhere in the app.
    """

    def __init__(
        self,
        manager: NMManager,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._nm = manager
        self._group_nodes: dict[tuple[str, int], _GroupNode] = {}

    # ------------------------------------------------------------------
    # Node traversal helpers

    def _group_node(self, kind: str, label: str, owner: NMObject) -> _GroupNode:
        """Return the canonical (cached) group node for (kind, owner).

        See ``_GroupNode``'s docstring for why this must be cached
        rather than constructed fresh on every call.
        """
        key = (kind, id(owner))
        node = self._group_nodes.get(key)
        if node is None:
            node = _GroupNode(kind=kind, label=label, owner=owner)
            self._group_nodes[key] = node
        return node

    def _group_children(self, owner: NMObject) -> list[_GroupNode]:
        """Non-empty synthetic group rows for a folder-like or dataseries owner."""
        if isinstance(owner, NMDataSeries):
            candidates = _DATASERIES_GROUPS
        else:
            candidates = _FOLDER_LIKE_GROUPS
            if isinstance(owner, NMFolder):
                candidates = candidates + _FOLDER_ONLY_GROUPS
        groups = []
        for label, kind in candidates:
            container = getattr(owner, kind, None)
            if container is not None and len(container) > 0:
                groups.append(self._group_node(kind, label, owner))
        return groups

    def _children_of(self, node: object | None) -> list[object]:
        """Ordered Qt children of *node* (``None`` = invisible root)."""
        if node is None:
            return list(self._nm.folders.values())
        if isinstance(node, _GroupNode):
            return node.children()
        if isinstance(node, (NMFolder, NMToolFolder, NMDataSeries)):
            return self._group_children(node)
        return []  # NMData, NMChannel, NMEpoch are leaves

    def _logical_parent(self, node: object) -> object | None:
        """The node (``_GroupNode``, ``NMObject``, or ``None``) that owns *node*."""
        if isinstance(node, _GroupNode):
            return node.owner
        if isinstance(node, NMFolder):
            return None
        if isinstance(node, NMData):
            return self._group_node("data", "Data", node._parent)
        if isinstance(node, NMDataSeries):
            return self._group_node("dataseries", "Data Series", node._parent)
        if isinstance(node, NMToolFolder):
            return self._group_node("toolfolders", "Tool Folders", node._parent)
        if isinstance(node, NMChannel):
            return self._group_node("channels", "Channels", node._parent)
        if isinstance(node, NMEpoch):
            return self._group_node("epochs", "Epochs", node._parent)
        return None

    def _index_for(self, node: object | None) -> QtCore.QModelIndex:
        """Build the QModelIndex for *node* (``None`` -> invisible root)."""
        if node is None:
            return QtCore.QModelIndex()
        siblings = self._children_of(self._logical_parent(node))
        for row, sibling in enumerate(siblings):
            if sibling is node:
                return self.createIndex(row, 0, node)
        return QtCore.QModelIndex()

    # ------------------------------------------------------------------
    # QAbstractItemModel interface

    def index(
        self,
        row: int,
        column: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> QtCore.QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        parent_node = parent.internalPointer() if parent.isValid() else None
        children = self._children_of(parent_node)
        if row < 0 or row >= len(children):
            return QtCore.QModelIndex()
        return self.createIndex(row, column, children[row])

    def parent(self, index: QtCore.QModelIndex) -> QtCore.QModelIndex:  # type: ignore[override]
        if not index.isValid():
            return QtCore.QModelIndex()
        logical_parent = self._logical_parent(index.internalPointer())
        return self._index_for(logical_parent)

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.column() > 0:
            return 0
        parent_node = parent.internalPointer() if parent.isValid() else None
        return len(self._children_of(parent_node))

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 1

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        node = index.internalPointer()
        if isinstance(node, _GroupNode):
            return node.label
        if isinstance(node, NMObject):
            return node.name
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
            and section == 0
        ):
            return "Name"
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        base = QtCore.Qt.ItemFlag.ItemIsEnabled
        if isinstance(index.internalPointer(), _GroupNode):
            return base  # synthetic rows are not selectable
        return base | QtCore.Qt.ItemFlag.ItemIsSelectable

    # ------------------------------------------------------------------
    # Refresh

    def refresh(self) -> None:
        """Re-read the hierarchy from ``NMManager``.

        The core object model has no change-notification hooks, so this
        must be called explicitly after mutating actions (adding data,
        running a tool, etc.) — it is not automatic. Uses a full
        model reset (loses expand/selection state); fine-grained
        updates can be added later if that proves annoying in practice.
        """
        self.beginResetModel()
        self._group_nodes.clear()
        self.endResetModel()
