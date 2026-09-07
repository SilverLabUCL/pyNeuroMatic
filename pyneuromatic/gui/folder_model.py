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


class _SetsNode:
    """Synthetic "Sets" row under a ``_GroupNode``, e.g. "Data" -> "Sets".

    Every ``NMObjectContainer`` has a ``.sets`` (``NMSets``); this exposes
    it as a browsable row. Cached the same way and for the same reason as
    ``_GroupNode`` (see its docstring) — one canonical instance per
    owning group, keyed by ``id(group)``.
    """

    __slots__ = ("group", "label")

    def __init__(self, group: _GroupNode) -> None:
        self.group = group
        self.label = "Sets"

    @property
    def container(self):
        """The NMObjectContainer whose .sets this represents."""
        return getattr(self.group.owner, self.group.kind, None)


class _SetNode:
    """One named set under a ``_SetsNode``, e.g. "Sets" -> "Set1".

    Cached per ``(id(sets_node), name)`` — see ``_GroupNode`` docstring.
    """

    __slots__ = ("sets_node", "name")

    def __init__(self, sets_node: _SetsNode, name: str) -> None:
        self.sets_node = sets_node
        self.name = name

    @property
    def label(self) -> str:
        return self.name

    @property
    def container(self):
        return self.sets_node.container


class _SetMemberNode:
    """One member of a set, as shown under "Sets" -> "Set1" -> member.

    A set's members are the *same* NMObjects already shown elsewhere in
    the tree (e.g. under "Data") — but a QAbstractItemModel node can only
    have one logical parent, and this member's parent here is the set,
    not "Data". Reusing the raw NMObject as internalPointer in both
    places would make ``parent()`` ambiguous for whichever wasn't
    hard-coded. This wrapper gives each (set, member) pairing its own
    distinct, unambiguous identity instead, the same way ``_GroupNode``
    gives "Data" its own identity separate from the NMFolder it groups.
    Cached per ``(id(set_node), id(obj))``.
    """

    __slots__ = ("set_node", "obj")

    def __init__(self, set_node: _SetNode, obj: NMObject) -> None:
        self.set_node = set_node
        self.obj = obj

    @property
    def label(self) -> str:
        return self.obj.name


# (label, container attribute name) candidates, in display order, per owner type
_FOLDER_LIKE_GROUPS = (("Data", "data"), ("Data Series", "dataseries"))
_FOLDER_ONLY_GROUPS = (("Tool Folders", "toolfolders"),)
_DATASERIES_GROUPS = (("Channels", "channels"), ("Epochs", "epochs"))

# Kinds shown even when empty, so a brand-new folder always has a "Data"
# and "Data Series" row to right-click "New..." on — otherwise a freshly
# created, still-empty folder would show no children at all, with no way
# to add anything through the tree's context menu (see _group_children).
_ALWAYS_SHOWN_KINDS = frozenset({"data", "dataseries"})


class FolderTreeModel(QtCore.QAbstractItemModel):
    """Read-only Qt tree model over ``NMManager.folders``.

    Wraps the manager directly — no state is copied.  ``rowCount()`` /
    ``data()`` / ``index()`` read live from the underlying
    ``NMObjectContainer``s on every call, so the model scales to large
    folders the same way ``QTreeView`` already lazily requests only
    expanded/visible rows.

    Tree shape (identical per tool folder, since ``NMToolFolder``
    exposes the same ``.data`` / ``.dataseries`` containers)::

        "Folders"  (always shown) -> NMFolder
        |-- "Sets"          (only if non-empty) -> Set leaves -> member leaves
        |-- "Data"          (always shown) -> NMData leaves
        |   `-- "Sets"      (only if non-empty) -> Set leaves -> member leaves
        |-- "Data Series"   (always shown) -> NMDataSeries
        |   |-- "Sets"      (only if non-empty) -> Set leaves -> member leaves
        |   |-- "Channels"  (only if non-empty) -> NMChannel leaves
        |   |   `-- "Sets"  (only if non-empty) -> Set leaves -> member leaves
        |   `-- "Epochs"    (only if non-empty) -> NMEpoch leaves
        |       `-- "Sets"  (only if non-empty) -> Set leaves -> member leaves
        `-- "Tool Folders"  (only if non-empty) -> NMToolFolder

    Every group row (including the top-level "Folders" row) can have a
    "Sets" child, since every ``NMObjectContainer`` has a ``.sets``
    (``NMSets``) — e.g. "Data" -> "Sets" -> "Set1" -> the NMData objects
    in Set1. A set's members are the *same* objects shown elsewhere in
    the tree (e.g. under "Data" directly), wrapped in a distinct
    ``_SetMemberNode`` identity so each occurrence has an unambiguous
    parent (see its docstring) — required because a QAbstractItemModel
    node can only have one logical parent, but a member's own container
    location and its set membership are two different things.

    Unlike "Folders"/"Data"/"Data Series" (always shown so there's a row
    to bootstrap from nothing), "Sets" is hidden when the container has
    no sets defined yet — it appears under *every* group, potentially
    many times per folder (every dataseries, every channel, every
    epoch...), so always showing it added a lot of visual noise for a
    feature most groups never use. Creating the *first* set therefore
    doesn't go through "Sets" at all: "New Set..." is offered directly
    on the parent group's own context menu too (see
    ``FolderBrowserWidget._show_tree_context_menu``), and once that set
    exists, "Sets" appears to browse/manage it like any other.
    "Tool Folders" and "Channels"/"Epochs" (as groups, not their "Sets"
    child) stay hidden when empty too; nothing creates those manually
    through this UI the way folders/data/dataseries are.

    The top-level "Folders" group node exists so it can be selected in
    the tree (e.g. to show the flat list of folders in a detail pane),
    consistent with every other group node — folders are not bare
    top-level rows.

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
        self._sets_nodes: dict[int, _SetsNode] = {}
        self._set_nodes: dict[tuple[int, str], _SetNode] = {}
        self._set_member_nodes: dict[tuple[int, int], _SetMemberNode] = {}

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

    def _sets_node(self, group: _GroupNode) -> _SetsNode:
        """Return the canonical (cached) "Sets" node for *group*."""
        key = id(group)
        node = self._sets_nodes.get(key)
        if node is None:
            node = _SetsNode(group)
            self._sets_nodes[key] = node
        return node

    def _set_node(self, sets_node: _SetsNode, name: str) -> _SetNode:
        """Return the canonical (cached) node for one named set."""
        key = (id(sets_node), name)
        node = self._set_nodes.get(key)
        if node is None:
            node = _SetNode(sets_node, name)
            self._set_nodes[key] = node
        return node

    def _set_member_node(self, set_node: _SetNode, obj: NMObject) -> _SetMemberNode:
        """Return the canonical (cached) wrapper for one set member."""
        key = (id(set_node), id(obj))
        node = self._set_member_nodes.get(key)
        if node is None:
            node = _SetMemberNode(set_node, obj)
            self._set_member_nodes[key] = node
        return node

    def _group_children(self, owner: NMObject) -> list[_GroupNode]:
        """Synthetic group rows for a folder-like or dataseries owner.

        Hidden when empty, except for ``_ALWAYS_SHOWN_KINDS`` ("Data",
        "Data Series") — those stay visible even on a brand-new,
        completely empty folder so there's always a row to right-click
        "New..." on.
        """
        if isinstance(owner, NMDataSeries):
            candidates = _DATASERIES_GROUPS
        else:
            candidates = _FOLDER_LIKE_GROUPS
            if isinstance(owner, NMFolder):
                candidates = candidates + _FOLDER_ONLY_GROUPS
        groups = []
        for label, kind in candidates:
            container = getattr(owner, kind, None)
            if container is None:
                continue
            if len(container) > 0 or kind in _ALWAYS_SHOWN_KINDS:
                groups.append(self._group_node(kind, label, owner))
        return groups

    def _children_of(self, node: object | None) -> list[object]:
        """Ordered Qt children of *node* (``None`` = invisible root)."""
        if node is None:
            # Always shown, even with zero folders — same reasoning as
            # _ALWAYS_SHOWN_KINDS: a brand-new, completely empty manager
            # still needs a row to right-click "New..." on to create the
            # first folder at all.
            return [self._group_node("folders", "Folders", self._nm)]
        if isinstance(node, _GroupNode):
            # "Sets" is hidden when the container has no sets defined
            # yet, same as Tool Folders/Channels/Epochs — unlike those,
            # it appears under *every* group (potentially many times per
            # folder: every dataseries, channel, epoch...), so always
            # showing it (as v1 did) added a lot of visual noise for a
            # feature most groups never use. Bootstrapping the first set
            # doesn't need "Sets" to be visible first: "New Set..." is
            # also offered directly on the parent group's own context
            # menu (see FolderBrowserWidget._show_tree_context_menu).
            children = node.children()
            container = getattr(node.owner, node.kind, None)
            if container is not None and len(container.sets) > 0:
                children = [self._sets_node(node)] + children
            return children
        if isinstance(node, _SetsNode):
            container = node.container
            if container is None:
                return []
            return [self._set_node(node, name) for name in container.sets.keys()]
        if isinstance(node, _SetNode):
            container = node.container
            if container is None:
                return []
            members = container.sets.get_items(node.name, default=[])
            if not isinstance(members, list):
                return []
            return [self._set_member_node(node, obj) for obj in members]
        if isinstance(node, (NMFolder, NMToolFolder, NMDataSeries)):
            return self._group_children(node)
        return []  # NMData, NMChannel, NMEpoch, _SetMemberNode are leaves

    def _logical_parent(self, node: object) -> object | None:
        """The node (``_GroupNode``, ``NMObject``, or ``None``) that owns *node*."""
        if isinstance(node, _SetMemberNode):
            return node.set_node
        if isinstance(node, _SetNode):
            return node.sets_node
        if isinstance(node, _SetsNode):
            return node.group
        if isinstance(node, _GroupNode):
            return None if node.kind == "folders" else node.owner
        if isinstance(node, NMFolder):
            return self._group_node("folders", "Folders", self._nm)
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
        if isinstance(node, (_GroupNode, _SetsNode, _SetNode, _SetMemberNode)):
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
        # Group rows (Data, Data Series, Tool Folders, Channels, Epochs,
        # Folders) are selectable too: selecting one is what drives the
        # detail pane (see FolderBrowserWidget), so it needs the normal
        # Qt highlight/feedback like any other row, not just a silent
        # "current" move with nothing visibly selected.
        return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable

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
        self._sets_nodes.clear()
        self._set_nodes.clear()
        self._set_member_nodes.clear()
        self.endResetModel()
