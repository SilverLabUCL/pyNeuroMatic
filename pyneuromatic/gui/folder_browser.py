# -*- coding: utf-8 -*-
"""
FolderBrowserWidget - master-detail view over an NMManager's folder hierarchy.

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

from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.gui.folder_model import (
    FolderTreeModel,
    _GroupNode,
    _SetMemberNode,
    _SetNode,
    _SetsNode,
)


class FolderBrowserWidget(QtWidgets.QWidget):
    """Master-detail view for navigating and editing within an NMManager.

    Left pane is a ``QTreeView`` over the full hierarchy (see
    ``FolderTreeModel``); right pane is a ``QTableView`` showing the
    *children* of whatever is currently selected on the left — the
    classic Finder-column / Mail folder-list-then-message-list pattern.
    Both views share the same model instance; the detail pane simply
    tracks the tree's current selection via ``setRootIndex()``. This is
    Qt-only wiring — neither pane calls ``NMManager.select_value_set()``
    or otherwise touches manager selection state (that coupling turned
    out to be confusing; see the v1 browser).

    Editing is split between the two panes:

    - **New...** — right-click a group row *in the tree* (Folders / Data /
      Data Series / Channels / Epochs — **not** Tool Folders, which NM
      Tools create when they run, not users) -> ``container.new(name)``.
      Deliberately a context menu rather than a modifier-click shortcut:
      modifier keys don't have one consistent meaning across platforms
      (Qt remaps Ctrl/Cmd on macOS, and Control-click there is the OS's
      own secondary-click convention), whereas right-click is unambiguous
      everywhere. This is also the extension point for future
      group-specific actions — e.g. an "Import..." action under
      "Folders" only — by checking the group's ``kind`` in
      :meth:`_show_tree_context_menu` before adding it.
    - **Rename... / Delete / Add to Set** — right-click one or more rows
      *in the detail pane*: ``container.rename(old, new)`` (exactly one
      row), ``del container[name]`` for each selected row (after
      confirmation), or ``container.sets.add(set_name, selected_objects)``
      (an existing set name or "New Set...").

    Both context menus only act on a synthetic group row (Data / Data
    Series / Tool Folders / Channels / Epochs / Folders) — i.e. a single,
    well-defined ``NMObjectContainer``. Selecting a real object (e.g. an
    ``NMFolder``) in the tree shows *its* group rows in the detail pane,
    which are not themselves editable.

    Every group can also have a "Sets" child (every ``NMObjectContainer``
    has a ``.sets``), with its own parallel editing surface. Unlike the
    other group rows, "Sets" is hidden when the container has no sets
    defined yet — it appears once per group instance (so potentially
    many times per folder), and always showing it turned out to be a lot
    of visual noise for a feature most groups never use. So "New Set..."
    is offered in *two* places: on the parent group's own context menu
    (works even before any set exists — this is how you create the
    first one) and on "Sets" itself once it's visible:

    - **New Set... / Rename Set / Delete Set** — right-click a group row
      or "Sets" *in the tree* -> ``container.sets.add(name)`` /
      ``.rename(old, new)`` / ``del container.sets[name]`` (the latter
      two via a submenu of existing set names, same shape as "Add to
      Set" above, and only offered once a set exists to target). Also
      reachable by selecting "Sets" (populating the detail pane with the
      set names) and right-clicking a row there.
    - **Remove from Set** — select "Sets" -> a specific set to show its
      *members* in the detail pane, then right-click selected row(s).
      This removes membership only (``container.sets.remove(...)``) — it
      never deletes the underlying object, unlike "Delete" above. A
      set's members are the same NMObjects shown elsewhere in the tree
      (e.g. under "Data" directly); see ``_SetMemberNode`` in
      folder_model.py for why they need a distinct wrapper identity
      there rather than reusing the object directly.

    Every edit calls :meth:`refresh` (a full model reset — the core
    object model has no change-notification hooks) rather than
    fine-grained ``insertRows``/``removeRows`` updates. A model reset
    would normally collapse the whole tree and drop the detail pane's
    root, so :meth:`refresh` records which rows were expanded and what
    was selected (by row-label path, since old ``QModelIndex``es don't
    survive a reset) and restores them afterward. If the exact selected
    row is gone (e.g. it was just deleted), it falls back to the nearest
    surviving ancestor instead of losing the selection entirely.

    Double-click / Enter ("activate") on the tree is wired to a
    placeholder for now — reserved for a future "open" action (e.g.
    quick-plot a leaf) that doesn't exist yet.
    """

    selection_requested = QtCore.pyqtSignal(object)
    content_changed = QtCore.pyqtSignal()

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
        self._tree.selectionModel().currentChanged.connect(self._on_tree_current_changed)
        self._tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_tree_context_menu)

        self._detail = QtWidgets.QTableView(self)
        self._detail.setModel(self._model)
        self._detail.horizontalHeader().setStretchLastSection(True)
        self._detail.verticalHeader().hide()
        self._detail.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._detail.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._detail.setRootIndex(QtCore.QModelIndex())
        self._detail.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._detail.customContextMenuRequested.connect(self._show_detail_context_menu)

        splitter = QtWidgets.QSplitter(self)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._detail)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    @property
    def model(self) -> FolderTreeModel:
        return self._model

    @property
    def tree(self) -> QtWidgets.QTreeView:
        return self._tree

    @property
    def detail(self) -> QtWidgets.QTableView:
        return self._detail

    def refresh(self) -> None:
        """Re-read the hierarchy from the manager (see FolderTreeModel.refresh).

        Preserves tree expand state and the current selection across the
        underlying model reset (see class docstring) — falls back to the
        nearest surviving ancestor if the exact selected row is gone.

        Note: this reads ``self._tree.currentIndex()`` *after* a delete
        may already have happened (``_delete_selected()`` deletes, then
        calls this). That's only safe because the tree's current index
        is always a ``_GroupNode`` — never a leaf NMObject — whenever a
        delete is actually reachable: the context menu that triggers
        deletes only appears when the detail pane's root (which mirrors
        the tree's current index) is a group. A `_GroupNode` is never
        freed by deleting one of the NMObjects inside it, so this is
        safe in practice. If the tree's current index ever *were* a leaf
        that just got deleted, reading it here would be a use-after-free:
        the container drops its last reference to the NMObject, CPython
        frees it immediately, and the QModelIndex's internalPointer()
        would dangle.
        """
        expanded_paths = self._expanded_paths()
        current_path = self._index_path(self._tree.currentIndex())

        self._model.refresh()
        self._detail.setRootIndex(QtCore.QModelIndex())

        for path in expanded_paths:
            index = self._index_from_path(path)
            if index.isValid():
                self._tree.setExpanded(index, True)

        current_index = self._index_from_path_or_ancestor(current_path)
        if current_index.isValid():
            # Fires currentChanged -> _on_tree_current_changed, which
            # restores the detail pane's root too.
            self._tree.setCurrentIndex(current_index)

    # ------------------------------------------------------------------
    # Tree state preservation across a model reset (see refresh())

    def _index_path(self, index: QtCore.QModelIndex) -> list[str]:
        """Row labels from the root down to *index* (``[]`` if invalid)."""
        path: list[str] = []
        while index.isValid():
            path.append(self._model.data(index))
            index = self._model.parent(index)
        path.reverse()
        return path

    def _index_from_path(self, path: list[str]) -> QtCore.QModelIndex:
        """Inverse of :meth:`_index_path`: walk the model matching labels."""
        index = QtCore.QModelIndex()
        for label in path:
            child = None
            for row in range(self._model.rowCount(index)):
                candidate = self._model.index(row, 0, index)
                if self._model.data(candidate) == label:
                    child = candidate
                    break
            if child is None:
                return QtCore.QModelIndex()
            index = child
        return index

    def _index_from_path_or_ancestor(self, path: list[str]) -> QtCore.QModelIndex:
        """Resolve *path*, or the nearest surviving ancestor prefix of it."""
        while path:
            index = self._index_from_path(path)
            if index.isValid():
                return index
            path = path[:-1]
        return QtCore.QModelIndex()

    def _expanded_paths(self) -> list[list[str]]:
        """Paths of all currently-expanded rows, deepest-first traversal.

        Only recurses into already-expanded branches — a row can only be
        expanded if its ancestors are too, so this is both correct and
        avoids walking the whole (possibly large) model.
        """
        paths: list[list[str]] = []

        def walk(index: QtCore.QModelIndex) -> None:
            for row in range(self._model.rowCount(index)):
                child = self._model.index(row, 0, index)
                if self._tree.isExpanded(child):
                    paths.append(self._index_path(child))
                    walk(child)

        walk(QtCore.QModelIndex())
        return paths

    def _show_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        """Right-click a group or "Sets" row in the tree.

        Group rows (Data / Data Series / Tool Folders / Channels /
        Epochs / Folders) get "New..." and "New Set...". "Sets" rows
        (only present once a group has >=1 set — see FolderTreeModel's
        docstring) get the same "New Set..." plus Rename/Delete
        submenus over existing set names. Right-clicking a real object
        (e.g. an NMFolder) or empty area shows nothing, same as the
        detail pane's context menu. Extension point for future
        group-specific actions, e.g.::

            if node.kind == "folders":
                menu.addAction("Import...", self._import_into_folders)

        "Tool Folders" gets no "New...": tool folders are created by NM
        Tools when they run, not manually by users through this browser
        ("New Set..." is still offered there — grouping *existing*
        tool folders into a set is unrelated to who creates them).
        """
        index = self._tree.indexAt(pos)
        if not index.isValid():
            return
        node = index.internalPointer()
        menu = QtWidgets.QMenu(self)
        if isinstance(node, _GroupNode):
            container = getattr(node.owner, node.kind, None)
            if container is None:
                return
            if node.kind != "toolfolders":
                menu.addAction("New...", lambda: self._add_item(container))
            menu.addAction("New Set...", lambda: self._add_set(self._model._sets_node(node)))
            if node.kind == "epochs":
                menu.addAction(
                    "New Group...",
                    lambda: self._new_group(container),
                )
        elif isinstance(node, _SetsNode):
            self._populate_sets_menu(menu, node)
        elif isinstance(node, _SetNode):
            menu.addAction("Select Set", lambda: self._request_selection(index))
        elif isinstance(node, (_SetMemberNode, NMObject)):
            menu.addAction("Select", lambda: self._request_selection(index))
        else:
            return
        if not menu.actions():
            return
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _populate_sets_menu(self, menu: QtWidgets.QMenu, sets_node: _SetsNode) -> None:
        """New Set... / Rename Set / Delete Set actions for *sets_node*.

        Shared by the tree's "Sets" row context menu and the detail
        pane's context menu when its root is a "Sets" row.
        """
        menu.addAction("New Set...", lambda: self._add_set(sets_node))
        container = sets_node.container
        existing = list(container.sets.keys()) if container is not None else []
        if existing:
            rename_menu = menu.addMenu("Rename Set")
            for name in existing:
                rename_menu.addAction(
                    name, lambda checked=False, n=name: self._rename_set(sets_node, n)
                )
            delete_menu = menu.addMenu("Delete Set")
            for name in existing:
                delete_menu.addAction(
                    name, lambda checked=False, n=name: self._delete_sets(sets_node, [n])
                )

    def _on_activated(self, index: QtCore.QModelIndex) -> None:
        """Placeholder for a future "open" action (e.g. quick-plot a leaf).

        No behavior yet — v1 has no viewer to open a leaf into. Reserved
        so wiring one in later doesn't require new signal plumbing.
        """
        pass

    def _on_tree_current_changed(
        self,
        current: QtCore.QModelIndex,
        previous: QtCore.QModelIndex,
    ) -> None:
        self._detail.setRootIndex(current if current.isValid() else QtCore.QModelIndex())

    def _request_selection(self, index: QtCore.QModelIndex) -> None:
        self.selection_requested.emit(self._selection_for_index(index))

    @staticmethod
    def _selection_for_index(index: QtCore.QModelIndex) -> dict[str, object | None]:
        if not index.isValid():
            return {"object": None, "set": None}

        node = index.internalPointer()
        if isinstance(node, _SetMemberNode):
            return {"object": node.obj, "set": node.set_node.name}
        if isinstance(node, _SetNode):
            owner = node.sets_node.group.owner
            return {
                "object": owner if isinstance(owner, NMObject) else None,
                "set": node.name,
            }
        if isinstance(node, _SetsNode):
            owner = node.group.owner
            return {
                "object": owner if isinstance(owner, NMObject) else None,
                "set": None,
            }
        if isinstance(node, _GroupNode):
            owner = node.owner
            return {
                "object": owner if isinstance(owner, NMObject) else None,
                "set": None,
            }
        if isinstance(node, NMObject):
            return {"object": node, "set": None}
        return {"object": None, "set": None}

    # ------------------------------------------------------------------
    # Detail pane: container/selection helpers

    def _current_container(self):
        """The NMObjectContainer whose children the detail pane is showing.

        None if the detail pane has no root, or its root is a real
        NMObject (its rows are group nodes, not editable items).
        """
        root = self._detail.rootIndex()
        if not root.isValid():
            return None
        node = root.internalPointer()
        if isinstance(node, _GroupNode):
            return getattr(node.owner, node.kind, None)
        return None

    def _selected_objects(self) -> list[NMObject]:
        sel_model = self._detail.selectionModel()
        if sel_model is None:
            return []
        objects = []
        for index in sel_model.selectedRows():
            node = index.internalPointer()
            if isinstance(node, NMObject):
                objects.append(node)
        return objects

    def _selected_set_nodes(self) -> list[_SetNode]:
        """Selected set-name rows, when the detail pane's root is "Sets"."""
        sel_model = self._detail.selectionModel()
        if sel_model is None:
            return []
        return [
            node
            for index in sel_model.selectedRows()
            if isinstance(node := index.internalPointer(), _SetNode)
        ]

    def _selected_set_members(self) -> list[NMObject]:
        """Selected members, when the detail pane's root is a specific set."""
        sel_model = self._detail.selectionModel()
        if sel_model is None:
            return []
        return [
            node.obj
            for index in sel_model.selectedRows()
            if isinstance(node := index.internalPointer(), _SetMemberNode)
        ]

    # ------------------------------------------------------------------
    # Context menu

    def _show_detail_context_menu(self, pos: QtCore.QPoint) -> None:
        """Context menu for the detail pane, dispatched by its root's kind.

        - Group root: Rename/Delete/Add to Set for selected item(s) (see
          :meth:`_show_detail_context_menu` body below).
        - "Sets" root: Rename Set / Delete Set for selected set name(s).
        - A specific set's root: Remove from Set for selected member(s) —
          removes membership only, never deletes the underlying object.

        "New..." / "New Set..." live in the tree's context menu instead
        (see :meth:`_show_tree_context_menu`) — nothing here to act on
        without a selection, so this shows nothing when empty.
        """
        root = self._detail.rootIndex()
        root_node = root.internalPointer() if root.isValid() else None

        if isinstance(root_node, _SetsNode):
            self._show_sets_list_context_menu(pos, root_node)
            return
        if isinstance(root_node, _SetNode):
            self._show_set_members_context_menu(pos, root_node)
            return

        container = self._current_container()
        if container is None:
            return  # detail root is a real object (group rows) - nothing editable

        selected = self._selected_objects()
        if not selected:
            return

        menu = QtWidgets.QMenu(self)
        if len(selected) == 1:
            menu.addAction("Rename...", self._rename_selected)
        menu.addAction("Delete", self._delete_selected)
        if hasattr(container, "sets"):
            set_menu = menu.addMenu("Add to Set")
            for set_name in container.sets.keys():
                set_menu.addAction(
                    set_name,
                    lambda checked=False, sn=set_name: self._add_selected_to_set(sn),
                )
            if container.sets.keys():
                set_menu.addSeparator()
            set_menu.addAction("New Set...", self._add_selected_to_new_set)
        selected_epochs = [obj for obj in selected if isinstance(obj, NMEpoch)]
        if selected_epochs and hasattr(container, "groups"):
            group_menu = menu.addMenu("Add to Group")
            for group_number in container.groups.group_numbers:
                group_menu.addAction(
                    f"Group {group_number}",
                    lambda checked=False, number=group_number:
                    self._assign_selected_to_group(container, selected_epochs, number),
                )
        menu.exec(self._detail.viewport().mapToGlobal(pos))

    def _show_sets_list_context_menu(self, pos: QtCore.QPoint, sets_node: _SetsNode) -> None:
        """Rename Set / Delete Set for whichever set-name row(s) are selected.

        "New Set..." is deliberately not offered here — it lives only in
        the tree's context menu on the "Sets" row, matching how "New..."
        for regular items is tree-only too.
        """
        selected = self._selected_set_nodes()
        if not selected:
            return
        menu = QtWidgets.QMenu(self)
        if len(selected) == 1:
            name = selected[0].name
            menu.addAction("Rename Set...", lambda: self._rename_set(sets_node, name))
        menu.addAction(
            "Delete Set",
            lambda: self._delete_sets(sets_node, [n.name for n in selected]),
        )
        menu.exec(self._detail.viewport().mapToGlobal(pos))

    def _show_set_members_context_menu(self, pos: QtCore.QPoint, set_node: _SetNode) -> None:
        """Remove from Set for whichever member row(s) are selected."""
        selected = self._selected_set_members()
        if not selected:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Remove from Set", lambda: self._remove_from_set(set_node, selected))
        menu.exec(self._detail.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Actions

    def _add_item(self, container=None) -> None:
        """Prompt for a name and add a new item to *container*.

        Defaults to the detail pane's current container (the "New..."
        context-menu action); pass one explicitly for the Ctrl+Click
        tree shortcut, which may target a different container than
        whatever the detail pane happens to be showing.
        """
        if container is None:
            container = self._current_container()
        if container is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Item", "Name (leave blank for auto):"
        )
        if not ok:
            return
        try:
            container.new(name=name or None)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Add Failed", str(e))
            return
        self.refresh()

    def _rename_selected(self) -> None:
        container = self._current_container()
        if container is None:
            return
        selected = self._selected_objects()
        if len(selected) != 1:
            return
        obj = selected[0]
        newname, ok = QtWidgets.QInputDialog.getText(
            self, "Rename", "New name:", text=obj.name
        )
        if not ok or not newname:
            return
        try:
            container.rename(obj.name, newname)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Rename Failed", str(e))
            return
        self.refresh()

    def _delete_selected(self) -> None:
        container = self._current_container()
        if container is None:
            return
        selected = self._selected_objects()
        if not selected:
            return
        names = ", ".join(o.name for o in selected)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete",
            "Delete %s?" % names,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            for obj in selected:
                del container[obj.name]
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Delete Failed", str(e))
        self.refresh()

    def _add_selected_to_set(self, set_name: str) -> None:
        container = self._current_container()
        if container is None:
            return
        selected = self._selected_objects()
        if not selected:
            return
        try:
            container.sets.add(set_name, selected)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Add to Set Failed", str(e))

    def _add_selected_to_new_set(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "New Set", "Set name:")
        if not ok or not name:
            return
        self._add_selected_to_set(name)

    def _add_set(self, sets_node: _SetsNode) -> None:
        """Create a new, empty named set (tree's "Sets" row -> New Set...)."""
        container = sets_node.container
        if container is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New Set", "Set name:")
        if not ok or not name:
            return
        try:
            container.sets.add(name)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Add Set Failed", str(e))
            return
        self.refresh()

    def _new_group(self, epoch_container) -> None:
        existing = epoch_container.groups.group_numbers
        group_count, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "New Group",
            "Number of groups to create:",
            value=1,
            min=1,
        )
        if not accepted:
            return
        try:
            for group_number in range(group_count):
                if group_number not in existing:
                    epoch_container.groups.add_group(group_number)
        except (KeyError, ValueError, TypeError) as error:
            QtWidgets.QMessageBox.warning(self, "Group Creation Failed", str(error))
            return
        self.content_changed.emit()

    def _assign_selected_to_group(
        self,
        epoch_container,
        epochs: list[NMObject],
        group_number: int,
    ) -> None:
        try:
            for epoch in epochs:
                epoch_container.groups.assign(epoch.name, group_number)
        except (KeyError, ValueError, TypeError) as error:
            QtWidgets.QMessageBox.warning(self, "Group Assignment Failed", str(error))
            return
        self.content_changed.emit()

    def _rename_set(self, sets_node: _SetsNode, old_name: str) -> None:
        container = sets_node.container
        if container is None:
            return
        newname, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Set", "New name:", text=old_name
        )
        if not ok or not newname:
            return
        try:
            # NMSets.rename() renames the set itself; rename_item() is a
            # different thing entirely (renaming a *member's* name across
            # all sets, used when the member object itself gets renamed).
            # rename() also doesn't validate `old_name` exists before
            # using it, raising AttributeError (not KeyError) if it's
            # missing - catch that too.
            container.sets.rename(old_name, newname)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            QtWidgets.QMessageBox.warning(self, "Rename Set Failed", str(e))
            return
        self.refresh()

    def _delete_sets(self, sets_node: _SetsNode, names: list[str]) -> None:
        container = sets_node.container
        if container is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Set",
            "Delete set%s %s?" % ("s" if len(names) != 1 else "", ", ".join(names)),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            for name in names:
                del container.sets[name]
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Delete Set Failed", str(e))
        self.refresh()

    def _remove_from_set(self, set_node: _SetNode, objects: list[NMObject]) -> None:
        """Remove *objects* from the set — membership only, never deletes them."""
        container = set_node.container
        if container is None:
            return
        try:
            container.sets.remove(set_node.name, objects)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Remove from Set Failed", str(e))
            return
        self.refresh()
