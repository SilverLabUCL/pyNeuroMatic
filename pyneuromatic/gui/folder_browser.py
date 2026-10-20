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

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_epoch import NMEpoch, NMEpochContainer
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.tools.nm_tool_folder import NMToolFolder
from pyneuromatic.gui.info_panel import INFO_TYPES, InfoPanel, NotesWidget
from pyneuromatic.gui.membership import MembershipModel, unassign_from_group
from pyneuromatic.gui.folder_model import (
    FolderTreeModel,
    _GroupMemberNode,
    _GroupNode,
    _GroupNumberNode,
    _GroupsNode,
    _SetMemberNode,
    _SetNode,
    _SetsNode,
)
from pyneuromatic.gui.tier_actions import (
    ask_new_data,
    copy_name,
    copy_target_name,
    create_data,
    name_error,
    new_data_defaults,
    series_name_error,
    series_name_hint,
)


# Rows with Set / Group actions (the same in the tree and the detail pane)
_SET_GROUP_NODES = (_SetsNode, _SetNode, _GroupsNode, _GroupNumberNode)


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
    # A data array was created with New…; the app shows it in the Table
    data_created = QtCore.pyqtSignal(object)
    # The info panel's Edit Scales... was clicked for a data array
    edit_scales_requested = QtCore.pyqtSignal(object)

    def __init__(
        self,
        manager: NMManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._nm = manager
        self._model = FolderTreeModel(manager, parent=self)
        self._folder_actions = None  # set by the app (see set_folder_actions)
        self._series_actions = None  # set by the app (see set_series_actions)

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

        # Edit mode: when a set or group is chosen in the tree, the second
        # column can list every item with a checkbox for membership
        self._edit_checkbox = QtWidgets.QCheckBox("Edit", self)
        self._edit_checkbox.setToolTip("List every item with a checkbox to add or remove it")
        self._all_button = QtWidgets.QToolButton(self)
        self._all_button.setText("All")
        self._none_button = QtWidgets.QToolButton(self)
        self._none_button.setText("Clear")
        self._invert_button = QtWidgets.QToolButton(self)
        self._invert_button.setText("Invert")
        edit_row = QtWidgets.QHBoxLayout()
        edit_row.setContentsMargins(4, 2, 4, 2)
        edit_row.addStretch(1)
        for widget in (self._all_button, self._none_button, self._invert_button,
                       self._edit_checkbox):
            edit_row.addWidget(widget)
        self._edit_bar = QtWidgets.QWidget(self)
        self._edit_bar.setLayout(edit_row)
        self._edit_bar.hide()

        self._membership_view = QtWidgets.QTableView(self)
        self._membership_view.verticalHeader().hide()
        self._membership_view.horizontalHeader().setStretchLastSection(True)
        self._membership_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._membership_view.viewport().installEventFilter(self)
        self._membership_model: MembershipModel | None = None
        self._last_toggled_row: int | None = None
        # Rows selected when the mouse went down (a click changes the selection)
        self._press_rows: list[int] | None = None
        self._reselect_rows: list[int] = []

        # A single object (folder, data array, epoch...) is described here
        # rather than listed: its rows are already in the tree
        self._info_panel = InfoPanel(self)
        self._info_panel.edit_scales_requested.connect(self.edit_scales_requested)

        self._detail_stack = QtWidgets.QStackedWidget(self)
        self._detail_stack.addWidget(self._detail)
        self._detail_stack.addWidget(self._membership_view)
        self._detail_stack.addWidget(self._info_panel)
        right = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._edit_bar)
        self._target_notes = NotesWidget(self)
        self._target_notes.hide()
        self._detail_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        self._detail_split.addWidget(self._detail_stack)
        self._detail_split.addWidget(self._target_notes)
        self._detail_split.setStretchFactor(0, 3)
        self._detail_split.setStretchFactor(1, 1)
        self._detail_split.setChildrenCollapsible(False)
        right_layout.addWidget(self._detail_split, stretch=1)

        self._edit_checkbox.toggled.connect(self._on_edit_toggled)
        self._all_button.clicked.connect(lambda: self._edit_all("all"))
        self._none_button.clicked.connect(lambda: self._edit_all("none"))
        self._invert_button.clicked.connect(lambda: self._edit_all("invert"))

        splitter = QtWidgets.QSplitter(self)
        splitter.addWidget(self._tree)
        splitter.addWidget(right)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Start with the top Folders row open, so the folders are listed, and
        # the current folder open too, showing its Data and Data Series
        folders = self._model.index(0, 0)
        self._tree.expand(folders)
        current = self._nm.folders.selected_value
        for row in range(self._model.rowCount(folders)):
            index = self._model.index(row, 0, folders)
            if current is not None and index.internalPointer() is current:
                self._tree.expand(index)

    def set_folder_actions(self, actions) -> None:
        """Use the strip's Folder actions (Duplicate, Rename, Delete...) for
        folders and subfolders, so they behave the same everywhere."""
        self._folder_actions = actions

    def set_series_actions(self, actions) -> None:
        """Use the strip's Data Series actions (the New Series dialog)."""
        self._series_actions = actions

    @property
    def model(self) -> FolderTreeModel:
        return self._model

    @property
    def tree(self) -> QtWidgets.QTreeView:
        return self._tree

    @property
    def detail(self) -> QtWidgets.QTableView:
        return self._detail

    @property
    def info_panel(self) -> InfoPanel:
        return self._info_panel

    @property
    def is_showing_info(self) -> bool:
        return self._detail_stack.currentWidget() is self._info_panel

    @property
    def target_notes(self) -> NotesWidget:
        """The notes of the set or group chosen in the tree (below its list)."""
        return self._target_notes

    @property
    def edit_checkbox(self) -> QtWidgets.QCheckBox:
        return self._edit_checkbox

    @property
    def membership_view(self) -> QtWidgets.QTableView:
        return self._membership_view

    @property
    def membership_model(self) -> MembershipModel | None:
        """The checkbox list shown in edit mode, if any."""
        return self._membership_model if self.is_editing else None

    @property
    def is_editing(self) -> bool:
        return self._detail_stack.currentWidget() is self._membership_view

    # ------------------------------------------------------------------
    # Edit mode: set / group membership with checkboxes

    @staticmethod
    def _edit_target(node) -> tuple | None:
        """(container, set name, group number, note) for an editable row."""
        if isinstance(node, _SetNode) and node.container is not None:
            note = None
            if node.container.sets.is_equation(node.name):
                note = "A combined (AND/OR) set: edit the sets it combines"
            return node.container, node.name, None, note
        if isinstance(node, _GroupNumberNode) and node.container is not None:
            return node.container, None, node.number, None
        return None

    def _update_edit_mode(self) -> None:
        """Show the Edit bar for a set or group row, and keep edit mode on
        while moving between sets and groups."""
        target = self._edit_target(self._tree.currentIndex().internalPointer())
        self._edit_bar.setVisible(target is not None)
        if target is None or target[3] is not None:
            self._edit_checkbox.blockSignals(True)
            self._edit_checkbox.setChecked(False)
            self._edit_checkbox.blockSignals(False)
            self._edit_checkbox.setEnabled(target is not None and target[3] is None)
            self._edit_checkbox.setToolTip(target[3] if target else "")
            self._show_edit_list(None)
        else:
            self._edit_checkbox.setEnabled(True)
            self._edit_checkbox.setToolTip("List every item with a checkbox to add or remove it")
            self._show_edit_list(target if self._edit_checkbox.isChecked() else None)
        self._update_target_notes()  # after Edit is settled: editing hides the notes

    def _notes_target(self):
        """(notes, owner name) of the set or group chosen in the tree, or None."""
        target = self._edit_target(self._tree.currentIndex().internalPointer())
        if target is None:
            return None
        container, set_name, group, _note = target
        if set_name is not None and set_name in container.sets:
            return container.sets.notes(set_name), set_name
        if group is not None and group in container.groups.group_numbers:
            return container.groups.notes(group), f"Group {group}"
        return None

    def _update_target_notes(self, *_args) -> None:
        """Show the chosen set's or group's notes below its list, except while
        editing its members (the checkbox list takes the column then)."""
        found = self._notes_target()
        editing = self._edit_checkbox.isChecked() and self._edit_checkbox.isEnabled()
        show = found is not None and not editing
        self._target_notes.setVisible(show)
        if show:
            self._target_notes.set_notes(*found)

    def _show_edit_list(self, target) -> None:
        editing = target is not None
        for button in (self._all_button, self._none_button, self._invert_button):
            button.setVisible(editing)
        if not editing:
            self._detail_stack.setCurrentWidget(self._detail)
            return
        container, set_name, group, _note = target
        model = self._membership_model
        if model is None or not model.same_target(container, set_name, group):
            model = MembershipModel(container, set_name=set_name, group=group, parent=self)
            model.membership_changed.connect(self._on_membership_changed)
            model.toggled.connect(self._on_member_toggled)
            self._membership_model = model
            self._membership_view.setModel(model)
            self._last_toggled_row = None
        else:
            model.reload()  # the set or group may have changed elsewhere
        noun = "epochs" if hasattr(container, "groups") else "items"
        # Groups are exclusive, so Invert (and All) would raid other groups
        self._all_button.setText("Ungrouped" if model.is_group else "All")
        self._all_button.setToolTip(
            f"Tick every {noun[:-1]} that is in no group" if model.is_group
            else f"Tick every {noun[:-1]}"
        )
        self._invert_button.setVisible(not model.is_group)
        self._detail_stack.setCurrentWidget(self._membership_view)

    def _on_edit_toggled(self, _checked: bool) -> None:
        self._update_edit_mode()

    def eventFilter(self, watched, event) -> bool:
        if watched is self._membership_view.viewport():
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                self._press_rows = self._selected_member_rows()
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                QtCore.QTimer.singleShot(0, self._forget_press_rows)
        return super().eventFilter(watched, event)

    def _forget_press_rows(self) -> None:
        self._press_rows = None

    def _selected_member_rows(self) -> list[int]:
        sel_model = self._membership_view.selectionModel()
        if sel_model is None:
            return []
        return sorted(index.row() for index in sel_model.selectedRows())

    def _reselect_member_rows(self) -> None:
        self._select_member_rows(self._reselect_rows)

    def _select_member_rows(self, rows: list[int]) -> None:
        model = self._membership_model
        sel_model = self._membership_view.selectionModel()
        if model is None or sel_model is None:
            return
        selection = QtCore.QItemSelection()
        for row in rows:
            selection.select(model.index(row, 0),
                             model.index(row, model.columnCount() - 1))
        sel_model.select(selection,
                         QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _on_member_toggled(self, row: int, checked: bool) -> None:
        # Ticking a box inside a multi-row selection ticks the whole selection
        rows = self._press_rows if self._press_rows is not None \
            else self._selected_member_rows()
        if row in rows and len(rows) > 1:
            self._membership_model.set_checked(rows, checked)
            self._last_toggled_row = row
            # Keep the selection, after the click has finished changing it
            self._reselect_rows = rows
            QtCore.QTimer.singleShot(0, self._reselect_member_rows)
            return
        # Shift-click applies the same state to every row since the last click
        last = self._last_toggled_row
        shift = QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
        if shift and last is not None and last != row:
            low, high = sorted((last, row))
            self._membership_model.set_checked(range(low, high + 1), checked)
        self._last_toggled_row = row

    def _edit_all(self, how: str) -> None:
        model = self._membership_model
        if model is None:
            return
        if how == "all":
            # A group takes only ungrouped epochs: All would empty other groups
            (model.check_ungrouped if model.is_group else model.check_all)()
        else:
            {"none": model.check_none, "invert": model.invert}[how]()

    def _on_membership_changed(self, _count: int) -> None:
        # Reload the tree (its set/group member rows) and tell the app
        self._changed()

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
        """Right-click any row in the tree.

        - Group rows (Folders, Data, Data Series, Channels, Epochs): New...
          ("Tool Folders" has none: tools create them when they run).
        - Sets / set / Groups / group rows: see :meth:`_populate_node_menu`.
        - Objects (folder, data array, epoch...): the same menu as in the
          second column (see :meth:`_populate_object_menu`).
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
            # New sets and groups are made on the Sets and Groups rows
            if node.kind == "folders" and self._folder_actions is not None:
                menu.addAction("New...", self._folder_actions.new_folder)
            elif node.kind == "dataseries" and self._series_actions is not None:
                # The New Series dialog (prefix, channels, epochs, points...)
                menu.addAction("New...", lambda: self._series_actions.new_series(node.owner))
            elif node.kind != "toolfolders":
                menu.addAction("New...", lambda: self._add_item(container))
        elif isinstance(node, _SET_GROUP_NODES):
            self._populate_node_menu(menu, index)
        elif isinstance(node, _SetMemberNode):
            menu.addAction("Select", lambda: self._request_selection(index))
            menu.addAction("Remove from Set",
                           lambda: self._remove_from_set(node.set_node, [node.obj]))
        elif isinstance(node, _GroupMemberNode):
            menu.addAction("Select", lambda: self._request_selection(index))
            menu.addAction("Remove from Group",
                           lambda: self._remove_from_group(node.group_number_node, [node.obj]))
        elif isinstance(node, NMObject):
            self._populate_object_menu(menu, [node], node._container, index)
        else:
            return
        if not menu.actions():
            return
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _populate_object_menu(self, menu: QtWidgets.QMenu, objects: list[NMObject],
                              container, index: QtCore.QModelIndex | None = None) -> None:
        """Actions for folders, data arrays, epochs... — the same in both
        columns (*index* is the row, for Select, when there is one object)."""
        if not objects or container is None:
            return
        single = objects[0] if len(objects) == 1 else None
        if single is not None and index is not None:
            menu.addAction("Select", lambda: self._request_selection(index))
        first = objects[0]
        actions = self._folder_actions
        # Folders and subfolders: the strip's Folder actions, as in its ⋯ menu
        if single is not None and actions is not None:
            if isinstance(first, NMFolder):
                menu.addAction("Duplicate...", lambda: actions.duplicate_folder(first.name))
                menu.addAction("Rename...", lambda: actions.rename_folder(first.name))
                menu.addAction("Delete...", lambda: actions.delete_folder(first.name))
                return
            if isinstance(first, NMToolFolder):
                menu.addAction("Duplicate...", lambda: actions.duplicate_subfolder(first))
                menu.addAction("Rename...", lambda: actions.rename_subfolder(first))
                menu.addAction("Delete...", lambda: actions.delete_subfolder(first))
                return
        # Channels and epochs are named by the data series: no Rename
        if single is not None and not isinstance(first, (NMChannel, NMEpoch)):
            menu.addAction("Rename...", lambda: self._rename_object(container, first))
        if isinstance(container, NMDataContainer):
            menu.addAction("Duplicate", lambda: self._duplicate_selected(container, objects))
            folders = [folder for folder in self._nm.folders.values()
                       if folder.data is not container]
            if folders:
                copy_menu = menu.addMenu("Copy to Folder")
                for folder in folders:
                    copy_menu.addAction(
                        folder.name,
                        lambda checked=False, f=folder: self._copy_selected_to(container, objects, f),
                    )
        # Sets are for data arrays and epochs (others may come later)
        if isinstance(container, (NMDataContainer, NMEpochContainer)):
            set_menu = menu.addMenu("Add to Set")
            for set_name in container.sets.keys():
                set_menu.addAction(
                    set_name,
                    lambda checked=False, name=set_name: self._add_to_set(container, objects, name),
                )
            if container.sets.keys():
                set_menu.addSeparator()
            set_menu.addAction("New...", lambda: self._add_to_new_set(container, objects))
        epochs = [obj for obj in objects if isinstance(obj, NMEpoch)]
        if epochs and hasattr(container, "groups") and container.groups.group_numbers:
            group_menu = menu.addMenu("Add to Group")
            for number in container.groups.group_numbers:
                group_menu.addAction(
                    f"Group {number}",
                    lambda checked=False, n=number:
                    self._assign_selected_to_group(container, epochs, n),
                )
        menu.addAction("Delete...", lambda: self._delete_objects(container, objects))  # last

    def _populate_node_menu(self, menu: QtWidgets.QMenu, index: QtCore.QModelIndex) -> None:
        """Actions for a Sets, set, Groups or group row — shared by both panes."""
        node = index.internalPointer()
        if isinstance(node, _GroupsNode) and node.container is not None:
            menu.addAction("New Groups...", lambda: self._new_group(node.container))
            numbers = node.container.groups.group_numbers
            if numbers:
                every = [self._model._group_number_node(node, n) for n in numbers]
                menu.addAction("Clear Groups", lambda: self._clear_groups(every))
                delete_menu = menu.addMenu("Delete Group")
                for number_node in every:
                    delete_menu.addAction(
                        str(number_node.number),
                        lambda checked=False, n=number_node: self._delete_groups([n]),
                    )
                delete_menu.addSeparator()
                delete_menu.addAction("All Groups", lambda: self._delete_groups(every))
        elif isinstance(node, _GroupNumberNode):
            menu.addAction("Select", lambda: self._request_selection(index))
            menu.addAction("Clear...", lambda: self._clear_groups([node]))
            menu.addAction("Delete...", lambda: self._delete_groups([node]))
        elif isinstance(node, _SetsNode):
            self._populate_sets_menu(menu, node)
        elif isinstance(node, _SetNode):
            menu.addAction("Select", lambda: self._request_selection(index))
            menu.addAction("Rename...",
                           lambda: self._rename_set(node.sets_node, node.name))
            if not self._is_combined_set(node):
                menu.addAction("Clear...",
                               lambda: self._clear_sets(node.sets_node, [node.name]))
            menu.addAction("Delete...",
                           lambda: self._delete_sets(node.sets_node, [node.name]))

    def _populate_sets_menu(self, menu: QtWidgets.QMenu, sets_node: _SetsNode) -> None:
        """New... for a "Sets" row (renaming, deleting and clearing are on
        each set's own row)."""
        menu.addAction("New...", lambda: self._add_set(sets_node))

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
        self._hide_set_and_group_rows(current)
        self._update_edit_mode()
        self._update_info_panel(current)

    def _hide_set_and_group_rows(self, root: QtCore.QModelIndex) -> None:
        """List only the items for Data and Epochs: their Sets (and Groups)
        rows are in the tree."""
        node = root.internalPointer() if root.isValid() else None
        items_only = isinstance(node, _GroupNode) and node.kind in ("data", "epochs")
        for row in range(self._model.rowCount(root)):
            child = self._model.index(row, 0, root).internalPointer()
            self._detail.setRowHidden(
                row, items_only and isinstance(child, (_SetsNode, _GroupsNode))
            )

    def _update_info_panel(self, current: QtCore.QModelIndex) -> None:
        """Describe a single object; group rows keep their list."""
        node = current.internalPointer() if current.isValid() else None
        if isinstance(node, (_SetMemberNode, _GroupMemberNode)):
            node = node.obj  # a set or group member is the object itself
        if isinstance(node, INFO_TYPES):
            self._info_panel.show_object(node)
            self._detail_stack.setCurrentWidget(self._info_panel)
        elif self._detail_stack.currentWidget() is self._info_panel:
            self._info_panel.show_object(None)
            self._detail_stack.setCurrentWidget(self._detail)

    def _request_selection(self, index: QtCore.QModelIndex) -> None:
        self.selection_requested.emit(self._selection_for_index(index))

    @staticmethod
    def _selection_for_index(index: QtCore.QModelIndex) -> dict[str, object | None]:
        if not index.isValid():
            return {"object": None, "set": None}

        node = index.internalPointer()
        if isinstance(node, _GroupMemberNode):
            return {"object": node.obj, "set": None, "group": node.group_number_node.number}
        if isinstance(node, _GroupNumberNode):
            return {"object": node.groups_node.group.owner, "set": None, "group": node.number}
        if isinstance(node, _GroupsNode):
            return {"object": node.group.owner, "set": None}
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
        return self._selected_members(_SetMemberNode)

    def _selected_group_members(self) -> list[NMObject]:
        """Selected epochs, when the detail pane's root is a group."""
        return self._selected_members(_GroupMemberNode)

    def _selected_members(self, member_class) -> list[NMObject]:
        sel_model = self._detail.selectionModel()
        if sel_model is None:
            return []
        return [
            node.obj
            for index in sel_model.selectedRows()
            if isinstance(node := index.internalPointer(), member_class)
        ]

    # ------------------------------------------------------------------
    # Context menu

    def _show_detail_context_menu(self, pos: QtCore.QPoint) -> None:
        """Context menu for the detail pane, dispatched by its root's kind.

        - Group root: the selected object(s)' menu, the same as in the tree
          (see :meth:`_populate_object_menu`).
        - "Sets" root: Rename Set / Delete Set / Clear Set for selected sets.
        - A specific set's root: Remove from Set for selected member(s) —
          removes membership only, never deletes the underlying object.
        - A group's root: Remove from Group for selected epoch(s).
        - "Groups" root: Select / Clear / Delete Group for selected group(s).

        A single set or group row (or empty space under a Sets or Groups
        list) gets the same menu as that row has in the tree.
        """
        root = self._detail.rootIndex()
        root_node = root.internalPointer() if root.isValid() else None

        # One set or group row, or empty space under a Sets or Groups list:
        # the same menu as that row has in the tree
        clicked = self._detail.indexAt(pos)
        sel_model = self._detail.selectionModel()
        selected = sel_model.selectedRows() if sel_model is not None else []
        target = None
        if clicked.isValid():
            clicked = clicked.siblingAtColumn(0)
            if isinstance(clicked.internalPointer(), _SET_GROUP_NODES) and (
                not selected or selected == [clicked]
            ):
                target = clicked
        elif isinstance(root_node, (_SetsNode, _GroupsNode)):
            target = root
        if target is not None:
            menu = QtWidgets.QMenu(self)
            self._populate_node_menu(menu, target)
            if menu.actions():
                menu.exec(self._detail.viewport().mapToGlobal(pos))
            return

        if isinstance(root_node, _SetsNode):
            self._show_sets_list_context_menu(pos, root_node)
            return
        if isinstance(root_node, _SetNode):
            self._show_set_members_context_menu(pos, root_node)
            return
        if isinstance(root_node, _GroupNumberNode):
            self._show_group_members_context_menu(pos, root_node)
            return
        if isinstance(root_node, _GroupsNode):
            self._show_groups_list_context_menu(pos)
            return

        container = self._current_container()
        if container is None:
            return  # detail root is a real object (group rows) - nothing editable

        selected = self._selected_objects()
        if not selected:
            return
        menu = QtWidgets.QMenu(self)
        row = clicked if clicked.isValid() and len(selected) == 1 else None
        self._populate_object_menu(menu, selected, container, row)
        if menu.actions():
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
            menu.addAction("Rename...", lambda: self._rename_set(sets_node, name))
        clearable = [n.name for n in selected if not self._is_combined_set(n)]
        if clearable:
            menu.addAction("Clear...", lambda: self._clear_sets(sets_node, clearable))
        menu.addAction(
            "Delete...",
            lambda: self._delete_sets(sets_node, [n.name for n in selected]),
        )
        menu.exec(self._detail.viewport().mapToGlobal(pos))

    def _show_groups_list_context_menu(self, pos: QtCore.QPoint) -> None:
        """Select / Clear... / Delete... for whichever group row(s) are selected."""
        sel_model = self._detail.selectionModel()
        rows = [] if sel_model is None else [
            index for index in sel_model.selectedRows()
            if isinstance(index.internalPointer(), _GroupNumberNode)
        ]
        if not rows:
            return
        selected = [index.internalPointer() for index in rows]
        menu = QtWidgets.QMenu(self)
        if len(rows) == 1:
            menu.addAction("Select", lambda: self._request_selection(rows[0]))
        menu.addAction("Clear...", lambda: self._clear_groups(selected))
        menu.addAction("Delete...", lambda: self._delete_groups(selected))
        menu.exec(self._detail.viewport().mapToGlobal(pos))

    def _show_set_members_context_menu(self, pos: QtCore.QPoint, set_node: _SetNode) -> None:
        """Remove from Set for whichever member row(s) are selected."""
        selected = self._selected_set_members()
        if not selected:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Remove from Set", lambda: self._remove_from_set(set_node, selected))
        menu.exec(self._detail.viewport().mapToGlobal(pos))

    def _show_group_members_context_menu(
        self, pos: QtCore.QPoint, group_node: _GroupNumberNode,
    ) -> None:
        """Remove from Group for whichever epoch row(s) are selected."""
        selected = self._selected_group_members()
        if not selected:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Remove from Group",
                       lambda: self._remove_from_group(group_node, selected))
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
        if isinstance(container, NMDataContainer):
            self._new_data(container)
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
        self._changed()

    def _rename_selected(self) -> None:
        selected = self._selected_objects()
        if len(selected) == 1:
            self._rename_object(self._current_container(), selected[0])

    def _rename_object(self, container, obj: NMObject) -> None:
        if container is None:
            return
        newname, ok = QtWidgets.QInputDialog.getText(
            self, "Rename", "New name:", text=obj.name
        )
        if not ok or not newname:
            return
        # Warn about joining a data series only if the name is free to use
        if (
            isinstance(container, NMDataContainer)
            and newname != obj.name
            and container._getkey(newname) is None
        ):
            hint = series_name_hint(container._parent, newname, joins_now=True)
            if hint and not self._confirm("Rename", hint + "\n\nRename anyway?"):
                return
        try:
            container.rename(obj.name, newname)
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            QtWidgets.QMessageBox.warning(self, "Rename Failed", str(e))
            return
        self._changed()

    def _delete_selected(self) -> None:
        self._delete_objects(self._current_container(), self._selected_objects())

    def _delete_objects(self, container, selected: list[NMObject]) -> None:
        if container is None or not selected:
            return
        names = ", ".join(o.name for o in selected)
        text = "Delete %s?" % names
        if isinstance(container, NMDataContainer):
            in_series = [o.name for o in selected if container.dataseries_of(o.name)]
            if in_series:
                text += (
                    "\n\n%d belong to a data series and will be removed from it "
                    "(data in a locked series is kept)." % len(in_series)
                )
        if not self._confirm("Delete", text):
            return
        problems = []
        for obj in selected:
            try:
                del container[obj.name]
            except (KeyError, ValueError, TypeError, RuntimeError) as e:
                problems.append(str(e))
        if problems:
            QtWidgets.QMessageBox.warning(self, "Delete Failed", "\n".join(problems))
        self._changed()

    def _ask_new_data(self, defaults: dict, validate, context) -> dict | None:
        """The New Data dialog (replaced in tests)."""
        return ask_new_data(defaults, validate, context=context, parent=self)

    def _new_data(self, container: NMDataContainer) -> None:
        """New data array with the same dialog as Data ⋯ › New…"""
        owner = container._parent
        values = self._ask_new_data(
            new_data_defaults(container),
            lambda values: name_error(container, values["name"])
            or series_name_error(owner, values["name"]),
            ("Folder", getattr(owner, "name", "")),
        )
        if values is None:
            return
        hint = series_name_hint(owner, values["name"])
        if hint and not self._confirm("New Data", hint + "\n\nCreate anyway?"):
            return
        data = create_data(container, values)
        self._changed()
        self.data_created.emit(data)

    def show_groups_of(self, series, number: int | None = None) -> None:
        """Select *series*' group *number* (or its Groups row, or Epochs row)."""
        model = self._model

        def find(parent: QtCore.QModelIndex, match):
            for row in range(model.rowCount(parent)):
                index = model.index(row, 0, parent)
                if match(index.internalPointer()):
                    return index
                found = find(index, match)
                if found is not None:
                    return found
            return None

        def is_groups(node) -> bool:
            return isinstance(node, _GroupsNode) and node.group.owner is series

        root = QtCore.QModelIndex()
        index = None
        if number is not None:
            index = find(root, lambda node: isinstance(node, _GroupNumberNode)
                         and is_groups(node.groups_node) and node.number == number)
        index = index or find(root, is_groups)
        if index is None:
            self.show_data_of(series, kind="epochs")
            return
        self._tree.expand(index)
        parent = index.parent()
        while parent.isValid():
            self._tree.expand(parent)
            parent = parent.parent()
        self._tree.setCurrentIndex(index)
        self._tree.scrollTo(index)
        self._tree.setFocus()

    def show_object(self, obj) -> None:
        """Select *obj* (e.g. a data series) in the tree, expanding its parents."""
        model = self._model

        def find(parent: QtCore.QModelIndex):
            for row in range(model.rowCount(parent)):
                index = model.index(row, 0, parent)
                if index.internalPointer() is obj:
                    return index
                found = find(index)
                if found is not None:
                    return found
            return None

        index = find(QtCore.QModelIndex())
        if index is None:
            return
        parent = index.parent()
        while parent.isValid():
            self._tree.expand(parent)
            parent = parent.parent()
        self._tree.setCurrentIndex(index)
        self._tree.scrollTo(index)
        self._tree.setFocus()

    def _changed(self) -> None:
        """Reload the tree after an edit and tell the app (strip, plots...)."""
        self.refresh()
        self.content_changed.emit()

    def _confirm(self, title: str, text: str) -> bool:
        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            text,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _duplicate_selected(self, container, selected: list[NMObject]) -> None:
        for obj in selected:
            container.duplicate(obj.name, copy_name(container, obj.name))
        self._changed()

    def _copy_selected_to(self, container, selected: list[NMObject], folder) -> None:
        for obj in selected:
            container.copy_to(obj.name, folder.data, copy_target_name(folder.data, obj.name))
        self._changed()

    def show_data_of(
        self, owner, sets: bool = False, select: str | None = None, kind: str = "data",
    ) -> None:
        """Show *owner*'s (a folder's or subfolder's) data in the browser.

        Without *sets*: expand Data and select the data array named
        *select* in the tree (the detail pane shows that array).
        With *sets*: expand Data > Sets and select the set named *select* in
        the tree, so the detail pane lists its arrays.
        Without a matching *select*, the Data (or Sets) row is selected; with
        no sets, the Data row is. *kind* "epochs" does the same for a data
        series' Epochs row and its epoch sets (*owner* is the series).
        """
        model = self._model

        def is_data_group(node) -> bool:
            return isinstance(node, _GroupNode) and node.kind == kind and node.owner is owner

        def is_data_sets(node) -> bool:
            return isinstance(node, _SetsNode) and is_data_group(node.group)

        def is_named_set(node) -> bool:
            return (
                isinstance(node, _SetNode)
                and is_data_sets(node.sets_node)
                and select is not None
                and node.name.lower() == select.lower()
            )

        def find(parent: QtCore.QModelIndex, match):
            for row in range(model.rowCount(parent)):
                index = model.index(row, 0, parent)
                if match(index.internalPointer()):
                    return index
                found = find(index, match)
                if found is not None:
                    return found
            return None

        def named_data(group_index: QtCore.QModelIndex):
            """The row of data *select* directly under the Data row."""
            if select is None:
                return None
            for row in range(model.rowCount(group_index)):
                index = model.index(row, 0, group_index)
                node = index.internalPointer()
                if isinstance(node, NMObject) and node.name.lower() == select.lower():
                    return index
            return None

        root = QtCore.QModelIndex()
        index = None
        if sets:
            index = find(root, is_named_set) or find(root, is_data_sets)
        if index is None:
            index = find(root, is_data_group)
            if index is not None and not sets:
                index = named_data(index) or index
        if index is None:
            return
        self._tree.expand(index)
        parent = index.parent()
        while parent.isValid():
            self._tree.expand(parent)
            parent = parent.parent()
        self._tree.setCurrentIndex(index)
        self._tree.scrollTo(index)
        self._tree.setFocus()

    def _add_selected_to_set(self, set_name: str) -> None:
        self._add_to_set(self._current_container(), self._selected_objects(), set_name)

    def _add_to_set(self, container, selected: list[NMObject], set_name: str) -> None:
        if container is None or not selected:
            return
        try:
            container.sets.add(set_name, selected)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Add to Set Failed", str(e))
            return
        self._changed()

    def _add_selected_to_new_set(self) -> None:
        self._add_to_new_set(self._current_container(), self._selected_objects())

    def _ask_set_name(self, container) -> tuple[str, bool]:
        """The New Set prompt, filled with the next free name (e.g. "Set3")."""
        return QtWidgets.QInputDialog.getText(
            self, "New Set", "Set name:", text=container.sets.auto_name_next()
        )

    def _add_to_new_set(self, container, selected: list[NMObject]) -> None:
        if container is None:
            return
        name, ok = self._ask_set_name(container)
        if not ok or not name:
            return
        self._add_to_set(container, selected, name)

    def _add_set(self, sets_node: _SetsNode) -> None:
        """Create a new, empty named set (a "Sets" row's New...)."""
        container = sets_node.container
        if container is None:
            return
        name, ok = self._ask_set_name(container)
        if not ok or not name:
            return
        try:
            container.sets.add(name)
        except (KeyError, ValueError, TypeError) as e:
            QtWidgets.QMessageBox.warning(self, "Add Set Failed", str(e))
            return
        self._changed()

    def _new_group(self, epoch_container) -> None:
        existing = epoch_container.groups.group_numbers
        group_count, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "New Groups",
            "Number of groups (empty groups 0, 1, ... are created):",
            value=max(len(existing) + 1, 2),
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
        self._changed()

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
        self._changed()

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
        self._changed()

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
        self._changed()

    @staticmethod
    def _is_combined_set(set_node: _SetNode) -> bool:
        """An AND/OR set: clearing it would replace its definition."""
        container = set_node.container
        return container is not None and container.sets.is_equation(set_node.name)

    def _clear_sets(self, sets_node: _SetsNode, names: list[str]) -> None:
        """Remove every member of the sets — the sets and members are kept."""
        container = sets_node.container
        if container is None or not names:
            return
        if not self._confirm(
            "Clear Set",
            "Remove every item from set%s %s? The items themselves are kept."
            % ("s" if len(names) != 1 else "", ", ".join(names)),
        ):
            return
        for name in names:
            container.sets.empty(name)
        self._changed()

    def _clear_groups(self, group_nodes: list[_GroupNumberNode]) -> None:
        """Take every epoch out of the groups — the (empty) groups are kept."""
        if not group_nodes or group_nodes[0].container is None:
            return
        groups = group_nodes[0].container.groups
        labels = ", ".join(str(node.number) for node in group_nodes)
        if not self._confirm(
            "Clear Group",
            "Remove every epoch from group%s %s? The epochs themselves are kept."
            % ("s" if len(group_nodes) != 1 else "", labels),
        ):
            return
        for node in group_nodes:
            unassign_from_group(groups, groups.get_items(node.number), node.number)
        self._changed()

    def _delete_groups(self, group_nodes: list[_GroupNumberNode]) -> None:
        """Delete the groups; their epochs are unassigned, never deleted."""
        if not group_nodes or group_nodes[0].container is None:
            return
        groups = group_nodes[0].container.groups
        labels = ", ".join(str(node.number) for node in group_nodes)
        if not self._confirm(
            "Delete Group",
            "Delete group%s %s? Their epochs are kept, with no group."
            % ("s" if len(group_nodes) != 1 else "", labels),
        ):
            return
        for node in group_nodes:
            if node.number in groups.group_numbers:
                groups.delete_group(node.number, quiet=True)
        self._changed()

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
        self._changed()

    def _remove_from_group(
        self, group_node: _GroupNumberNode, objects: list[NMObject],
    ) -> None:
        """Take *objects* out of the group — they stay in the data series."""
        container = group_node.container
        if container is None:
            return
        unassign_from_group(container.groups, [obj.name for obj in objects],
                            group_node.number)
        self._changed()
