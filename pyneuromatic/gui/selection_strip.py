"""Selection strip: Folder, Data and Data Series sections above the workspace."""
from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtWidgets

from pyneuromatic.gui.selection_model import (
    ALL_CHANNELS,
    ALL_GROUPS,
    ALL_SETS,
    BUILTIN_EPOCH_SETS,
    SelectionModel,
    all_set_names,
    is_stored_epoch_set,
    scope_runs,
    scoped_epoch_names,
)

# Shown for "no subfolder", i.e. the folder itself
NO_SUBFOLDER = "—"


class FlowLayout(QtWidgets.QLayout):
    """Lay items out left to right, wrapping onto new lines when out of room."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setSpacing(spacing)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )

    def _arrange(self, rect: QtCore.QRect, move: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + self.spacing()
                line_height = 0
            if move:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def is_text_array(data) -> bool:
    """True for a text (string / object) array."""
    array = getattr(data, "nparray", None)
    return array is not None and array.dtype.kind in "OUS"


class SelectionStrip(QtWidgets.QWidget):
    """Folder, Data and Data Series sections, each two rows of selectors.

    Each selector is a dropdown, most with a ``⋯`` actions button. The Data
    and Data Series titles are radio buttons showing which one tools analyse
    (the selection ``source``); the active section is outlined.
    """

    # Section layout: rows of (selection key, label, has actions button)
    _SECTIONS: tuple[tuple[str, str | None, tuple[tuple[tuple[str, str, bool], ...], ...]], ...] = (
        ("Folder", None, (
            (("folder", "Folder", True),),
            (("toolfolder", "Subfolder", True),),
        )),
        ("Data", "data", (
            (("data", "Data", True),),
            (("data_set", "Data Set", True),),
        )),
        ("Data Series", "dataseries", (
            (("dataseries", "Prefix", True), ("channel", "Channel", False)),
            (("epoch_set", "Epoch Set", True), ("group_operator", "", False),
             ("group", "Group", True)),
        )),
    )
    _MIN_CONTENT_LENGTH = {
        "folder": 14,
        "toolfolder": 9,
        "data": 14,
        "data_set": 9,
        "dataseries": 10,
        "channel": 1,
        "epoch_set": 7,
        "group_operator": 1,
        "group": 1,
    }

    # Narrower than the style's smallest dropdown: they hold "A" or "12"
    _FIXED_WIDTH = {"channel": 66, "group": 66}  # room for "All"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._selection_model: SelectionModel | None = None
        self._combos: dict[str, QtWidgets.QComboBox] = {}
        self._labels: dict[str, str] = {}
        self._action_buttons: dict[str, QtWidgets.QToolButton] = {}
        self._source_buttons: dict[str, QtWidgets.QRadioButton] = {}
        self._sections: dict[str, QtWidgets.QFrame] = {}
        self._source_group = QtWidgets.QButtonGroup(self)
        self._source_group.setExclusive(True)

        self.setStyleSheet(
            "QFrame#selectionSection { border: 1px solid palette(mid);"
            " border-radius: 4px; }"
            "QFrame#selectionSection[active=\"true\"] {"
            " border: 2px solid palette(highlight); }"
            "QToolButton::menu-indicator { image: none; width: 0px; }"
        )
        layout = FlowLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        for title, source, rows in self._SECTIONS:
            layout.addWidget(self._build_section(title, source, rows))

    # --- construction -----------------------------------------------------

    def _build_section(self, title, source, rows) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame(self)
        frame.setObjectName("selectionSection")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(6, 3, 6, 4)
        layout.setSpacing(3)
        if source is None:
            header = QtWidgets.QLabel(title, frame)
            header.setStyleSheet("font-weight: bold;")
        else:
            header = QtWidgets.QRadioButton(title, frame)
            header.setStyleSheet("font-weight: bold;")
            header.setToolTip(f"Analyse the selected {title.lower()}")
            self._source_group.addButton(header)
            self._source_buttons[source] = header
            self._sections[source] = frame
        layout.addWidget(header)
        for row in rows:
            row_layout = QtWidgets.QHBoxLayout()
            row_layout.setSpacing(4)
            for key, label, has_actions in row:
                if label:
                    row_layout.addWidget(QtWidgets.QLabel(label, frame))
                combo = QtWidgets.QComboBox(frame)
                combo.setObjectName(f"select_{key}")
                combo.setMinimumContentsLength(self._MIN_CONTENT_LENGTH[key])
                combo.setSizeAdjustPolicy(
                    QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                )
                if key in self._FIXED_WIDTH:
                    combo.setFixedWidth(self._FIXED_WIDTH[key])
                if key == "group_operator":
                    combo.setToolTip("Combine the Epoch Set and Group with AND or OR")
                if key == "group":
                    combo.setToolTip("A group, or All (All Groups): Run goes once per group")
                self._combos[key] = combo
                self._labels[key] = label or "Operator"
                row_layout.addWidget(combo)
                if has_actions:
                    button = QtWidgets.QToolButton(frame)
                    button.setText("⋯")
                    button.setObjectName(f"actions_{key}")
                    button.setEnabled(False)
                    button.setToolTip(f"{self._labels[key]} actions (coming soon)")
                    self._action_buttons[key] = button
                    row_layout.addWidget(button)
                if key == "data_set":
                    # How many data arrays the Data Set selects (as for epochs)
                    self.data_count_label = QtWidgets.QLabel("", frame)
                    self.data_count_label.setObjectName("dataCount")
                    row_layout.addWidget(self.data_count_label)
                if key == "group":
                    # How many epochs the Epoch Set/Group selects
                    self.epoch_count_label = QtWidgets.QLabel("", frame)
                    self.epoch_count_label.setObjectName("epochCount")
                    self.epoch_count_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
                    row_layout.addWidget(self.epoch_count_label)
            row_layout.addStretch(1)
            layout.addLayout(row_layout)
        return frame

    # --- access -----------------------------------------------------------

    def keys(self) -> list[str]:
        """Selection keys of the dropdowns, in display order."""
        return list(self._combos)

    def combo(self, key: str) -> QtWidgets.QComboBox:
        return self._combos[key]

    def label(self, key: str) -> str:
        return self._labels[key]

    def action_button(self, key: str) -> QtWidgets.QToolButton:
        return self._action_buttons[key]

    def set_actions_menu(self, key: str, menu: QtWidgets.QMenu) -> None:
        """Show *menu* from the ⋯ button of the *key* selector."""
        button = self._action_buttons[key]
        button.setMenu(menu)
        button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setEnabled(True)
        button.setToolTip(f"{self._labels[key]} actions")

    def source_button(self, source: str) -> QtWidgets.QRadioButton:
        return self._source_buttons[source]

    def is_active_section(self, source: str) -> bool:
        return bool(self._sections[source].property("active"))

    # --- selection model --------------------------------------------------

    def bind_selection_model(self, model: SelectionModel) -> None:
        self._selection_model = model
        model.selection_changed.connect(self._update_from_selection)
        for key, combo in self._combos.items():
            combo.currentIndexChanged.connect(
                lambda index, key=key, combo=combo: self._on_selector_changed(
                    key, combo.itemData(index)
                )
            )
        for source, button in self._source_buttons.items():
            button.clicked.connect(
                lambda _checked=False, source=source: self._on_source_clicked(source)
            )
        self._update_from_selection(model.selection)

    def _on_selector_changed(self, key: str, value: Any) -> None:
        if self._selection_model is None:
            return
        if key == "channel" and value == ALL_CHANNELS:
            # Every channel for runs; the current channel stays selected
            self._selection_model.update(all_channels=True)
        elif key == "channel" and value is not None:
            self._selection_model.update(channel=value, all_channels=False)
        else:
            self._selection_model.update(**{key: value})

    def _on_source_clicked(self, source: str) -> None:
        if self._selection_model is not None:
            self._selection_model.update(source=source)

    def refresh_options(self) -> None:
        if self._selection_model is not None:
            self._update_from_selection(self._selection_model.selection)

    @staticmethod
    def runs_breakdown(selection: dict) -> str:
        """For All Sets / All Groups, the epochs each run analyses, e.g.
        "Set0: 5, Set1: 4"; "" otherwise."""
        series = selection.get("dataseries")
        if series is None:
            return ""
        try:
            runs = scope_runs(series, selection)
        except RuntimeError as error:
            return str(error)
        if not runs:
            return ""
        parts = []
        for label, run_selection in runs:
            try:
                names = scoped_epoch_names(series, run_selection) or []
            except RuntimeError:
                names = []
            parts.append(f"{label}: {len(names)}")
        return ", ".join(parts)

    @staticmethod
    def data_count(selection: dict) -> int | None:
        """How many data arrays the Data Set selects (1, the current array,
        without one); None with no data array or Data Set."""
        context = selection.get("toolfolder") or selection.get("folder")
        data_set = selection.get("data_set")
        if context is None:
            return None
        if data_set is not None and data_set in context.data.sets:
            members = context.data.sets.get_items(data_set, get_keys=True) or []
            return sum(1 for name in members if name in context.data)
        return 1 if selection.get("data") is not None else None

    @staticmethod
    def epoch_count_text(selection: dict) -> str:
        """e.g. "n=12-3": 12 epochs in the Epoch Set/Group, 3 of them excluded (SetX)."""
        dataseries = selection.get("dataseries")
        if dataseries is None:
            return ""
        try:
            scoped = scoped_epoch_names(dataseries, selection)
            unscoped = scoped_epoch_names(dataseries, selection, exclude=False)
        except RuntimeError:
            return "Choose AND or OR"
        if scoped is None:  # Current Epoch
            epoch = selection.get("epoch")
            unscoped = [epoch.name] if epoch is not None else []
            excluded = dataseries.epochs.excluded_names()
            scoped = [name for name in unscoped if name not in excluded]
        left_out = len(unscoped) - len(scoped)
        return f"n={len(unscoped)}" + (f"-{left_out}" if left_out else "")

    def _update_from_selection(self, selection: dict) -> None:
        if self._selection_model is None:
            return
        options = self._options_for_selection(selection)
        for key, combo in self._combos.items():
            empty_text = {"toolfolder": NO_SUBFOLDER}.get(key, "")
            selected = selection.get(key)
            if key == "channel" and selection.get("all_channels"):
                selected = ALL_CHANNELS
            self._set_combo_options(combo, options.get(key, []), selected, empty_text)
        count = self.data_count(selection)
        self.data_count_label.setText("" if count is None else f"n={count}")
        self.data_count_label.setToolTip(
            "" if count is None
            else f"{count} data array{'s' * (count != 1)} selected"
        )
        text = self.epoch_count_text(selection)
        if text.startswith("n="):
            total, _, left_out = text[2:].partition("-")
            # The excluded epochs (SetX) in red, as on the SetX checkbox
            shown = f"n={total}"
            if left_out:
                shown += f'<span style="color: #c62828;">-{left_out}</span>'
            self.epoch_count_label.setText(shown)
            tip = f"{total} epoch{'s' * (total != '1')} selected"
            if left_out:
                tip += f", {left_out} excluded (SetX): {int(total) - int(left_out)} analysed"
            runs = self.runs_breakdown(selection)
            if runs:
                tip += f"\nRun goes once per set or group: {runs}"
            self.epoch_count_label.setToolTip(tip)
        else:
            self.epoch_count_label.setText(text)
            self.epoch_count_label.setToolTip("")
        source = selection.get("source")
        for name, button in self._source_buttons.items():
            button.setEnabled(selection.get(name) is not None)
            self._source_group.setExclusive(False)
            button.setChecked(name == source)
            self._source_group.setExclusive(True)
            frame = self._sections[name]
            frame.setProperty("active", name == source)
            frame.style().unpolish(frame)
            frame.style().polish(frame)

    def _options_for_selection(self, selection: dict) -> dict[str, list[tuple[str, object]]]:
        manager = self._selection_model.manager
        options: dict[str, list[tuple[str, object]]] = {
            "folder": [(name, name) for name in manager.folders.keys()],
        }
        folder = selection.get("folder")
        if folder is None:
            return options
        options["toolfolder"] = [(name, name) for name in folder.toolfolders.keys()]
        context = selection.get("toolfolder") or folder
        # Numeric arrays only: text arrays (e.g. ST_w0_data) can't be analysed
        options["data"] = [(name, name) for name, data in context.data.items()
                           if not is_text_array(data)]
        options["data_set"] = [(name, name) for name in context.data.sets.keys()]
        # A lock marks series whose arrays cannot be changed
        options["dataseries"] = [
            (f"🔒 {name}" if series.locked else name, name)
            for name, series in context.dataseries.items()
        ]

        dataseries = selection.get("dataseries")
        if dataseries is not None:
            # "All" runs once per channel (when there is more than one)
            channels = list(dataseries.channels.keys())
            options["channel"] = ([("All", ALL_CHANNELS)] if len(channels) > 1 else []) + [
                (name, name) for name in channels
            ]
            # All Sets runs once per set (there must be one besides SetX)
            all_sets = [(ALL_SETS, ALL_SETS)] if all_set_names(dataseries) else []
            options["epoch_set"] = [(name, name) for name in BUILTIN_EPOCH_SETS] + all_sets + [
                (name, name) for name in dataseries.epochs.sets.keys()
            ]
            numbers = dataseries.epochs.groups.group_numbers
            # "All" here is All Groups: once per group (fits the narrow box)
            options["group"] = ([("All", ALL_GROUPS)] if numbers else []) + [
                (str(number), number) for number in numbers
            ]
        # AND/OR combines a stored Epoch Set (or All Sets) with a Group (or
        # All Groups); All needs neither
        epoch_set = selection.get("epoch_set")
        if ((is_stored_epoch_set(epoch_set) or epoch_set == ALL_SETS)
                and selection.get("group") is not None):
            options["group_operator"] = [("AND", "AND"), ("OR", "OR")]
        return options

    @staticmethod
    def _set_combo_options(
        combo: QtWidgets.QComboBox,
        options: list[tuple[str, object]],
        selected,
        empty_text: str | None = "",
    ) -> None:
        if hasattr(selected, "name"):
            selected = selected.name
        combo.blockSignals(True)
        combo.clear()
        if empty_text is not None:
            combo.addItem(empty_text, None)
        for text, value in options:
            combo.addItem(text, value)
        selected_index = combo.findData(selected)
        combo.setCurrentIndex(max(selected_index, 0))
        combo.setEnabled(bool(options))
        combo.blockSignals(False)


__all__ = ["FlowLayout", "NO_SUBFOLDER", "SelectionStrip", "is_text_array"]
