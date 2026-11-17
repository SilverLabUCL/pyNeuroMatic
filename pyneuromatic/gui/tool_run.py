"""A tool tab that runs its tool on the strip's selection (Stats1, Spike1).

What a run analyses, as Stats1 has done (#528 shares it): the data series'
channel and its Epoch Set, Group (AND / OR) or current epoch, leaving out
SetX; or a folder's Data Set or data array. All Channels, All Sets and All
Groups run once per channel, set or group, each saved in its own subfolder.
Beside Run, how many epochs or arrays it will analyse ("n=9").
"""
from __future__ import annotations

from typing import Any

from PyQt6 import QtWidgets

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_epoch import EXCLUDE_SET, NMEpoch
from pyneuromatic.core.nm_manager import HIERARCHY_SELECT_KEYS, NMManager
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.gui.selection_model import (
    ALL_EPOCHS,
    combine_scope_names,
    scope_runs,
    scoped_epoch_names,
)
from pyneuromatic.gui.tool_tab import ToolTabWidget


class SelectionRunTab(ToolTabWidget):
    """A tool tab whose Run runs its NMTool (:meth:`analysis_tool`) on the
    strip's selection, as one ``run_selection`` command per run (#448).

    Subclasses set ``self._tool`` and may add:

    - :meth:`check_run`: before a run, raise RuntimeError if it can't
    - :meth:`check_results`: after a single run, e.g. warn of no results
    - :meth:`show_run_results`: after the run(s), e.g. show the subfolder
    """

    def __init__(self, title: str, manager: NMManager,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._manager = manager
        self._tool: Any = None

    def _on_selection_changed(self, selection: dict[str, Any]) -> None:
        super()._on_selection_changed(selection)
        self._update_run_info()

    def _update_run_info(self) -> None:
        """Show beside Run how many epochs or data arrays it will analyse, e.g. "n=9"."""
        try:
            planned = self._run_batches()
            targets = self._selected_targets() if planned is None else []
        except RuntimeError as error:
            self.run_info_label.setText(str(error))
            return
        if planned is not None:  # All Sets / All Groups
            batches, skipped = planned
            runs = len(batches)
            self.run_info_label.setText(f"n={runs} run{'s' * (runs != 1)}")
            tip = "Run goes once per set or group: " + ", ".join(
                f"{label}: {len(targets)}" for label, targets, _selection in batches
            ) if batches else "No set or group has epochs to analyze"
            if skipped:
                tip += f"\nSkipped (no epochs): {', '.join(skipped)}"
            self.run_info_label.setToolTip(tip)
            return
        count = len(targets)
        if self.selection.get("source") == "dataseries":
            noun = "epoch" if count == 1 else "epochs"
            tip = f"Run analyses {count} {noun}"
            excluded = self._excluded_in_scope()
            if excluded:
                tip += f" ({excluded} excluded by SetX)"
        else:
            noun = "data array" if count == 1 else "data arrays"
            tip = f"Run analyses {count} {noun}"
        self.run_info_label.setText(f"n={count}")
        self.run_info_label.setToolTip(tip)

    @staticmethod
    def _excluded_names(dataseries, selection) -> set[str]:
        """Epochs left out of runs (SetX), unless SetX is the Epoch Set."""
        if (selection.get("epoch_set") or "").lower() == EXCLUDE_SET.lower():
            return set()
        return dataseries.epochs.excluded_names()

    def _excluded_in_scope(self) -> int:
        """How many epochs of the Epoch Set/Group scope SetX leaves out."""
        selection = self.selection
        dataseries = selection.get("dataseries")
        if dataseries is None:
            return 0
        excluded = self._excluded_names(dataseries, selection)
        try:
            names = scoped_epoch_names(dataseries, selection, exclude=False)
        except RuntimeError:
            return 0
        if names is None:
            epoch = selection.get("epoch")
            names = [epoch.name] if epoch is not None else []
        return sum(name in excluded for name in names)

    def analysis_tool(self) -> Any:
        return self._tool  # reports a run's progress (#450)

    def _run_selection_args(self, targets: list[dict], selection: dict) -> tuple[Any, dict]:
        """The run as the tool's ``run_selection`` arguments (its command,
        #448): the data series, its channel, Epoch Set ("all", "Set1"...) or
        the current epoch, with any Group and AND / OR; or a data array
        run's folder and Data Set (or array)."""
        if selection.get("source") != "dataseries":
            data = selection.get("data_set") or targets[0]["data"].name
            return selection.get("toolfolder") or selection.get("folder"), {"data": data}
        series = selection["dataseries"]
        epoch_set = selection.get("epoch_set")
        group = selection.get("group")
        if epoch_set == ALL_EPOCHS:
            epochs = "all"
        elif epoch_set is not None:
            epochs = epoch_set
        elif group is None:  # the current epoch
            epochs = targets[0]["epoch"].name
        else:
            epochs = None  # the Group alone
        args: dict[str, Any] = {"channel": selection["channel"].name, "epochs": epochs}
        if group is not None:
            args["group"] = group
            if epochs not in (None, "all"):
                args["group_operator"] = selection.get("group_operator")
        return series, args

    def _run_selection(self, targets: list[dict], selection: dict) -> bool:
        source, args = self._run_selection_args(targets, selection)
        return self._tool.run_selection(source, **args)

    def _scope_runs(self) -> list[tuple[str, dict]] | None:
        """For All Channels / All Sets / All Groups: (label, selection) per
        run, each naming one channel, set or group; None for a single run.
        Channels combine with sets or groups (channel x set). Raises
        RuntimeError for All Sets with All Groups."""
        selection = self.selection
        series = selection.get("dataseries")
        if selection.get("source") != "dataseries" or series is None:
            return None
        runs = scope_runs(series, selection)
        if not selection.get("all_channels"):
            return runs
        per_channel = runs if runs is not None else [("", selection)]
        return [
            (f"{channel.name} {label}".strip(),
             {**run_selection, "channel": channel, "all_channels": None})
            for channel in series.channels.values()
            for label, run_selection in per_channel
        ]

    def _run_batches(self) -> tuple[list[tuple[str, list, dict]], list[str]] | None:
        """For All Sets / All Groups: the runs with epochs to analyse, as
        (label, targets, selection), and the labels of empty ones (skipped);
        None for a single run."""
        runs = self._scope_runs()
        if runs is None:
            return None
        batches, skipped = [], []
        for label, run_selection in runs:
            targets = self._selected_targets(run_selection)
            if targets:
                batches.append((label, targets, run_selection))
            else:
                skipped.append(label)
        return batches, skipped

    def _selected_targets(self, selection: dict | None = None) -> list[dict[str, NMObject]]:
        """Data to analyse from the selection's source (data or data series).

        A data series uses the epochs in the Epoch Set/Group, or the current
        epoch when neither is chosen. A data array uses the Data Set, or the
        current data array when none is chosen. *selection* defaults to the
        strip's.
        """
        selection = self.selection if selection is None else selection
        source = selection.get("source")

        if source == "dataseries":
            dataseries = selection.get("dataseries")
            if dataseries is None:  # the source, with nothing chosen (#521)
                raise RuntimeError("Choose a data series")
            channel = selection.get("channel")
            if not isinstance(channel, NMChannel):
                raise RuntimeError(f"Select a channel before running {self.title_label.text()}")
            target_names = scoped_epoch_names(dataseries, selection)
            if target_names is None:  # Current Epoch, unless it is excluded
                epoch = selection.get("epoch")
                target_names = (
                    [epoch.name]
                    if isinstance(epoch, NMEpoch)
                    and epoch.name not in self._excluded_names(dataseries, selection)
                    else []
                )
            targets = []
            wanted = set(target_names)
            found = dataseries.epoch_data_map(channel.name)
            for epoch in dataseries.epochs.values():
                if epoch.name not in wanted:
                    continue
                item = found.get(epoch.name)
                if isinstance(item, NMData):
                    target = self._manager_target(selection)
                    target.update(
                        {
                            "dataseries": dataseries,
                            "channel": channel,
                            "epoch": epoch,
                            "data": item,
                        }
                    )
                    targets.append(target)
            return targets

        if source == "data":
            data = selection.get("data")
            context = selection.get("toolfolder") or selection.get("folder")
            if context is None:
                return []
            data_set = selection.get("data_set")
            if data_set is None:
                if data is None:  # the source, with nothing chosen (#521)
                    raise RuntimeError("Choose a data array or Data Set")
                data_names = [data.name]
            else:
                members = context.data.sets.get_items(data_set, get_keys=True) or []
                data_names = combine_scope_names(
                    list(context.data.keys()), members, None, None
                )
            targets = []
            for name in data_names:
                item = context.data.get(name)
                if isinstance(item, NMData):
                    target = self._manager_target(selection)
                    target["data"] = item
                    # Analyse the data array, not the selected data series:
                    # None (not left out), so an earlier series run's
                    # selection doesn't stay in the tool (or its folder name)
                    for tier in ("dataseries", "channel", "epoch"):
                        target[tier] = None
                    targets.append(target)
            return targets

        raise RuntimeError("Select a data array or a Data Series channel and epoch first")

    def _manager_target(self, selection: dict[str, Any]) -> dict[str, NMObject]:
        return {
            tier: value
            for tier in HIERARCHY_SELECT_KEYS
            if isinstance((value := selection.get(tier)), NMObject)
        }

    def _confirm(self, title: str, text: str) -> bool:
        """Yes/No question (replaced in tests)."""
        answer = QtWidgets.QMessageBox.question(self, title, text)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def check_run(self) -> None:
        """Before a run: raise RuntimeError if it can't run."""

    def check_results(self) -> None:
        """After a single run (not All Sets...), e.g. warn of no results."""

    def show_run_results(self) -> None:
        """After the run(s), e.g. show the last run's subfolder."""

    def _confirm_excluded(self) -> None:
        """Running on SetX (the excluded epochs): asks first."""
        selection = self.selection
        if (
            selection.get("source") == "dataseries"
            and (selection.get("epoch_set") or "").lower() == EXCLUDE_SET.lower()
            and not self._confirm(
                "Run on SetX",
                f"{EXCLUDE_SET} lists epochs excluded from analysis. Run on them anyway?",
            )
        ):
            raise RuntimeError(f"Cancelled: {EXCLUDE_SET} lists excluded epochs")

    def run_tool(self) -> Any:
        self.check_run()
        self._confirm_excluded()
        planned = self._run_batches()
        if planned is not None:  # All Sets / All Groups: once per set or group
            self._run_each(*planned)
        else:
            targets = self._selected_targets()
            if not targets:
                raise RuntimeError("The current selection has no data to analyze")
            # One command, naming what it runs on (the subfolder too, e.g. _S1_)
            if not self._run_selection(targets, self.selection):
                raise RuntimeError(f"{self.title_label.text()} analysis did not complete")
            self.check_results()
        self.show_run_results()
        return None

    def run_summary(self) -> str:
        """e.g. "Complete → Stats_Record_A_S1_0" (the last run's subfolder)."""
        folder = self._tool.toolfolder
        status = self.status_label.text()
        return f"{status} → {folder.name}" if folder is not None else status


    def _run_each(self, batches: list[tuple[str, list, dict]], skipped: list[str]) -> None:
        """Run once per set or group, each saved in its own subfolder."""
        if not batches:
            raise RuntimeError("None of the sets or groups has epochs to analyze")
        self.set_run_total(sum(len(targets) for _label, targets, _sel in batches))
        for _label, targets, selection in batches:  # a command each
            if not self._run_selection(targets, selection):
                raise RuntimeError(f"{self.title_label.text()} analysis did not complete")
            if self.stop_requested:  # Stop: not the sets or groups after it
                break
        if skipped:
            self.show_warning(f"Skipped (no epochs to analyze): {', '.join(skipped)}")


__all__ = ["SelectionRunTab"]
