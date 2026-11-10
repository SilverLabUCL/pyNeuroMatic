# -*- coding: utf-8 -*-
"""
NM Tool - Base class for analysis tools.

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

import datetime
import math
from typing import Callable

import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_data import NMData, nm_ref
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_manager import HIERARCHY_SELECT_KEYS
from pyneuromatic.tools.nm_tool_config import NMToolConfig
from pyneuromatic.tools.nm_tool_folder import NMToolFolder


def toolfolder_prefix(name: str) -> str | None:
    """The prefix of the tool whose subfolders *name* is one of (e.g.
    ``"Stats_"`` for ``Stats_Record_A_All_0``), or None (#494). From each
    tool's :attr:`NMTool.TOOLFOLDER_PREFIX`."""
    tools = list(NMTool.__subclasses__())
    while tools:
        tool = tools.pop()
        tools.extend(tool.__subclasses__())
        prefix = tool.TOOLFOLDER_PREFIX
        if prefix and name.startswith(prefix):
            return prefix
    return None


def set_code(name: str) -> str:
    """A set's short code for subfolder names: ``Set1`` -> ``S1``, ``SetX``
    -> ``SX``; other names are cut to 8 characters (e.g. imported sets)."""
    suffix = name[3:]
    if name[:3].lower() == "set" and (suffix.isdigit() or suffix.upper() == "X"):
        return "S" + suffix
    return name[:8]


class NMTool:
    """
    NM Tool - Base class for analysis tools.

    Tools receive selection state from NMManager via select_values and
    perform analysis during run(). Subclasses override run() to implement
    specific analysis.

    Selection is set by NMManager - tools have read-only access to
    individual hierarchy tier select keys (folder, data, dataseries, channel, epoch).

    Example:
        class MyTool(NMTool):
            def run(self) -> bool:
                data = self.dataseries.get_data(self.channel, self.epoch)
                # analyze data...
                return True
    """

    # The start of the names of the subfolders a tool lists (e.g. Stats2
    # lists "Stats_..."), or None if it lists none. A subfolder can't be
    # renamed or copied to a name without it (#494).
    TOOLFOLDER_PREFIX: str | None = None

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._select: dict[str, NMObject | None] = {
            tier: None for tier in HIERARCHY_SELECT_KEYS
        }
        self._run_meta: dict = {}
        self._config: NMToolConfig | None = None

        # Output flags — conservative defaults; subclasses init from config.
        self._ignore_nans: bool = True
        self._overwrite: bool = False
        self._results_to_history: bool = False
        self._results_to_cache: bool = True
        self._results_to_numpy: bool = False

        # X-axis window — subclasses init from config.
        self._xbgn: float = -math.inf
        self._xend: float = math.inf
        # Called after each target as progress(done, total); returning False
        # stops the run (e.g. the GUI's progress bar and Stop, #450)
        self.progress: Callable[[int, int], bool] | None = None
        self._stopped: tuple[int, int] | None = None
        self._omitted: list[str] = []  # left out during the run (omit_last)

    @property
    def name(self) -> str:
        """Tool name used in command history (e.g. ``'stats'``, ``'tool_main'``)."""
        return self._name

    @property
    def config(self) -> NMToolConfig | None:
        """Tool configuration object, or None if the tool has no config."""
        return self._config

    @property
    def ignore_nans(self) -> bool:
        """If True, use NaN-ignoring numpy functions (e.g. ``np.nanmean``)."""
        return self._ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, value: bool) -> None:
        self._ignore_nans_set(value)

    @nmch.records_command
    def _ignore_nans_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self._ignore_nans = value
        nmch.add_nm_command("%s.ignore_nans = %r" % (self._name, self._ignore_nans))

    @property
    def overwrite(self) -> bool:
        """If True, reuse the existing toolfolder (clearing it); otherwise create a new one."""
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        self._overwrite_set(value)

    @nmch.records_command
    def _overwrite_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "overwrite", "boolean"))
        self._overwrite = value
        nmch.add_nm_command("%s.overwrite = %r" % (self._name, self._overwrite))

    @property
    def results_to_history(self) -> bool:
        """If True, print results to the history log after run."""
        return self._results_to_history

    @results_to_history.setter
    def results_to_history(self, value: bool) -> None:
        self._results_to_history_set(value)

    @nmch.records_command
    def _results_to_history_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_history", "boolean"))
        self._results_to_history = value
        nmch.add_nm_command(
            "%s.results_to_history = %r" % (self._name, self._results_to_history)
        )

    @property
    def results_to_cache(self) -> bool:
        """If True, save results to the NMFolder tool-results cache after run."""
        return self._results_to_cache

    @results_to_cache.setter
    def results_to_cache(self, value: bool) -> None:
        self._results_to_cache_set(value)

    @nmch.records_command
    def _results_to_cache_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_cache", "boolean"))
        self._results_to_cache = value
        nmch.add_nm_command(
            "%s.results_to_cache = %r" % (self._name, self._results_to_cache)
        )

    @property
    def results_to_numpy(self) -> bool:
        """If True, write results as NMData arrays in a toolfolder after run."""
        return self._results_to_numpy

    @results_to_numpy.setter
    def results_to_numpy(self, value: bool) -> None:
        self._results_to_numpy_set(value)

    @nmch.records_command
    def _results_to_numpy_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_numpy", "boolean"))
        self._results_to_numpy = value
        nmch.add_nm_command(
            "%s.results_to_numpy = %r" % (self._name, self._results_to_numpy)
        )

    @property
    def xbgn(self) -> float:
        """X-axis window start. Default ``-inf`` (no lower bound)."""
        return self._xbgn

    @xbgn.setter
    def xbgn(self, value: float) -> None:
        self._xbgn_set(value)

    @nmch.records_command
    def _xbgn_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "xbgn", "float"))
        if math.isnan(value):
            raise ValueError("xbgn cannot be NaN")
        self._xbgn = float(value)
        nmch.add_nm_command("%s.xbgn = %r" % (self._name, self._xbgn))

    @property
    def xend(self) -> float:
        """X-axis window end. Default ``+inf`` (no upper bound)."""
        return self._xend

    @xend.setter
    def xend(self, value: float) -> None:
        self._xend_set(value)

    @nmch.records_command
    def _xend_set(self, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "xend", "float"))
        if math.isnan(value):
            raise ValueError("xend cannot be NaN")
        self._xend = float(value)
        nmch.add_nm_command("%s.xend = %r" % (self._name, self._xend))

    # Read-only convenience properties for accessing selection
    @property
    def folder(self) -> NMFolder | None:
        """Get the currently selected folder."""
        v = self._select.get("folder")
        return v if isinstance(v, NMFolder) else None

    @property
    def data(self) -> NMData | None:
        """Get the currently selected data."""
        v = self._select.get("data")
        return v if isinstance(v, NMData) else None

    @property
    def dataseries(self) -> NMDataSeries | None:
        """Get the currently selected dataseries."""
        v = self._select.get("dataseries")
        return v if isinstance(v, NMDataSeries) else None

    @property
    def channel(self) -> NMChannel | None:
        """Get the currently selected channel."""
        v = self._select.get("channel")
        return v if isinstance(v, NMChannel) else None

    @property
    def epoch(self) -> NMEpoch | None:
        """Get the currently selected epoch."""
        v = self._select.get("epoch")
        return v if isinstance(v, NMEpoch) else None

    def transformed_data(self) -> NMData | None:
        """The current data as tools analyse it: with its channel's filters
        and transforms (a copy; the array is never changed).

        The data itself when it isn't in the selected channel or the channel
        has no transforms turned on (e.g. tools that change arrays, such as
        Main ops, use :attr:`data`).
        """
        data, channel = self.data, self.channel
        if data is None or channel is None:
            return data
        if not any(d is data for d in channel.data):
            return data
        return channel.transformed(data)

    @property
    def toolfolder(self) -> NMToolFolder | None:
        """Get the currently selected toolfolder (set when running in toolfolder mode)."""
        v = self._select.get("toolfolder")
        return v if isinstance(v, NMToolFolder) else None

    @property
    def select_values(self) -> dict[str, NMObject | None]:
        """Get the current selection as a dictionary of NMObjects."""
        return self._select.copy()

    @select_values.setter
    def select_values(self, values: dict[str, NMObject | None]) -> None:
        """
        Set selection from a dictionary of NMObjects.

        Called by NMManager.run_tool() to set the context for run execution.

        Args:
            values: Dictionary mapping hierarchy tier select keys to NMObjects.
                    Keys should be from HIERARCHY_SELECT_KEYS.
        """
        for tier in HIERARCHY_SELECT_KEYS:
            if tier in values:
                self._select[tier] = values[tier]

    @property
    def select_keys(self) -> dict[str, str | None]:
        """Get the current selection as a dictionary of names."""
        return {
            key: (obj.name if isinstance(obj, NMObject) else None)
            for key, obj in self._select.items()
        }

    @property
    def run_meta(self) -> dict:
        """Get run context metadata populated during the most recent run_all().

        Contains:
            date: ISO timestamp of when run_all() started.
            run_keys: Copy of the run configuration dict passed from
                ``NMManager.run_keys_set()`` (e.g.
                ``{"folder": "selected", "channel": "ChanSet1",
                "epoch": "all"}``). Empty dict if run_all() was called
                directly without run_keys.
            folders: Unique folder names processed, in order of first
                encounter.
            dataseries: Unique dataseries names processed.
            channels: Unique channel names processed.
            epochs: Unique epoch names processed.
        """
        meta = self._run_meta.copy()
        if "run_keys" in meta:
            meta["run_keys"] = dict(meta["run_keys"])
        return meta

    def _update_run_meta(self, target: dict) -> None:
        """Update run_meta lists with unique names from a single target dict.

        Called once per target during run_all(). Tracks folder, toolfolder,
        dataseries, channel, and epoch names (not data).

        Args:
            target: Selection dict mapping tier names to NMObjects.
        """
        for tier, meta_key in (
            ("folder",      "folders"),
            ("toolfolder",  "toolfolders"),
            ("dataseries",  "dataseries"),
            ("channel",     "channels"),
            ("epoch",       "epochs"),
        ):
            obj = target.get(tier)
            if isinstance(obj, NMObject) and obj.name not in self._run_meta[meta_key]:
                self._run_meta[meta_key].append(obj.name)

    def run_init(self) -> bool:
        """Called once before run loop. Override in subclass."""
        return True

    def run(self) -> bool:
        """Called for each run target. Override in subclass."""
        print(self.select_keys)
        return True

    def run_finish(self) -> bool:
        """Called once after run loop. Override in subclass."""
        return True

    def _make_toolfolder(
        self,
        prefix: str,
        overwrite: bool = False,
    ) -> NMToolFolder:
        """Return the target ``{prefix}_{dataseries}_{channel}_{epochs}_N`` subfolder.

        Assembles the name from ``self.dataseries``, ``self.channel`` and a
        short code for the epochs or data analysed (see :meth:`_selection_code`),
        e.g. ``Stats_Record_A_S1_0`` or ``Stats_S2_0`` for a Data Set.

        Args:
            prefix: Tool-specific prefix, e.g. ``"Spike"`` or ``"Stats"``.
            overwrite: If True, clear and reuse ``{base}_0``; otherwise pick
                the next unused ``{base}_N``.
        """
        parts = [prefix]
        if self.dataseries is not None:
            parts.append(self.dataseries.name)
        if self.channel is not None:
            parts.append(self.channel.name)
        code = self._selection_code()
        if code:
            parts.append(code)
        base = "_".join(parts)
        folder = self.folder.toolfolders.get_or_create(base, overwrite=overwrite)
        # Noted, so run_each() can say which subfolder each run wrote
        made = getattr(self, "_made_toolfolders", None)
        if made is not None and folder not in made:
            made.append(folder)
        return folder

    def _selection_code(self) -> str | None:
        """A short code for what the run analysed, from its ``run_keys``.

        Epochs of a data series (``"epoch"``, plus ``"group"`` and
        ``"group_operator"`` when both are chosen): ``All``, ``S1`` (Set1),
        ``G0`` (Group 0, or ``"group0"``), ``S1andG0`` / ``S1orG0``, or an
        epoch's name (e.g. ``E3``). Data arrays: the Data Set's code
        (``"data"``), e.g. ``S2``. None when there is nothing to add.
        """
        keys = self._run_meta.get("run_keys", {})
        if self.dataseries is None and "epoch" not in keys:
            data_set = keys.get("data")
            sets = getattr(getattr(self.folder, "data", None), "sets", None)
            if data_set and sets is not None and data_set in sets:
                return set_code(data_set)
            return None
        value = keys.get("epoch")
        group = keys.get("group")
        code = None
        if value:
            lower = str(value).lower()
            if lower == "all":
                code = "All"
            elif lower.startswith("group") and lower[5:].isdigit():
                code = "G%d" % int(lower[5:])
            elif lower == "selected":
                epochs = self._run_meta.get("epochs", [])
                code = epochs[0] if len(epochs) == 1 else None
            elif self.dataseries is not None and value in self.dataseries.epochs:  # e.g. E3
                code = str(value)
            else:
                code = set_code(str(value))
        if group is not None:
            group_code = "G%d" % int(group)
            if code is None or code == "All":
                return group_code  # the group alone
            operator = "or" if str(keys.get("group_operator", "")).upper() == "OR" else "and"
            return code + operator + group_code
        return code

    @nmch.records_command
    def run_selection(
        self,
        source: NMDataSeries | NMFolder | NMToolFolder,
        channel: str | None = None,
        epochs: str | list[str] | None = None,
        group: int | None = None,
        group_operator: str | None = None,
        data: str | list[str] | None = None,
        omit: list[str] | None = None,
    ) -> bool:
        """Run the tool on data named by a data series or a folder, as one
        command (#448), e.g.::

            nm.stats.run_selection(nm.folders['Demo'].dataseries['Record'],
                            channel='A', epochs='Set1')
            nm.stats.run_selection(nm.folders['Demo'], data='RecordSet')

        Args:
            source: A data series, or the folder (or tool subfolder) whose
                data arrays to analyse.
            channel: The data series' channel (default: its selected one).
            epochs: The data series' epochs: ``"all"``, a set (e.g.
                ``"Set1"``), an epoch (e.g. ``"E3"``) or a list of epochs.
                Epochs in SetX are left out, unless *epochs* is SetX.
            group: The data series' group number, alone or with *epochs*.
            group_operator: ``"AND"`` or ``"OR"``, when *epochs* and
                *group* are both given.
            data: The folder's data: an array, a Data Set or a list of arrays.
            omit: Epochs (or data arrays) to leave out, by name, e.g.
                ``["E3"]``. Those omitted during the run (:meth:`omit_last`)
                are added to the command, which is recorded when the run ends,
                so the script repeats the run as it was made (#472).

        Returns:
            Return value of :meth:`run_all`.
        """
        targets, run_keys = self._run_selection_targets(
            source, channel, epochs, group, group_operator, data)
        omit = list(omit or [])
        targets = [t for t in targets if self._target_name(t) not in omit]
        if not targets:
            raise RuntimeError("nothing to analyze")
        self._omitted = []
        try:
            return self.run_all(targets, run_keys=run_keys)
        finally:
            # Recorded as the run was made: with the epochs omitted during it
            omitted = omit + [name for name in self._omitted if name not in omit]
            values = {"channel": channel, "epochs": epochs, "group": group,
                      "group_operator": group_operator, "data": data,
                      "omit": omitted or None}
            text = {"omit": nmu.names_repr(omitted)} if omitted else {}
            if isinstance(epochs, list):
                text["epochs"] = nmu.names_repr(epochs)
            call = nmu.kwargs_repr(self.run_selection, values, text=text or None)
            nmch.add_nm_command("%s.run_selection(%s%s)" % (self._name, nm_ref(source), call))

    @staticmethod
    def _target_name(target: dict) -> str | None:
        """A target's name for omit: its epoch's, else its data array's."""
        obj = target.get("epoch") or target.get("data")
        return obj.name if isinstance(obj, NMObject) else None

    # --- omitting the epoch just analysed (#472) ------------------------------

    @property
    def can_omit(self) -> bool:
        """True if :meth:`omit_last` works (the tool has :meth:`discard_last`)."""
        return type(self).discard_last is not NMTool.discard_last

    def discard_last(self) -> None:
        """Drop the results of the target :meth:`run` just analysed.
        Override in subclass (e.g. NMToolStats)."""
        raise NotImplementedError("%s can't omit an epoch" % type(self).__name__)

    def omit_last(self) -> str | None:
        """Leave the target just analysed out of the run (e.g. a bad epoch,
        from the run prompt): its results are dropped, and its name is kept
        in :attr:`omitted`. Returns the name."""
        target = dict(self._select)
        name = self._target_name(target)
        self.discard_last()
        epoch = target.get("epoch")
        if isinstance(epoch, NMObject) and epoch.name in self._run_meta.get("epochs", []):
            self._run_meta["epochs"].remove(epoch.name)  # as if not run on
        if name is not None:
            self._omitted.append(name)
        return name

    @property
    def omitted(self) -> list[str]:
        """Names left out during the last run (:meth:`omit_last`)."""
        return list(self._omitted)

    @staticmethod
    def _run_selection_targets(source, channel, epochs, group, group_operator, data,
                        ) -> tuple[list[dict[str, NMObject | None]], dict]:
        """The targets and run keys of :meth:`run_selection`."""
        folder = toolfolder = None
        for obj in getattr(source, "path_objects", []):
            if isinstance(obj, NMToolFolder):
                toolfolder = obj
            elif isinstance(obj, NMFolder):
                folder = obj
        context = {"folder": folder, "toolfolder": toolfolder}
        if isinstance(source, NMDataSeries):
            if data is not None:
                raise ValueError("data: use epochs for a data series")
            c = source.channels.selected_value if channel is None else source.channels.get(channel)
            if not isinstance(c, NMChannel):
                raise KeyError("no channel '%s' in %s" % (channel, source.name))
            names = source.epochs.scope_names(epochs, group, group_operator)
            found = source.epoch_data_map(c.name)
            targets = [
                {**context, "dataseries": source, "channel": c,
                 "epoch": source.epochs[name], "data": found[name]}
                for name in names if isinstance(found.get(name), NMData)
            ]
            epoch = epochs.lower() if isinstance(epochs, str) and epochs.lower() == "all" \
                else epochs if isinstance(epochs, str) else None
            run_keys: dict = {"epoch": epoch}
            if group is not None:
                run_keys["group"] = group
                if group_operator:
                    run_keys["group_operator"] = group_operator.upper()
            return targets, run_keys
        if not isinstance(source, (NMFolder, NMToolFolder)):
            raise TypeError(nmu.type_error_str(source, "source", "data series or folder"))
        if channel is not None or epochs is not None or group is not None:
            raise ValueError("channel, epochs and group are for a data series")
        if data is None:
            raise ValueError("data: name an array, a Data Set or a list of arrays")
        sets = source.data.sets
        run_keys = {}
        if isinstance(data, (list, tuple)):
            names = list(data)
        elif sets._getkey(data) is not None:
            members = sets.get_items(data, get_keys=True) or []
            names = [name for name in source.data.keys() if name in members]
            run_keys = {"data": data}
        else:
            names = [data]
        targets = []
        for name in names:
            item = source.data.get(name)
            if not isinstance(item, NMData):
                raise KeyError("no data '%s' in %s" % (name, source.name))
            # None (not left out), so an earlier series run's selection
            # doesn't stay in the tool (or its subfolder's name)
            targets.append({**context, "dataseries": None, "channel": None,
                            "epoch": None, "data": item})
        return targets, run_keys

    def run_each(
        self,
        batches: list[tuple[list[dict[str, NMObject]], dict | None]],
    ) -> list[tuple[dict, NMToolFolder | None]]:
        """Run the tool once per batch, in turn (e.g. once per epoch set or
        group, as Igor NeuroMatic's "All Sets" / "All Groups").

        Each batch is ``(targets, run_keys)``; each run is a full
        :meth:`run_all`, so it saves its results under its own subfolder,
        named from its run keys (e.g. ``Stats_Record_A_S0_0`` then
        ``Stats_Record_A_S1_0``). Batches without targets (e.g. an empty
        set) are skipped; a run that fails stops the rest.

        Args:
            batches: ``(targets, run_keys)`` pairs, as for :meth:`run_all`.

        Returns:
            ``(run_keys, subfolder)`` for each batch that ran, in order; the
            subfolder is the last one the run wrote, or None.
        """
        done: list[tuple[dict, NMToolFolder | None]] = []
        for targets, run_keys in batches:
            if not targets:
                continue
            succeeded = self.run_all(targets, run_keys=run_keys)
            made = getattr(self, "_made_toolfolders", [])
            done.append((dict(run_keys or {}), made[-1] if made else None))
            if not succeeded:
                break
        return done

    # recorded so a run's notes (e.g. each compute) are inner calls under
    # the command that ran it (run_selection, run_tool); it records none itself
    @nmch.records_command
    def run_all(
        self,
        targets: list[dict[str, NMObject]],
        run_keys: dict[str, str] | None = None,
    ) -> bool:
        """Run the tool over a list of selection targets.

        Calls ``run_init()`` once, then ``run()`` for each target (setting
        ``select_values`` before each call), then ``run_finish()`` once.
        Stops early if ``run()`` returns False.

        This is the standard entry point for executing a tool.
        ``NMManager.run_tool()`` calls this method after resolving run
        targets from the project hierarchy.  Tools can also be driven
        directly (e.g. in tests or scripts) by passing targets explicitly.

        Populates ``run_meta`` before ``run_init()`` is called, so subclasses
        can access it from ``run_init()``, ``run()``, and ``run_finish()``.

        Args:
            targets: List of selection dicts mapping hierarchy tier keys to
                NMObjects, as returned by ``NMManager.run_values()``.
            run_keys: Optional run configuration dict as passed to
                ``NMManager.run_keys_set()`` (e.g.
                ``{"epoch": "Set1", "channel": "A"}``). Used to record
                the epoch set name in ``run_meta``. ``None`` when the tool
                is driven directly (e.g. in tests or scripts).

        Returns:
            Return value of ``run_finish()``.
        """
        self._run_meta = {
            "date": datetime.datetime.now().isoformat(" ", "seconds"),
            "run_keys": dict(run_keys) if run_keys else {},
            "folders": [],
            "toolfolders": [],
            "dataseries": [],
            "channels": [],
            "epochs": [],
        }
        self._made_toolfolders: list[NMToolFolder] = []
        self._stopped = None
        if not self.run_init():
            return False
        total = len(targets)
        for done, target in enumerate(targets, start=1):
            self.select_values = target
            self._update_run_meta(target)
            if not self.run():
                break
            if self.progress is not None and self.progress(done, total) is False:
                self._stopped = (done, total)
                break
        finished = self.run_finish()  # a stopped run keeps what it did
        if self._omitted:
            for folder in self._made_toolfolders:
                folder.notes.add("omitted: %s" % ", ".join(self._omitted))
        if self._stopped is not None:
            for folder in self._made_toolfolders:
                folder.notes.add("cancelled after %d of %d: partial results" % self._stopped)
        return finished

    @property
    def stopped(self) -> tuple[int, int] | None:
        """(done, total) if the last run was stopped by :attr:`progress`,
        else None."""
        return self._stopped
