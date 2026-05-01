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

import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_dataseries import NMDataSeries
from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_manager import HIERARCHY_SELECT_KEYS
from pyneuromatic.analysis.nm_tool_config import NMToolConfig
from pyneuromatic.analysis.nm_tool_folder import NMToolFolder


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

    def _ignore_nans_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self._ignore_nans = value
        nmh.history("set ignore_nans=%s" % value, quiet=quiet)
        nmch.add_nm_command("%s.ignore_nans = %r" % (self._name, self._ignore_nans))

    @property
    def overwrite(self) -> bool:
        """If True, reuse the existing toolfolder (clearing it); otherwise create a new one."""
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        self._overwrite_set(value)

    def _overwrite_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "overwrite", "boolean"))
        self._overwrite = value
        nmh.history("set overwrite=%s" % value, quiet=quiet)
        nmch.add_nm_command("%s.overwrite = %r" % (self._name, self._overwrite))

    @property
    def results_to_history(self) -> bool:
        """If True, print results to the history log after run."""
        return self._results_to_history

    @results_to_history.setter
    def results_to_history(self, value: bool) -> None:
        self._results_to_history_set(value)

    def _results_to_history_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_history", "boolean"))
        self._results_to_history = value
        nmh.history("set results_to_history=%s" % value, quiet=quiet)
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

    def _results_to_cache_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_cache", "boolean"))
        self._results_to_cache = value
        nmh.history("set results_to_cache=%s" % value, quiet=quiet)
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

    def _results_to_numpy_set(self, value: bool, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "results_to_numpy", "boolean"))
        self._results_to_numpy = value
        nmh.history("set results_to_numpy=%s" % value, quiet=quiet)
        nmch.add_nm_command(
            "%s.results_to_numpy = %r" % (self._name, self._results_to_numpy)
        )

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
        """Return the target ``{prefix}_{dataseries}_{channel}_{epoch_set}_N`` subfolder.

        Assembles the name from ``self.dataseries``, ``self.channel``, and
        ``run_keys["epoch"]`` (when present).

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
        epoch_set = self._run_meta.get("run_keys", {}).get("epoch")
        if epoch_set:
            parts.append(epoch_set)
        base = "_".join(parts)
        return self.folder.toolfolders.get_or_create(base, overwrite=overwrite)

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
        if not self.run_init():
            return False
        for target in targets:
            self.select_values = target
            self._update_run_meta(target)
            if not self.run():
                break
        return self.run_finish()
