# -*- coding: utf-8 -*-
"""
NMToolMain - Main tool that is always loaded.

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

from pyneuromatic.analysis.nm_main_op import NMMainOp, NMMainOpAverage, op_from_name
from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_utilities as nmu


class NMToolMain(NMTool):
    """Main NM Tool — always loaded by default.

    Provides core waveform operations (Average, Scale, …) via a pluggable
    ``op`` property.  Setting ``op`` to a string name (e.g. ``"average"``)
    looks up and instantiates the corresponding :class:`NMMainOp` subclass
    from the registry.  Setting it to an :class:`NMMainOp` instance allows
    full parameter control::

        tool = NMToolMain()
        tool.op = NMMainOpScale(factor=2.0)
        nm.run_tool(tool)

    Overrides :meth:`run_all` (rather than the per-item ``run()``) to
    collect the full data list first, then delegate to
    ``self.op.run_all(data_items, folder)``.  This lets aggregating ops
    (Average) receive all data at once while pointwise ops (Scale) use the
    default per-item loop in :class:`NMMainOp`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op: NMMainOp = NMMainOpAverage()

    # ------------------------------------------------------------------
    # op property

    @property
    def op(self) -> NMMainOp:
        """Current operation (NMMainOp instance)."""
        return self._op

    @op.setter
    def op(self, value: NMMainOp | str) -> None:
        """Set the current operation.

        Args:
            value: An :class:`NMMainOp` instance, or a string name looked up
                in the op registry (e.g. ``"average"``, ``"scale"``).

        Raises:
            TypeError: If value is neither an NMMainOp nor a string.
            ValueError: If value is a string not in the registry.
        """
        if isinstance(value, str):
            self._op = op_from_name(value)
        elif isinstance(value, NMMainOp):
            self._op = value
        else:
            raise TypeError(
                nmu.type_error_str(value, "op", "NMMainOp or string")
            )

    # ------------------------------------------------------------------
    # run_all override

    def run_all(
        self,
        targets: list[dict[str, NMObject]],
        run_keys: dict[str, str] | None = None,
    ) -> bool:
        """Run the current op over a list of selection targets.

        Collects ``(NMData, channel_name)`` pairs from all targets, then
        calls ``self.op.run_all(data_items, folder)``.  Populates
        ``run_meta`` in the same format as :meth:`NMTool.run_all`.

        Args:
            targets: List of selection dicts as returned by
                ``NMManager.run_values()``.
            run_keys: Optional run configuration dict (recorded in
                ``run_meta``).

        Returns:
            True on success.
        """
        # 1. Populate run_meta
        self._run_meta = {
            "date": datetime.datetime.now().isoformat(" ", "seconds"),
            "run_keys": dict(run_keys) if run_keys else {},
            "folders": [],
            "dataseries": [],
            "channels": [],
            "epochs": [],
        }

        # 2. Collect (data, channel_name) pairs; track meta
        data_items: list[tuple[NMData, str | None]] = []
        folder: NMFolder | None = None
        prefix: str | None = None

        for target in targets:
            self.select_values = target

            self._update_run_meta(target)

            if (self.dataseries is not None
                    and self.channel is not None
                    and self.epoch is not None):
                d = self.dataseries.get_data(self.channel.name, self.epoch.name)
            else:
                d = self.data
            if d is not None:
                channel_name = (
                    self.channel.name if self.channel is not None else None
                )
                data_items.append((d, channel_name))

            if folder is None and self.folder is not None:
                folder = self.folder

            if prefix is None and self.dataseries is not None:
                prefix = self.dataseries.name

        # 3. Delegate to op
        self._op.run_all(data_items, folder, prefix=prefix)
        return True

