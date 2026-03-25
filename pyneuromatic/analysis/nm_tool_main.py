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

from pyneuromatic.analysis.nm_main_op import NMMainOp, NMMainOpAverage, op_from_name
from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu


class NMToolMain(NMTool):
    """Main NM Tool — always loaded by default.

    Provides core array operations (Average, Scale, …) via a pluggable
    ``op`` property.  Setting ``op`` to a string name (e.g. ``"average"``)
    looks up and instantiates the corresponding :class:`NMMainOp` subclass
    from the registry.  Setting it to an :class:`NMMainOp` instance allows
    full parameter control::

        tool = NMToolMain()
        tool.op = NMMainOpArithmetic(factor=2.0)
        nm.run_tool(tool)

    Delegates to the current ``op`` via the standard ``run_init / run /
    run_finish`` lifecycle inherited from :class:`NMTool`.
    """

    def __init__(self) -> None:
        super().__init__(name="tool_main")
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
        from pyneuromatic.core.nm_command_history import add_nm_command
        params = self._op._op_params_str()
        if params is None:
            params = ""
        add_nm_command("%s.op = %s(%s)" % (self._name, type(self._op).__name__, params))

    # ------------------------------------------------------------------
    # Lifecycle hooks

    def run_init(self) -> bool:
        """Reset per-run context and initialise the current op."""
        self._run_folder: NMFolder | None = None
        self._run_prefix: str | None = None
        self._op.run_init()
        return True

    def run(self) -> bool:
        """Process the currently selected data item via the current op."""
        if (self.dataseries is not None
                and self.channel is not None
                and self.epoch is not None):
            d = self.dataseries.get_data(self.channel.name, self.epoch.name)
        else:
            d = self.data
        if d is None:
            return True  # skip silently, don't stop the loop

        channel_name = self.channel.name if self.channel is not None else None

        if self._run_folder is None and self.folder is not None:
            self._run_folder = self.folder
        if self._run_prefix is None and self.dataseries is not None:
            self._run_prefix = self.dataseries.name

        self._op.run(d, channel_name)
        return True

    def run_finish(self) -> bool:
        """Finalise the current op with the folder and prefix captured during run()."""
        self._op.run_finish(self._run_folder, self._run_prefix)
        meta = self.run_meta
        nmh.history(
            "%s: folders=%s, dataseries=%s, channels=%s, epochs=%s"
            % (
                self._op.__class__.__name__,
                meta.get("folders", []),
                meta.get("dataseries", []),
                meta.get("channels", []),
                meta.get("epochs", []),
            ),
            path="NMToolMain.run_finish",
        )
        return True
