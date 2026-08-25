# -*- coding: utf-8 -*-
"""
Matplotlib plotting utilities for NMData and NMFolder.

Provides two public functions:

- :func:`plot_nmdata` — plot a single NMData array.
- :func:`plot_folder`  — plot all epochs for a given prefix, one subplot
  per channel, with epochs overlaid.

No GUI framework integration is required.  Figures are displayed via
``matplotlib.pyplot.show()`` (or inline in Jupyter when the appropriate
backend is active).

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

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import pyneuromatic.core.nm_utilities as nmu

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pyneuromatic.core.nm_data import NMData
    from pyneuromatic.core.nm_folder import NMFolder


def _x_array(nmdata: "NMData") -> np.ndarray:
    """Build the x-axis array from nmdata.xscale."""
    arr = nmdata.nparray
    n = len(arr) if arr is not None else 0
    start = nmdata.xscale.start
    delta = nmdata.xscale.delta
    return np.arange(n) * delta + start


def _axis_label(label: str, units: str) -> str:
    """Format 'Label (units)' or just 'Label' when units is empty."""
    if units:
        return "%s (%s)" % (label, units)
    return label


def plot_nmdata(
    nmdata: "NMData",
    ax: "Axes | None" = None,
    title: str | None = None,
    show: bool = True,
    **line_kwargs,
) -> tuple["Figure", "Axes"]:
    """Plot a single NMData array.

    Args:
        nmdata:      The NMData object to plot.
        ax:          Existing Matplotlib Axes to draw on.  If None a new
                     figure and axes are created.
        title:       Axes title.  Defaults to ``nmdata.name``.
        show:        If True call ``plt.show()`` after plotting.
        **line_kwargs: Passed directly to ``ax.plot()``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    arr = nmdata.nparray
    if arr is None or len(arr) == 0:
        return fig, ax

    x = _x_array(nmdata)
    ax.plot(x, arr, **line_kwargs)

    xlabel = _axis_label(nmdata.xscale.label, nmdata.xscale.units)
    ylabel = _axis_label(nmdata.yscale.label, nmdata.yscale.units)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else nmdata.name)

    if show:
        plt.show()

    return fig, ax


def plot_folder(
    folder: "NMFolder",
    prefix: str,
    channels: list[str] | None = None,
    epochs: list[int] | None = None,
    title: str | None = None,
    alpha: float = 0.7,
    show: bool = True,
    **line_kwargs,
) -> tuple["Figure", list["Axes"]]:
    """Plot all epochs for a given prefix from a folder, one subplot per channel.

    Epochs are overlaid on the same axes so trial-to-trial variability is
    visible at a glance.

    Args:
        folder:    The NMFolder containing the data.
        prefix:    Data name prefix (e.g. ``"Record"``).
        channels:  Channel letters to include (e.g. ``["A", "B"]``).
                   Default: all channels found in the folder.
        epochs:    Epoch indices to include (e.g. ``[0, 1, 2]``).
                   Default: all epochs found in the folder.
        title:     Figure suptitle.  Defaults to
                   ``"<folder.name> — <prefix>"``.
        alpha:     Line opacity for overlaid epochs (default 0.7).
        show:      If True call ``plt.show()`` after plotting.
        **line_kwargs: Passed to ``ax.plot()`` for every epoch line.

    Returns:
        ``(fig, axes)`` where *axes* is a list with one entry per channel.

    Raises:
        ValueError: if no matching data are found for *prefix*.
    """
    import matplotlib.pyplot as plt

    # Collect matching NMData objects, grouped by channel then sorted by epoch
    channel_data: dict[str, list[tuple[int, "NMData"]]] = defaultdict(list)

    for nmdata in folder.data.values():
        parsed = nmu.parse_data_name(nmdata.name)
        if parsed is None:
            continue
        data_prefix, chan, epoch = parsed
        if data_prefix != prefix:
            continue
        if channels is not None and chan not in channels:
            continue
        if epochs is not None and epoch not in epochs:
            continue
        channel_data[chan].append((epoch, nmdata))

    if not channel_data:
        raise ValueError(
            "no data found for prefix %r in folder %r" % (prefix, folder.name)
        )

    # Sort channels alphabetically; sort epochs within each channel
    sorted_channels = sorted(channel_data.keys())
    for chan in sorted_channels:
        channel_data[chan].sort(key=lambda t: t[0])

    n_channels = len(sorted_channels)
    fig, axes = plt.subplots(
        n_channels, 1,
        sharex=True,
        squeeze=False,
        figsize=(8, 3 * n_channels),
    )
    axes = [axes[i][0] for i in range(n_channels)]

    for ax, chan in zip(axes, sorted_channels):
        epoch_list = channel_data[chan]
        for _epoch_idx, nmdata in epoch_list:
            arr = nmdata.nparray
            if arr is None or len(arr) == 0:
                continue
            x = _x_array(nmdata)
            ax.plot(x, arr, alpha=alpha, **line_kwargs)

        # Labels from the last nmdata in this channel
        last = epoch_list[-1][1]
        xlabel = _axis_label(last.xscale.label, last.xscale.units)
        ylabel = _axis_label(last.yscale.label, last.yscale.units)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_title("Channel %s  (%d epoch%s)"
                     % (chan, len(epoch_list), "s" if len(epoch_list) != 1 else ""))

    if xlabel:
        axes[-1].set_xlabel(xlabel)

    fig_title = title if title is not None else "%s — %s" % (folder.name, prefix)
    fig.suptitle(fig_title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
