# -*- coding: utf-8 -*-
"""
Axon Binary Format (.abf) file reader.

Reads ABF files (ABF1 and ABF2) using the pyabf library.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

References:
    - NeuroMatic Igor import: NM_ImportPclamp.ipf
    - pyabf library: https://github.com/swharden/pyABF
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_folder import NMFolder

from pyneuromatic.io.base import (
    FileInfo, check_options, crop, import_numbers, make_data_name, note_import,
)
import pyneuromatic.core.nm_utilities as nmu


def read_abf(
    filepath: str | Path,
    folder: "NMFolder | None" = None,
    prefix: str = "Record",
    make_dataseries: bool = True,
    channels: list[str] | None = None,
    epochs: list[int] | range | None = None,
    xrange: tuple[float | None, float | None] | None = None,
) -> "NMFolder":
    """Read an Axon Binary Format file into an NMFolder.

    Reads ABF files created by pClamp/Clampex using the pyabf library.
    Requires pyabf (pip install pyabf).

    Args:
        filepath: Path to the ABF file.
        folder: Optional existing folder to add data to. If None, creates new.
        prefix: Prefix for data names (default "Record").
        make_dataseries: If True, automatically create dataseries from data.
        channels: Channel letters to import (e.g. ["A"]); None for all.
        epochs: Epoch (sweep) numbers to import; None for all. The
            channels and epochs imported are renumbered from A and 0.
        xrange: (xbgn, xend) to crop each array to; None for an open end.

    Returns:
        NMFolder containing the imported data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If pyabf is not installed.
    """
    try:
        import pyabf
    except ImportError:
        raise ImportError(
            "pyabf is required to read ABF files. "
            "Install it with: pip install pyabf"
        )

    check_options(channels, epochs, xrange)
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Import here to avoid circular imports
    from pyneuromatic.core.nm_folder import NMFolder

    # Create or use provided folder
    if folder is None:
        import re

        folder_name = re.sub(r"[^a-zA-Z0-9_]", "_", filepath.stem)
        if folder_name and not folder_name[0].isalpha():
            folder_name = "F" + folder_name
        folder = NMFolder(name=folder_name)

    # Load the ABF file
    abf = pyabf.ABF(str(filepath), loadData=True)

    # Store metadata
    folder.metadata["root"] = {
        "FileFormat": f"ABF{abf.abfVersionString}",
        "AcqMode": abf.protocol,
        "NumWaves": abf.sweepCount,
        "NumChannels": abf.channelCount,
        "SamplesPerWave": abf.sweepPointCount,
        "SampleInterval": abf.dataSecPerPoint * 1000,  # sec -> ms
        "SampleRate": abf.sampleRate,
        "WavePrefix": prefix,
        "xLabel": "ms",
        "Creator": abf.creator,
    }

    if abf.abfDateTime:
        folder.metadata["root"]["FileDateTime"] = str(abf.abfDateTime)

    # x-scaling: convert seconds to ms
    x_start = 0.0
    x_delta = abf.dataSecPerPoint * 1000  # sec -> ms
    x_units = "ms"

    # The channels and sweeps to import, renumbered from A and 0
    letters = [nmu.channel_char(c) for c in range(abf.channelCount)]
    channel_numbers, epoch_numbers = import_numbers(
        letters, list(range(abf.sweepCount)), channels, epochs)
    note_import(folder, channel_numbers, epoch_numbers, channels, epochs, xrange)

    # Process sweeps and channels; build matches dict as we go
    matches = {}
    for sweep in range(abf.sweepCount):
        for channel in range(abf.channelCount):
            if letters[channel] not in channel_numbers or sweep not in epoch_numbers:
                continue
            abf.setSweep(sweep, channel=channel)
            new_channel, new_epoch = channel_numbers[letters[channel]], epoch_numbers[sweep]

            name = make_data_name(prefix, new_channel, new_epoch)
            y, start = crop(abf.sweepY, x_start, x_delta, xrange)

            # Build scale dicts
            xscale = {"start": start, "delta": x_delta, "units": x_units}
            yscale = {}
            if channel < len(abf.adcNames):
                yscale["label"] = abf.adcNames[channel]
            if channel < len(abf.adcUnits):
                yscale["units"] = abf.adcUnits[channel]

            data = folder.data.new(
                name, xscale=xscale, yscale=yscale if yscale else None
            )
            if data is None:
                continue

            # Set y data
            data.nparray = y.copy()

            ch_char = nmu.channel_char(new_channel)
            if ch_char:
                matches[(ch_char, new_epoch)] = data

    # Optionally create dataseries directly from the matches dict
    if make_dataseries and matches:
        dataseries = folder.build_dataseries(prefix, matches)
        if dataseries is not None:
            dataseries.locked = True  # protect raw recordings

    return folder


def abf_info(filepath: str | Path) -> FileInfo:
    """What an ABF file holds (its header only; the data isn't read)."""
    try:
        import pyabf
    except ImportError:
        raise ImportError(
            "pyabf is required to read ABF files. "
            "Install it with: pip install pyabf"
        )
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    abf = pyabf.ABF(str(filepath), loadData=False)
    channels = []
    for channel in range(abf.channelCount):
        label = abf.adcNames[channel] if channel < len(abf.adcNames) else ""
        units = abf.adcUnits[channel] if channel < len(abf.adcUnits) else ""
        channels.append((nmu.channel_char(channel), label, units))
    return FileInfo(
        prefix="Record",
        channels=channels,
        epochs=list(range(abf.sweepCount)),
        n_points=int(abf.sweepPointCount),
        x_start=0.0,
        x_delta=abf.dataSecPerPoint * 1000,  # sec -> ms
        x_units="ms",
        file_format=f"Axon Binary Format (ABF {abf.abfVersionString})",
    )
