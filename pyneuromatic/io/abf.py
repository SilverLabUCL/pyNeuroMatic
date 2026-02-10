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

from pyneuromatic.io.base import make_data_name


def read_abf(
    filepath: str | Path,
    folder: "NMFolder | None" = None,
    prefix: str = "Record",
    make_dataseries: bool = True,
) -> "NMFolder":
    """Read an Axon Binary Format file into an NMFolder.

    Reads ABF files created by pClamp/Clampex using the pyabf library.
    Requires pyabf (pip install pyabf).

    Args:
        filepath: Path to the ABF file.
        folder: Optional existing folder to add data to. If None, creates new.
        prefix: Prefix for data names (default "Record").
        make_dataseries: If True, automatically create dataseries from data.

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

    # Process sweeps and channels
    for sweep in range(abf.sweepCount):
        for channel in range(abf.channelCount):
            abf.setSweep(sweep, channel=channel)

            name = make_data_name(prefix, channel, sweep)

            data = folder.data.new(name)
            if data is None:
                continue

            # Set y data
            data.nparray = abf.sweepY.copy()

            # Set y label/units from ADC info
            if channel < len(abf.adcNames):
                data.yscale["label"] = abf.adcNames[channel]
            if channel < len(abf.adcUnits):
                data.yscale["units"] = abf.adcUnits[channel]

            # Set x scaling
            data.xscale["start"] = x_start
            data.xscale["delta"] = x_delta
            data.xscale["units"] = x_units

    # Optionally create dataseries
    if make_dataseries:
        prefixes = folder.detect_prefixes()
        for p in prefixes:
            folder.make_dataseries(p)

    return folder
