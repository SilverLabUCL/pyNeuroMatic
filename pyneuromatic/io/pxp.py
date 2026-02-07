# -*- coding: utf-8 -*-
"""
Igor Pro packed experiment (.pxp) file reader.

Reads PXP files created by NeuroMatic in Igor Pro using the igor2 library.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

References:
    - NeuroMatic Igor: https://github.com/SilverLabUCL/NeuroMatic
    - igor2 library: https://github.com/AFM-analysis/igor2
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_folder import NMFolder

from pyneuromatic.io.base import parse_units_from_label, make_data_name
import pyneuromatic.core.nm_utilities as nmu


def read_pxp(
    filepath: str | Path,
    folder: "NMFolder | None" = None,
    prefix: str | None = None,
    make_dataseries: bool = True,
) -> "NMFolder":
    """Read an Igor Pro PXP file into an NMFolder.

    Reads packed experiment files created by NeuroMatic in Igor Pro.
    Requires the igor2 library (pip install igor2).

    Args:
        filepath: Path to the PXP file.
        folder: Optional existing folder to add data to. If None, creates new.
        prefix: Prefix for data names. If None, auto-detects from file's
            WavePrefix variable (falls back to "Record").
        make_dataseries: If True, automatically create dataseries from data.

    Returns:
        NMFolder containing the imported data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If igor2 is not installed.
    """
    try:
        import igor2.packed as igor_packed
    except ImportError:
        raise ImportError(
            "igor2 is required to read PXP files. "
            "Install it with: pip install igor2"
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
        folder = NMFolder(name=folder_name)

    # Load the PXP file
    records, _ = igor_packed.load(str(filepath))

    # Walk records, extracting metadata and collecting wave records
    wave_records = []
    folder_stack = ["root"]

    for record in records:
        rtype = type(record).__name__

        if rtype == "FolderStartRecord":
            name = record.null_terminated_text
            if isinstance(name, bytes):
                name = name.decode("utf-8").rstrip("\x00")
            folder_stack.append(name)

        elif rtype == "FolderEndRecord":
            if len(folder_stack) > 1:
                folder_stack.pop()

        elif rtype == "VariablesRecord":
            current_folder = folder_stack[-1]
            variables = record.variables.get("variables", {})
            user_vars = variables.get("userVars", {})
            user_strs = variables.get("userStrs", {})

            # Merge vars and strings into metadata
            if user_vars or user_strs:
                if current_folder not in folder.metadata:
                    folder.metadata[current_folder] = {}
                for k, v in user_vars.items():
                    key = k.decode("utf-8") if isinstance(k, bytes) else k
                    folder.metadata[current_folder][key] = v
                for k, v in user_strs.items():
                    key = k.decode("utf-8") if isinstance(k, bytes) else k
                    val = v.decode("utf-8") if isinstance(v, bytes) else v
                    folder.metadata[current_folder][key] = val

        elif rtype == "WaveRecord":
            wave_records.append(record)

    # Resolve prefix
    if prefix is None:
        root_meta = folder.metadata.get("root", {})
        prefix = root_meta.get("WavePrefix", "Record")

    # Find the yLabel text wave for y-axis labels/units
    y_labels = _find_ylabel_wave(wave_records)

    # Get x-units from root variables
    root_meta = folder.metadata.get("root", {})
    x_units = root_meta.get("xLabel", "ms")

    # Process data wave records
    for record in wave_records:
        wave = record.wave["wave"]
        wave_header = wave["wave_header"]

        bname = wave_header["bname"]
        if isinstance(bname, bytes):
            wave_name = bname.decode("utf-8").rstrip("\x00")
        else:
            wave_name = str(bname).rstrip("\x00")

        # Parse using NeuroMatic naming convention
        parsed = nmu.parse_data_name(wave_name)
        if parsed is None:
            continue

        wave_prefix, channel_char, epoch_num = parsed
        if wave_prefix != prefix:
            continue

        channel_num = ord(channel_char) - ord("A")

        # Build the data name using the requested prefix
        name = make_data_name(prefix, channel_num, epoch_num)

        # Create NMData
        data = folder.data.new(name)
        if data is None:
            continue

        # Set y data
        data.y.data = wave["wData"]

        # Set y label/units from yLabel config wave
        if channel_num < len(y_labels):
            y_parsed = parse_units_from_label(y_labels[channel_num])
            data.y.label = y_parsed.label
            data.y.units = y_parsed.units

        # Set x scaling from wave header
        data.x.start = float(wave_header.get("hsB", 0.0))
        data.x.delta = float(wave_header.get("hsA", 1.0))
        data.x.units = x_units

    # Optionally create dataseries
    if make_dataseries:
        prefixes = folder.detect_prefixes()
        for p in prefixes:
            folder.make_dataseries(p)

    return folder


def _find_ylabel_wave(wave_records: list) -> list[str]:
    """Find and parse the yLabel text wave from PXP records.

    Returns list of y-axis label strings, one per channel.
    """
    for record in wave_records:
        wave = record.wave["wave"]
        bname = wave["wave_header"]["bname"]
        if isinstance(bname, bytes):
            name = bname.decode("utf-8").rstrip("\x00")
        else:
            name = str(bname).rstrip("\x00")

        if name == "yLabel":
            wdata = wave["wData"]
            labels = []
            for item in wdata:
                if isinstance(item, bytes):
                    labels.append(item.decode("utf-8").rstrip("\x00"))
                else:
                    labels.append(str(item).rstrip("\x00"))
            return labels

    return []
