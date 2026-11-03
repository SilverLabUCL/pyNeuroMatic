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

import functools

from pyneuromatic.io.base import (
    FileInfo, check_options, crop, import_numbers, make_data_name, note_import,
    parse_units_from_label,
)
import pyneuromatic.core.nm_utilities as nmu


def read_pxp(
    filepath: str | Path,
    folder: "NMFolder | None" = None,
    prefix: str | None = None,
    make_dataseries: bool = True,
    channels: list[str] | None = None,
    epochs: list[int] | range | None = None,
    xrange: tuple[float | None, float | None] | None = None,
    wave_prefix: str | None = None,
) -> "NMFolder":
    """Read an Igor Pro PXP file into an NMFolder.

    Reads packed experiment files created by NeuroMatic in Igor Pro.
    Requires the igor2 library (pip install igor2).

    Args:
        filepath: Path to the PXP file.
        folder: Optional existing folder to add data to. If None, creates new.
        prefix: Prefix for the imported data's names; default: *wave_prefix*.
        make_dataseries: If True, automatically create dataseries from data.
        channels: Channel letters to import (e.g. ["A"]); None for all.
        epochs: Epoch numbers to import; None for all. The channels and
            epochs imported are renumbered from A and 0.
        xrange: (xbgn, xend) to crop each array to; None for an open end.
        wave_prefix: The prefix of the file's waves to read. If None,
            the file's WavePrefix variable (falls back to "Record").

    Returns:
        NMFolder containing the imported data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If igor2 is not installed.
    """
    check_options(channels, epochs, xrange)
    metadata, wave_records = _load(filepath)

    # Import here to avoid circular imports
    from pyneuromatic.core.nm_folder import NMFolder

    # Create or use provided folder
    if folder is None:
        import re

        folder_name = re.sub(r"[^a-zA-Z0-9_]", "_", Path(filepath).stem)
        if folder_name and not folder_name[0].isalpha():
            folder_name = "F" + folder_name
        folder = NMFolder(name=folder_name)

    for igor_folder, values in metadata.items():
        folder.metadata.setdefault(igor_folder, {}).update(values)

    root_meta = metadata.get("root", {})
    if wave_prefix is None:
        wave_prefix = root_meta.get("WavePrefix", "Record")
    if prefix is None:
        prefix = wave_prefix

    # Find the yLabel text wave for y-axis labels/units
    y_labels = _find_ylabel_wave(wave_records)

    # Get x-units from root variables
    x_units = root_meta.get("xLabel", "ms")

    waves = _prefix_waves(wave_records, wave_prefix)  # (channel, epoch, record)
    channel_numbers, epoch_numbers = import_numbers(
        sorted({c for c, _e, _r in waves}), sorted({e for _c, e, _r in waves}),
        channels, epochs)
    note_import(folder, channel_numbers, epoch_numbers, channels, epochs, xrange)

    # Process data wave records; build matches dict as we go
    matches = {}
    for channel_char, epoch_num, record in waves:
        if channel_char not in channel_numbers or epoch_num not in epoch_numbers:
            continue
        wave = record.wave["wave"]
        wave_header = wave["wave_header"]
        channel_num = ord(channel_char) - ord("A")  # the file's, for its y label
        new_channel, new_epoch = channel_numbers[channel_char], epoch_numbers[epoch_num]

        # Build the data name using the requested prefix
        name = make_data_name(prefix, new_channel, new_epoch)

        delta = float(wave_header.get("hsA", 1.0))
        y, start = crop(wave["wData"], float(wave_header.get("hsB", 0.0)), delta, xrange)

        # Build scale dicts
        xscale = {"start": start, "delta": delta, "units": x_units}
        yscale = None
        if channel_num < len(y_labels):
            y_parsed = parse_units_from_label(y_labels[channel_num])
            yscale = {"label": y_parsed.label, "units": y_parsed.units}

        # Create NMData
        data = folder.data.new(name, xscale=xscale, yscale=yscale)
        if data is None:
            continue

        # Set y data (a copy: the file's contents are kept for re-reading)
        data.nparray = y.copy()

        matches[(nmu.channel_char(new_channel), new_epoch)] = data

    # Optionally create dataseries directly from the matches dict
    if make_dataseries and matches:
        dataseries = folder.build_dataseries(prefix, matches)
        if dataseries is not None:
            dataseries.locked = True  # protect raw recordings

    return folder


def pxp_info(filepath: str | Path, wave_prefix: str | None = None) -> FileInfo:
    """What a PXP file holds: its waves with the file's WavePrefix (or
    *wave_prefix*). The file is read once and kept for :func:`read_pxp`."""
    metadata, wave_records = _load(filepath)
    root_meta = metadata.get("root", {})
    if wave_prefix is None:
        wave_prefix = root_meta.get("WavePrefix", "Record")
    waves = _prefix_waves(wave_records, wave_prefix)
    y_labels = _find_ylabel_wave(wave_records)
    channels = []
    for letter in sorted({c for c, _e, _r in waves}):
        number = ord(letter) - ord("A")
        parsed = parse_units_from_label(y_labels[number]) if number < len(y_labels) else None
        channels.append((letter, parsed.label if parsed else "", parsed.units if parsed else ""))
    first = waves[0][2].wave["wave"] if waves else None
    return FileInfo(
        prefix=wave_prefix,
        channels=channels,
        epochs=sorted({e for _c, e, _r in waves}),
        n_points=max((len(r.wave["wave"]["wData"]) for _c, _e, r in waves), default=0),
        x_start=float(first["wave_header"].get("hsB", 0.0)) if first else 0.0,
        x_delta=float(first["wave_header"].get("hsA", 1.0)) if first else 1.0,
        x_units=root_meta.get("xLabel", "ms"),
        file_format="Igor Pro packed experiment",
    )


def clear_cache() -> None:
    """Forget the last file read (kept so describing then importing a file
    reads it once)."""
    _load_cached.cache_clear()


def _load(filepath: str | Path) -> tuple[dict, list]:
    """The file's metadata ({Igor folder: {name: value}}) and wave records."""
    try:
        import igor2.packed  # noqa: F401
    except ImportError:
        raise ImportError(
            "igor2 is required to read PXP files. "
            "Install it with: pip install igor2"
        )
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    stat = filepath.stat()
    return _load_cached(str(filepath.resolve()), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=1)
def _load_cached(path: str, _mtime: int, _size: int) -> tuple[dict, list]:
    import igor2.packed as igor_packed

    records, _ = igor_packed.load(path)

    # Walk records, extracting metadata and collecting wave records
    metadata: dict[str, dict] = {}
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
                values = metadata.setdefault(current_folder, {})
                for k, v in user_vars.items():
                    key = k.decode("utf-8") if isinstance(k, bytes) else k
                    values[key] = v
                for k, v in user_strs.items():
                    key = k.decode("utf-8") if isinstance(k, bytes) else k
                    val = v.decode("utf-8") if isinstance(v, bytes) else v
                    values[key] = val

        elif rtype == "WaveRecord":
            wave_records.append(record)

    return metadata, wave_records


def _prefix_waves(wave_records: list, wave_prefix: str) -> list[tuple[str, int, object]]:
    """(channel letter, epoch number, record) of the waves named
    ``{wave_prefix}{channel}{epoch}`` (NeuroMatic's naming), in file order."""
    waves = []
    for record in wave_records:
        bname = record.wave["wave"]["wave_header"]["bname"]
        if isinstance(bname, bytes):
            wave_name = bname.decode("utf-8").rstrip("\x00")
        else:
            wave_name = str(bname).rstrip("\x00")
        parsed = nmu.parse_data_name(wave_name)
        if parsed is not None and parsed[0] == wave_prefix:
            waves.append((parsed[1], parsed[2], record))
    return waves


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
