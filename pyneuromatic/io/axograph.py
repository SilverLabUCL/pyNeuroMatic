# -*- coding: utf-8 -*-
"""
Axograph file format reader.

Supports Axograph X files (.axgx) and classic Axograph files (.axgd).

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

References:
    - NeuroMatic Igor import: NM_ImportAxograph.ipf
    - axographio Python library
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


def read_axograph(
    filepath: str | Path,
    folder: "NMFolder | None" = None,
    prefix: str = "Record",
    make_dataseries: bool = True,
    channels: list[str] | None = None,
    epochs: list[int] | range | None = None,
    xrange: tuple[float | None, float | None] | None = None,
) -> "NMFolder":
    """Read an Axograph file into an NMFolder.

    Args:
        filepath: Path to the Axograph file (.axgx or .axgd).
        folder: Optional existing folder to add data to. If None, creates new.
        prefix: Prefix for data names (default "Record").
        make_dataseries: If True, automatically create dataseries from data.
        channels: Channel letters to import (e.g. ["A"]); None for all.
        epochs: Epoch numbers to import; None for all. The channels and
            epochs imported are renumbered from A and 0.
        xrange: (xbgn, xend) to crop each array to; None for an open end.

    Returns:
        NMFolder containing the imported data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not recognized.

    Example:
        >>> folder = read_axograph("recording.axgx")
        >>> print(folder.data.keys())
        ['RecordA0', 'RecordA1', 'RecordB0', 'RecordB1']
    """
    check_options(channels, epochs, xrange)
    filepath = Path(filepath)

    # Import here to avoid circular imports
    from pyneuromatic.core.nm_folder import NMFolder

    # Create or use provided folder
    if folder is None:
        # Sanitize filename: replace non-alphanumeric chars with underscores
        import re

        folder_name = re.sub(r"[^a-zA-Z0-9_]", "_", filepath.stem)
        if folder_name and not folder_name[0].isalpha():
            folder_name = "F" + folder_name
        folder = NMFolder(name=folder_name)

    x_start, x_delta, x_units, data_columns = _layout(filepath)
    if not data_columns:
        return folder

    channel_numbers, epoch_numbers = import_numbers(
        [nmu.channel_char(c) for c in sorted({col["channel"] for col in data_columns})],
        sorted({col["epoch"] for col in data_columns}), channels, epochs)
    note_import(folder, channel_numbers, epoch_numbers, channels, epochs, xrange)

    # Process data columns; build matches dict as we go
    matches = {}
    for col in data_columns:
        letter = nmu.channel_char(col["channel"])
        if letter not in channel_numbers or col["epoch"] not in epoch_numbers:
            continue
        channel, epoch = channel_numbers[letter], epoch_numbers[col["epoch"]]
        name = make_data_name(prefix, channel, epoch)

        # Build y data and scale (a copy: the file's contents are kept
        # for re-reading)
        y_data = col["data"]
        parsed = parse_units_from_label(col["title"])

        # Apply scale if needed
        y_data = y_data * parsed.scale if parsed.scale != 1.0 else y_data.copy()
        y_data, start = crop(y_data, x_start, x_delta, xrange)

        xscale = {"start": start, "delta": x_delta, "units": x_units}
        yscale = {"label": parsed.label, "units": parsed.units}

        # Create NMData
        data = folder.data.new(name, xscale=xscale, yscale=yscale)
        if data is None:
            continue

        data.nparray = y_data

        ch_char = nmu.channel_char(channel)
        if ch_char:
            matches[(ch_char, epoch)] = data

    # Optionally create dataseries directly from the matches dict
    if make_dataseries and matches:
        dataseries = folder.build_dataseries(prefix, matches)
        if dataseries is not None:
            dataseries.locked = True  # protect raw recordings

    return folder


def axograph_info(filepath: str | Path) -> FileInfo:
    """What an Axograph file holds. The file is read once and kept for
    :func:`read_axograph`."""
    x_start, x_delta, x_units, data_columns = _layout(Path(filepath))
    channels = {}
    for col in data_columns:
        if col["channel"] not in channels:
            parsed = parse_units_from_label(col["title"])
            channels[col["channel"]] = (nmu.channel_char(col["channel"]),
                                        parsed.label, parsed.units)
    return FileInfo(
        prefix="Record",
        channels=[channels[c] for c in sorted(channels)],
        epochs=sorted({col["epoch"] for col in data_columns}),
        n_points=max((len(col["data"]) for col in data_columns), default=0),
        x_start=x_start,
        x_delta=x_delta,
        x_units=x_units,
        file_format=_format_name(Path(filepath)),
    )


def _format_name(filepath: Path) -> str:
    """ "AxoGraph" (classic) or "AxoGraph X", from the file's header."""
    with open(filepath, "rb") as f:
        header = f.read(4).lower()
    return "AxoGraph X" if header == b"axgx" else "AxoGraph"


def clear_cache() -> None:
    """Forget the last file read (kept so describing then importing a file
    reads it once)."""
    _read_cached.cache_clear()


def _layout(filepath: Path) -> tuple[float, float, str, list[dict]]:
    """The file's X scale (start, delta, units) and data columns, each
    with its "channel" and "epoch" numbers."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    stat = filepath.stat()
    columns = _read_cached(str(filepath.resolve()), stat.st_mtime_ns, stat.st_size)
    if not columns:
        return 0.0, 1.0, "ms", []

    # First column is typically time (x-axis)
    time_col = columns[0]
    x_start = 0.0
    x_delta = 1.0
    x_units = "ms"

    if len(time_col["data"]) > 1:
        x_start = float(time_col["data"][0])
        x_delta = float(time_col["data"][1] - time_col["data"][0])

    # Parse time units
    parsed = parse_units_from_label(time_col["title"])
    if parsed.units:
        x_units = parsed.units
        x_start *= parsed.scale
        x_delta *= parsed.scale

    # Detect channels and epochs from column titles.
    # Columns with the same title are the same channel across epochs.
    # When a title repeats, it signals a new epoch.
    data_columns = []
    channel_map: dict[str, int] = {}  # title -> channel index
    epoch_counter: dict[str, int] = {}  # title -> next epoch

    for col in columns[1:]:
        title = col["title"]
        if title not in channel_map:
            channel_map[title] = len(channel_map)
            epoch_counter[title] = 0
        data_columns.append({**col, "channel": channel_map[title],
                             "epoch": epoch_counter[title]})
        epoch_counter[title] += 1
    return x_start, x_delta, x_units, data_columns


@functools.lru_cache(maxsize=1)
def _read_cached(path: str, _mtime: int, _size: int) -> list[dict]:
    return _read_axograph_file(Path(path))


def _read_axograph_file(filepath: Path) -> list[dict]:
    """Read raw column data from an Axograph file.

    Args:
        filepath: Path to the Axograph file.

    Returns:
        List of column dicts with 'title' and 'data' keys.

    Raises:
        ValueError: If file format is not recognized.
    """
    import struct
    import numpy as np

    columns: list[dict] = []

    with open(filepath, "rb") as f:
        # Read header
        header = f.read(4)

        # Handle both cases - some files use lowercase headers
        header_lower = header.lower()
        if header_lower == b"axgr":
            # Classic Axograph format
            columns = _read_axograph_classic(f)
        elif header_lower == b"axgx":
            # Axograph X format
            columns = _read_axograph_x(f)
        else:
            raise ValueError(
                f"Unrecognized Axograph file format. "
                f"Header: {header!r}"
            )

    return columns


def _read_axograph_classic(f) -> list[dict]:
    """Read classic Axograph format (AxGr).

    Format versions 1 and 2.
    """
    import struct
    import numpy as np

    columns: list[dict] = []

    # Read version and column count
    version_data = f.read(4)
    version = struct.unpack(">h", version_data[:2])[0]
    num_columns = struct.unpack(">h", version_data[2:4])[0]

    for col_idx in range(num_columns):
        # Read column header
        num_points = struct.unpack(">i", f.read(4))[0]
        col_type = struct.unpack(">i", f.read(4))[0]

        # Read title (80 bytes, null-terminated)
        title_bytes = f.read(80)
        title = title_bytes.split(b"\x00")[0].decode("latin-1")

        if col_idx == 0:
            # First column: time/x-axis
            # Read sample interval as float
            sample_interval = struct.unpack(">f", f.read(4))[0]
            # Generate time array
            data = np.arange(num_points) * sample_interval
        else:
            # Data column: read scale factor and data
            scale_factor = struct.unpack(">f", f.read(4))[0]
            # Read short integers
            raw_data = np.frombuffer(f.read(num_points * 2), dtype=">i2")
            data = raw_data.astype(float) * scale_factor

        columns.append({"title": title, "data": data})

    return columns


def _read_axograph_x(f) -> list[dict]:
    """Read Axograph X format (AxGx).

    Format version 3+.
    """
    import struct
    import numpy as np

    columns: list[dict] = []

    # Read version
    version = struct.unpack(">i", f.read(4))[0]

    # Read number of columns
    num_columns = struct.unpack(">i", f.read(4))[0]

    for col_idx in range(num_columns):
        # Read number of points
        num_points = struct.unpack(">i", f.read(4))[0]

        # Read data type
        data_type = struct.unpack(">i", f.read(4))[0]

        # Read title length (in bytes) and title
        title_len = struct.unpack(">i", f.read(4))[0]
        title_bytes = f.read(title_len)  # UTF-16-BE encoded
        title = title_bytes.decode("utf-16-be").rstrip("\x00")

        # Read data based on type
        if data_type == 4:
            # Short integers (no scale)
            raw_data = np.frombuffer(f.read(num_points * 2), dtype=">i2")
            data = raw_data.astype(float)

        elif data_type == 5:
            # Long integers (no scale)
            raw_data = np.frombuffer(f.read(num_points * 4), dtype=">i4")
            data = raw_data.astype(float)

        elif data_type == 6:
            # Float
            data = np.frombuffer(f.read(num_points * 4), dtype=">f4")

        elif data_type == 7:
            # Double
            data = np.frombuffer(f.read(num_points * 8), dtype=">f8")

        elif data_type == 9:
            # Series (start + delta, for x-axis)
            start = struct.unpack(">d", f.read(8))[0]
            delta = struct.unpack(">d", f.read(8))[0]
            data = start + np.arange(num_points) * delta

        elif data_type == 10:
            # Scaled short: scale + offset + raw short data
            scale = struct.unpack(">d", f.read(8))[0]
            offset = struct.unpack(">d", f.read(8))[0]
            raw_data = np.frombuffer(f.read(num_points * 2), dtype=">i2")
            data = raw_data.astype(float) * scale + offset

        else:
            raise ValueError(
                f"Unsupported Axograph data type: {data_type}"
            )

        columns.append({"title": title, "data": data})

    return columns
