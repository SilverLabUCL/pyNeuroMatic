# -*- coding: utf-8 -*-
"""
Igor Text (.itx) file writer.

Writes NMFolder data to Igor Text format for comparison with Igor/NeuroMatic.

Part of pyNeuroMatic.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_folder import NMFolder


def write_itx(
    folder: "NMFolder",
    filepath: str | Path,
) -> Path:
    """Write NMFolder data to an Igor Text file.

    Args:
        folder: NMFolder containing data to export.
        filepath: Output file path (.itx).

    Returns:
        Path to the written file.
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        f.write("IGOR\n")

        for name in folder.data:
            nmdata = folder.data[name]
            if nmdata.nparray is None:
                continue

            f.write(f"WAVES/D\t{name}\n")
            f.write("BEGIN\n")
            for val in nmdata.nparray:
                f.write(f"\t{val}\n")
            f.write("END\n")

            # x scaling: SetScale/P x, offset, delta, "units", waveName
            x_start = nmdata.xscale.start
            x_delta = nmdata.xscale.delta
            x_units = nmdata.xscale.units
            f.write(
                f'X SetScale/P x, {x_start}, {x_delta},'
                f' "{x_units}", {name}\n'
            )

            # y units: SetScale d, 0, 0, "units", waveName
            y_units = nmdata.yscale.units
            f.write(
                f'X SetScale d, 0, 0,'
                f' "{y_units}", {name}\n'
            )

    return filepath
