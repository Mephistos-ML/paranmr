# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read spectrum data from CSV files.

Provides helpers to parse two-column shift–intensity spectra into numeric arrays.
"""

import csv

import pandas as pd

from paranmr.io.csv.csv_util import read_csv_safe, write_csv_safe


def read_spectrum(file_name: str):
    """Loads spectrum data from a CSV file.

    The input file must contain two columns with no header: the first column
    is ppm (shift) and the second column is intensity.

    Args:
        file_name: Path to the CSV file.
    """

    df = read_csv_safe(
        file_name,
        header=None,
        quoting=csv.QUOTE_NONE,  # treat quotes as normal characters
        converters={
            0: lambda s: float(s.strip().strip("\"'")),
            1: lambda s: float(s.strip().strip("\"'")),
        },
    )

    if df.shape[1] != 2:
        raise ValueError(
            "Spectrum file must contain exactly two columns: ppm and intensity"
        )

    spectrum = df.to_numpy(dtype=float)

    return spectrum


def write_spectrum(
    file_name: str,
    shift_ppm,
    intensity,
):
    """Write spectrum data to a CSV file.

    Writes a two-column CSV with header:
    - shift (ppm)
    - intensity (a.u.)

    Args:
        file_name: Output CSV path.
        shift_ppm: 1D array-like of chemical shifts in ppm.
        intensity: 1D array-like of intensities (arbitrary units).
    """

    df = pd.DataFrame(
        {
            "shift (ppm)": shift_ppm,
            "intensity (a.u.)": intensity,
        }
    )

    write_csv_safe(df, file_name)
