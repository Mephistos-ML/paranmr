# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Provide safe CSV I/O utilities.

Defines helpers to read and write CSV files with explicit UTF-8 encoding,
basic validation, and normalization suitable for cross-platform use
(including Windows and Excel).
"""

import datetime
import os

import pandas as pd

from paranmr.__version__ import __version__
from paranmr.io.csv.csv_valid import validate_csv_delimiters


def read_csv_safe(
    file_name: str | os.PathLike,
    **kwargs,
) -> pd.DataFrame:
    try:
        # Detect delimiter from first non-comment, non-empty line
        delimiter_is_comma = False
        with open(file_name, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "," in stripped:
                    delimiter_is_comma = True
                break

        if delimiter_is_comma:
            # Validate only comma-separated CSV files
            validate_csv_delimiters(file_name)
            df = pd.read_csv(
                file_name,
                sep=",",
                skipinitialspace=True,
                comment="#",
                encoding="utf-8-sig",
                **kwargs,
            )
        else:
            # Whitespace-separated file
            df = pd.read_csv(
                file_name,
                sep=r"\s+",
                engine="python",
                comment="#",
                encoding="utf-8-sig",
                **kwargs,
            )

        # Normalize column names (strip BOM and surrounding whitespace)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

        return df
    except FileNotFoundError:
        raise ValueError(f"CSV file not found: {os.path.basename(file_name)}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {os.path.basename(file_name)}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {os.path.basename(file_name)}: {e}")


def write_csv_safe(
    df: pd.DataFrame,
    file_name: str | os.PathLike,
    comment: str | list[str] | None = None,
    *,
    sep: str = ",",
    float_format: str = "%.6f",
    index: bool = False,
    encoding: str = "utf-8-sig",
    newline: str = "",
    **kwargs,
) -> None:
    """Write DataFrame to CSV with explicit UTF-8 encoding (Excel-friendly by default).

    Symmetric to read_csv_safe. No delimiter sniffing or validation.

    Optionally, write comment lines (prefixed with '#') before the CSV header.
    """
    with open(file_name, "w", encoding=encoding, newline=newline) as fh:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        fh.write(
            f"# This file was generated with ParaNMR v{__version__} at {timestamp}\n"
        )
        if comment is not None:
            if isinstance(comment, str):
                fh.write(f"# {comment}\n")
            else:
                for line in comment:
                    fh.write(f"# {line}\n")
        df.to_csv(fh, sep=sep, index=index, float_format=float_format, **kwargs)
