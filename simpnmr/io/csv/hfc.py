# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write hyperfine coupling data as CSV.

Provides helpers to serialize hyperfine tensors for all nuclei and to export
them in tabular CSV form.
"""

import datetime
import logging

import numpy as np
import pandas as pd

from simpnmr.__version__ import __version__
from simpnmr.io.csv.csv_util import write_csv_safe

logger = logging.getLogger(__name__)


def save_to_csv(
    hyperfine_data,
    file_name: str = "dft_hyperfines.csv",
    verbose: bool = True,
    comment: str = "",
    delimiter: str = ",",
) -> None:
    """Save hyperfine data to a CSV file.

    Args:
        hyperfine_data: Hyperfine container (e.g., QC A-tensor reader result) exposing
            `a_iso`, `a_dtensor`, and `a_units`.
        file_name: Output CSV file name.
        verbose: If True, prints the output file name.
        comment: Optional additional comment line (including comment marker).
        delimiter: Delimiter used in the CSV.
    """

    # Save hyperfine data to file
    out = np.array(
        [
            "{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                label, iso, *tensor[0, :], *tensor[1, 1:], tensor[2, 2]
            )
            for iso, (label, tensor) in zip(
                hyperfine_data.a_iso.values(), hyperfine_data.a_dtensor.items()
            )
        ]
    )

    _comments = (
        f"#This file was generated with SimpNMR v{__version__} on {{}}\n".format(
            datetime.datetime.now().strftime("%H:%M:%S %d-%m-%Y ")
        )
    )

    _comments += comment + "\n"

    header = (
        f"atom_label, "
        f"Aiso ({hyperfine_data.a_units}), "
        f"dA_xx ({hyperfine_data.a_units}), "
        f"dA_xy ({hyperfine_data.a_units}), "
        f"dA_xz ({hyperfine_data.a_units}), "
        f"dA_yy ({hyperfine_data.a_units}), "
        f"dA_yz ({hyperfine_data.a_units}), "
        f"dA_zz ({hyperfine_data.a_units})"
    )

    # Save to file
    np.savetxt(
        file_name,
        out,
        delimiter=delimiter,
        header=header,
        fmt="%s",
        comments=_comments,
    )

    if verbose:
        logger.info("Raw DFT Hyperfine data written to %s", file_name)

    return


def save_hyperfines_to_csv(
    hyperfines=list,
    file_name: str = "dft_hyperfines.csv",
    verbose: bool = True,
    comment: str = "",
) -> None:
    """Save hyperfine data for all nuclei to a CSV file.

    Args:
        file_name: Output CSV file name.
        verbose: If True, prints the output file path.
        comment: Optional additional comment line (including comment marker).
        delimiter: CSV delimiter.
    """

    df = _build_hyperfines_df(hyperfines)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Hyperfine data written to %s", file_name)

    return


def _build_hyperfines_df(molecule):
    """Build a hyperfine-couplings table for CSV export."""

    columns = [
        "atom_label ()",
        "chem_label ()",
        "Aiso (ppm Å^-3)",
        "dA_xx (ppm Å^-3)",
        "dA_xy (ppm Å^-3)",
        "dA_xz (ppm Å^-3)",
        "dA_yy (ppm Å^-3)",
        "dA_yz (ppm Å^-3)",
        "dA_zz (ppm Å^-3)",
    ]

    nuclei = molecule.nuclei

    data = {
        "atom_label ()": [nuc.label for nuc in nuclei],
        "chem_label ()": [nuc.chem_label for nuc in nuclei],
        "Aiso (ppm Å^-3)": [nuc.A.iso for nuc in nuclei],
        "dA_xx (ppm Å^-3)": [nuc.A.dtensor[0, 0] for nuc in nuclei],
        "dA_xy (ppm Å^-3)": [nuc.A.dtensor[0, 1] for nuc in nuclei],
        "dA_xz (ppm Å^-3)": [nuc.A.dtensor[0, 2] for nuc in nuclei],
        "dA_yy (ppm Å^-3)": [nuc.A.dtensor[1, 1] for nuc in nuclei],
        "dA_yz (ppm Å^-3)": [nuc.A.dtensor[1, 2] for nuc in nuclei],
        "dA_zz (ppm Å^-3)": [nuc.A.dtensor[2, 2] for nuc in nuclei],
    }

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.reset_index(drop=True)
