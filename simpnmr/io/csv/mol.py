# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Read and write molecular structure and hyperfine data as CSV.

Provides CSV parsing and serialization helpers for atom labels, coordinates,
optional chemical labels, and hyperfine tensors.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from simpnmr.io.csv.csv_util import read_csv_safe, write_csv_safe

logger = logging.getLogger(__name__)


def read_molecule_csv(file_name: str) -> dict:
    """Read a molecule CSV file and return raw structure + (optional) hyperfine + labels

    This function preserves the legacy parsing behaviour previously implemented
    in `Molecule.from_csv` (domain), but keeps IO in the IO layer.

    Hyperfine tensor payload is optional at the file level in this phase-1
    contract: structure-only CSV files are accepted and return ``tensors=None``.

    Returns dict with keys:
      - labels: list[str]
      - coords: np.ndarray shape (n, 3)
      - tensors: list[np.ndarray] | None
      - chem_labels: list[str] | None
      - chem_math_labels: list[str] | None
    """
    data = read_csv_safe(file_name)

    required_cols = ["atom_label ()", "x (Å)", "y (Å)", "z (Å)"]
    legacy_split_hyperfine_cols = [
        "Adip_xx (ppm Å^-3)",
        "Adip_xy (ppm Å^-3)",
        "Adip_xz (ppm Å^-3)",
        "Adip_yy (ppm Å^-3)",
        "Adip_yz (ppm Å^-3)",
        "Adip_zz (ppm Å^-3)",
    ]
    split_hyperfine_cols = [
        "A_fc_iso (ppm Å^-3)",
        "A_sd_xx (ppm Å^-3)",
        "A_sd_xy (ppm Å^-3)",
        "A_sd_xz (ppm Å^-3)",
        "A_sd_yy (ppm Å^-3)",
        "A_sd_yz (ppm Å^-3)",
        "A_sd_zz (ppm Å^-3)",
    ]
    split_hyperfine_cols_alt_iso = [
        "Aiso (ppm Å^-3)",
        "dA_xx (ppm Å^-3)",
        "dA_xy (ppm Å^-3)",
        "dA_xz (ppm Å^-3)",
        "dA_yy (ppm Å^-3)",
        "dA_yz (ppm Å^-3)",
        "dA_zz (ppm Å^-3)",
    ]
    full_hyperfine_cols = [
        "A_xx (ppm Å^-3)",
        "A_xy (ppm Å^-3)",
        "A_xz (ppm Å^-3)",
        "A_yy (ppm Å^-3)",
        "A_yz (ppm Å^-3)",
        "A_zz (ppm Å^-3)",
    ]

    # Standardise column names (kept identical to old domain implementation)
    name_convertor = {
        "atom_labels": "atom_label ()",
        "atom_labels ()": "atom_label ()",
        "chem_label": "chem_label ()",
        "chem_labels ()": "chem_label ()",
        "chem_math_label": "chem_math_label ()",
        "chem_math_labels ()": "chem_math_label ()",
        "x": "x (Å)",
        "x (A)": "x (Å)",
        "y": "y (Å)",
        "y (A)": "y (Å)",
        "z": "z (Å)",
        "z (A)": "z (Å)",
    }
    others = {}
    for key, val in name_convertor.items():
        others[key.capitalize()] = val
        others[val.capitalize()] = val
    name_convertor.update(others)
    data.rename(columns=name_convertor, inplace=True)

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing header(s) {missing} in {file_name}")

    # Detect hyperfine encoding (split vs full) exactly like before
    has_new_split = all(col in data.columns for col in split_hyperfine_cols)
    has_new_split_alt_iso = all(
        col in data.columns for col in split_hyperfine_cols_alt_iso
    )
    has_legacy_split = all(col in data.columns for col in legacy_split_hyperfine_cols)
    has_full = all(col in data.columns for col in full_hyperfine_cols)

    use_legacy = False
    if has_new_split:
        split = True
    elif not has_new_split and has_new_split_alt_iso:
        split = True
        logger.warning(
            "Legacy hyperfine column 'Aiso' detected. Please rename to 'Aiso_eff'; "
            "support for 'Aiso' will be removed in a future release."
        )
    elif has_legacy_split:
        split = True
        use_legacy = True
        logger.warning(
            "Legacy hyperfine columns 'Adip_*' detected. Please migrate to 'dA_*'; "
            "support for 'Adip_*' will be removed in a future release."
        )
    elif has_full:
        split = False
    else:
        split = None

    labels = data["atom_label ()"].tolist()

    coords = np.array([data["x (Å)"], data["y (Å)"], data["z (Å)"]]).T

    if split is None:
        tensors = None

    elif split:
        aiso_key = (
            "Aiso_eff (ppm Å^-3)"
            if "Aiso_eff (ppm Å^-3)" in data.columns
            else "Aiso (ppm Å^-3)"
        )
        if use_legacy:
            tensors = [
                np.array(
                    [
                        [
                            row["Adip_xx (ppm Å^-3)"],
                            row["Adip_xy (ppm Å^-3)"],
                            row["Adip_xz (ppm Å^-3)"],
                        ],
                        [
                            row["Adip_xy (ppm Å^-3)"],
                            row["Adip_yy (ppm Å^-3)"],
                            row["Adip_yz (ppm Å^-3)"],
                        ],
                        [
                            row["Adip_xz (ppm Å^-3)"],
                            row["Adip_yz (ppm Å^-3)"],
                            row["Adip_zz (ppm Å^-3)"],
                        ],
                    ],
                    dtype=float,
                )
                + np.eye(3) * float(row[aiso_key])
                for _, row in data.iterrows()
            ]
        else:
            tensors = [
                np.array(
                    [
                        [
                            row["A_sd_xx (ppm Å^-3)"],
                            row["A_sd_xy (ppm Å^-3)"],
                            row["A_sd_xz (ppm Å^-3)"],
                        ],
                        [
                            row["A_sd_xy (ppm Å^-3)"],
                            row["A_sd_yy (ppm Å^-3)"],
                            row["A_sd_yz (ppm Å^-3)"],
                        ],
                        [
                            row["A_sd_xz (ppm Å^-3)"],
                            row["A_sd_yz (ppm Å^-3)"],
                            row["A_sd_zz (ppm Å^-3)"],
                        ],
                    ],
                    dtype=float,
                )
                + np.eye(3) * float(row[aiso_key])
                for _, row in data.iterrows()
            ]
    else:
        tensors = [
            np.array(
                [
                    [
                        row["A_xx (ppm Å^-3)"],
                        row["A_xy (ppm Å^-3)"],
                        row["A_xz (ppm Å^-3)"],
                    ],
                    [
                        row["A_xy (ppm Å^-3)"],
                        row["A_yy (ppm Å^-3)"],
                        row["A_yz (ppm Å^-3)"],
                    ],
                    [
                        row["A_xz (ppm Å^-3)"],
                        row["A_yz (ppm Å^-3)"],
                        row["A_zz (ppm Å^-3)"],
                    ],
                ],
                dtype=float,
            )
            for _, row in data.iterrows()
        ]

    chem_labels = None
    chem_math_labels = None

    if "chem_label ()" in data.columns:
        chem_labels = data["chem_label ()"].tolist()

    if "chem_math_label ()" in data.columns:
        chem_math_labels = data["chem_math_label ()"].tolist()

    return {
        "labels": labels,
        "coords": coords,
        "tensors": tensors,
        "chem_labels": chem_labels,
        "chem_math_labels": chem_math_labels,
    }


def save_molecule_to_csv(
    molecule,
    file_name: str = "molecule.csv",
    verbose: bool = True,
    comment: str = "",
    delimiter: str = ",",
) -> None:
    """Save molecule structure, hyperfine data, and shifts to a CSV file.

    Args:
        file_name: Output CSV file name.
        verbose: If True, prints the output file path.
        comment: Optional additional comment line (including comment marker).
        delimiter: CSV delimiter.
    """

    df = _build_molecule_df(molecule)

    write_csv_safe(df, file_name, comment)

    if verbose:
        logger.info("Molecule data written to %s", file_name)

    return


def _build_molecule_df(molecule):
    """Build a full molecule table for CSV export."""

    nuclei = molecule.nuclei

    hyperfine_meta = molecule.metadata.get("hyperfine", {})
    has_orb = hyperfine_meta.get("orbital_contribution") == "available"

    base_specs = [
        ("atom_label ()", lambda nuc: nuc.label),
        ("chem_label ()", lambda nuc: nuc.chem_label),
        ("x (Å)", lambda nuc: nuc.coord[0]),
        ("y (Å)", lambda nuc: nuc.coord[1]),
        ("z (Å)", lambda nuc: nuc.coord[2]),
        ("A_fc_iso (ppm Å^-3)", lambda nuc: 1.0 / 3.0 * np.trace(nuc.A.fc)),
        ("A_sd_xx (ppm Å^-3)", lambda nuc: nuc.A.sd[0, 0]),
        ("A_sd_xy (ppm Å^-3)", lambda nuc: nuc.A.sd[0, 1]),
        ("A_sd_xz (ppm Å^-3)", lambda nuc: nuc.A.sd[0, 2]),
        ("A_sd_yy (ppm Å^-3)", lambda nuc: nuc.A.sd[1, 1]),
        ("A_sd_yz (ppm Å^-3)", lambda nuc: nuc.A.sd[1, 2]),
        ("A_sd_zz (ppm Å^-3)", lambda nuc: nuc.A.sd[2, 2]),
        ("δ_total_avg (ppm)", lambda nuc: nuc.shift.avg),
        ("δ_total (ppm)", lambda nuc: nuc.shift.total),
        ("δ_dia (ppm)", lambda nuc: nuc.shift.dia),
        ("δ_fc (ppm)", lambda nuc: nuc.shift.fc),
        ("δ_pc (ppm)", lambda nuc: nuc.shift.pc),
    ]

    orb_specs = [
        ("δ_orb (ppm)", lambda nuc: getattr(nuc.shift, "orb", 0.0)),
        ("δ_orb_iso (ppm)", lambda nuc: getattr(nuc.shift, "orb_iso", 0.0)),
        (
            "δ_orb_aniso (ppm)",
            lambda nuc: getattr(nuc.shift, "orb_aniso", 0.0),
        ),
    ]

    tail_specs = [
        (
            "linewidth (ppm)",
            lambda nuc: nuc.shift.lw if hasattr(nuc.shift, "lw") else 1.0,
        ),
    ]

    spec_groups = [
        base_specs,
        orb_specs if has_orb else [],
        tail_specs,
    ]
    specs = [spec for group in spec_groups for spec in group]

    columns = [name for name, _ in specs]
    data = {name: [getter(nuc) for nuc in nuclei] for name, getter in specs}

    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return df

    return df.reset_index(drop=True)
