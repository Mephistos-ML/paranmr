# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load signal label mappings from CSV.

Reads external data and returns plain mappings for downstream use.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from paranmr.io.csv.csv_util import read_csv_safe


def load_signal_labels_from_csv(
    file_name: str,
) -> Tuple[
    Dict[str, str],
    Optional[Dict[str, str]],
]:
    """Load signal labels from a CSV file.

    The CSV must include columns ``atom_label`` and ``signal_label``.
    Optionally, it may include ``signal_math_label``.

    Args:
        file_name: Path to the CSV file.

    Returns:
        al_to_sl: Mapping atom_label -> signal_label.
        al_to_sml: Mapping atom_label -> signal_math_label, or None if not provided.

    Raises:
        KeyError: If duplicate atom labels exist or required entries are missing.
    """

    table = read_csv_safe(file_name)

    # Required columns
    for col in ("atom_label", "signal_label"):
        if col not in table.columns:
            raise KeyError(f"Missing required column '{col}' in signal labels file")

    # Check for duplicate atom labels
    counts = table["atom_label"].value_counts()
    dupes = counts[counts > 1]
    if not dupes.empty:
        raise KeyError(
            f"Duplicate atom_label(s) in signal labels file: {list(dupes.index)}"
        )

    # Check for missing signal_label entries
    if table["signal_label"].isnull().any():
        missing = table.loc[table["signal_label"].isnull(), "atom_label"].iloc[0]
        raise KeyError(f"Missing signal_label for atom_label '{missing}'")

    al_to_sl = dict(zip(table["atom_label"], table["signal_label"]))

    # Optional signal_math_label
    al_to_sml: Optional[Dict[str, str]] = None
    if "signal_math_label" in table.columns:
        if table["signal_math_label"].isnull().any():
            missing = table.loc[
                table["signal_math_label"].isnull(),
                "atom_label",
            ].iloc[0]
            raise KeyError(f"Missing signal_math_label for atom_label '{missing}'")
        al_to_sml = {
            al: str(sml).strip()
            for al, sml in zip(table["atom_label"], table["signal_math_label"])
        }

    return al_to_sl, al_to_sml
