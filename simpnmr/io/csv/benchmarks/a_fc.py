# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Write A_fc benchmark reports as CSV."""

import logging
from pathlib import Path

import pandas as pd

from simpnmr.io.csv.csv_util import write_csv_safe

logger = logging.getLogger(__name__)

A_FC_BENCHMARK_MAX_COLUMNS = [
    "chem_label",
    "nucleus",
    "functional",
    "max (ppm A-3)",
    "min (ppm A-3)",
    "range",
]


def save_a_fc_benchmark_max_csv(
    rows: list[dict[str, object]],
    file_name: str | Path,
    *,
    verbose: bool = True,
) -> None:
    """Save A_fc max benchmark rows to CSV."""
    out = Path(file_name)
    if out.parent:
        out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=A_FC_BENCHMARK_MAX_COLUMNS)
    write_csv_safe(df, out)

    if verbose:
        logger.info("A_fc benchmark max CSV saved to %s", out)
