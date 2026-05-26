# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Summaries for A_fc benchmark data."""

from __future__ import annotations

import numpy as np

from simpnmr.core.benchmarks.hyperfine.summary import (
    summarize_hyperfine_metric_max_by_nucleus,
    summarize_hyperfine_metric_ranges_by_functional_and_nucleus,
)
from simpnmr.core.domain.mol import Molecule


def summarize_a_fc_ranges_by_functional_and_nucleus(
    functional_sources: dict[str, list[tuple[str, Molecule]]],
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Summarize A_fc values by functional, nucleus, and chemical label."""
    return summarize_hyperfine_metric_ranges_by_functional_and_nucleus(
        functional_sources,
        metric_key="a_fc",
        metric_getter=_get_a_fc,
    )


def summarize_a_fc_max_by_nucleus(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    *,
    max_label_tolerance: float = 0.0,
) -> dict[str, list[dict[str, object]]]:
    """Summarize maximum A_fc values by nucleus across functionals."""
    return summarize_hyperfine_metric_max_by_nucleus(
        summary,
        max_label_tolerance=max_label_tolerance,
    )


def summarize_a_fc_max_report_rows(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    max_by_nucleus: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Build tabular A_fc max benchmark rows for CSV export."""
    rows: list[dict[str, object]] = []

    for nucleus_label, max_rows in max_by_nucleus.items():
        for row in max_rows:
            functional = str(row["functional"])
            chem_label = str(row["chem_label"])
            max_value = float(row["max"])
            min_value = float(summary[functional][nucleus_label][chem_label]["min"])
            range_value = (
                (max_value - min_value) / max_value
                if max_value != 0.0
                else np.nan
            )

            rows.append(
                {
                    "chem_label": chem_label,
                    "nucleus": nucleus_label,
                    "functional": functional,
                    "max (ppm A-3)": max_value,
                    "min (ppm A-3)": min_value,
                    "range": range_value,
                }
            )

    return rows


def _get_a_fc(nucleus) -> float:
    """Return isotropic A_fc for one nucleus."""
    return float(np.trace(nucleus.A.fc) / 3.0)
