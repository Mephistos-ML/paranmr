# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generic summaries for hyperfine benchmark metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from simpnmr.core.domain.mol import Molecule


def summarize_hyperfine_metric_ranges_by_functional_and_nucleus(
    functional_sources: dict[str, list[tuple[str, Molecule]]],
    *,
    metric_key: str,
    metric_getter: Callable[[Any], float],
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Summarize a scalar hyperfine metric by functional, nucleus, and label."""
    grouped_values: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}

    for functional, sources in functional_sources.items():
        grouped_values.setdefault(functional, {})
        for source_id, molecule in sources:
            for nucleus in molecule.nuclei:
                nucleus_label = nucleus.label_nn
                chem_label = nucleus.chem_label
                metric_value = float(metric_getter(nucleus))
                grouped_values[functional].setdefault(nucleus_label, {})
                grouped_values[functional][nucleus_label].setdefault(
                    chem_label, []
                ).append(
                    {
                        "source_id": source_id,
                        "atom_label": nucleus.label,
                        "chem_math_label": nucleus.chem_math_label,
                        metric_key: metric_value,
                    }
                )

    summary: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for functional, nucleus_values in grouped_values.items():
        summary[functional] = {}
        for nucleus_label, chem_label_values in nucleus_values.items():
            summary[functional][nucleus_label] = {}
            for chem_label, values in chem_label_values.items():
                metric_values = [float(entry[metric_key]) for entry in values]
                summary[functional][nucleus_label][chem_label] = {
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "mean": float(np.mean(metric_values)),
                    "count": len(metric_values),
                    "chem_math_label": values[0]["chem_math_label"],
                    "values": values,
                }

    return summary


def summarize_hyperfine_metric_max_by_nucleus(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    *,
    max_label_tolerance: float = 0.0,
) -> dict[str, list[dict[str, object]]]:
    """Summarize maximum scalar hyperfine metric values by nucleus."""
    raw_rows_by_nucleus: dict[str, list[dict[str, object]]] = {}

    for functional, nucleus_summary in summary.items():
        for nucleus_label, chem_label_summary in nucleus_summary.items():
            max_chem_label, max_summary = max(
                chem_label_summary.items(),
                key=lambda item: float(item[1]["max"]),
            )
            raw_rows_by_nucleus.setdefault(nucleus_label, []).append(
                {
                    "functional": functional,
                    "max": float(max_summary["max"]),
                    "chem_label": max_chem_label,
                    "raw_max": float(max_summary["max"]),
                    "raw_chem_label": max_chem_label,
                    "adjusted": False,
                }
            )

    max_by_nucleus: dict[str, list[dict[str, object]]] = {}
    for nucleus_label, raw_rows in raw_rows_by_nucleus.items():
        majority_label = Counter(
            str(row["raw_chem_label"]) for row in raw_rows
        ).most_common(1)[0][0]

        adjusted_rows: list[dict[str, object]] = []
        for row in raw_rows:
            adjusted_row = dict(row)
            raw_max = float(row["raw_max"])
            adjusted_row["majority_chem_label"] = majority_label
            majority_max = _get_chem_label_max(
                summary=summary,
                functional=str(row["functional"]),
                nucleus_label=nucleus_label,
                chem_label=majority_label,
            )

            if majority_max is not None:
                drop_fraction = (
                    abs(raw_max - majority_max) / abs(raw_max)
                    if raw_max != 0.0
                    else abs(raw_max - majority_max)
                )
                if drop_fraction <= max_label_tolerance:
                    adjusted_row["max"] = majority_max
                    adjusted_row["chem_label"] = majority_label
                    adjusted_row["adjusted"] = majority_label != row["raw_chem_label"]
                    adjusted_row["drop_fraction"] = drop_fraction

            adjusted_rows.append(adjusted_row)

        max_by_nucleus[nucleus_label] = sorted(
            adjusted_rows,
            key=lambda row: float(row["max"]),
            reverse=True,
        )

    return max_by_nucleus


def _get_chem_label_max(
    *,
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    functional: str,
    nucleus_label: str,
    chem_label: str,
) -> float | None:
    """Return a chem-label maximum from nested benchmark summary."""
    try:
        return float(summary[functional][nucleus_label][chem_label]["max"])
    except KeyError:
        return None
