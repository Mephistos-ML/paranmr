# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Summaries for A_fc benchmark data."""

from __future__ import annotations

from collections import Counter

import numpy as np

from simpnmr.core.domain.mol import Molecule


def summarize_a_fc_ranges_by_functional_and_nucleus(
    functional_sources: dict[str, list[tuple[str, Molecule]]],
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Summarize A_fc values by functional, nucleus, and chemical label.

    Args:
        functional_sources: Mapping from functional name to source-labelled
            molecules. Source labels are caller-defined identifiers used only
            for diagnostics.

    Returns:
        Nested mapping ``functional -> nucleus -> chem_label -> summary``. Each
        summary contains ``min``, ``max``, ``mean``, ``count``, and
        source-resolved ``values``.
    """
    grouped_values: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}

    for functional, sources in functional_sources.items():
        grouped_values.setdefault(functional, {})
        for source_id, molecule in sources:
            for nucleus in molecule.nuclei:
                nucleus_label = nucleus.label_nn
                chem_label = nucleus.chem_label
                a_fc = float(np.trace(nucleus.A.fc) / 3.0)
                grouped_values[functional].setdefault(nucleus_label, {})
                grouped_values[functional][nucleus_label].setdefault(
                    chem_label, []
                ).append(
                    {
                        "source_id": source_id,
                        "atom_label": nucleus.label,
                        "chem_math_label": nucleus.chem_math_label,
                        "a_fc": a_fc,
                    }
                )

    summary: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for functional, nucleus_values in grouped_values.items():
        summary[functional] = {}
        for nucleus_label, chem_label_values in nucleus_values.items():
            summary[functional][nucleus_label] = {}
            for chem_label, values in chem_label_values.items():
                a_fc_values = [float(entry["a_fc"]) for entry in values]
                summary[functional][nucleus_label][chem_label] = {
                    "min": min(a_fc_values),
                    "max": max(a_fc_values),
                    "mean": float(np.mean(a_fc_values)),
                    "count": len(a_fc_values),
                    "chem_math_label": values[0]["chem_math_label"],
                    "values": values,
                }

    return summary


def summarize_a_fc_max_by_nucleus(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    *,
    max_label_tolerance: float = 0.0,
) -> dict[str, list[dict[str, object]]]:
    """Summarize maximum A_fc values by nucleus across functionals.

    Args:
        summary: Nested A_fc summary produced by
            ``summarize_a_fc_ranges_by_functional_and_nucleus``.
        max_label_tolerance: Relative tolerance used to replace a raw maximum
            label with the majority maximum label for that nucleus.

    Returns:
        Mapping from nucleus label to sorted rows. Each row contains
        ``functional`` and the maximum A_fc value observed for that functional
        and nucleus. Rows are sorted by descending maximum A_fc.
    """
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
    """Return a chem-label maximum from nested A_fc summary."""
    try:
        return float(summary[functional][nucleus_label][chem_label]["max"])
    except KeyError:
        return None
