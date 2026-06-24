# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shared linewidth helpers for susceptibility fitting."""

from __future__ import annotations

from dataclasses import dataclass

from paranmr.core.fitting.linewidth import predict_r6_linewidths


@dataclass(frozen=True)
class SusceptibilityLinewidthInputs:
    """Shared linewidth-forward inputs for susceptibility fitting."""

    mean_inv_r6_by_atom_label: dict[str, float] | None


def predict_r6_widths_by_atom_label(
    *,
    linewidth_inputs: SusceptibilityLinewidthInputs,
    linewidth_vars_by_name: dict[str, float],
) -> dict[str, float]:
    """Return per-atom-label linewidths from an ``r^-6`` susceptibility model."""

    if linewidth_inputs.mean_inv_r6_by_atom_label is None:
        raise ValueError(
            "Susceptibility fitting requires an R6 linewidth model for "
            "calculated packages."
        )
    if "p1" not in linewidth_vars_by_name or "p2" not in linewidth_vars_by_name:
        raise ValueError(
            "Susceptibility fitting requires linewidth variables p1 and p2 "
            "for the R6 linewidth model."
        )
    return predict_r6_linewidths(
        linewidth_inputs.mean_inv_r6_by_atom_label,
        linewidth_vars_by_name["p1"],
        linewidth_vars_by_name["p2"],
    )
