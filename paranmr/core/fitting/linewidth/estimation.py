# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Estimate ``r^-6`` linewidth-model parameters from experimental widths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class R6LinewidthParameterEstimate:
    """Estimated parameters of an ``r^-6`` linewidth model.

    Args:
        linewidth_method: Linewidth model identifier.
        estimate_mode: Estimate mode identifier.
        p1: Distance-dependent linewidth coefficient in ``ppm Å^6``.
        p2: Distance-independent linewidth offset in ``ppm``.
        rmse: Root-mean-square error of the fitted widths in ``ppm``.
        mean_inv_r6_by_label: Mean ``1/r^6`` values used for the estimate.
        observed_widths_by_label: Experimental linewidths in ``ppm``.
        predicted_widths_by_label: Back-calculated linewidths in ``ppm``.
    """

    linewidth_method: str
    estimate_mode: str
    p1: float
    p2: float
    rmse: float
    mean_inv_r6_by_label: dict[str, float]
    observed_widths_by_label: dict[str, float]
    predicted_widths_by_label: dict[str, float]


def estimate_r6_linewidth_parameters(
    *,
    mean_inv_r6_by_label: Mapping[str, float],
    observed_widths_by_label: Mapping[str, float],
    fit_offset: bool,
) -> R6LinewidthParameterEstimate:
    """Estimate a non-negative ``r^-6`` linewidth model from experimental widths.

    Args:
        mean_inv_r6_by_label: Mapping from signal label to mean ``1/r^6`` in
            ``Å^-6``.
        observed_widths_by_label: Mapping from signal label to observed
            linewidth in ``ppm``.
        fit_offset: If ``True``, fit both ``p1`` and ``p2``. If ``False``,
            fit only ``p1`` and keep ``p2 = 0``.

    Returns:
        Structured estimate with fitted coefficients and
        back-calculated linewidths.

    Raises:
        ValueError: If the inputs are empty, inconsistent, or non-finite.
    """

    labels = tuple(observed_widths_by_label.keys())
    if not labels:
        raise ValueError("Linewidth-parameter estimation requires at least one signal")

    missing = [label for label in labels if label not in mean_inv_r6_by_label]
    if missing:
        raise ValueError(
            "Missing mean 1/r^6 values for signal label(s): "
            + ", ".join(sorted(missing))
        )

    x = np.asarray(
        [
            _validate_nonnegative_finite(
                f"mean_inv_r6_by_label[{label!r}]",
                mean_inv_r6_by_label[label],
            )
            for label in labels
        ],
        dtype=float,
    )
    y = np.asarray(
        [
            _validate_nonnegative_finite(
                f"observed_widths_by_label[{label!r}]",
                observed_widths_by_label[label],
            )
            for label in labels
        ],
        dtype=float,
    )

    design = x[:, None]
    estimate_mode = "p1"
    if fit_offset:
        design = np.column_stack((x, np.ones_like(x)))
        estimate_mode = "p1_p2"

    fit = lsq_linear(design, y, bounds=(0.0, np.inf))
    if not fit.success:
        raise ValueError(
            "Linewidth-parameter estimation failed: "
            + (fit.message if isinstance(fit.message, str) else str(fit.message))
        )

    coeffs = np.asarray(fit.x, dtype=float)
    if fit_offset:
        p1, p2 = float(coeffs[0]), float(coeffs[1])
    else:
        p1, p2 = float(coeffs[0]), 0.0

    predicted = design @ coeffs
    rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))

    predicted_widths_by_label = {
        label: float(width) for label, width in zip(labels, predicted)
    }
    return R6LinewidthParameterEstimate(
        linewidth_method="r6",
        estimate_mode=estimate_mode,
        p1=p1,
        p2=p2,
        rmse=rmse,
        mean_inv_r6_by_label={
            str(label): float(mean_inv_r6_by_label[label]) for label in labels
        },
        observed_widths_by_label={
            str(label): float(observed_widths_by_label[label]) for label in labels
        },
        predicted_widths_by_label=predicted_widths_by_label,
    )


def _validate_nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value
