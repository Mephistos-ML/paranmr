# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""R^-6 linewidth prediction helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from paranmr.core.domain.mol import Nucleus


def mean_inv_r6_by_label(
    nuclei: Sequence[Nucleus],
    paramagnetic_centre: ArrayLike,
    isotope_filter: str | None = None,
    label_kind: str = "signal_label",
) -> dict[str, float]:
    """Compute mean ``1/r^6`` values grouped by signal label.

    Args:
        nuclei: Nuclei to group.
        paramagnetic_centre: Cartesian coordinates of the paramagnetic centre
            in Angstrom.
        isotope_filter: Optional isotope selector, for example ``"1H"``.
        label_kind: Label used for grouping. Supported values are
            ``"signal_label"`` and ``"atom_label"``.

    Returns:
        Mapping from signal label to mean ``1/r^6`` in ``Angstrom^-6``.

    Raises:
        ValueError: If the paramagnetic centre is invalid, a nucleus lies at the
            paramagnetic centre, a pre-averaged ``r_inv6`` value is invalid, or
            no usable nuclei remain after filtering.
    """

    centre = np.asarray(paramagnetic_centre, dtype=float)
    if centre.shape != (3,):
        raise ValueError("paramagnetic_centre must be a length-3 coordinate")

    if label_kind not in {"signal_label", "atom_label"}:
        raise ValueError("label_kind must be 'signal_label' or 'atom_label'")

    grouped_inv_r6: dict[str, list[float]] = defaultdict(list)
    for nuc in nuclei:
        if isotope_filter is not None and nuc.isotope != isotope_filter:
            continue
        label = nuc.signal_label if label_kind == "signal_label" else nuc.label
        grouped_inv_r6[label].append(_nucleus_inv_r6(nuc=nuc, centre=centre))

    if not grouped_inv_r6:
        raise ValueError("No nuclei available for r^-6 linewidth prediction")

    return {
        signal_label: float(np.mean(values))
        for signal_label, values in grouped_inv_r6.items()
    }


def predict_r6_linewidths(
    mean_inv_r6: Mapping[str, float],
    p1: float,
    p2: float,
) -> dict[str, float]:
    """Predict linewidths from a two-parameter ``r^-6`` model.

    The model is ``linewidth = p1 * mean(1/r^6) + p2``. The returned linewidth
    unit follows the units of ``p1`` and ``p2``.

    Args:
        mean_inv_r6: Mapping from signal label to mean ``1/r^6`` values in
            ``Angstrom^-6``.
        p1: Distance-dependent linewidth coefficient.
        p2: Distance-independent linewidth offset.

    Returns:
        Mapping from signal label to predicted linewidth.

    Raises:
        ValueError: If model parameters or ``mean_inv_r6`` values are invalid.
    """

    p1_value = _validate_nonnegative_finite("p1", p1)
    p2_value = _validate_nonnegative_finite("p2", p2)

    linewidths: dict[str, float] = {}
    for signal_label, inv_r6 in mean_inv_r6.items():
        inv_r6_value = _validate_nonnegative_finite(
            f"mean_inv_r6[{signal_label!r}]",
            inv_r6,
        )
        linewidths[str(signal_label)] = p1_value * inv_r6_value + p2_value
    return linewidths


def _nucleus_inv_r6(nuc: Nucleus, centre: np.ndarray) -> float:
    precomputed = getattr(nuc.A, "r_inv6", None)
    if precomputed is not None:
        return _validate_positive_finite(
            f"r_inv6 for nucleus {nuc.label!r}",
            precomputed,
        )

    distance = float(np.linalg.norm(np.asarray(nuc.coord, dtype=float) - centre))
    if distance <= 1e-12:
        raise ValueError(f"Nucleus {nuc.label!r} is at the paramagnetic centre")
    return 1.0 / distance**6


def _validate_nonnegative_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value
