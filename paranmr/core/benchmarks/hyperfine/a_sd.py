# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Summaries for A_sd benchmark data."""

from __future__ import annotations

import numpy as np

from paranmr.core.benchmarks.hyperfine.summary import (
    summarize_hyperfine_metric_max_by_nucleus,
    summarize_hyperfine_metric_ranges_by_functional_and_nucleus,
)
from paranmr.core.domain.mol import Molecule


def summarize_a_sd_ranges_by_functional_and_nucleus(
    functional_sources: dict[str, list[tuple[str, Molecule]]],
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Summarize axial A_sd values by functional, nucleus, and chemical label."""
    return summarize_hyperfine_metric_ranges_by_functional_and_nucleus(
        functional_sources,
        metric_key="a_sd",
        metric_getter=_get_a_sd_ax,
    )


def summarize_a_sd_max_by_nucleus(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    *,
    max_label_tolerance: float = 0.0,
) -> dict[str, list[dict[str, object]]]:
    """Summarize maximum axial A_sd values by nucleus across functionals."""
    return summarize_hyperfine_metric_max_by_nucleus(
        summary,
        max_label_tolerance=max_label_tolerance,
    )


def _get_a_sd_ax(nucleus) -> float:
    """Return axial A_sd for one nucleus."""
    sd = nucleus.A.sd
    return float(2.0 / 3.0 * (sd[2, 2] - np.trace(sd) / 3.0))
