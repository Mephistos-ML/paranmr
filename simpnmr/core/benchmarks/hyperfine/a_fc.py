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


def _get_a_fc(nucleus) -> float:
    """Return isotropic A_fc for one nucleus."""
    return float(np.trace(nucleus.A.fc) / 3.0)
