# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Linewidth fitting and prediction helpers."""

from paranmr.core.fitting.linewidth.r6 import (
    mean_inv_r6_by_label,
    predict_r6_linewidths,
)

__all__ = [
    "mean_inv_r6_by_label",
    "predict_r6_linewidths",
]
