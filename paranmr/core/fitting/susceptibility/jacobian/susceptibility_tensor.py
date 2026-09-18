# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Analytical Cartesian susceptibility-tensor derivatives for GMM."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_SPLIT_TENSOR_DERIVATIVES = {
    "iso": np.eye(3, dtype=float),
    "dxx": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    "dyy": np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
    "dxy": np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "dxz": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    "dyz": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
}


def differentiate_tensor_by_split_parameter(name: str) -> NDArray[np.float64]:
    """Return the constant ``dχ/dparameter`` basis tensor for ``SplitFitter``."""

    try:
        return _SPLIT_TENSOR_DERIVATIVES[name].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown split susceptibility parameter {name!r}") from exc
