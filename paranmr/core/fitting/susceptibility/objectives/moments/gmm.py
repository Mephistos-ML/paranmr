# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Placeholder for the generalized method of moments objective."""

from __future__ import annotations

from typing import NoReturn


def raise_gmm_not_implemented() -> NoReturn:
    """Raise the public placeholder error for the GMM moment objective."""
    raise NotImplementedError(
        "Moment objective 'gmm' is not implemented yet. "
        "Use assignment:moment_objective:type 'ls' instead."
    )
