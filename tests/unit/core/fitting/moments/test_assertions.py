# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import pytest

from tests.integration.experimental.susceptibility_fit.assertions import (
    range_based_ppm_tolerance,
)


@pytest.mark.unit
def test_range_based_ppm_tolerance_scales_with_reference_range() -> None:
    centers = {frozenset({"H1"}): -10.0, frozenset({"H2"}): 30.0}

    assert range_based_ppm_tolerance(centers, fraction=0.05) == pytest.approx(2.0)
