# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

from pathlib import Path

import pandas as pd
import pytest

from paranmr.core.fitting.susceptibility.jacobian.types import (
    MOMENT_JACOBIAN_PARAMETER_NAMES,
    MomentJacobianResult,
)
from paranmr.core.fitting.susceptibility.moments.descriptors import MOMENT_NAMES
from paranmr.io.csv.fit import save_moment_jacobian


@pytest.mark.unit
def test_moment_jacobian_result_validates_shape_and_contract():
    values = [
        [10.0 * row + col for col in range(len(MOMENT_JACOBIAN_PARAMETER_NAMES))]
        for row in range(len(MOMENT_NAMES))
    ]

    result = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_NAMES,
        parameter_names=MOMENT_JACOBIAN_PARAMETER_NAMES,
        values=values,
    )

    assert result.values.shape == (6, 8)


@pytest.mark.unit
def test_moment_jacobian_result_rejects_noncanonical_parameter_order():
    values = [[0.0] * 8 for _ in range(6)]

    with pytest.raises(ValueError, match="canonical parameter order"):
        MomentJacobianResult(
            temperature=302.15,
            moment_names=MOMENT_NAMES,
            parameter_names=(
                "p2",
                "p1",
                "chi_iso",
                "chi_ax",
                "chi_rh_over_ax",
                "alpha",
                "beta",
                "gamma",
            ),
            values=values,
        )


@pytest.mark.unit
def test_save_moment_jacobian_writes_expected_csv_layout(tmp_path: Path):
    values = [
        [10.0 * row + col for col in range(len(MOMENT_JACOBIAN_PARAMETER_NAMES))]
        for row in range(len(MOMENT_NAMES))
    ]
    result = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_NAMES,
        parameter_names=MOMENT_JACOBIAN_PARAMETER_NAMES,
        values=values,
    )

    output = tmp_path / "moment_jacobian_302.15_K.csv"
    save_moment_jacobian(result, str(output), verbose=False)

    df = pd.read_csv(output, comment="#")
    assert list(df.columns) == [
        "quantity",
        *MOMENT_JACOBIAN_PARAMETER_NAMES,
    ]
    assert list(df["quantity"]) == list(MOMENT_NAMES)
    assert df.iloc[0]["p1"] == pytest.approx(0.0)
    assert df.iloc[0]["gamma"] == pytest.approx(7.0)
    assert df.iloc[5]["p1"] == pytest.approx(50.0)
    assert df.iloc[5]["gamma"] == pytest.approx(57.0)
