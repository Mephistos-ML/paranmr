# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

from pathlib import Path

import pandas as pd
import pytest

from paranmr.core.fitting.susceptibility.jacobian.types import (
    MomentJacobianResult,
)
from paranmr.io.csv.fit import save_moment_jacobian
from paranmr.viz.plots.jacobian import plot_moment_jacobian_heatmap
from paranmr.viz.style.theme import build_spec

MOMENT_LABELS = tuple(f"m{order}" for order in range(1, 7))


@pytest.mark.unit
def test_moment_jacobian_result_validates_shape_and_contract():
    parameter_names = ("ax", "rho_over_ax", "alpha")
    values = [
        [10.0 * row + col for col in range(len(parameter_names))]
        for row in range(len(MOMENT_LABELS))
    ]

    result = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_LABELS,
        parameter_names=parameter_names,
        values=values,
    )

    assert result.values.shape == (6, 3)


@pytest.mark.unit
def test_moment_jacobian_result_rejects_duplicate_parameter_names():
    values = [[0.0] * 3 for _ in range(6)]

    with pytest.raises(ValueError, match="column labels must be unique"):
        MomentJacobianResult(
            temperature=302.15,
            moment_names=MOMENT_LABELS,
            parameter_names=("ax", "ax", "alpha"),
            values=values,
        )


@pytest.mark.unit
def test_save_moment_jacobian_writes_expected_csv_layout(tmp_path: Path):
    parameter_names = ("ax", "rho_over_ax", "alpha")
    values = [
        [10.0 * row + col for col in range(len(parameter_names))]
        for row in range(len(MOMENT_LABELS))
    ]
    result = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_LABELS,
        parameter_names=parameter_names,
        values=values,
    )

    output = tmp_path / "moment_jacobian_302.15_K.csv"
    save_moment_jacobian(result, str(output), verbose=False)

    df = pd.read_csv(output, comment="#")
    assert list(df.columns) == [
        "quantity",
        *parameter_names,
    ]
    assert list(df["quantity"]) == list(MOMENT_LABELS)
    assert df.iloc[0]["ax"] == pytest.approx(0.0)
    assert df.iloc[0]["alpha"] == pytest.approx(2.0)
    assert df.iloc[5]["ax"] == pytest.approx(50.0)
    assert df.iloc[5]["alpha"] == pytest.approx(52.0)


@pytest.mark.unit
def test_plot_moment_jacobian_heatmap_writes_pdf(tmp_path: Path):
    parameter_names = ("ax", "rho_over_ax", "alpha")
    values = [
        [10.0 * row + col for col in range(len(parameter_names))]
        for row in range(len(MOMENT_LABELS))
    ]
    result = MomentJacobianResult(
        temperature=302.15,
        moment_names=MOMENT_LABELS,
        parameter_names=parameter_names,
        values=values,
    )

    output = tmp_path / "moment_jacobian_heatmap_302.15_K"
    spec = build_spec("paper")
    with spec.context():
        plot_moment_jacobian_heatmap(
            result,
            spec=spec,
            save=True,
            show=False,
            save_name=str(output),
            verbose=False,
        )

    assert Path(f"{output}.pdf").exists()
