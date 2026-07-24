# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

from pathlib import Path

import pytest

from paranmr.core.fitting.susceptibility.objective_map import ObjectiveMapResult
from paranmr.viz.plots.objective_map import plot_objective_map
from paranmr.viz.style.theme import build_spec


@pytest.mark.unit
def test_plot_moment_objective_map_writes_pdf(tmp_path: Path):
    result = ObjectiveMapResult(
        temperature=302.15,
        objective_type="gmm",
        parameter_names=("ax", "rho_over_ax"),
        center_values=(0.1, 0.05),
        x_values=[0.0, 0.5, 1.0],
        y_values=[0.0, 0.5, 1.0],
        score_grid=[[3.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0]],
        gradient_x=[[0.0, 0.0, 0.0]] * 3,
        gradient_y=[[0.0, 0.0, 0.0]] * 3,
    )

    output = tmp_path / "objective_map_ax_rho_over_ax_302.15_K"
    spec = build_spec("paper")
    with spec.context():
        plot_objective_map(
            result,
            spec=spec,
            save=True,
            show=False,
            save_name=str(output),
            verbose=False,
        )

    assert Path(f"{output}.pdf").exists()
