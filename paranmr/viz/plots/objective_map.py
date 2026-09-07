# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot two-parameter objective maps for susceptibility fitting."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from paranmr.core.fitting.susceptibility.objective_map import ObjectiveMapResult
from paranmr.viz.layout.canvas import create_canvas
from paranmr.viz.layout.export import render_figure
from paranmr.viz.style.theme import PlotSpec
from paranmr.viz.utils.labels import parameter_label_mathtext

logger = logging.getLogger(__name__)


def plot_objective_map(
    objective_map: ObjectiveMapResult,
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "objective_map.pdf",
    window_title: str = "Objective Map",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a contour/heat map of the objective score surface."""

    fig, ax = create_canvas(
        spec.profile,
        variant="horizontal",
        window_title=window_title,
        layout="constrained",
    )
    spec.skin_axes(ax)

    x_values = np.asarray(objective_map.x_values, dtype=float)
    y_values = np.asarray(objective_map.y_values, dtype=float)
    score_grid = np.asarray(objective_map.score_grid, dtype=float)
    score_scale_exponent = _score_scale_exponent(score_grid)
    score_scale = 10.0**score_scale_exponent
    display_score_grid = score_grid / score_scale
    gradient_x = (
        None
        if objective_map.gradient_x is None
        else np.asarray(objective_map.gradient_x, dtype=float)
    )
    gradient_y = (
        None
        if objective_map.gradient_y is None
        else np.asarray(objective_map.gradient_y, dtype=float)
    )

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    contour_fill = ax.contourf(
        x_grid,
        y_grid,
        display_score_grid,
        levels=30,
        cmap="viridis",
    )
    ax.contour(
        x_grid,
        y_grid,
        display_score_grid,
        levels=12,
        colors="white",
        linewidths=0.6,
        alpha=0.75,
    )

    if gradient_x is not None and gradient_y is not None and not np.allclose(
        gradient_x, 0.0
    ):
        step = max(1, len(x_values) // 15)
        ax.quiver(
            x_grid[::step, ::step],
            y_grid[::step, ::step],
            -gradient_x[::step, ::step],
            -gradient_y[::step, ::step],
            color="black",
            alpha=0.65,
            linewidth=0.4,
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.003,
        )

    ax.plot(
        objective_map.center_values[0],
        objective_map.center_values[1],
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
    )

    ax.set_xlabel(parameter_label_mathtext(objective_map.parameter_names[0]))
    ax.set_ylabel(parameter_label_mathtext(objective_map.parameter_names[1]))
    ax.set_title("Objective Map at Convergence")

    colorbar = fig.colorbar(contour_fill, ax=ax, pad=0.02)
    colorbar.set_label(_score_label(score_scale_exponent))

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if verbose and save:
        logger.info("Objective map written to %s", save_name)

    return fig, ax


def _score_scale_exponent(score_grid: np.ndarray) -> int:
    max_abs = float(np.max(np.abs(score_grid)))
    if max_abs == 0.0:
        return 0
    return int(np.floor(np.log10(max_abs)))


def _score_label(exponent: int) -> str:
    if exponent == 0:
        return "Score"
    return rf"Score $\times 10^{{{exponent}}}$"
