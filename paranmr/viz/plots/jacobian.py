# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot normalized moment-Jacobian heat maps."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from paranmr.core.fitting.susceptibility.jacobian.types import (
    MomentJacobianResult,
)
from paranmr.viz.layout.canvas import create_canvas
from paranmr.viz.layout.export import render_figure
from paranmr.viz.style.theme import PlotSpec
from paranmr.viz.utils.labels import (
    moment_label_mathtext,
    parameter_label_mathtext,
)

logger = logging.getLogger(__name__)


def plot_moment_jacobian_heatmap(
    jacobian: MomentJacobianResult,
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "moment_jacobian_heatmap.pdf",
    window_title: str = "Moment Jacobian Heat Map",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a signed log-scaled heat map of the normalized moment Jacobian."""

    fig, ax = create_canvas(
        spec.profile,
        variant="horizontal",
        window_title=window_title,
        layout="constrained",
    )
    spec.skin_axes(ax)
    values = np.asarray(jacobian.values, dtype=float)
    display_values = np.sign(values) * np.log10(1.0 + np.abs(values))
    vmax = float(np.max(np.abs(display_values)))
    if vmax == 0.0:
        vmax = 1.0

    image = ax.imshow(
        display_values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(jacobian.parameter_names)))
    ax.set_xticklabels(
        [parameter_label_mathtext(label) for label in jacobian.parameter_names],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(jacobian.moment_names)))
    ax.set_yticklabels(
        [moment_label_mathtext(label) for label in jacobian.moment_names]
    )
    ax.set_xlabel("Active fitted parameter")
    ax.set_ylabel("Normalized moment")
    ax.set_title("Normalized Jacobian Matrix at Convergence")

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"$\mathrm{sign}(J)\ \log_{10}(1 + |J|)$")

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if verbose and save:
        logger.info("Moment Jacobian heat map written to %s", save_name)

    return fig, ax
