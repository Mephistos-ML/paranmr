# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot moment-covariance heat maps for GMM fitting."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from paranmr.core.fitting.susceptibility.objectives.moments.gmm.covariance import (
    MomentCovarianceEstimate,
)
from paranmr.viz.layout.canvas import create_canvas
from paranmr.viz.layout.export import render_figure
from paranmr.viz.style.theme import PlotSpec
from paranmr.viz.utils.labels import moment_label_mathtext

logger = logging.getLogger(__name__)


def plot_moment_covariance_heatmap(
    covariance: MomentCovarianceEstimate,
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "moment_covariance_heatmap.pdf",
    window_title: str = "Moment Covariance Heat Map",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a signed log-scaled heat map of the experimental moment covariance."""

    fig, ax = create_canvas(
        spec.profile,
        variant="horizontal",
        window_title=window_title,
        layout="constrained",
    )
    spec.skin_axes(ax)

    values = np.asarray(covariance.covariance, dtype=float)
    display_values = np.sign(values) * np.log10(1.0 + np.abs(values))
    vmax = float(np.max(np.abs(display_values)))
    if vmax == 0.0:
        vmax = 1.0

    image = ax.imshow(
        display_values,
        cmap="RdBu_r",
        aspect="equal",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    tick_labels = [moment_label_mathtext(label) for label in covariance.moment_names]
    ax.set_xticks(np.arange(len(covariance.moment_names)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(covariance.moment_names)))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Experimental moment")
    ax.set_ylabel("Experimental moment")
    ax.set_title("Moment Covariance Matrix")

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"$\mathrm{sign}(C)\ \log_{10}(1 + |C|)$")

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if verbose and save:
        logger.info("Moment covariance heat map written to %s", save_name)

    return fig, ax
