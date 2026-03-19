# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Correlation-time fit diagnostic plots."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import to_rgba

from simpnmr.viz.layout.canvas import create_canvas, create_header_plot_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.label import resolve_label_layout
from simpnmr.viz.layout.table import render_compact_table
from simpnmr.viz.style.theme import PlotSpec

logger = logging.getLogger(__name__)


def plot_corr_time_scatter(
    *,
    theory_r1: np.ndarray,
    exp_r1: np.ndarray,
    chem_labels: list[str],
    rsquared: float,
    fix_param: str | None = None,
    tau_R_fit: float | None = None,
    tau_E_fit: float | None = None,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "experimental_vs_fitted_R1.pdf",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot experimental versus fitted ``R1`` values.

    This helper renders a scatter plot of fitted theoretical ``R1`` values
    against experimental ``R1`` values and overlays an ``x = y`` reference
    line. Optional fit diagnostics are shown in the plot annotation area.

    Args:
        theory_r1: Fitted theoretical ``R1`` values.
        exp_r1: Experimental ``R1`` values.
        chem_labels: Chemical labels corresponding to the plotted points.
        rsquared: Coefficient of determination for the fit.
        fix_param: Optional fit mode. Supported values are ``"tau_r"``,
            ``"tau_e"``, ``"none"``, ``""``, or ``None``.
        tau_R_fit: Fitted rotational correlation time.
        tau_E_fit: Fitted electronic correlation time.
        spec: Optional plot specification used for styling.
        save: Whether to save the figure.
        show: Whether to display the figure interactively.
        save_name: Output filename for the figure.
        verbose: Whether to emit an info log when the figure is saved.

    Returns:
        Tuple of ``(figure, axes)`` for the rendered scatter plot.
    """

    glyphs = spec.glyphs
    palette = spec.palette

    fig, summary_ax, ax = create_header_plot_canvas(
        spec.profile,
        variant="vertical",
        layout="constrained",
        header_ratio=0.55,
        plot_ratio=3.05,
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    scale = spec.skin_axes(ax)
    ax.minorticks_on()
    ax.grid(True, which="major", color=palette.grid, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    marker_size = glyphs.ms
    annotation_size = scale.annotation
    axis_label_size = scale.axis_label
    fit_lines = [f"R²: {rsquared:.4f}"]

    model_lines = ["—"]
    if fix_param == "tau_r" and tau_E_fit is not None:
        model_lines = [f"τE: {tau_E_fit:.3e} s"]
    elif fix_param == "tau_e" and tau_R_fit is not None:
        model_lines = [f"τR: {tau_R_fit:.3e} s"]
    elif fix_param in {None, "", "none"}:
        tau_lines: list[str] = []
        if tau_R_fit is not None:
            tau_lines.append(f"τR: {tau_R_fit:.3e} s")
        if tau_E_fit is not None:
            tau_lines.append(f"τE: {tau_E_fit:.3e} s")
        if tau_lines:
            model_lines = tau_lines

    blocks = [
        ("Fit Stats", fit_lines),
        ("Corr. Time", model_lines),
    ]

    render_compact_table(
        summary_ax,
        blocks,
        spec,
        bbox=[0.0, 0.0, 1.0, 1.0],
        cell_align="center",
        remove_outer_frame=True,
    )

    scatter_color = palette.primary

    ax.scatter(
        theory_r1,
        exp_r1,
        marker="o",
        facecolors=to_rgba(scatter_color, alpha=0.55),
        edgecolors=scatter_color,
        linewidths=0.8,
        s=(marker_size**2),
    )

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    shared_min = min(x_limits[0], y_limits[0])
    shared_max = max(x_limits[1], y_limits[1])
    ax.set_xlim(shared_min, shared_max)
    ax.set_ylim(shared_min, shared_max)
    ax.set_aspect("equal", adjustable="box")

    diag_line = ax.plot(
        [shared_min, shared_max],
        [shared_min, shared_max],
        linestyle="-",
        color="black",
        lw=1.0,
    )[0]

    label_entries = [
        (label, float(theory_val), float(exp_val))
        for label, theory_val, exp_val in zip(chem_labels, theory_r1, exp_r1)
    ]
    resolve_label_layout(
        ax,
        label_entries,
        fontsize=annotation_size,
        marker_size=marker_size,
        diag_line=diag_line,
    )

    ax.set_xlabel("Theoretical $R_1$ (s$^{-1}$)", fontsize=axis_label_size)
    ax.set_ylabel("Experimental $R_1$ (s$^{-1}$)", fontsize=axis_label_size)

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if save and verbose:
        logger.info("Saved correlation-time scatter plot to %s", save_name)
    return fig, ax


def plot_corr_time_by_label(
    *,
    theory_r1: np.ndarray,
    exp_r1: np.ndarray,
    chem_labels: list[str],
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "r1_fit_comparison.pdf",
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot experimental and fitted ``R1`` values by chemical label.

    This helper renders a categorical comparison plot that shows experimental
    and fitted theoretical ``R1`` values for each chemical label.

    Args:
        theory_r1: Fitted theoretical ``R1`` values.
        exp_r1: Experimental ``R1`` values.
        chem_labels: Chemical labels corresponding to the plotted points.
        spec: Optional plot specification used for styling.
        save: Whether to save the figure.
        show: Whether to display the figure interactively.
        save_name: Output filename for the figure.
        verbose: Whether to emit an info log when the figure is saved.

    Returns:
        Tuple of ``(figure, axes)`` for the rendered per-label comparison plot.
    """

    glyphs = spec.glyphs
    palette = spec.palette

    fig, ax = create_canvas(
        spec.profile,
        variant="standard",
        layout="constrained",
    )
    fig.patch.set_facecolor(palette.annotation_bg)
    ax.set_facecolor(palette.annotation_bg)
    scale = spec.skin_axes(ax)
    ax.grid(axis="x", ls="--", which="minor", color=palette.grid)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    marker_size = glyphs.ms
    axis_label_size = scale.axis_label
    title_size = scale.title

    xvals = np.arange(len(chem_labels), dtype=float)
    xpos = xvals + 0.5

    ax.plot(
        xpos,
        exp_r1,
        linestyle="none",
        marker="o",
        label="Experimental $R_1$",
        markersize=marker_size,
        markerfacecolor=to_rgba(palette.highlight, alpha=0.55),
        markeredgecolor=palette.highlight,
        markeredgewidth=0.8,
    )

    ax.plot(
        xpos,
        theory_r1,
        linestyle="none",
        marker="s",
        label="Fitted Theory $R_1$",
        markersize=marker_size,
        markerfacecolor=to_rgba(palette.auxiliary, alpha=0.55),
        markeredgecolor=palette.auxiliary,
        markeredgewidth=0.8,
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels(chem_labels, rotation=45)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_tick_params("major", length=0)
    ax.xaxis.set_tick_params("minor", length=0)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Chemical Label", fontsize=axis_label_size)
    ax.set_ylabel("$R_1$ (s$^{-1}$)", fontsize=axis_label_size)
    ax.set_title("Experimental vs Theoretical $R_1$ (s$^{-1}$)", fontsize=title_size)
    ax.legend(loc="best")

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if save and verbose:
        logger.info("Saved correlation-time per-label plot to %s", save_name)
    return fig, ax
