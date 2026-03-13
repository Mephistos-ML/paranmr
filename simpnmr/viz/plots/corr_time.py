# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Correlation-time fit diagnostic plots."""

from __future__ import annotations

import logging

import numpy as np
from matplotlib import pyplot as plt

from simpnmr.viz.layout.export import render_figure
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
    spec: PlotSpec | None = None,
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
    glyphs = spec.glyphs if spec is not None else None
    palette = spec.palette if spec is not None else None

    fig, ax = plt.subplots(figsize=(6, 6))
    if spec is not None:
        fig.patch.set_facecolor(palette.annotation_bg)
        ax.set_facecolor(palette.annotation_bg)
        scale = spec.skin_axes(ax)
        ax.grid(True, which="major", color=palette.grid, linewidth=1.0)
        ax.grid(True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        marker_size = glyphs.ms
        annotation_size = scale.annotation
        axis_label_size = scale.axis_label
        title_size = scale.title
    else:
        marker_size = 6.0
        annotation_size = 12.0
        axis_label_size = 14.0
        title_size = 16.0

    scatter_color = palette.primary if palette is not None else "blue"
    reference_color = palette.reference if palette is not None else "black"

    ax.scatter(
        theory_r1,
        exp_r1,
        marker="x",
        color=scatter_color,
        s=(marker_size**2),
    )

    for x_val, y_val, label in zip(theory_r1, exp_r1, chem_labels):
        ax.text(x_val, y_val, label, fontsize=annotation_size)

    min_val = float(min(np.min(theory_r1), np.min(exp_r1)))
    max_val = float(max(np.max(theory_r1), np.max(exp_r1)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color=reference_color,
        lw=1.0,
        label="x = y",
    )

    ax.set_xlabel("Fitted $R_1$ (s$^{-1}$)", fontsize=axis_label_size)
    ax.set_ylabel("Experimental $R_1$ (s$^{-1}$)", fontsize=axis_label_size)
    ax.set_title("Experimental vs Fitted $R_1$", fontsize=title_size)
    ax.text(
        0.01,
        0.96,
        f"$r^2$ = {rsquared:.3f}",
        fontsize=annotation_size,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    if fix_param == "tau_r" and tau_E_fit is not None:
        ax.text(
            0.01,
            0.91,
            f"Fitted $\\tau_{{\\mathrm{{E}}}}$: {tau_E_fit:.3e} s",
            fontsize=annotation_size,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
    elif fix_param == "tau_e" and tau_R_fit is not None:
        ax.text(
            0.01,
            0.91,
            f"Fitted $\\tau_{{\\mathrm{{R}}}}$: {tau_R_fit:.3e} s",
            fontsize=annotation_size,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
    elif fix_param in {None, "", "none"}:
        fit_lines: list[str] = []
        if tau_R_fit is not None:
            fit_lines.append(f"Fitted $\\tau_{{\\mathrm{{R}}}}$: {tau_R_fit:.3e} s")
        if tau_E_fit is not None:
            fit_lines.append(f"Fitted $\\tau_{{\\mathrm{{E}}}}$: {tau_E_fit:.3e} s")
        if fit_lines:
            ax.text(
                0.01,
                0.91,
                "\n".join(fit_lines),
                fontsize=annotation_size,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )

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
    spec: PlotSpec | None = None,
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
    glyphs = spec.glyphs if spec is not None else None
    palette = spec.palette if spec is not None else None

    fig, ax = plt.subplots(figsize=(8, 5))
    if spec is not None:
        fig.patch.set_facecolor(palette.annotation_bg)
        ax.set_facecolor(palette.annotation_bg)
        scale = spec.skin_axes(ax)
        ax.grid(True, which="major", color=palette.grid, linewidth=1.0)
        ax.grid(True, which="minor", color=palette.grid, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        marker_size = glyphs.ms
        axis_label_size = scale.axis_label
        title_size = scale.title
    else:
        marker_size = 6.0
        axis_label_size = 14.0
        title_size = 16.0

    series_theory_color = palette.highlight if palette is not None else "red"

    ax.plot(
        chem_labels,
        exp_r1,
        "o",
        label="Experimental $R_1$",
        markersize=marker_size,
    )
    ax.plot(
        chem_labels,
        theory_r1,
        "s",
        label="Fitted Theory $R_1$",
        markersize=marker_size,
    )
    ax.plot(
        chem_labels,
        theory_r1,
        "x",
        color=series_theory_color,
        label="Theory X",
        markersize=marker_size,
    )

    ax.set_xlabel("Chemical Label", fontsize=axis_label_size)
    ax.set_ylabel("$R_1$ (s$^{-1}$)", fontsize=axis_label_size)
    ax.set_title("Experimental vs Fitted $R_1$", fontsize=title_size)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=45)

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )
    if save and verbose:
        logger.info("Saved correlation-time per-label plot to %s", save_name)
    return fig, ax
