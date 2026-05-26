# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot benchmark summaries."""

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from simpnmr.viz.layout.canvas import create_canvas
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.layout.violin import set_violin_colours
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.fmt import isotope_format


def plot_a_fc_spread(
    functional: str,
    nucleus_label: str,
    chem_label_summary: dict[str, dict[str, object]],
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "a_fc_spread.pdf",
    window_title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot A_fc distributions by chemical label for one functional.

    Args:
        functional: Functional name used as the plot title.
        nucleus_label: Nucleus label plotted in this figure.
        chem_label_summary: Mapping from chemical label to summary entries with
            source-resolved ``values`` and precomputed ``mean`` values.
        spec: Resolved plotting style.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Matplotlib window title.

    Returns:
        A tuple ``(fig, ax)``.
    """
    return _plot_hyperfine_metric_spread(
        functional=functional,
        nucleus_label=nucleus_label,
        chem_label_summary=chem_label_summary,
        spec=spec,
        value_key="a_fc",
        y_label=r"$A_\mathregular{FC}$ (ppm Å$^\mathregular{-3}$)",
        window_title=window_title or f"A_fc benchmark: {functional} {nucleus_label}",
        save=save,
        show=show,
        save_name=save_name,
    )


def plot_a_fc_functional_max_curve(
    nucleus_label: str,
    max_rows: list[dict[str, object]],
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "a_fc_functional_max.pdf",
    window_title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot sorted maximum A_fc values across functionals for one nucleus.

    Args:
        nucleus_label: Nucleus label plotted in this figure.
        max_rows: Rows containing ``functional`` and ``max`` values, sorted in
            the desired plotting order.
        spec: Resolved plotting style.
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Matplotlib window title.

    Returns:
        A tuple ``(fig, ax)``.
    """
    return _plot_hyperfine_metric_functional_max_curve(
        nucleus_label=nucleus_label,
        max_rows=max_rows,
        spec=spec,
        y_label=r"max $A_\mathregular{FC}$ (ppm Å$^\mathregular{-3}$)",
        title_metric=r"$A_\mathregular{FC}$",
        window_title=window_title or f"A_fc max benchmark: {nucleus_label}",
        save=save,
        show=show,
        save_name=save_name,
    )


def plot_a_sd_spread(
    functional: str,
    nucleus_label: str,
    chem_label_summary: dict[str, dict[str, object]],
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "a_sd_spread.pdf",
    window_title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot axial A_sd distributions by chemical label for one functional."""
    return _plot_hyperfine_metric_spread(
        functional=functional,
        nucleus_label=nucleus_label,
        chem_label_summary=chem_label_summary,
        spec=spec,
        value_key="a_sd",
        y_label=r"$A_{\mathregular{SD}}^{\mathregular{ax}}$ (ppm Å$^\mathregular{-3}$)",
        window_title=window_title or f"A_sd benchmark: {functional} {nucleus_label}",
        save=save,
        show=show,
        save_name=save_name,
    )


def plot_a_sd_functional_max_curve(
    nucleus_label: str,
    max_rows: list[dict[str, object]],
    *,
    spec: PlotSpec,
    save: bool = True,
    show: bool = True,
    save_name: str = "a_sd_functional_max.pdf",
    window_title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot sorted maximum axial A_sd values across functionals for one nucleus."""
    return _plot_hyperfine_metric_functional_max_curve(
        nucleus_label=nucleus_label,
        max_rows=max_rows,
        spec=spec,
        y_label=(
            r"max $A_{\mathregular{SD}}^{\mathregular{ax}}$ "
            r"(ppm Å$^\mathregular{-3}$)"
        ),
        title_metric=r"$A_{\mathregular{SD}}^{\mathregular{ax}}$",
        window_title=window_title or f"A_sd max benchmark: {nucleus_label}",
        save=save,
        show=show,
        save_name=save_name,
    )


def _plot_hyperfine_metric_spread(
    functional: str,
    nucleus_label: str,
    chem_label_summary: dict[str, dict[str, object]],
    *,
    spec: PlotSpec,
    value_key: str,
    y_label: str,
    save: bool,
    show: bool,
    save_name: str,
    window_title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot hyperfine metric distributions by chemical label."""
    formatted_nucleus = _format_benchmark_nucleus(nucleus_label)
    fig, ax = create_canvas(
        spec.profile,
        variant="horizontal",
        window_title=window_title,
        layout="constrained",
    )

    glyphs = spec.glyphs
    scale = spec.skin_axes(ax)
    palette = spec.palette

    ordered_labels = list(chem_label_summary)
    xvals = np.arange(1, len(ordered_labels) + 1)
    x_positions = xvals + 0.5

    datasets = [
        [float(entry[value_key]) for entry in chem_label_summary[label]["values"]]
        for label in ordered_labels
    ]
    mean_values = [
        float(chem_label_summary[label]["mean"]) for label in ordered_labels
    ]

    violin_colour = palette.secondary if palette is not None else "C1"
    line_colour = palette.primary if palette is not None else "k"

    violin = ax.violinplot(
        dataset=datasets,
        positions=x_positions,
        widths=0.65,
        vert=True,
        showmeans=True,
    )
    set_violin_colours(violin, violin_colour)

    ax.plot(
        x_positions,
        mean_values,
        color=line_colour,
        lw=0,
        marker="o",
        markerfacecolor="none",
        markeredgecolor=line_colour,
        markersize=(glyphs.ms if glyphs is not None else 7),
        label="Mean",
    )

    ax.hlines(
        0.0,
        0.5,
        len(ordered_labels) + 1.5,
        color=line_colour,
        lw=(glyphs.line_lw if glyphs is not None else 0.5),
    )

    ax.grid(axis="x", ls="--", which="minor")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            str(chem_label_summary[label].get("chem_math_label", label))
            for label in ordered_labels
        ],
        rotation=90,
    )
    ax.tick_params(axis="x", labelsize=scale.axis_label)
    ax.grid(axis="x", ls="--", which="minor")
    ax.set_xlim(0.5, len(ordered_labels) + 1.5)

    ax.set_ylabel(y_label)
    ax.set_title(f"{functional} {formatted_nucleus}")
    ax.legend(
        handles=[
            lines.Line2D(
                [0],
                [0],
                color=line_colour,
                lw=0,
                marker="o",
                markerfacecolor="none",
                markeredgecolor=line_colour,
                label="Mean",
            ),
        ],
        loc="best",
    )

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    return fig, ax


def _plot_hyperfine_metric_functional_max_curve(
    nucleus_label: str,
    max_rows: list[dict[str, object]],
    *,
    spec: PlotSpec,
    y_label: str,
    title_metric: str,
    save: bool,
    show: bool,
    save_name: str,
    window_title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot sorted maximum hyperfine metric values across functionals."""
    formatted_nucleus = _format_benchmark_nucleus(nucleus_label)
    variant = "horizontal_extended" if len(max_rows) > 10 else "horizontal"
    fig, ax = create_canvas(
        spec.profile,
        variant=variant,
        window_title=window_title,
        layout="constrained",
    )

    glyphs = spec.glyphs
    scale = spec.skin_axes(ax)
    palette = spec.palette

    functionals = [str(row["functional"]) for row in max_rows]
    max_values = [float(row["max"]) for row in max_rows]
    xvals = np.arange(1, len(functionals) + 1)
    x_positions = xvals + 0.5

    line_colour = palette.primary if palette is not None else "k"

    ax.plot(
        x_positions,
        max_values,
        color=line_colour,
        lw=(glyphs.line_lw if glyphs is not None else 1.0),
        marker="o",
        markerfacecolor="none",
        markeredgecolor=line_colour,
        markersize=(glyphs.ms if glyphs is not None else 7),
    )

    ax.hlines(
        0.0,
        0.5,
        len(functionals) + 1.5,
        color=line_colour,
        lw=(glyphs.line_lw if glyphs is not None else 0.5),
    )

    ax.grid(axis="x", ls="--", which="minor")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_xticks(x_positions)
    ax.set_xticklabels(functionals, rotation=90)
    ax.tick_params(axis="x", labelsize=scale.axis_label)
    ax.set_xlim(0.5, len(functionals) + 1.5)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.1 * (ymax - ymin))

    ax.set_ylabel(y_label)
    ax.set_title(f"{formatted_nucleus} max {title_metric}")

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    return fig, ax


def _format_benchmark_nucleus(nucleus_label: str) -> str:
    """Format benchmark nucleus labels for plot display."""
    isotope_by_label = {
        "H": "1H",
        "C": "13C",
    }
    isotope = isotope_by_label.get(nucleus_label)
    if isotope is None:
        return nucleus_label
    return isotope_format(isotope)
