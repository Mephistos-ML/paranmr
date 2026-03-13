# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Plot fitted theoretical-versus-experimental chemical shifts.

Provides the fitted-shift scatter plot together with its label and summary-table
helpers.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.constants as constants

from simpnmr.core.const import ptable
from simpnmr.core.domain.exp import Experiment
from simpnmr.core.domain.mol import Molecule
from simpnmr.core.fitting import models
from simpnmr.viz.layout.export import render_figure
from simpnmr.viz.style.theme import PlotSpec
from simpnmr.viz.utils.label_layout import resolve_label_layout

logger = logging.getLogger(__name__)


def _render_summary_table(
    header_ax: plt.Axes,
    molecule: Molecule,
    susc_model: models.SusceptibilityModel,
    *,
    scale,
    susc_units: str,
) -> None:
    """Build and draw the fitted-shift summary header table.

    Args:
        header_ax: Header axes used only for the summary table.
        molecule: Molecule containing the fitted susceptibility.
        susc_model: Fitted susceptibility model.
        scale: Plot scale bundle returned by the theme skinning helper.
        susc_units: Units for reporting susceptibility values in the header.
            Supported: ``"A3"``, ``"A3 mol-1"``, ``"cm3"``, ``"cm3 mol-1"``.
    """
    if susc_units == "A3":
        conv = 1.0
        model_unit_label = "Å³"
    elif susc_units == "A3 mol-1":
        conv = constants.Avogadro
        model_unit_label = "Å³ mol⁻¹"
    elif susc_units == "cm3":
        conv = 1e-24
        model_unit_label = "cm³"
    elif susc_units == "cm3 mol-1":
        conv = 1e-24 * constants.Avogadro / (4 * np.pi)
        model_unit_label = "cm³ mol⁻¹"

    fit_lines = [
        f"R²adj  {susc_model.adj_r2:.4f}",
        f"MAE    {susc_model.mae:.1f} ppm",
        f"RMSE   {susc_model.rmse:.1f} ppm",
    ]

    model_lines = []
    for name in susc_model.VARNAMES:
        val = float(susc_model.final_var_values[name]) * conv
        label = susc_model.VARNAMES_MM[name]

        err = susc_model.fit_stdev.get(name)
        if name in susc_model.fit_vars and err is not None and err > 0:
            err_val = float(err) * conv
            model_lines.append(f"{label}  {val:.3f} ± {err_val:.3f}")
        else:
            model_lines.append(f"{label}  {val:.3f}")

    euler_lines = [
        f"α  {int(round(molecule.susc.alpha))}°",
        f"β  {int(round(molecule.susc.beta))}°",
        f"γ  {int(round(molecule.susc.gamma))}°",
    ]

    n_rows = max(len(fit_lines), len(model_lines), len(euler_lines))
    fit_lines += [""] * (n_rows - len(fit_lines))
    model_lines += [""] * (n_rows - len(model_lines))
    euler_lines += [""] * (n_rows - len(euler_lines))

    header_table = header_ax.table(
        cellText=[
            [fit_entry, model_entry, euler_entry]
            for fit_entry, model_entry, euler_entry in zip(
                fit_lines,
                model_lines,
                euler_lines,
            )
        ],
        colLabels=["Fit", f"Model ({model_unit_label})", "Euler Angles"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.96],
    )
    header_table.auto_set_font_size(False)
    header_table.set_fontsize(scale.annotation)
    max_row = max(row for row, _ in header_table.get_celld().keys())
    max_col = max(col for _, col in header_table.get_celld().keys())

    for (row, col), cell in header_table.get_celld().items():
        visible_edges = "LTRB"
        if row == 0:
            visible_edges = visible_edges.replace("T", "")
        if row == max_row:
            visible_edges = visible_edges.replace("B", "")
        if col == 0:
            visible_edges = visible_edges.replace("L", "")
        if col == max_col:
            visible_edges = visible_edges.replace("R", "")

        cell.visible_edges = visible_edges
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        cell.set_facecolor("white")
        cell.PAD = 0.12
        cell.set_text_props(ha="center", va="center")
        if row == 0:
            cell.set_text_props(weight="bold", ha="center", va="center")


def plot_fitted_shifts(
    molecule: Molecule,
    experiment: Experiment,
    susc_model: models.SusceptibilityModel,
    spec: PlotSpec,
    average: bool = True,
    save: bool = True,
    show: bool = True,
    save_name: str = "nmr_shifts.pdf",
    window_title: str = "Fitted Shifts",
    susc_units: str = "A3",
    verbose: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plots theoretical vs experimental shifts for a fitted susceptibility model.

    Args:
        molecule: Molecule containing theoretical shift data.
        experiment: Experimental shift data.
        susc_model: Fitted susceptibility model.
        average: If ``True``, averages equivalent nuclei (same chemical label).
        save: If ``True``, saves the plot to `save_name`.
        show: If ``True``, shows the plot.
        save_name: Output image file name.
        window_title: Figure window title.
        susc_units: Units for reporting susceptibility values in the annotation.
            Supported: ``"A3"``, ``"A3 mol-1"``, ``"cm3"``, ``"cm3 mol-1"``.
        verbose: If ``True``, prints the output file name when saving.

    Returns:
        A tuple ``(fig, ax)``.
    """

    seen = set()
    unique_nuclei = [
        seen.add(nuc.chem_label) or nuc
        for nuc in molecule.nuclei
        if nuc.chem_label not in seen
    ]

    if average:
        # Theoretical shifts, averaged over equivalent nuclei
        calc_shifts = {nuc.chem_label: nuc.shift.avg for nuc in unique_nuclei}
        # Experimental shifts, same order as theoretical
        exp = {label: experiment[label].shift for label in calc_shifts.keys()}
    else:
        # One signal per nucleus
        calc_shifts = {nuc.chem_label: [] for nuc in unique_nuclei}
        for nuc in molecule.nuclei:
            calc_shifts[nuc.chem_label].append(nuc.shift.total)

        # Experimental shifts, same order as theoretical
        exp = {
            label: [experiment[label].shift] * len(calc_shifts[label])
            for label in calc_shifts.keys()
        }

    # Element specific markers with consistent order
    _unique_elements = [
        ele for ele in ptable.elements if ele in [nuc.label_nn for nuc in unique_nuclei]
    ]
    _markers = {
        ele: mrkr for (ele, mrkr) in zip(_unique_elements, ["x", "o", "v", "s", "*"])
    }

    markers = {nuc.chem_label: _markers[nuc.label_nn] for nuc in molecule.nuclei}

    # if math labels are present then use these instead
    if all([len(nuc.chem_math_label) for nuc in molecule.nuclei]):
        for nuc in unique_nuclei:
            calc_shifts[nuc.chem_math_label] = calc_shifts.pop(nuc.chem_label)
            markers[nuc.chem_math_label] = markers.pop(nuc.chem_label)
            exp[nuc.chem_math_label] = exp.pop(nuc.chem_label)

    fig = plt.figure(figsize=(5.6, 6.4), num=window_title)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 5.05], hspace=0.02)
    header_ax = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])

    fig.patch.set_facecolor("white")
    header_ax.set_facecolor("white")
    header_ax.axis("off")
    ax.set_facecolor("#f3f3f7")
    glyphs = spec.glyphs
    palette = spec.palette
    scale = spec.skin_axes(ax)
    ax.grid(True, which="major", color="white", linewidth=1.0)
    ax.grid(True, which="minor", color="white", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        ax.plot(
            calc,
            expt,
            lw=0,
            marker=markers[label],
            color=palette.primary,
            markersize=glyphs.ms,
        )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    shared_min = np.min([x_lim, y_lim])
    shared_max = np.max([x_lim, y_lim])
    ax.set_xlim([shared_min, shared_max])
    ax.set_ylim([shared_min, shared_max])
    ax.set_aspect("equal", adjustable="box")
    diag_line = ax.plot(
        [shared_min, shared_max],
        [shared_min, shared_max],
        color=palette.primary,
        lw=0.75,
    )[0]

    ax.set_xlabel("Theoretical Shift (ppm)")
    ax.set_ylabel("Experimental Shift (ppm)")

    label_entries: list[tuple[str, float, float]] = []
    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        if average:
            label_entries.append((label, float(calc), float(expt)))
        else:
            for calc_value, exp_value in zip(calc, expt):
                label_entries.append((label, float(calc_value), float(exp_value)))

    _render_summary_table(
        header_ax,
        molecule,
        susc_model,
        scale=scale,
        susc_units=susc_units,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.10,
        top=0.97,
        hspace=0.04,
    )
    fig.canvas.draw()
    plot_pos = ax.get_position()
    header_pos = header_ax.get_position()
    header_ax.set_position(
        [
            plot_pos.x0,
            header_pos.y0,
            plot_pos.width,
            header_pos.height,
        ]
    )

    ax.invert_xaxis()
    ax.invert_yaxis()
    resolve_label_layout(
        ax,
        label_entries,
        fontsize=scale.annotation,
        marker_size=glyphs.ms,
        diag_line=diag_line,
    )

    render_figure(
        fig,
        save=save,
        show=show,
        save_name=save_name,
    )

    if save and verbose:
        logger.info("Chemical shift plot saved to %s", f"{save_name}.pdf")

    return fig, ax
