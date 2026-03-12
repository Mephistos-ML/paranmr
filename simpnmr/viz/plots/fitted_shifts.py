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

logger = logging.getLogger(__name__)


def _place_labels(
    ax: plt.Axes,
    calc_shifts: dict[str, float | list[float]],
    exp: dict[str, float | list[float]],
    *,
    average: bool,
    fontsize: float,
    marker_size: float,
    diag_line: plt.Line2D | None = None,
) -> list[plt.Text]:
    """Build and draw fitted-shift labels with obstacle-aware placement.

    Labels are created only after the plot geometry is fully defined. Placement
    is resolved in display coordinates against already drawn objects: marker
    footprints, the reference diagonal, previously placed labels, and the
    visible axes area.

    This function assumes that the axes geometry is already final, including
    limits, aspect ratio, and axis inversion.

    Args:
        ax: Target axes.
        calc_shifts: Calculated shifts keyed by display label.
        exp: Experimental shifts keyed by display label, in the same order as
            ``calc_shifts``.
        average: Whether each label maps to a single averaged point or multiple
            point entries.
        fontsize: Font size used for the labels.
        marker_size: Marker size used in the scatter plot.
        diag_line: Optional plotted y=x reference line used as an obstacle.

    Returns:
        Drawn label artists.
    """
    # Screen-space layout constants.
    axes_inner_margin_px = 4.0
    point_pad_px = max(5.0, float(marker_size) * 0.9)
    diag_pad_px = 3.0
    label_bbox_pad_px = 2.0

    offset_near = 5.0
    offset_mid = 7.0
    offset_far = 9.0
    offset_outer = 11.0

    weight_label_overlap = 2000.0
    weight_point_overlap = 1500.0
    weight_line_overlap = 1200.0
    weight_out_of_bounds = 400.0
    weight_offset_distance = 0.08

    # Flatten the input mapping into one anchor per drawn label.
    # Collect label anchor points in data coordinates.
    label_entries: list[tuple[str, float, float]] = []
    for (label, calc), expt in zip(calc_shifts.items(), exp.values()):
        if average:
            label_entries.append((label, float(calc), float(expt)))
        else:
            for calc_value, exp_value in zip(calc, expt):
                label_entries.append((label, float(calc_value), float(exp_value)))

    # Exit early when there is nothing to annotate.
    if not label_entries:
        return []

    # Create label artists first; positions are resolved after the plot is drawn.
    label_texts = [
        ax.annotate(
            label,
            (x, y),
            xytext=(0.0, 0.0),
            textcoords="offset points",
            fontsize=fontsize,
            ha="left",
            va="bottom",
        )
        for label, x, y in label_entries
    ]

    # Resolve all obstacle geometry in display coordinates.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Reserve a small inner safe region so labels do not cling to the axes frame.
    axes_bbox = ax.get_window_extent(renderer=renderer)
    inner_margin_px = axes_inner_margin_px
    safe_bbox = plt.matplotlib.transforms.Bbox.from_extents(
        axes_bbox.x0 + inner_margin_px,
        axes_bbox.y0 + inner_margin_px,
        axes_bbox.x1 - inner_margin_px,
        axes_bbox.y1 - inner_margin_px,
    )

    # Convert label anchors from data space to display space for collision checks.
    anchor_points = np.array(
        [ax.transData.transform((x, y)) for _, x, y in label_entries],
        dtype=float,
    )

    # Approximate each plotted marker by a padded screen-space bounding box.
    point_obstacles = [
        plt.matplotlib.transforms.Bbox.from_extents(
            px - point_pad_px,
            py - point_pad_px,
            px + point_pad_px,
            py + point_pad_px,
        )
        for px, py in anchor_points
    ]

    # Approximate the diagonal reference line by a sequence of small obstacle boxes.
    line_obstacles: list[plt.matplotlib.transforms.Bbox] = []
    if diag_line is not None:
        xdata = np.asarray(diag_line.get_xdata(), dtype=float)
        ydata = np.asarray(diag_line.get_ydata(), dtype=float)
        if xdata.size >= 2 and ydata.size >= 2:
            samples = np.linspace(0.0, 1.0, 48)
            xs = xdata[0] + samples * (xdata[-1] - xdata[0])
            ys = ydata[0] + samples * (ydata[-1] - ydata[0])
            diag_points = ax.transData.transform(np.column_stack([xs, ys]))
            line_obstacles = [
                plt.matplotlib.transforms.Bbox.from_extents(
                    px - diag_pad_px,
                    py - diag_pad_px,
                    px + diag_pad_px,
                    py + diag_pad_px,
                )
                for px, py in diag_points
            ]

    # Candidate label offsets are tried from near to far around each anchor point.
    candidate_offset_specs = [
        (
            offset_near,
            [
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ],
        ),
        (
            offset_mid,
            [
                (0.0, 1.0),
                (1.0, 0.0),
                (0.0, -1.0),
                (-1.0, 0.0),
            ],
        ),
        (
            offset_far,
            [
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ],
        ),
        (
            offset_outer,
            [
                (0.0, 1.0),
                (1.0, 0.0),
                (0.0, -1.0),
                (-1.0, 0.0),
            ],
        ),
    ]
    candidate_offsets = [
        (radius * unit_dx, radius * unit_dy)
        for radius, directions in candidate_offset_specs
        for unit_dx, unit_dy in directions
    ]

    # Place the most crowded labels first so later labels fit around them.
    if len(anchor_points) == 1:
        order = [0]
    else:
        diffs = anchor_points[:, None, :] - anchor_points[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=1)
        order = list(np.argsort(nearest))

    placed_bboxes: list[plt.matplotlib.transforms.Bbox] = []

    # Greedily assign the least-colliding screen-space candidate to each label.
    for idx in order:
        text = label_texts[idx]
        best_state: (
            tuple[
                float,
                tuple[float, float],
                str,
                str,
                plt.matplotlib.transforms.Bbox,
            ]
            | None
        ) = None

        # Evaluate each candidate in screen space and keep the least-penalized one.
        for dx, dy in candidate_offsets:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")

            text.set_position((dx, dy))
            text.set_ha(ha)
            text.set_va(va)

            bbox = text.get_window_extent(renderer=renderer)
            bbox = plt.matplotlib.transforms.Bbox.from_extents(
                bbox.x0 - label_bbox_pad_px,
                bbox.y0 - label_bbox_pad_px,
                bbox.x1 + label_bbox_pad_px,
                bbox.y1 + label_bbox_pad_px,
            )

            score = 0.0
            score += weight_label_overlap * sum(
                bbox.overlaps(prev) for prev in placed_bboxes
            )
            score += weight_point_overlap * sum(
                bbox.overlaps(obs) for obs in point_obstacles
            )
            score += weight_line_overlap * sum(
                bbox.overlaps(obs) for obs in line_obstacles
            )

            # Penalize labels that leave the visible plotting area.
            out_of_bounds = (
                max(safe_bbox.x0 - bbox.x0, 0.0)
                + max(bbox.x1 - safe_bbox.x1, 0.0)
                + max(safe_bbox.y0 - bbox.y0, 0.0)
                + max(bbox.y1 - safe_bbox.y1, 0.0)
            )
            score += weight_out_of_bounds * out_of_bounds
            score += weight_offset_distance * float(np.hypot(dx, dy))

            state = (score, (dx, dy), ha, va, bbox)
            if best_state is None or score < best_state[0]:
                best_state = state

            if score == 0.0:
                break

        # Commit the best candidate and reserve its screen-space footprint.
        assert best_state is not None
        _, (dx, dy), ha, va, bbox = best_state
        text.set_position((dx, dy))
        text.set_ha(ha)
        text.set_va(va)
        placed_bboxes.append(bbox)

    # Request a redraw so the final label positions are reflected on screen/export.
    fig.canvas.draw_idle()
    return label_texts


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
    _place_labels(
        ax,
        calc_shifts,
        exp,
        average=average,
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
