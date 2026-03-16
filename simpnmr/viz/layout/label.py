# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Shared obstacle-aware label placement utilities for scatter plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

LabelEntry = tuple[str, float, float]


def resolve_label_layout(
    ax: plt.Axes,
    entries: Sequence[LabelEntry],
    *,
    fontsize: float,
    marker_size: float,
    diag_line: plt.Line2D | None = None,
) -> list[plt.Text]:
    """Build and draw scatter-plot labels with obstacle-aware placement.

    Labels are created only after the plot geometry is fully defined. Placement
    is resolved in display coordinates against already drawn objects: marker
    footprints, the reference diagonal, previously placed labels, and the
    visible axes area.

    This helper assumes that the axes geometry is already final, including
    limits, aspect ratio, and any axis inversion.

    Args:
        ax: Target axes.
        entries: Normalized ``(label, x, y)`` tuples in data coordinates.
        fontsize: Font size used for the labels.
        marker_size: Marker size used in the scatter plot.
        diag_line: Optional plotted reference diagonal used as an obstacle.

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

    if not entries:
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
        for label, x, y in entries
    ]

    # Resolve all obstacle geometry in display coordinates.
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Reserve a small inner safe region so labels do not cling to the axes frame.
    axes_bbox = ax.get_window_extent(renderer=renderer)
    safe_bbox = plt.matplotlib.transforms.Bbox.from_extents(
        axes_bbox.x0 + axes_inner_margin_px,
        axes_bbox.y0 + axes_inner_margin_px,
        axes_bbox.x1 - axes_inner_margin_px,
        axes_bbox.y1 - axes_inner_margin_px,
    )

    # Convert label anchors from data space to display space for collision checks.
    anchor_points = np.array(
        [ax.transData.transform((x, y)) for _, x, y in entries],
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

    # Approximate the reference diagonal by a sequence of small obstacle boxes.
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
