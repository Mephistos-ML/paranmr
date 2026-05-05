# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Canonical figure-size tokens for visualization profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from simpnmr.app.params.plot_cfg import PlotProfile

FigureVariant = Literal[
    "horizontal",
    "horizontal_extended",
    "vertical",
    "vertical_extended",
]


@dataclass(frozen=True, slots=True)
class FigureSize:
    """Canonical figure size in inches."""

    width: float
    height: float

    def as_tuple(self) -> tuple[float, float]:
        """Return the figure size as a Matplotlib ``figsize`` tuple."""
        return (self.width, self.height)


_FIGURE_SIZES: dict[PlotProfile, dict[FigureVariant, FigureSize]] = {
    "paper": {
        "horizontal": FigureSize(width=3.54, height=2.40),  # 9.00 × 6.10 cm
        "horizontal_extended": FigureSize(width=7.08, height=2.40),  # 18.00 x 6.10 cm
        "vertical": FigureSize(width=3.54, height=4.05),  # 9.00 x 10.28 cm
        "vertical_extended": FigureSize(width=3.54, height=4.33),  # 9.00 x 11.00 cm
    },
    "poster": {
        "horizontal": FigureSize(width=3.54, height=2.40),  # 9.00 × 6.10 cm
        "horizontal_extended": FigureSize(width=7.08, height=2.40),  # 18.00 x 6.10 cm
        "vertical": FigureSize(width=3.54, height=4.05),  # 9.00 x 10.28 cm
        "vertical_extended": FigureSize(width=3.54, height=4.33),  # 9.00 x 11.00 cm
    },
}


def get_figsize(
    profile: PlotProfile,
    variant: FigureVariant = "horizontal",
) -> tuple[float, float]:
    """Return the canonical ``figsize`` tuple for a profile and variant.

    Args:
        profile: Plotting profile that selects the publication context.
        variant: Figure geometry variant within the selected profile.

    Returns:
        Canonical Matplotlib ``figsize`` tuple in inches.

    Raises:
        ValueError: If the profile/variant combination is not supported.
    """
    try:
        return _FIGURE_SIZES[profile][variant].as_tuple()
    except KeyError as exc:
        raise ValueError(
            "Unsupported figsize combination: "
            f"profile={profile!r}, variant={variant!r}."
        ) from exc
