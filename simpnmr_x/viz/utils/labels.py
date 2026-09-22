# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Public label helpers for human-facing visualization output."""

from __future__ import annotations

_PARAMETER_MATHTEXT = {
    "iso": r"$\chi_\mathrm{iso}$",
    "ax": r"$\Delta\chi_\mathrm{ax}$",
    "rho_over_ax": r"$\Delta\chi_\mathrm{rh} / \Delta\chi_\mathrm{ax}$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "p1": r"$p_1$",
    "p2": r"$p_2$",
}


def parameter_label_mathtext(label: str) -> str:
    """Return the public mathtext label for a fitted parameter name."""

    return _PARAMETER_MATHTEXT.get(label, label)


def moment_label_mathtext(label: str) -> str:
    """Return the public mathtext label for a moment label."""

    if isinstance(label, str) and label.startswith("m") and label[1:].isdigit():
        return rf"$m_{{{int(label[1:])}}}$"
    return label
