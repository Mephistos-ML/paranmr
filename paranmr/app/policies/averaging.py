# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Application-layer averaging policy helpers for susceptibility fitting."""

from __future__ import annotations

from collections.abc import Sequence

from paranmr.core.domain.mol import Molecule


def resolve_average_shift_groups(
    *,
    molecule: Molecule,
    average_shifts: str | Sequence[str] | None,
) -> list[list[str]]:
    """Resolve user-facing averaging policy into explicit atom-label groups."""

    if average_shifts in (None, "", []):
        return []

    if isinstance(average_shifts, str):
        normalized = average_shifts.strip().lower()
        if normalized == "all":
            signal_labels = sorted({nuc.signal_label for nuc in molecule.nuclei})
            return _groups_from_signal_labels(
                molecule=molecule,
                signal_labels=signal_labels,
            )
        return _groups_from_signal_labels(
            molecule=molecule,
            signal_labels=[average_shifts],
        )

    signal_labels = [str(value) for value in average_shifts]
    if not signal_labels:
        return []
    if any(value.strip().lower() == "all" for value in signal_labels):
        if len(signal_labels) != 1:
            raise ValueError(
                "susc_fit:average_shifts cannot combine 'all' with "
                "selected signal labels."
            )
        signal_labels = sorted({nuc.signal_label for nuc in molecule.nuclei})

    return _groups_from_signal_labels(
        molecule=molecule,
        signal_labels=signal_labels,
    )


def _groups_from_signal_labels(
    *,
    molecule: Molecule,
    signal_labels: Sequence[str],
) -> list[list[str]]:
    groups: list[list[str]] = []
    available_signal_labels = {nuc.signal_label for nuc in molecule.nuclei}
    missing = [
        signal_label
        for signal_label in signal_labels
        if signal_label not in available_signal_labels
    ]
    if missing:
        raise ValueError(
            "Unknown signal label(s) requested for susc_fit:average_shifts: "
            + ", ".join(sorted(set(missing)))
        )

    for signal_label in signal_labels:
        group = [
            nuc.label for nuc in molecule.nuclei if nuc.signal_label == signal_label
        ]
        if group:
            groups.append(group)
    return groups
