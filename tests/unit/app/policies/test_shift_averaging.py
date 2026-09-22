# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import pytest

from simpnmr_x.app.policies.averaging import (
    resolve_average_shift_groups,
)
from simpnmr_x.core.domain.mol import Molecule
from simpnmr_x.core.fitting.susceptibility.moments.forward import (
    calculated_moments_from_parameters,
    calculated_signal_packages_from_parameters,
)
from simpnmr_x.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)


class _DummyModel:
    def model(self, parameters, nuclei):
        return {nucleus.label: parameters[nucleus.label] for nucleus in nuclei}


def _assign_methyl_chemical_label(molecule: Molecule) -> None:
    for nucleus in molecule.nuclei:
        if nucleus.label in {"H1", "H2", "H3"}:
            nucleus.signal_label = "Me-1"


@pytest.mark.unit
def test_moment_forward_collapses_chemical_label_group_into_one_signal():
    molecule = Molecule.from_labels_coords(
        labels=["C1", "H1", "H2", "H3", "C2", "H4"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-1.52, 0.0, 0.0],
            [-2.61, 0.0, 0.0],
        ],
        elements="H",
    )

    _assign_methyl_chemical_label(molecule)
    average_labels = resolve_average_shift_groups(
        molecule=molecule, average_shifts="all"
    )
    packages = calculated_signal_packages_from_parameters(
        model=_DummyModel(),
        parameters={"H1": 1.0, "H2": 2.0, "H3": 4.0, "H4": 10.0},
        nuclei=molecule.nuclei,
        average_labels=tuple(tuple(group) for group in average_labels),
    )

    assert len(packages) == 2
    packages_by_atoms = {package.atom_labels: package for package in packages}
    methyl_package = packages_by_atoms[("H1", "H2", "H3")]
    assert methyl_package.center == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)
    assert methyl_package.area == pytest.approx(3.0)
    assert packages_by_atoms[("H4",)].area == pytest.approx(1.0)


@pytest.mark.unit
def test_calculated_moments_weight_collapsed_packages_by_theoretical_area():
    molecule = Molecule.from_labels_coords(
        labels=["C1", "H1", "H2", "H3", "C2", "H4"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-1.52, 0.0, 0.0],
            [-2.61, 0.0, 0.0],
        ],
        elements="H",
    )
    _assign_methyl_chemical_label(molecule)
    average_labels = resolve_average_shift_groups(
        molecule=molecule, average_shifts="all"
    )
    widths_by_label = {
        "H1": 1.0,
        "H2": 1.0,
        "H3": 1.0,
        "H4": 1.0,
    }

    moments = calculated_moments_from_parameters(
        model=_DummyModel(),
        parameters={"H1": 1.0, "H2": 2.0, "H3": 4.0, "H4": 10.0},
        nuclei=molecule.nuclei,
        linewidths_by_label=widths_by_label,
        include_diamagnetic=False,
        moment_labels=("m1", "m2", "m3", "m4", "m5", "m6"),
        average_labels=tuple(tuple(group) for group in average_labels),
    )

    expected_peaks = gaussian_peak_representation(
        centers=[(1.0 + 2.0 + 4.0) / 3.0, 10.0],
        fwhm=[1.0, 1.0],
        areas=[3.0, 1.0],
    )
    expected_m1 = float(sum(expected_peaks["area_norm"] * expected_peaks["center"]))
    assert moments["m1"] == pytest.approx(expected_m1)


@pytest.mark.unit
def test_average_all_keeps_singleton_signal_groups_for_assignment_workflows():
    molecule = Molecule.from_labels_coords(
        labels=["C1", "C2", "C3"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        elements="all",
    )
    molecule.nuclei[0].signal_label = "A"
    molecule.nuclei[1].signal_label = "A"
    molecule.nuclei[2].signal_label = "B"

    groups = resolve_average_shift_groups(
        molecule=molecule,
        average_shifts="all",
    )

    assert groups == [["C1", "C2"], ["C3"]]
