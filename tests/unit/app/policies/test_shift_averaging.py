# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

import pytest

from paranmr.app.policies.averaging import (
    apply_methyl_signal_labels,
    detect_methyl_group_records,
    resolve_average_shift_groups,
)
from paranmr.core.domain.mol import Molecule
from paranmr.core.fitting.susceptibility.moments.forward import (
    calculated_moments_from_parameters,
    calculated_signal_packages_from_parameters,
)
from paranmr.core.fitting.susceptibility.moments.gaussian import (
    gaussian_peak_representation,
)


class _DummyModel:
    def model(self, parameters, nuclei):
        return {
            nucleus.label: parameters[nucleus.label]
            for nucleus in nuclei
        }


@pytest.mark.unit
def test_detect_methyl_group_records_finds_three_protons_on_one_carbon():
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

    groups = detect_methyl_group_records(molecule)

    assert len(groups) == 1
    assert groups[0].carbon_label == "C1"
    assert groups[0].proton_labels == ("H1", "H2", "H3")


@pytest.mark.unit
def test_detect_methyl_group_records_requires_h_only_nuclei_selection():
    molecule = Molecule.from_labels_coords(
        labels=["C1", "H1", "H2", "H3", "C2"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-1.52, 0.0, 0.0],
        ],
        elements="all",
    )

    with pytest.raises(ValueError, match="only H nuclei"):
        detect_methyl_group_records(molecule)


@pytest.mark.unit
def test_moment_forward_collapses_methyl_group_into_one_signal():
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

    average_labels = resolve_average_shift_groups(
        molecule=molecule,
        average_shifts="methyls",
    )
    packages = calculated_signal_packages_from_parameters(
        model=_DummyModel(),
        parameters={"H1": 1.0, "H2": 2.0, "H3": 4.0, "H4": 10.0},
        nuclei=molecule.nuclei,
        average_labels=tuple(tuple(group) for group in average_labels),
    )

    assert len(packages) == 2
    assert packages[0].atom_labels == ("H1", "H2", "H3")
    assert packages[0].center == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)
    assert packages[1].atom_labels == ("H4",)


@pytest.mark.unit
def test_calculated_moments_treat_collapsed_packages_with_equal_weight():
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
    average_labels = resolve_average_shift_groups(
        molecule=molecule,
        average_shifts="methyls",
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
        areas=[1.0, 1.0],
    )
    expected_m1 = float(
        sum(expected_peaks["area_norm"] * expected_peaks["center"])
    )
    assert moments["m1"] == pytest.approx(expected_m1)


@pytest.mark.unit
def test_apply_methyl_signal_labels_assigns_shared_synthetic_labels():
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

    apply_methyl_signal_labels(molecule)

    ch3_label = "CH3(C1)"
    assert [nuc.signal_label for nuc in molecule.nuclei] == [
        ch3_label,
        ch3_label,
        ch3_label,
        "H4",
    ]
    assert [nuc.signal_math_label for nuc in molecule.nuclei[:3]] == [
        ch3_label,
        ch3_label,
        ch3_label,
    ]


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
