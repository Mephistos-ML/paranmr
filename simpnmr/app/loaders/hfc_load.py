# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Suturina Group

"""Application-layer loader for constructing a base Molecule from hyperfine inputs.

This loader encapsulates the selection of hyperfine source (DFT/QC, point-dipole
approximation, or CSV) and returns a domain `Molecule` that is ready for downstream
prediction workflows.

Domain rules are implemented in factories and domain objects; this module orchestrates
IO + factories according to the provided configuration.
"""

from __future__ import annotations

import os
from pathlib import Path  # noqa
from typing import Any

from simpnmr.app.loaders.mol_load import load_molecule_from_csv
from simpnmr.app.policies.hfc import (
    OrbitalContribution,
    normalise_orbital_contribution,
)
from simpnmr.core.build.mol import (
    build_molecule_from_qca,
    build_molecule_with_pdip,
)
from simpnmr.core.domain.mol import Molecule
from simpnmr.io.csv.hfc import save_to_csv as save_hfc_csv  # noqa
from simpnmr.io.qc import gateway as rdrs
from simpnmr.io.qc.backends.orca.detect import detect_hfc_has_orb
from simpnmr.tools.coords import xyz_fmt as xyzf


def load_base_molecule_from_hyperfines(config: Any, delimiter: str) -> Molecule:
    """Load or construct the base Molecule including hyperfine information.

    Args:
        config: Application configuration object. Expected attributes:
            - hyperfine_method: One of {"dft", "pdip", "csv"}.
            - hyperfine_file: Path to the input file.
            - project_name: Output directory for generated artifacts.
            - nuclei_include: Elements to retain.
            - hyperfine_pdip_centres: PDIP centres for point-dipole calculation.
        delimiter: CSV delimiter for writing raw QC hyperfine data.

    Returns:
        A base Molecule populated with hyperfine data as required by the selected
        method.

    Raises:
        ValueError: If the hyperfine method or file format is unsupported.
    """
    method = config.hyperfine_method

    # DFT/QC-derived hyperfine data.
    if method == "dft":
        requested_orbital_contribution = normalise_orbital_contribution(
            config.hyperfine_orbital_contribution
        )
        has_orb = detect_hfc_has_orb(config.hyperfine_file)

        if requested_orbital_contribution is OrbitalContribution.ON and not has_orb:
            raise ValueError(
                "Requested orbital hyperfine contribution, but no A(ORB) term "
                "was found in the input file."
            )

        if requested_orbital_contribution is OrbitalContribution.AUTO:
            effective_orbital_contribution = (
                OrbitalContribution.ON if has_orb else OrbitalContribution.OFF
            )
        else:
            effective_orbital_contribution = requested_orbital_contribution

        # Record final, domain-level hyperfine model choice
        hyperfine_orbital_availability = (
            "available"
            if effective_orbital_contribution is OrbitalContribution.ON
            else "unavailable"
        )

        qc_hyperfine_data = rdrs.QCA.guess_from_file(config.hyperfine_file)

        # TODO(io/csv): Re-enable raw hyperfine CSV export after migrating the CSV
        # writer to the component-based QCA contract (a_fc/a_sd/a_orb).
        # Write raw calculation data to an output file for traceability.
        # save_hfc_csv(
        #     qc_hyperfine_data,
        #     file_name=os.path.join(config.project_name, "dft_hyperfines.csv"),
        #     verbose=True,
        #     delimiter=delimiter,
        #     comment=f"Data taken from file {Path(config.hyperfine_file).name}",
        # )

        # Retain only the atoms that are given in the labels file.
        base_molecule = build_molecule_from_qca(
            qc_hyperfine_data,
            converter="MHz_to_Ang-3",
            elements=config.nuclei_include,
            orbital_contribution=effective_orbital_contribution.value,
        )

        base_molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
            hyperfine_orbital_availability
        )

    # Point dipole approximation.
    elif method == "pdip":
        ext = os.path.splitext(config.hyperfine_file)[1]

        if ext == ".xyz":
            labels, coords = xyzf.load_xyz(config.hyperfine_file)
        elif ext in [".log", ".out"]:
            qcs = rdrs.QCStructure.guess_from_file(config.hyperfine_file)
            labels = qcs.labels
            coords = qcs.coords
        else:
            raise ValueError(
                "Specified hyperfine file format "
                f"{os.path.splitext(config.hyperfine_file)[1]} unsupported"
            )

        # Build and populate PDIP hyperfines via factories.
        base_molecule = build_molecule_with_pdip(
            labels=labels,
            coords=coords,
            elements=config.nuclei_include,
            centres=config.hyperfine_pdip_centres,
        )

        base_molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
            "unavailable"
        )

    # CSV-provided hyperfines.
    elif method == "csv":
        base_molecule = load_molecule_from_csv(
            config.hyperfine_file,
            elements=config.nuclei_include,
        )

        # TODO(policy): For CSV hyperfines, orbital contribution handling must be
        # driven strictly by explicit yaml configuration. CSV provides only the final
        # tensor without provenance, so automatic detection is not meaningful here.
        base_molecule.metadata.setdefault("hyperfine", {})["orbital_contribution"] = (
            "unavailable"
        )

    else:
        raise ValueError(
            "Unsupported hyperfine_method. Expected one of {'dft', 'pdip', 'csv'}, "
            f"got: {method!r}"
        )

    return base_molecule
