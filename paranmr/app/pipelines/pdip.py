# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Compute point-dipole hyperfine deviatoric (traceless) tensors.

Loads structural data, evaluates point-dipole A_dip tensors, optionally applies
signal labels, and writes results to CSV with optional plots.
"""

import logging
import os

import numpy as np

from paranmr.app.loaders.labels_load import load_signal_labels_from_csv
from paranmr.app.params.options import CalcPdipRunOptions
from paranmr.app.policies.hfc import has_missing_selected_signal_labels
from paranmr.core.domain.mol import Molecule
from paranmr.io.qc import gateway as rdrs
from paranmr.tools.coords import xyz_fmt as xyzf
from paranmr.viz.plots.hfc import plot_hyperfine, plot_hyperfine_spread
from paranmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_calc_pdip(
    structure_file: str,
    centres: list[str],
    elements: list[str] | str,
    signal_labels: str | None,
    plot_components: list[str],
    options: CalcPdipRunOptions,
) -> int:
    """
    Compute point-dipole hyperfine dipolar tensors for a structure and optionally plot.
    """

    # Normalise centres
    centres = [centre.lower().capitalize() for centre in centres]

    # Load structure
    ext = os.path.splitext(structure_file)[1]
    if ext == ".xyz":
        labels, coords = xyzf.load_xyz(structure_file)
    elif ext in {".log", ".out"}:
        qcs = rdrs.QCStructure.guess_from_file(structure_file)
        labels = qcs.labels
        coords = qcs.coords
    else:
        raise ValueError(f"Unsupported structure file format: {ext}")

    # Create molecule
    molecule = Molecule.from_labels_coords(labels, coords, elements=elements)

    # Calculate point-dipole hyperfine deviatoric (traceless) tensor
    molecule.calc_pdip(centres)

    if signal_labels is not None:
        al_to_sl, al_to_sml = load_signal_labels_from_csv(signal_labels)
        if has_missing_selected_signal_labels(molecule, al_to_sl):
            logger.warning(
                "Signal labels file does not define labels for all selected nuclei; "
                "missing labels will use atom labels."
            )
        molecule.apply_signal_labels(al_to_sl, al_to_sml)

    # Save hyperfine data to file
    out = np.array(
        [
            "{}, {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(
                nuc.label,
                nuc.signal_label,
                *nuc.A.sd[0, :],
                *nuc.A.sd[1, 1:],
                nuc.A.sd[2, 2],
            )
            for nuc in molecule.nuclei
        ]
    )

    file_head = os.path.splitext(structure_file)[0]
    file_name = f"point_dipole_dA_{file_head}.csv"

    header = (
        "Label, dA_xx (ppm Å^-3), "
        "dA_xy (ppm Å^-3), "
        "dA_xz (ppm Å^-3), "
        "dA_yy (ppm Å^-3), "
        "dA_yz (ppm Å^-3), "
        "dA_zz (ppm Å^-3)"
    )

    np.savetxt(
        file_name,
        out,
        delimiter=options.runtime.csv_delimiter,
        header=header,
        fmt="%s",
    )
    logger.info("Point-dipole deviatoric hyperfine tensors saved to %s", file_name)

    # Build the resolved plotting contract once per run.
    spec = apply_profile(options.runtime.plot_profile)

    if plot_components:
        with spec.context():
            plot_hyperfine(
                molecule.nuclei,
                plot_components,
                spec=spec,
                save=True,
                show=options.runtime.show_plots,
                save_name=f"point_dipole_dA_{file_head}",
                verbose=True,
                window_title="Point-Dipole Hyperfines",
            )

            if signal_labels is not None:
                plot_hyperfine_spread(
                    molecule.nuclei,
                    plot_components,
                    spec=spec,
                    save=True,
                    show=options.runtime.show_plots,
                    save_name=f"spread_point_dipole_dA_{file_head}",
                    verbose=True,
                    window_title="Point-Dipole Hyperfines Spread",
                )

    return 0
