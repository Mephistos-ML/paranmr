# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Load hyperfine benchmark input sources."""

from types import SimpleNamespace

from simpnmr.app.loaders.hfc_load import load_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_base_molecule


def load_hyperfine_benchmark_sources(config) -> list[dict[str, object]]:
    """Load all configured hyperfine benchmark sources."""
    al_to_cl, al_to_cml = load_chem_labels_from_csv(config.chem_labels_file)

    signals: list[dict[str, object]] = []
    for index, source in enumerate(config.hyperfine, start=1):
        source_config = SimpleNamespace(
            hyperfine_method=source.method,
            hyperfine_file=source.file,
            nuclei_include=config.nuclei_include,
            hyperfine_orbital_contribution="auto",
        )

        molecule = load_base_molecule(source_config)
        molecule = load_hyperfines(molecule=molecule, config=source_config)
        molecule.apply_chem_labels(al_to_cl, al_to_cml)

        signals.append(
            {
                "source_id": f"source_{index}",
                "functional": source.functional,
                "molecule": molecule,
            }
        )

    return signals


def group_loaded_sources_by_functional(
    signals: list[dict[str, object]],
) -> dict[str, list[tuple[str, object]]]:
    """Group loaded benchmark molecules by functional name."""
    grouped: dict[str, list[tuple[str, object]]] = {}
    for signal in signals:
        functional = str(signal["functional"])
        source_id = str(signal["source_id"])
        grouped.setdefault(functional, []).append((source_id, signal["molecule"]))
    return grouped
