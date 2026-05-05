# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Run A_fc benchmark workflows.

This module owns application-level orchestration for the A_fc benchmark command.
Scientific parsing and numerical comparison logic must stay in the IO and core
layers respectively.
"""

import logging
import os
import re
from types import SimpleNamespace

from simpnmr.app.loaders.hfc_load import load_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_base_molecule
from simpnmr.app.params.options import BenchmarkAfcRunOptions
from simpnmr.core.benchmarks.a_fc import (
    summarize_a_fc_max_by_nucleus,
    summarize_a_fc_ranges_by_functional_and_nucleus,
)
from simpnmr.viz.plots.benchmarks import (
    plot_a_fc_functional_max_curve,
    plot_a_fc_spread,
)
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_benchmark_a_fc(config, options: BenchmarkAfcRunOptions | None = None) -> int:
    """Run the A_fc benchmark workflow from a YAML configuration.

    Args:
        config: A_fc benchmark configuration loaded from YAML.
        options: Runtime options supplied by the CLI.

    Returns:
        Exit code: 0 on success.

    Raises:
        ValueError: If benchmark run options are not supplied.
    """
    if options is None:
        raise ValueError("BenchmarkAfcRunOptions is required")

    os.makedirs(config.project_name, exist_ok=True)
    spec = apply_profile(options.runtime.plot_profile)

    if options.dry_run:
        return 0

    signals = load_a_fc_benchmark_sources(config)

    a_fc_summary = summarize_a_fc_ranges_by_functional_and_nucleus(
        _group_loaded_sources_by_functional(signals)
    )
    plot_a_fc_benchmark_summary(
        a_fc_summary,
        output_dir=config.project_name,
        spec=spec,
        show=options.runtime.show_plots,
    )
    plot_a_fc_functional_max_summary(
        summarize_a_fc_max_by_nucleus(
            a_fc_summary,
            max_label_tolerance=config.max_label_tolerance,
        ),
        output_dir=config.project_name,
        spec=spec,
        show=options.runtime.show_plots,
    )

    return 0


def load_a_fc_benchmark_sources(config) -> list[dict[str, object]]:
    """Load all configured A_fc benchmark sources.

    Args:
        config: A_fc benchmark configuration loaded from YAML.

    Returns:
        Loaded benchmark source signals in the same order as the input
        ``hyperfine`` blocks.
    """
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


def _group_loaded_sources_by_functional(
    signals: list[dict[str, object]],
) -> dict[str, list[tuple[str, object]]]:
    """Group loaded benchmark molecules by functional name."""
    grouped: dict[str, list[tuple[str, object]]] = {}
    for signal in signals:
        functional = str(signal["functional"])
        source_id = str(signal["source_id"])
        grouped.setdefault(functional, []).append((source_id, signal["molecule"]))
    return grouped


def plot_a_fc_benchmark_summary(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    *,
    output_dir: str,
    spec,
    show: bool,
) -> None:
    """Plot one A_fc spread violin chart per functional and nucleus."""
    for functional, nucleus_summary in summary.items():
        safe_functional = _safe_filename_token(functional)
        plotted_nuclei: list[str] = []
        for nucleus_label, chem_label_summary in nucleus_summary.items():
            safe_nucleus = _safe_filename_token(nucleus_label)
            save_name = os.path.join(
                output_dir,
                f"{safe_functional}_{safe_nucleus}_A_FC_benchmark_spread",
            )
            plot_a_fc_spread(
                functional=functional,
                nucleus_label=nucleus_label,
                chem_label_summary=chem_label_summary,
                spec=spec,
                save=True,
                show=show,
                save_name=save_name,
                window_title=f"A_fc benchmark: {functional} {nucleus_label}",
            )
            plotted_nuclei.append(nucleus_label)
        logger.info(
            "%s %s benchmark plots saved to %s",
            ", ".join(plotted_nuclei),
            functional,
            output_dir,
        )


def plot_a_fc_functional_max_summary(
    summary: dict[str, list[dict[str, object]]],
    *,
    output_dir: str,
    spec,
    show: bool,
) -> None:
    """Plot sorted maximum A_fc curves for each nucleus."""
    plotted_nuclei: list[str] = []
    for nucleus_label, max_rows in summary.items():
        for row in max_rows:
            if row.get("adjusted"):
                logger.warning(
                    "%s %s max label replaced: %s -> %s",
                    nucleus_label,
                    row["functional"],
                    row["raw_chem_label"],
                    row["chem_label"],
                )
            elif row["raw_chem_label"] != row.get("majority_chem_label"):
                logger.warning(
                    "%s %s max label differs from majority label: %s vs %s. "
                    "Set or increase benchmark:max_label_tolerance to allow "
                    "majority-label replacement.",
                    nucleus_label,
                    row["functional"],
                    row["raw_chem_label"],
                    row["majority_chem_label"],
                )
        safe_nucleus = _safe_filename_token(nucleus_label)
        save_name = os.path.join(
            output_dir,
            f"{safe_nucleus}_A_FC_benchmark_max_curve",
        )
        plot_a_fc_functional_max_curve(
            nucleus_label=nucleus_label,
            max_rows=max_rows,
            spec=spec,
            save=True,
            show=show,
            save_name=save_name,
            window_title=f"A_fc max benchmark: {nucleus_label}",
        )
        plotted_nuclei.append(nucleus_label)

    logger.info(
        "%s maximum A_fc benchmark plots saved to %s",
        ", ".join(plotted_nuclei),
        output_dir,
    )


def _safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated benchmark plot names."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("_") or "functional"
