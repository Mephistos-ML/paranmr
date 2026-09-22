# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Generic plotting orchestration for hyperfine benchmark pipelines."""

import logging
import os
import re

logger = logging.getLogger(__name__)


def plot_hyperfine_benchmark_summary(
    summary: dict[str, dict[str, dict[str, dict[str, object]]]],
    *,
    output_dir: str,
    spec,
    show: bool,
    plot_spread,
    filename_metric: str,
    window_metric: str,
) -> None:
    """Plot one hyperfine benchmark spread chart per functional and nucleus."""
    for functional, nucleus_summary in summary.items():
        safe_functional = safe_filename_token(functional)
        plotted_nuclei: list[str] = []
        for nucleus_label, signal_label_summary in nucleus_summary.items():
            safe_nucleus = safe_filename_token(nucleus_label)
            save_name = os.path.join(
                output_dir,
                f"{safe_functional}_{safe_nucleus}_{filename_metric}_benchmark_spread",
            )
            plot_spread(
                functional=functional,
                nucleus_label=nucleus_label,
                signal_label_summary=signal_label_summary,
                spec=spec,
                save=True,
                show=show,
                save_name=save_name,
                window_title=f"{window_metric} benchmark: {functional} {nucleus_label}",
            )
            plotted_nuclei.append(nucleus_label)
        logger.info(
            "%s %s benchmark plots saved to %s",
            ", ".join(plotted_nuclei),
            functional,
            output_dir,
        )


def plot_hyperfine_functional_max_summary(
    summary: dict[str, list[dict[str, object]]],
    *,
    output_dir: str,
    spec,
    show: bool,
    plot_max_curve,
    filename_metric: str,
    log_metric: str,
    window_metric: str,
) -> None:
    """Plot sorted maximum hyperfine benchmark curves for each nucleus."""
    plotted_nuclei: list[str] = []
    for nucleus_label, max_rows in summary.items():
        _log_max_label_diagnostics(nucleus_label, max_rows)
        safe_nucleus = safe_filename_token(nucleus_label)
        save_name = os.path.join(
            output_dir,
            f"{safe_nucleus}_{filename_metric}_benchmark_max_curve",
        )
        plot_max_curve(
            nucleus_label=nucleus_label,
            max_rows=max_rows,
            spec=spec,
            save=True,
            show=show,
            save_name=save_name,
            window_title=f"{window_metric} max benchmark: {nucleus_label}",
        )
        plotted_nuclei.append(nucleus_label)

    logger.info(
        "%s maximum %s benchmark plots saved to %s",
        ", ".join(plotted_nuclei),
        log_metric,
        output_dir,
    )


def safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated benchmark plot names."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("_") or "functional"


def _log_max_label_diagnostics(
    nucleus_label: str,
    max_rows: list[dict[str, object]],
) -> None:
    """Log majority-label replacement warnings for max benchmark curves."""
    for row in max_rows:
        if row.get("adjusted"):
            logger.warning(
                "%s %s max label replaced: %s -> %s",
                nucleus_label,
                row["functional"],
                row["raw_signal_label"],
                row["signal_label"],
            )
        elif row["raw_signal_label"] != row.get("majority_signal_label"):
            logger.warning(
                "%s %s max label differs from majority label: %s vs %s. "
                "Set or increase benchmark:max_label_tolerance to allow "
                "majority-label replacement.",
                nucleus_label,
                row["functional"],
                row["raw_signal_label"],
                row["majority_signal_label"],
            )
