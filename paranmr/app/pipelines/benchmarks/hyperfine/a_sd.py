# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Run A_sd benchmark workflows."""

import os

from paranmr.app.params.options import BenchmarkAsdRunOptions
from paranmr.app.pipelines.benchmarks.hyperfine.runner import (
    plot_hyperfine_benchmark_summary,
    plot_hyperfine_functional_max_summary,
)
from paranmr.app.pipelines.benchmarks.hyperfine.sources import (
    group_loaded_sources_by_functional,
    load_hyperfine_benchmark_sources,
)
from paranmr.core.benchmarks.hyperfine.a_sd import (
    summarize_a_sd_max_by_nucleus,
    summarize_a_sd_ranges_by_functional_and_nucleus,
)
from paranmr.viz.plots.benchmarks import (
    plot_a_sd_functional_max_curve,
    plot_a_sd_spread,
)
from paranmr.viz.style.theme import apply_profile


def run_benchmark_a_sd(config, options: BenchmarkAsdRunOptions | None = None) -> int:
    """Run the A_sd benchmark workflow from a YAML configuration."""
    if options is None:
        raise ValueError("BenchmarkAsdRunOptions is required")

    os.makedirs(config.project_name, exist_ok=True)
    spec = apply_profile(options.runtime.plot_profile)

    if options.dry_run:
        return 0

    signals = load_hyperfine_benchmark_sources(config)

    a_sd_summary = summarize_a_sd_ranges_by_functional_and_nucleus(
        group_loaded_sources_by_functional(signals)
    )
    plot_hyperfine_benchmark_summary(
        a_sd_summary,
        output_dir=config.project_name,
        spec=spec,
        show=options.runtime.show_plots,
        plot_spread=plot_a_sd_spread,
        filename_metric="A_SD",
        window_metric="A_sd",
    )
    plot_hyperfine_functional_max_summary(
        summarize_a_sd_max_by_nucleus(
            a_sd_summary,
            max_label_tolerance=config.max_label_tolerance,
        ),
        output_dir=config.project_name,
        spec=spec,
        show=options.runtime.show_plots,
        plot_max_curve=plot_a_sd_functional_max_curve,
        filename_metric="A_SD",
        log_metric="A_sd",
        window_metric="A_sd",
    )

    return 0



