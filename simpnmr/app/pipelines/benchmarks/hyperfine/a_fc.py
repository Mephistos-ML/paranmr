# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Run A_fc benchmark workflows."""

import os

from simpnmr.app.params.options import BenchmarkAfcRunOptions
from simpnmr.app.pipelines.benchmarks.hyperfine.runner import (
    plot_hyperfine_benchmark_summary,
    plot_hyperfine_functional_max_summary,
)
from simpnmr.app.pipelines.benchmarks.hyperfine.sources import (
    group_loaded_sources_by_functional,
    load_hyperfine_benchmark_sources,
)
from simpnmr.core.benchmarks.hyperfine.a_fc import (
    summarize_a_fc_max_by_nucleus,
    summarize_a_fc_ranges_by_functional_and_nucleus,
)
from simpnmr.viz.plots.benchmarks import (
    plot_a_fc_functional_max_curve,
    plot_a_fc_spread,
)
from simpnmr.viz.style.theme import apply_profile


def run_benchmark_a_fc(config, options: BenchmarkAfcRunOptions | None = None) -> int:
    """Run the A_fc benchmark workflow from a YAML configuration."""
    if options is None:
        raise ValueError("BenchmarkAfcRunOptions is required")

    os.makedirs(config.project_name, exist_ok=True)
    spec = apply_profile(options.runtime.plot_profile)

    if options.dry_run:
        return 0

    signals = load_hyperfine_benchmark_sources(config)

    a_fc_summary = summarize_a_fc_ranges_by_functional_and_nucleus(
        group_loaded_sources_by_functional(signals)
    )
    plot_hyperfine_benchmark_summary(
        a_fc_summary,
        output_dir=config.project_name,
        spec=spec,
        show=options.runtime.show_plots,
        plot_spread=plot_a_fc_spread,
        filename_metric="A_FC",
        window_metric="A_fc",
    )
    plot_hyperfine_functional_max_summary(
        summarize_a_fc_max_by_nucleus(
            a_fc_summary,
            max_label_tolerance=config.max_label_tolerance,
        ),
        output_dir=config.project_name,
        spec=spec,
        show=options.runtime.show_plots,
        plot_max_curve=plot_a_fc_functional_max_curve,
        filename_metric="A_FC",
        log_metric="A_fc",
        window_metric="A_fc",
    )

    return 0



