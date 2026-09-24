# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Run the temperature-dependent susceptibility plotting workflow."""

from __future__ import annotations

import numpy as np

from simpnmr_x.app.params.options import PlotChiTRunOptions
from simpnmr_x.cfg.plot_chit import ChiTSourceConfig, PlotChiTConfig
from simpnmr_x.core.conv.a3_to_cm3mol import a3_to_cm3mol
from simpnmr_x.core.domain.tensor import canonical_principal_axes
from simpnmr_x.core.fitting.variable_temperatures.components import (
    calculate_E_D_components,
    compute_analytic_component,
    compute_curie_prefactor,
    compute_g_components,
    compute_g_sq_components,
    rotate_tensors_to_frame,
    validate_common_principal_axes,
)
from simpnmr_x.io.qc import gateway as rdrs
from simpnmr_x.viz.plots.susc import plot_chit_comparison
from simpnmr_x.viz.style.theme import apply_profile

_G_FRAME_ALIGNMENT_TOLERANCE = 1.0e-2


def run_plot_chit(config: PlotChiTConfig, options: PlotChiTRunOptions) -> int:
    """Load configured ORCA sources and render a χT comparison plot.

    Args:
        config: Validated ``plot_chit`` configuration.
        options: Runtime plotting options from the CLI.

    Returns:
        Zero after the plot has been rendered.
    """

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if config.xrd is not None:
        series["XRD"] = _read_chi_t_series(config.xrd)
    if config.opt is not None:
        opt_series = _read_chi_t_series(config.opt)
        series["OPT"] = opt_series
        if config.tip is not None:
            series["OPT without TIP"] = _remove_analytic_tip(
                config.opt,
                opt_series,
                config.tip.reference_temperature,
            )

    spec = apply_profile(options.runtime.plot_profile)
    with spec.context():
        plot_chit_comparison(
            series,
            spec,
            show=options.runtime.show_plots,
            save=True,
            save_name=config.output_file,
        )

    return 0


def _read_chi_t_series(
    source: ChiTSourceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = rdrs.read_orca_chi_t(source.file, source.section)
    temperatures = np.asarray(list(values.keys()), dtype=float)
    chi_t = np.asarray(list(values.values()), dtype=float)
    return temperatures, chi_t


def _remove_analytic_tip(
    source: ChiTSourceConfig,
    opt_series: tuple[np.ndarray, np.ndarray],
    reference_temperature: str | float,
) -> tuple[np.ndarray, np.ndarray]:
    temperatures, chi_t = opt_series
    reference_index = _reference_index(temperatures, reference_temperature)
    reference = float(temperatures[reference_index])

    tensors = rdrs.read_orca_susceptibility(source.file, source.section)
    tensor = _get_temperature_value(tensors, reference)
    _, chi_frame = canonical_principal_axes(tensor)

    g_tensor = rdrs.read_g_tensor_ab_initio(source.file, source.section)
    eff_h = rdrs.read_eff_hamiltonian_tensor(source.file, source.section)
    if g_tensor is None or eff_h is None:
        raise ValueError(
            "Analytic TIP removal requires both the ORCA g-tensor and "
            "effective Hamiltonian"
        )

    eff_h_frame, g_frame = rotate_tensors_to_frame(eff_h, g_tensor, chi_frame)
    validate_common_principal_axes(
        eff_h_frame,
        g_frame,
        _G_FRAME_ALIGNMENT_TOLERANCE,
    )

    spin = rdrs.read_orca_spin(source.file)
    g_components = compute_g_components(g_frame)
    g_components_sq = compute_g_sq_components(g_frame)
    D_J, E_J = calculate_E_D_components(eff_h_frame)
    analytic_chi = compute_analytic_component(
        "iso",
        np.asarray([reference]),
        g_components_sq,
        g_components,
        D_J,
        E_J,
        spin,
    )[0]

    prefactor = compute_curie_prefactor(spin)
    analytic_chi_t = a3_to_cm3mol(
        np.asarray([analytic_chi * reference * prefactor])
    )[0]
    tip_chi_t = float(chi_t[reference_index] - analytic_chi_t)

    return temperatures, chi_t - tip_chi_t


def _reference_index(
    temperatures: np.ndarray,
    reference_temperature: str | float,
) -> int:
    if reference_temperature == "max":
        return int(np.argmax(temperatures))

    matches = np.flatnonzero(
        np.isclose(temperatures, float(reference_temperature), rtol=0.0, atol=1.0e-8)
    )
    if matches.size == 0:
        raise ValueError(
            "TIP reference_temperature is not present in the OPT temperature grid"
        )
    return int(matches[0])


def _get_temperature_value(values: dict[float, np.ndarray], temperature: float):
    for value_temperature, value in values.items():
        if np.isclose(value_temperature, temperature, rtol=0.0, atol=1.0e-8):
            return value
    raise ValueError(
        "TIP reference_temperature is not present in the OPT tensor grid"
    )
