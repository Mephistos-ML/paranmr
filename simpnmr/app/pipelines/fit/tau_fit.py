# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit correlation-time parameters to experimental R1 data.

Loads experiments and hyperfine data, evaluates relaxation models, and fits
correlation times using non-linear least squares.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

from simpnmr.app.loaders.elstate_load import load_electronic_state
from simpnmr.app.loaders.exp_load import load_experiments
from simpnmr.app.loaders.hfc_load import load_hyperfines
from simpnmr.app.loaders.labels_load import load_chem_labels_from_csv
from simpnmr.app.loaders.mol_load import load_base_molecule
from simpnmr.app.params.options import FitCorrTimeRunOptions
from simpnmr.core.const.gammas import NUCLEAR_GAMMAS
from simpnmr.core.const.physics import EGAMMA
from simpnmr.core.conv.ang_to_freq import angstrom_to_mhz
from simpnmr.core.relaxation.eval import evaluate_relaxation_rates
from simpnmr.core.util.strings import remove_numbers
from simpnmr.io.csv.relax import save_corr_time_fit_data
from simpnmr.io.xyz import xyz_write
from simpnmr.viz.plots.corr_time import plot_corr_time_by_label, plot_corr_time_scatter
from simpnmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_fit_corr_time(config, options: FitCorrTimeRunOptions | None = None) -> int:
    """Fit correlation-time parameters to experimental R1 values.

    This pipeline reads experiments and hyperfine data, evaluates the requested
    relaxation model, and uses `scipy.optimize.curve_fit` to fit correlation times.

    Args:
        config: FitCorrTimeConfig loaded from YAML.

    Returns:
        Exit code: 0 on success.
    """

    if options is None:
        raise ValueError("FitCorrTimeRunOptions is required")

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Build the resolved plotting contract for this run.
    spec = apply_profile(options.runtime.plot_profile)

    tau_R_mode, tau_R_guess = (
        config.fit_corr_time_tau_R[0].lower(),
        config.fit_corr_time_tau_R[1],
    )
    tau_R_bounds = (
        config.fit_corr_time_tau_R[2] if len(config.fit_corr_time_tau_R) > 2 else None
    )

    tau_E_mode, tau_E_guess = (
        config.fit_corr_time_tau_E[0].lower(),
        config.fit_corr_time_tau_E[1],
    )
    tau_E_bounds = (
        config.fit_corr_time_tau_E[2] if len(config.fit_corr_time_tau_E) > 2 else None
    )

    if tau_R_mode == "fix" and tau_E_mode == "fit":
        fix_param = "tau_r"
    elif tau_R_mode == "fit" and tau_E_mode == "fix":
        fix_param = "tau_e"
    elif tau_R_mode == "fit" and tau_E_mode == "fit":
        fix_param = None  # Fit both
    elif tau_R_mode == "fix" and tau_E_mode == "fix":
        raise ValueError(
            "Both tau_R and tau_E cannot be fixed. At least one must be set to 'fit'."
        )
    else:
        raise ValueError(
            "Use syntax 'tau_C: [fit/fix, guess, [upper-bound, lower-bound]]', "
            "with bounds optional (tau_C refers to tau_R or tau_E)."
        )

    # Placeholders for fitted parameters and covariance
    tau_R_fit = None
    tau_E_fit = None
    pcov = None
    initial_guess = None

    if (
        getattr(config, "fit_corr_time_tau_R", None) is not None
        and getattr(config, "relaxation_model", None) is not None
    ):
        experiments = load_experiments(config.experiment_files)

        # Filter signals to only those with valid R1 values
        # Only include signals for specified elements (e.g., 'C')

        elements = (
            config.nuclei_include
            if isinstance(config.nuclei_include, list)
            else [config.nuclei_include]
        )

        exp_blocks = []
        for experiment in experiments:
            labels_this = []
            r1_this = []
            for signal in experiment.signals:
                if (
                    signal.r1 is not None
                    and np.isfinite(signal.r1)
                    and any(signal.assignment.startswith(e) for e in elements)
                ):
                    labels_this.append(signal.assignment)
                    r1_this.append(signal.r1)
            if len(labels_this) > 0:
                exp_blocks.append(
                    (experiment, np.array(labels_this), np.array(r1_this))
                )

        if not exp_blocks:
            raise ValueError("No valid experimental R1 values found for fitting.")

        chem_labels = np.concatenate([blk[1] for blk in exp_blocks])
        exp_r1 = np.concatenate([blk[2] for blk in exp_blocks])
        xdata = np.arange(len(exp_r1))

        # Load Molecule
        base_molecule = load_base_molecule(config)

        # Load Hyperfines
        base_molecule = load_hyperfines(
            molecule=base_molecule,
            config=config,
        )

        # Add chemical labels if provided
        if len(config.chem_labels_file):
            al_to_cl, al_to_cml = load_chem_labels_from_csv(config.chem_labels_file)
            base_molecule.apply_chem_labels(al_to_cl, al_to_cml)
        label_to_chem_label = {
            nuc.label: nuc.chem_label for nuc in base_molecule.nuclei
        }

        # Prepare relaxation model inputs
        nuclei_coords = {nuc.label: nuc.coord for nuc in base_molecule.nuclei}
        # TODO(domain): Replace relaxation-specific electron coordinates
        # with a domain-level paramagnetic centre entity.
        paramagnetic_centre = config.hyperfine_paramagnetic_centre

        # Dictionaries for relaxation calculations
        A_fc_dict = {
            nuc.label: float(
                angstrom_to_mhz(
                    np.trace(nuc.A.fc) / 3.0,
                    NUCLEAR_GAMMAS[remove_numbers(nuc.label)],
                )
            )
            * 1e6
            for nuc in base_molecule.nuclei
            if nuc.A is not None
        }
        gamma_I_dict = {
            label: NUCLEAR_GAMMAS[remove_numbers(label)] * 2 * np.pi * 1e6
            for label in nuclei_coords
        }

        # Load electronic state
        base_molecule.electronic = load_electronic_state(
            spin_S=config.spin_S,
            orbit_L=config.orbit,
            total_J=config.total_momentum_J,
            hyperfine_file=config.hyperfine_file,
            hyperfine_method=config.hyperfine_method,
        )
        spin = base_molecule.electronic.spin_S
        orbit = base_molecule.electronic.orbit_L
        total_momentum_J = base_molecule.electronic.total_J

        # --- Model function for curve_fit ---
        # TODO(core): Collapse duplicated tau-fit theory evaluation
        # into a single reusable R1 model function.
        if fix_param == "tau_r":
            tau_R = float(tau_R_guess)
            initial_guess = [float(tau_E_guess)]

            def r1_model(_, tau_E):
                """Compute model R1 values for the current tau_E with tau_R fixed."""
                tau_c1 = 1.0 / ((1.0 / tau_R) + (1.0 / tau_E))
                tau_c2 = tau_c1

                theory_all = []

                for experiment, labels_this, _r1_this in exp_blocks:
                    b0 = experiment.magnetic_field
                    temperature = experiment.temperature
                    omega_I_dict = {
                        label: -gamma_I_dict[label] * b0 for label in nuclei_coords
                    }
                    omega_S = -EGAMMA * b0 * 2 * np.pi * 1e6

                    rates_r1, _ = evaluate_relaxation_rates(
                        relaxation_model=config.relaxation_model,
                        nuclei_coords=nuclei_coords,
                        electron_coords=paramagnetic_centre,
                        gamma_I_dict=gamma_I_dict,
                        omega_I_dict=omega_I_dict,
                        omega_S=omega_S,
                        spin=spin,
                        orbit=orbit,
                        total_momentum_J=total_momentum_J,
                        A_iso_dict=A_fc_dict,
                        temperature=temperature,
                        tau_R=tau_R,
                        tau_c1=tau_c1,
                        tau_c2=tau_c2,
                        tau_e2=tau_E,
                        compute_r1=True,
                        compute_r2=False,
                    )

                    if rates_r1 is None:
                        raise ValueError(
                            "Shared relaxation evaluator returned no R1 rates"
                        )

                    r1_by_chem_label = defaultdict(list)
                    for nuc in base_molecule.nuclei:
                        if nuc.label in rates_r1:
                            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

                    avg_r1_by_chem_label = {
                        chem_label: np.mean(rate_list)
                        for chem_label, rate_list in r1_by_chem_label.items()
                    }

                    for label in labels_this:
                        chem_label = label_to_chem_label.get(label, label)
                        theory_all.append(avg_r1_by_chem_label.get(chem_label, np.nan))

                return np.array(theory_all)

            # --- Run the fit ---
            if tau_E_bounds:
                popt, pcov = curve_fit(
                    r1_model, xdata, exp_r1, p0=initial_guess, bounds=tau_E_bounds
                )
            elif tau_E_bounds is None:
                popt, pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

            tau_E_fit = popt[0]
            theory_r1 = r1_model(xdata, tau_E_fit)
            if tau_E_fit <= 0:
                raise ValueError(f"Fitted tau_E is negative: {tau_E_fit:.3e} s.")

        elif fix_param == "tau_e":
            tau_E = float(tau_E_guess)
            initial_guess = [float(tau_R_guess)]

            def r1_model(_, tau_R):
                """Compute model R1 values for the current tau_R with tau_E fixed."""
                tau_c1 = 1.0 / ((1.0 / tau_R) + (1.0 / tau_E))
                tau_c2 = tau_c1

                theory_all = []

                for experiment, labels_this, _r1_this in exp_blocks:
                    b0 = experiment.magnetic_field
                    temperature = experiment.temperature
                    omega_I_dict = {
                        label: -gamma_I_dict[label] * b0 for label in nuclei_coords
                    }
                    omega_S = -EGAMMA * b0 * 2 * np.pi * 1e6

                    rates_r1, _ = evaluate_relaxation_rates(
                        relaxation_model=config.relaxation_model,
                        nuclei_coords=nuclei_coords,
                        electron_coords=paramagnetic_centre,
                        gamma_I_dict=gamma_I_dict,
                        omega_I_dict=omega_I_dict,
                        omega_S=omega_S,
                        spin=spin,
                        orbit=orbit,
                        total_momentum_J=total_momentum_J,
                        A_iso_dict=A_fc_dict,
                        temperature=temperature,
                        tau_R=tau_R,
                        tau_c1=tau_c1,
                        tau_c2=tau_c2,
                        tau_e2=tau_E,
                        compute_r1=True,
                        compute_r2=False,
                    )

                    if rates_r1 is None:
                        raise ValueError(
                            "Shared relaxation evaluator returned no R1 rates"
                        )

                    r1_by_chem_label = defaultdict(list)
                    for nuc in base_molecule.nuclei:
                        if nuc.label in rates_r1:
                            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

                    avg_r1_by_chem_label = {
                        chem_label: np.mean(rate_list)
                        for chem_label, rate_list in r1_by_chem_label.items()
                    }

                    for label in labels_this:
                        chem_label = label_to_chem_label.get(label, label)
                        theory_all.append(avg_r1_by_chem_label.get(chem_label, np.nan))

                return np.array(theory_all)

            # --- Run the fit ---
            if tau_R_bounds:
                popt, pcov = curve_fit(
                    r1_model, xdata, exp_r1, p0=initial_guess, bounds=tau_R_bounds
                )
            elif tau_R_bounds is None:
                popt, pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

            tau_R_fit = popt[0]
            theory_r1 = r1_model(xdata, tau_R_fit)
            if tau_R_fit <= 0:
                raise ValueError(f"Fitted tau_R is negative: {tau_R_fit:.3e} s.")
            else:
                logger.info("Fitted tau_R: %.3e s", tau_R_fit)

        elif not fix_param or fix_param in ["none", ""]:
            # Fit both tau_R and tau_E
            initial_guess = [float(tau_R_guess), float(tau_E_guess)]
            bounds = None
            if tau_R_bounds and tau_E_bounds:
                bounds = (
                    [tau_R_bounds[0], tau_E_bounds[0]],
                    [tau_R_bounds[1], tau_E_bounds[1]],
                )

            def r1_model(_, tau_R, tau_E):
                """Compute model R1 values for the current (tau_R, tau_E)."""
                tau_c1 = 1.0 / ((1.0 / tau_R) + (1.0 / tau_E))
                tau_c2 = tau_c1

                theory_all = []

                for experiment, labels_this, _r1_this in exp_blocks:
                    b0 = experiment.magnetic_field
                    temperature = experiment.temperature
                    omega_I_dict = {
                        label: -gamma_I_dict[label] * b0 for label in nuclei_coords
                    }
                    omega_S = -EGAMMA * b0 * 2 * np.pi * 1e6

                    rates_r1, _ = evaluate_relaxation_rates(
                        relaxation_model=config.relaxation_model,
                        nuclei_coords=nuclei_coords,
                        electron_coords=paramagnetic_centre,
                        gamma_I_dict=gamma_I_dict,
                        omega_I_dict=omega_I_dict,
                        omega_S=omega_S,
                        spin=spin,
                        orbit=orbit,
                        total_momentum_J=total_momentum_J,
                        A_iso_dict=A_fc_dict,
                        temperature=temperature,
                        tau_R=tau_R,
                        tau_c1=tau_c1,
                        tau_c2=tau_c2,
                        tau_e2=tau_E,
                        compute_r1=True,
                        compute_r2=False,
                    )

                    if rates_r1 is None:
                        raise ValueError(
                            "Shared relaxation evaluator returned no R1 rates"
                        )

                    r1_by_chem_label = defaultdict(list)
                    for nuc in base_molecule.nuclei:
                        if nuc.label in rates_r1:
                            r1_by_chem_label[nuc.chem_label].append(rates_r1[nuc.label])

                    avg_r1_by_chem_label = {
                        chem_label: np.mean(rate_list)
                        for chem_label, rate_list in r1_by_chem_label.items()
                    }

                    for label in labels_this:
                        chem_label = label_to_chem_label.get(label, label)
                        theory_all.append(avg_r1_by_chem_label.get(chem_label, np.nan))

                return np.array(theory_all)

            # --- Run the fit ---
            if bounds:
                popt, pcov = curve_fit(
                    r1_model, xdata, exp_r1, p0=initial_guess, bounds=bounds
                )
            else:
                popt, pcov = curve_fit(r1_model, xdata, exp_r1, p0=initial_guess)

            tau_R_fit, tau_E_fit = popt
            theory_r1 = r1_model(xdata, tau_R_fit, tau_E_fit)
            if tau_R_fit <= 0 and tau_E_fit > 0:
                raise ValueError(f"tau_R is negative: {tau_R_fit:.3e} s.")
            elif tau_E_fit <= 0 and tau_R_fit > 0:
                raise ValueError(f"tau_E is negative: {tau_E_fit:.3e} s.")
            elif tau_R_fit <= 0 and tau_E_fit <= 0:
                raise ValueError(
                    f"Both tau_R and tau_E are negative: "
                    f"tau_R = {tau_R_fit:.3e} s, tau_E = {tau_E_fit:.3e} s."
                )
        else:
            raise ValueError("Correlation times must be 'tau_r' or 'tau_e'.")

        rsquared = 1 - (
            np.sum((exp_r1 - theory_r1) ** 2) / np.sum((exp_r1 - np.mean(exp_r1)) ** 2)
        )

        # Write fit diagnostics data
        save_corr_time_fit_data(
            xdata=xdata,
            exp_r1=exp_r1,
            chem_labels=chem_labels,
            file_name=os.path.join(
                config.project_name, "corr_time_fit_diagnostics.csv"
            ),
            initial_guess=initial_guess,
            fitted_tau_r=tau_R_fit,
            fitted_tau_e=tau_E_fit,
            covariance=pcov,
            comment=f"r2: {rsquared:.6f}",
            verbose=True,
        )

        # Plot fit diagnostics data
        with spec.context():
            plot_corr_time_scatter(
                theory_r1=theory_r1,
                exp_r1=exp_r1,
                chem_labels=list(chem_labels),
                rsquared=rsquared,
                fix_param=fix_param,
                tau_R_fit=tau_R_fit,
                tau_E_fit=tau_E_fit,
                spec=spec,
                save=True,
                show=options.runtime.show_plots,
                save_name=os.path.join(
                    config.project_name, "experimental_vs_fitted_R1.pdf"
                ),
                verbose=True,
            )
            plot_corr_time_by_label(
                theory_r1=theory_r1,
                exp_r1=exp_r1,
                chem_labels=list(chem_labels),
                spec=spec,
                save=True,
                show=options.runtime.show_plots,
                save_name=os.path.join(config.project_name, "r1_fit_comparison.pdf"),
                verbose=True,
            )

        if len(config.chem_labels_file):
            xyz_write.save_chemcraft_xyz(
                file_name=os.path.join(config.project_name, "chemcraft_structure.xyz"),
                labels=base_molecule.labels,
                coords=base_molecule.coords,
                chem_labels={nuc.label: nuc.chem_label for nuc in base_molecule.nuclei},
            )

        xyz_write.save_xyz(
            file_name=os.path.join(config.project_name, "structure.xyz"),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
            comment=f"Structure from {config.hyperfine_file}",
        )

    else:
        raise ValueError(
            "fit_corr_time and relaxation_model must be specified in the input file."
        )

    return 0
