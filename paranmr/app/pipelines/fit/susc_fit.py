# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Suturina Group

"""Fit susceptibility tensors to experimental shift data.

Loads inputs, fits a selected susceptibility model, and writes outputs and plots.
"""

import copy
import logging
import os

import numpy as np
from pathos import multiprocessing as mp

# Application layer
from paranmr.app.loaders.dia_load import load_diamagnetic_shifts
from paranmr.app.loaders.elstate_load import load_electronic_state
from paranmr.app.loaders.exp_load import load_experiments, save_experiment
from paranmr.app.loaders.hfc_load import load_hyperfines
from paranmr.app.loaders.labels_load import load_signal_labels_from_csv
from paranmr.app.loaders.mol_load import load_base_molecule
from paranmr.app.loaders.paramag_centre_load import load_paramagnetic_centre
from paranmr.app.loaders.sh_load import load_g_tensor_dft
from paranmr.app.params.options import FitSuscRunOptions
from paranmr.app.pipelines.fit.linewidth_r6 import resolve_r6_linewidth_inputs
from paranmr.app.pipelines.fit.vt_fit import fit_vt
from paranmr.app.policies.assignment import resolve_assignment_search_settings
from paranmr.app.policies.hfc import has_missing_selected_signal_labels
from paranmr.app.policies.output_linewidth import resolve_output_linewidths
from paranmr.app.policies.susc import resolve_susc_fit_variables

# Core / domain
from paranmr.core.domain.exp import Experiment
from paranmr.core.domain.mol import Molecule
from paranmr.core.domain.tensor import Hyperfine
from paranmr.core.conv.freq_to_ppm import signal_widths_hz_to_ppm
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.fitting.susceptibility.models.isoaxrho import IsoAxRhoFitter
from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)
from paranmr.core.fitting.susceptibility.models.split import SplitFitter
from paranmr.core.fitting.susceptibility.assignment.hungarian import (
    fit_with_hungarian_assignment,
)
from paranmr.core.fitting.susceptibility.assignment.permutations import (
    generate_assignment_permutations,
)
from paranmr.core.fitting.susceptibility.fitters.moments import fit_model_to_moments
from paranmr.core.fitting.susceptibility.fitters.shifts import fit_model_to_shifts
from paranmr.core.pcs.isosurf import compute_pcs_isosurface

# IO layer
from paranmr.io.csv.fit import save_moment_fit_diagnostics
from paranmr.io.csv.mol import save_molecule_to_csv
from paranmr.io.csv.susc import save_susc
from paranmr.io.cube.pcs_iso_write import write_pcs_cube
from paranmr.io.xyz import xyz_write
from paranmr.viz.plots.fitted_shifts import plot_fitted_shifts

# Visualisation
from paranmr.viz.plots.shifts import plot_shift_contrib, plot_shift_spread
from paranmr.viz.plots.spect import plot_pred_spectrum
from paranmr.viz.style.theme import apply_profile

logger = logging.getLogger(__name__)


def run_fit_susc(config, options: FitSuscRunOptions | None = None) -> int:
    """Fit susceptibility tensor(s) defined by a YAML configuration file.

    The pipeline builds a Molecule from the requested hyperfine source, loads
    experimental data, fits the chosen susceptibility model, generates plots, and
    writes outputs into the project directory.

    Args:
        config: FitSuscConfig loaded from YAML.
        options: Runtime options supplied by the CLI.

    Returns:
        Exit code: 0 on success.
    """
    if options is None:
        raise ValueError("FitSuscRunOptions is required")

    delimiter = options.runtime.csv_delimiter

    # Build the resolved plotting contract once per run
    spec = apply_profile(options.runtime.plot_profile)

    # Make output directory and file
    os.makedirs(config.project_name, exist_ok=True)

    # Load Molecule
    base_molecule = load_base_molecule(config)

    # Load canonical paramagnetic centre into the molecule domain container
    base_molecule = load_paramagnetic_centre(
        molecule=base_molecule,
        paramagnetic_centre=config.hyperfine_paramagnetic_centre,
    )

    # Load DFT g-tensor (if available)
    base_molecule.sh.g_tensor_dft = load_g_tensor_dft(
        config=config,
    )

    # Load Hyperfines
    base_molecule = load_hyperfines(
        molecule=base_molecule,
        config=config,
    )

    # Load electronic state
    base_molecule.electronic = load_electronic_state(
        spin_S=config.spin_S,
        orbit_L=config.orbit,
        total_J=config.total_momentum_J,
        hyperfine_file=config.hyperfine_file,
        hyperfine_method=config.hyperfine_method,
    )

    # Apply rotation matrix to all hyperfine tensors if requested
    if len(config.hyperfine_rotate):
        _rot_a = np.loadtxt(config.hyperfine_rotate)
        base_molecule.rotate_hyperfines(_rot_a)

    # Load experiments
    experiments = load_experiments(config.experiment_files)

    # Load diamagnetic shift file
    if len(config.diamagnetic_file):
        load_diamagnetic_shifts(
            file_name=config.diamagnetic_file,
            file_type=config.diamagnetic_method,
            ref_file_name=config.diamagnetic_ref_file,
            ref_file_type=config.diamagnetic_ref_method,
        )

    # Add signal labels for assignment-based workflows before any assignment
    # or shift-fit logic consumes `molecule.nuclei`.
    if config.assignment_method != "moments" and len(config.signal_labels_file):
        try:
            al_to_sl, al_to_sml = load_signal_labels_from_csv(config.signal_labels_file)
            if has_missing_selected_signal_labels(base_molecule, al_to_sl):
                logger.warning(
                    "Signal labels file does not define labels for all selected "
                    "nuclei; missing labels will use atom labels."
                )
            base_molecule.apply_signal_labels(al_to_sl, al_to_sml)
        except ValueError as err:
            raise ValueError(f"{err}\nCheck signal_labels and hyperfine files.")
        except KeyError as err:
            # treat missing labels/keys as a user input error
            raise ValueError(str(err))

    # Rotationally average hyperfines
    if len(config.hyperfine_average):
        base_molecule.average_hyperfine(config.hyperfine_average)

    # Check the number of experiments is consistent across the files
    # and issue warning if not
    if len(np.unique([len(exp.signals) for exp in experiments])) > 1:
        logger.warning("Some experiments have more signals than others!")

    # Create a molecule object to accompany each experiment object
    molecules = [copy.deepcopy(base_molecule) for _ in range(len(experiments))]

    name_to_susc_fit: dict[str, SusceptibilityModel] = {
        "split": SplitFitter,
        "isoaxrho": IsoAxRhoFitter,
        "isoaxrho_euler": IsoAxRhoEulerFitter,
    }

    model_to_use = name_to_susc_fit[config.susc_fit_type]

    # Create one susceptibility model per molecule/experiment pair. Reduced
    # input units depend on experiment temperature, so normalization happens
    # per experiment here rather than once at config-load time.
    susc_models: list[SusceptibilityModel] = []
    for experiment in experiments:
        fit_vars, fix_vars = resolve_susc_fit_variables(
            raw_variables=config.susc_fit_variables,
            input_units=config.susc_fit_input_units,
            temperature=experiment.temperature,
            spin=base_molecule.electronic.spin_S,
        )
        susc_models.append(model_to_use(fit_vars, fix_vars))

    if options.dry_run:
        logger.info("Dry run successful — no computations executed")
        return 0

    average_labels = []
    if config.assignment_method != "moments" and len(config.susc_fit_average_shifts):
        if "all" in config.susc_fit_average_shifts:
            config.susc_fit_average_shifts = list(
                {nuc.signal_label for nuc in base_molecule.nuclei}
            )
        average_labels = [
            [nuc.label for nuc in base_molecule.nuclei if nuc.signal_label == _cl]
            for _cl in config.susc_fit_average_shifts
        ]

    # Shift terms for plots
    # does not affect fit!
    _terms = ["pc", "fc", "d"]
    if config.hyperfine_method == "pdip":
        _terms.pop(_terms.index("fc"))
    if not config.diamagnetic_file:
        _terms.pop(_terms.index("d"))

    fitted_molecules: list[Molecule] = []
    fitted_susc_models: list[SusceptibilityModel] = []

    # Run fit for all experiments
    for molecule, susc_model, experiment in zip(molecules, susc_models, experiments):
        model_already_fitted = False

        # If permuting assignments, then first
        # run all assignment permutations to find best one
        if config.assignment_method == "permute":
            # If no permutation groups provided, permute all
            if not len(config.assignment_groups):
                config.assignment_groups = [
                    list({nuc.signal_label for nuc in molecule.nuclei})
                ]
            # For the current experiment, generate a new set in which
            # the assignment is permuted according to user defined groups
            permed_assignments = generate_assignment_permutations(
                experiment=experiment, groups=config.assignment_groups
            )

            logger.info("There are %s possible permutations", len(permed_assignments))

            # For each permutation, fit tensor and store r2_adjusted

            # Number of threads
            if config.num_threads == "auto":
                num_threads = mp.cpu_count() - 1
            else:
                num_threads = config.num_threads

            if num_threads > len(permed_assignments):
                num_threads = len(permed_assignments)

            # Create parallel pool
            pool = mp.Pool(num_threads)
            logger.info(
                "Parallel permutation search: %s worker processes",
                num_threads,
            )

            echo_r2 = options.runtime.echo_r2

            iterables = [
                (
                    molecule,
                    permed_assgn,
                    susc_model,
                    copy.deepcopy(experiment),
                    average_labels,
                    echo_r2,
                )
                for permed_assgn in permed_assignments
            ]

            # Calculate each assignment's r2 in parallel
            results = pool.starmap(_obtain_r2a, iterables)

            # Close Pool and let all the processes complete
            pool.close()
            pool.join()

            # Find assignment with largest r2
            # and use in subsequent (re)fitting
            assignment = permed_assignments[np.nanargmax(results)]
            opt_r2 = np.nanmax(results)
            logger.info("Optimal assignment with adj R² = %.6f", opt_r2)

            # and swap in new, permuted, assignments
            for it, new in enumerate(assignment):
                experiment.signals[it].signal_label = new

            # Save assigned experiment to file
            save_experiment(
                experiment,
                file_name=os.path.join(
                    config.project_name,
                    f"assigned_experiment_{experiment.temperature:.2f}_K.csv",
                ),
                delimiter=delimiter,
                comment=(
                    f"Optimal Assignment\n"
                    f"r2 = {opt_r2:f}\n"
                    f"T = {experiment.temperature:.2f} K"
                ),
            )

        elif config.assignment_method == "moments":
            observed_widths_ppm = signal_widths_hz_to_ppm(experiment)
            linewidth_inputs = resolve_r6_linewidth_inputs(
                molecule=molecule,
                isotope_filter=experiment.isotope,
                variables=config.linewidth_variables,
                label_kind="atom_label",
            )
            moment_fit_result = fit_model_to_moments(
                model=susc_model,
                molecule=molecule,
                centers_ppm=np.asarray(
                    [signal.shift for signal in experiment.signals],
                    dtype=float,
                ),
                widths_ppm=observed_widths_ppm,
                areas=np.asarray([signal.area for signal in experiment.signals], dtype=float),
                temperature=experiment.temperature,
                moment_objective=config.assignment_moment_objective,
                linewidth_mean_inv_r6_by_label=linewidth_inputs.mean_inv_r6_by_label,
                linewidth_variables=config.linewidth_variables,
            )
            if moment_fit_result is not None:
                save_moment_fit_diagnostics(
                    diagnostics=moment_fit_result,
                    file_name=os.path.join(
                        config.project_name,
                        "moment_fit_diagnostics_"
                        f"{experiment.temperature:.2f}_K.csv",
                    ),
                )
            model_already_fitted = True

        elif config.assignment_method == "hungarian":
            search_settings = resolve_assignment_search_settings(
                mode=config.assignment_search,
                n_attempts=config.assignment_n_attempts,
                max_iter=config.assignment_max_iter,
                r2_threshold=config.assignment_r2_threshold,
            )
            logger.info(
                "Hungarian search policy resolved: mode=%s, n_attempts=%d, "
                "max_iter=%d, r2_threshold=%.6f",
                search_settings.mode,
                search_settings.n_attempts,
                search_settings.max_iter,
                search_settings.r2_threshold,
            )

            # Call Hungarian assignment function
            opt_r2, assignment = fit_with_hungarian_assignment(
                molecule=molecule,
                susc_model=susc_model,
                experiment=experiment,
                average_labels=average_labels,
                n_attempts=search_settings.n_attempts,
                max_iter=search_settings.max_iter,
                r2_threshold=search_settings.r2_threshold,
            )
            logger.info("Hungarian completed: best R² = %.6f", opt_r2)

            # Save assigned experiment to file
            save_experiment(
                experiment,
                file_name=os.path.join(
                    config.project_name,
                    f"assigned_experiment_{experiment.temperature:.2f}_K.csv",
                ),
                delimiter=delimiter,
                comment=(
                    f"# Optimal Assignment (Hungarian)\n"
                    f"# r2 = {opt_r2:f}\n"
                    f"# T = {experiment.temperature:.2f} K"
                ),
            )

        # Fit susceptibility model to experimental chemical shifts.
        if not model_already_fitted:
            fit_model_to_shifts(
                model=susc_model,
                molecule=molecule,
                experiment=experiment,
                average_labels=average_labels,
            )

        # Skip if fit fails
        if not susc_model.fit_status:
            continue

        # Update susceptibility tensor of Molecule using model
        molecule.susc = susc_model.tosusceptibility()

        # Calculate shifts using new susceptibility tensor
        molecule.calculate_shifts()
        molecule.average_shifts()

        if not config.assignment_method == "moments":
            with spec.context():
                plot_fitted_shifts(
                    molecule,
                    experiment,
                    susc_model,
                    spec=spec,
                    show=options.runtime.show_plots,
                    susc_units=options.susc_units,
                    average=len(config.susc_fit_average_shifts),
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"shifts_{experiment.temperature:.2f}_K",
                    ),
                    verbose=True,
                    window_title=f"Fitted shifts at {experiment.temperature:.2f} K",
                )

            with spec.context():
                plot_shift_spread(
                    molecule,
                    experiment,
                    spec=spec,
                    terms=_terms,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"shift_spread_{molecule.susc.temperature:.2f}_K",
                    ),
                    verbose=True,
                    window_title=(
                        f"Spread of predicted shift components "
                        f"at {experiment.temperature:.2f} K"
                    ),
                    order="descending",
                )

            with spec.context():
                plot_shift_contrib(
                    molecule,
                    experiment,
                    spec=spec,
                    terms=_terms,
                    show=options.runtime.show_plots,
                    save=True,
                    save_name=os.path.join(
                        config.project_name,
                        f"mean_components_{experiment.temperature:.2f}_K",
                    ),
                    verbose=True,
                    window_title=(
                        f"Predicted shift components at {experiment.temperature:.2f} K"
                    ),
                    order="descending",
                )

        fitted_molecules.append(molecule)
        fitted_susc_models.append(susc_model)

    # Write shift data to file
    if not fitted_molecules:
        raise RuntimeError(
            "All susceptibility fits failed; no fitted susceptibility tensor is "
            "available to write. Check the optimizer status and moment objective "
            "diagnostics above."
        )

    _comment_base = f"Hyperfines from file {config.hyperfine_file}\n"
    if len(config.diamagnetic_file):
        _comment_base += f"Diamagnetic shifts from file {config.diamagnetic_file}\n"
    if len(config.diamagnetic_ref_file):
        _comment_base += (
            f"Diamagnetic reference from file {config.diamagnetic_ref_file}\n"
        )

    for molecule in fitted_molecules:
        comment = _comment_base + f"T = {molecule.susc.temperature:.2f} K"
        save_molecule_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                f"hyperfines_and_fitted_shifts_{molecule.susc.temperature:.2f}_K.csv",
            ),
            delimiter=delimiter,
            comment=comment,
            verbose=True,
        )

    # Write susceptibility tensor with model terms
    save_susc(
        fitted_molecules,
        os.path.join(config.project_name, "susceptibility_tensor.csv"),
        susc_models=fitted_susc_models,
        susc_units=options.susc_units,
    )

    if options.pcs_isosurface:
        for molecule in fitted_molecules:
            # Generate and save PCS isosurface
            molecule.susc.calc_irred()

            labels_arr = np.asarray(molecule.labels)
            coords_arr = np.asarray(molecule.coords, dtype=float)

            center_atom = molecule.labels[0]
            center_idx = np.where(labels_arr == center_atom)[0]
            if center_idx.size == 0:
                raise ValueError(f"Center atom {center_atom} not found in labels")

            coords_bohr = coords_arr * 1.88973
            coords_bohr = coords_bohr - coords_bohr[center_idx[0]]

            values, origin_bohr, step_bohr, grid_shape = compute_pcs_isosurface(
                chi_dtensor=molecule.susc.dtensor,
                labels=labels_arr,
                center_atom=center_atom,
                pdip_fn=Hyperfine.calc_pdip,
            )

            file_name = os.path.join(
                config.project_name,
                f"pcs_isosurf_{molecule.susc.temperature:.2f}_K.cube",
            )

            write_pcs_cube(
                file_name=file_name,
                comment=f"PCS Isosurface (T = {molecule.susc.temperature:.2f} K)",
                labels=labels_arr,
                coords_bohr=coords_bohr,
                origin_bohr=origin_bohr,
                step_bohr=step_bohr,
                grid_shape=grid_shape,
                values=values,
            )

            logger.info("PCS isosurface written to %s", file_name)

    mol = molecules[-1]

    shift_range = [
        np.min([nuc.shift.avg for nuc in mol.nuclei]),
        np.max([nuc.shift.avg for nuc in mol.nuclei]),
    ]

    extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]

    shift_range = [
        shift_range[0] + np.negative(np.max(extras)),
        shift_range[1] + np.positive(np.max(extras)),
    ]
    linewidth_output = resolve_output_linewidths(mol, shift_range)

    if base_molecule.electronic.spin_S is not None:
        temps_fit = np.array([mol.susc.temperature for mol in molecules], dtype=float)
        if temps_fit.size > 1:
            fit_vt(
                config=config,
                molecules=molecules,
                spin=base_molecule.electronic.spin_S,
                susc_models=susc_models,
                plot_profile=options.runtime.plot_profile,
                show_plots=options.runtime.show_plots,
            )

    with spec.context():
        plot_pred_spectrum(
            mol,
            isotope=mol.nuclei[0].isotope,
            shift_range=shift_range,
            spec=spec,
            effective_linewidths_by_label=linewidth_output.values_by_label,
            save=True,
            show=options.runtime.show_plots,
            save_name=os.path.join(
                config.project_name,
                f"pred_spectrum_{molecule.susc.temperature:.2f}_K",
            ),
        )

    # Save xyz file with signal labels for chemcraft
    if len(config.signal_labels_file):
        xyz_write.save_chemcraft_xyz(
            file_name=os.path.join(config.project_name, "chemcraft_structure.xyz"),
            labels=base_molecule.labels,
            coords=base_molecule.coords,
            signal_labels={nuc.label: nuc.signal_label for nuc in base_molecule.nuclei},
        )

    # Save xyz file with signal labels for chemcraft
    xyz_write.save_xyz(
        file_name=os.path.join(config.project_name, "structure.xyz"),
        labels=base_molecule.labels,
        coords=base_molecule.coords,
        comment=f"Structure from {config.hyperfine_file}",
    )

    return 0


def _moment_observed_centers_minus_signal_label_dia(
    experiment: Experiment,
    dia_by_signal_label: dict[str, float],
) -> np.ndarray:
    """Return moment-fit observed centers corrected by signal-label dia."""

    centers = []
    for signal in experiment.signals:
        try:
            dia_shift = dia_by_signal_label[signal.signal_label]
        except KeyError as exc:
            raise KeyError(
                "Cannot find signal label "
                f"{signal.signal_label!r} in diamagnetic shift mapping"
            ) from exc
        centers.append(float(signal.shift) - float(dia_shift))
    return np.asarray(centers, dtype=float)


def _obtain_r2a(
    molecule: Molecule,
    assignment: list[str],
    model: SusceptibilityModel,
    experiment: Experiment,
    average_labels: list[list[str]],
    echo_r2: bool,
):
    """
    Fit a susceptibility model for a proposed assignment and return adjusted R^2.

    This helper is designed to be run in parallel when searching over assignment
    permutations.

    Args:
        molecule (Molecule): Molecule instance used for shift prediction.
        assignment (list[str]): Proposed assignment list (one per signal).
        model: Model instance to fit.
        experiment (Experiment): Experiment data to fit against.
        average_labels (list[list[str]]): Groups of labels to average during fitting.

    Returns:
        float: Adjusted R^2 value for this assignment.
    """

    # and swap in new, permuted, assignments
    for it, new in enumerate(assignment):
        experiment.signals[it].signal_label = new

    # Fit susceptibility model to experimental chemical shifts
    fit_model_to_shifts(
        model=model,
        molecule=molecule,
        experiment=experiment,
        average_labels=average_labels,
    )

    return model.adj_r2
