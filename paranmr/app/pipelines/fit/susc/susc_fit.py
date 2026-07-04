"""Fit susceptibility tensors to experimental shift data.

Loads inputs, fits a selected susceptibility model, and writes outputs and plots.
"""

import copy
import logging
import os

import numpy as np

# Application layer
from paranmr.app.loaders.dia_load import load_diamagnetic_shifts
from paranmr.app.loaders.elstate_load import load_electronic_state
from paranmr.app.loaders.exp_load import load_experiments
from paranmr.app.loaders.hfc_load import load_hyperfines
from paranmr.app.loaders.labels_load import load_signal_labels_from_csv
from paranmr.app.loaders.mol_load import load_base_molecule
from paranmr.app.loaders.paramag_centre_load import load_paramagnetic_centre
from paranmr.app.loaders.sh_load import load_g_tensor_dft
from paranmr.app.params.options import FitSuscRunOptions
from paranmr.app.pipelines.fit.vt_fit import fit_vt
from paranmr.app.pipelines.fit.susc.fixed import fit_assigned_shifts
from paranmr.app.pipelines.fit.susc.hungarian import fit_hungarian_assignment
from paranmr.app.pipelines.fit.susc.moments import fit_moment_assignment
from paranmr.app.pipelines.fit.susc.permute import fit_permuted_assignments
from paranmr.app.policies.hfc import has_missing_selected_signal_labels
from paranmr.app.policies.output_linewidth import resolve_output_linewidths
from paranmr.app.policies.susc import resolve_susc_fit_variables

# Core / domain
from paranmr.core.domain.mol import Molecule
from paranmr.core.domain.tensor import Hyperfine
from paranmr.core.fitting.susceptibility.models.base import SusceptibilityModel
from paranmr.core.fitting.susceptibility.models.isoaxrho import IsoAxRhoFitter
from paranmr.core.fitting.susceptibility.models.isoaxrho_euler import (
    IsoAxRhoEulerFitter,
)
from paranmr.core.fitting.susceptibility.models.split import SplitFitter
from paranmr.core.pcs.isosurf import compute_pcs_isosurface

# IO layer
from paranmr.io.csv.mol import save_molecule_to_csv
from paranmr.io.csv.peaks import save_peak_data_to_csv
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

    if len(config.diamagnetic_file):
        dia_by_key, key_kind, ref_avg_by_label_nn = load_diamagnetic_shifts(
            file_name=config.diamagnetic_file,
            file_type=config.diamagnetic_method,
            ref_file_name=config.diamagnetic_ref_file,
            ref_file_type=config.diamagnetic_ref_method,
        )
        base_molecule.apply_diamagnetic_shifts(
            dia_by_key=dia_by_key,
            key_kind=key_kind,
            ref_avg_by_label_nn=ref_avg_by_label_nn,
        )

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
    fitted_linewidth_outputs = []

    # Run fit for all experiments
    for molecule, susc_model, experiment in zip(molecules, susc_models, experiments):
        model_already_fitted = False
        fitted_linewidth_by_label: dict[str, float] | None = None
        fitted_linewidth_method: str | None = None

        # If permuting assignments, then first
        # run all assignment permutations to find best one
        if config.assignment_method == "permute":
            _opt_r2, _assignment = fit_permuted_assignments(
                molecule=molecule,
                experiment=experiment,
                model=susc_model,
                average_labels=average_labels,
                assignment_groups=config.assignment_groups,
                num_threads=config.num_threads,
                project_name=config.project_name,
                delimiter=delimiter,
            )

        elif config.assignment_method == "moments":
            moment_fit_result = fit_moment_assignment(
                model=susc_model,
                molecule=molecule,
                experiment=experiment,
                project_name=config.project_name,
                assignment_moment_objective=config.assignment_moment_objective,
                linewidth_variables=config.linewidth_variables,
            )
            if moment_fit_result is None:
                fitted_linewidth_by_label = None
            else:
                fitted_linewidth_by_label = (
                    moment_fit_result.calculated_linewidths_by_label
                )
                fitted_linewidth_method = moment_fit_result.linewidth_method
            model_already_fitted = True

        elif config.assignment_method == "hungarian":
            fit_hungarian_assignment(
                molecule=molecule,
                susc_model=susc_model,
                experiment=experiment,
                average_labels=average_labels,
                assignment_search=config.assignment_search,
                n_attempts=config.assignment_n_attempts,
                max_iter=config.assignment_max_iter,
                r2_threshold=config.assignment_r2_threshold,
                project_name=config.project_name,
                delimiter=delimiter,
            )

        # Fit susceptibility model to experimental chemical shifts.
        if not model_already_fitted:
            fit_assigned_shifts(
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

        shift_range = [
            np.min([nuc.shift.avg for nuc in molecule.nuclei]),
            np.max([nuc.shift.avg for nuc in molecule.nuclei]),
        ]
        extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]
        shift_range = [
            shift_range[0] + np.negative(np.max(extras)),
            shift_range[1] + np.positive(np.max(extras)),
        ]
        linewidth_output = resolve_output_linewidths(
            molecule,
            shift_range,
            experiment=experiment,
            explicit_linewidth_by_label=fitted_linewidth_by_label,
            explicit_column_name=(
                None
                if fitted_linewidth_by_label is None or fitted_linewidth_method is None
                else f"linewidth_{fitted_linewidth_method}_fit (ppm)"
            ),
        )

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
        fitted_linewidth_outputs.append(linewidth_output)

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

    for molecule, linewidth_output in zip(fitted_molecules, fitted_linewidth_outputs):
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
        save_peak_data_to_csv(
            molecule=molecule,
            file_name=os.path.join(
                config.project_name,
                f"peak_data_{molecule.susc.temperature:.2f}_K.csv",
            ),
            linewidth_by_label=linewidth_output.values_by_label,
            linewidth_column_name=linewidth_output.column_name,
            comment=f"T = {molecule.susc.temperature:.2f} K",
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

    mol = fitted_molecules[-1]
    linewidth_output = fitted_linewidth_outputs[-1]

    shift_range = [
        np.min([nuc.shift.avg for nuc in mol.nuclei]),
        np.max([nuc.shift.avg for nuc in mol.nuclei]),
    ]
    extras = [0.1 * abs(shift_range[0]), 0.1 * abs(shift_range[1])]
    shift_range = [
        shift_range[0] + np.negative(np.max(extras)),
        shift_range[1] + np.positive(np.max(extras)),
    ]

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
