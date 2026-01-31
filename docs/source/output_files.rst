.. _output_files:

Output files
============

All output files are written to the ``project`` directory specified in the input file. The name of each file created by ``fit_susc`` is printed to the screen
with a short description. For completeness, the following files may be created by ``fit_susc``:

1. ``assigned_experiment_<TEMPERATURE>_K.csv`` – If ``assignment: method: permute``: assigned experiment file at a given temperature, with the same format as the input experiment.
2. ``dft_hyperfines.csv`` – If ``hyperfine: method: dft``: raw hyperfine coupling constants from the DFT output file.
3. ``hyperfines_and_shifts_<TEMPERATURE>_K.csv`` – Hyperfine coupling constants, chemical shifts, coordinates, and labels of each atom in the system for the specified temperature.
4. ``pcs_isosurf_<TEMPERATURE>_K.cube`` – Pseudocontact shift isosurface cube file.
5. ``susceptibility_components_chi.png`` – :math:`\chi` vs :math:`\mathrm{T}` – if more than one temperature is specified and ``--isoaxrho_plots on`` or ``save``.
6. ``susceptibility_components_chiT.png`` – :math:`\chi \mathrm{T}` vs :math:`\mathrm{T}` – if more than one temperature is specified and ``--isoaxrho_plots on`` or ``save``.
7. ``susceptibility_tensor.csv`` – Susceptibility tensors as a function of temperature, along with the standard deviation of fitted parameters, :math:`r^2` and :math:`r^2_\mathrm{adj}`, the mean absolute error (MAE), eigenvalues of the susceptibility tensor, and an eigenvector representation using Euler angles.
