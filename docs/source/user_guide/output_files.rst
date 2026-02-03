.. _output_files:

Output files
============


All output files are written to the output directory defined in the ``project`` block of the input YAML file.

For susceptibility fitting workflows (``fit_susc``), the names of all generated files are printed to the terminal together with a short description. Depending on the selected options and workflow configuration, the following output files may be produced:


1. ``assigned_experiment_<TEMPERATURE>_K.csv``  
   Assigned experimental data at a given temperature.

   The file has the same format as the input experimental CSV, but with assignments applied.

2. ``dft_hyperfines.csv``  
   Raw hyperfine coupling constants extracted from a DFT calculation.

   Generated only when ``hyperfine: method: dft`` is selected.

3. ``hyperfines_and_shifts_<TEMPERATURE>_K.csv``  
   Combined hyperfine and shift data at a given temperature.

   Contains hyperfine coupling constants, chemical shifts, atomic coordinates, and chemical labels for each nucleus in the system.

4. ``pcs_isosurf_<TEMPERATURE>_K.cube``  
   Pseudocontact shift isosurface.

   Generated as a Gaussian cube file for visualisation of PCS fields in real space.

5. ``susceptibility_components_chi.png``  
   Temperature dependence of the magnetic susceptibility components.

   Shows :math:`\chi` versus :math:`T`.  
   Generated when more than one temperature is specified and ``--isoaxrho_plots on`` or ``--save`` is enabled.

6. ``susceptibility_components_chiT.png``  
   Temperature-scaled susceptibility components.

   Shows :math:`\chi T` versus :math:`T`.  
   Generated when more than one temperature is specified and ``--isoaxrho_plots on`` or ``--save`` is enabled.

7. ``susceptibility_tensor.csv``  
   Fitted magnetic susceptibility tensors.

   Contains susceptibility tensors as a function of temperature, standard deviations of fitted parameters, goodness-of-fit metrics (:math:`r^2`, :math:`r^2_\mathrm{adj}`, MAE), eigenvalues of the susceptibility tensor, and an eigenvector representation using Euler angles.
