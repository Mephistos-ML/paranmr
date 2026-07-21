.. _output_files:

Output files
============


All output files are written to the output directory defined in the ``project`` block of the input YAML file.

For susceptibility fitting workflows (``fit_susc``), the names of all generated files are printed to the terminal together with a short description. Depending on the selected options and workflow configuration, the following output files may be produced:


1. ``assigned_experiment_<TEMPERATURE>_K.csv``  
   Assigned experimental data at a given temperature.

   The file has the same format as the input experimental CSV, but with signal_labels applied.

2. ``dft_hyperfines.csv``  
   Raw hyperfine coupling constants extracted from a DFT calculation.

   Generated only when ``hyperfine: method: dft`` is selected.

3. ``hyperfines_and_shifts_<TEMPERATURE>_K.csv``  
   Combined hyperfine and shift data at a given temperature.

   Contains hyperfine coupling constants, chemical shifts, atomic coordinates, and signal labels for each nucleus in the system.

4. ``pcs_isosurf_<TEMPERATURE>_K.cube``  
   Pseudocontact shift isosurface.

   Generated as a Gaussian cube file for visualisation of PCS fields in real space.

5. ``susceptibility_components_chi.pdf``  
   Temperature dependence of the magnetic susceptibility components.

   Shows :math:`\chi` versus :math:`T`.  
   Generated when more than one temperature is specified and ``--isoaxrho_plots on`` or ``--save`` is enabled.

6. ``susceptibility_components_chiT.pdf``  
   Temperature-scaled susceptibility components.

   Shows :math:`\chi T` versus :math:`T`.  
   Generated when more than one temperature is specified and ``--isoaxrho_plots on`` or ``--save`` is enabled.

7. ``susceptibility_tensor.csv``  
   Fitted magnetic susceptibility tensors.

   Contains susceptibility tensors as a function of temperature, standard deviations of fitted parameters, goodness-of-fit metrics (:math:`r^2`, :math:`r^2_\mathrm{adj}`, MAE), eigenvalues of the susceptibility tensor, and an eigenvector representation using Euler angles.

8. ``linewidth_estimate_<TEMPERATURE>_K.csv``  
   Estimated ``r^-6`` linewidth-model parameters for a fixed-assignment fit.

   Generated only when ``linewidth: estimate: p1_p2`` is enabled. Reports
   the fitted global ``p1`` and ``p2`` coefficients together with the
   estimation RMSE in ppm.

9. ``moment_fit_diagnostics_<TEMPERATURE>_K.csv``  
   Normalized moment descriptors for moment-based susceptibility fitting.

   Generated for ``assignment: method: moments``. Reports the observed and
   calculated normalized descriptor vectors as ``m1_norm`` to ``mN_norm``,
   together with the objective type and final objective score. In the current
   definition, ``m1`` is the spectral mean and ``m2`` to ``mN`` are central
   moments of order two through ``N``. The normalized observed vector is
   therefore unity by definition, while the calculated normalized vector
   reports the component-wise ratios relative to the observed descriptor
   values. The
   reported ``score`` is the value of the configured moment objective itself:
   for ``ls`` it is the norm of the weighted normalized residual vector, while
   for ``gmm`` it is the norm implied by the current GMM weighting matrix.

10. ``linewidth_model_<TEMPERATURE>_K.csv``  
    Fitted ``r^-6`` linewidth-model parameters used in a moment-based fit.

    Generated for ``assignment: method: moments`` when the ``r6`` linewidth
    model is active. Reports the final fitted ``p1`` and ``p2`` values used to
    generate the calculated spectrum.
