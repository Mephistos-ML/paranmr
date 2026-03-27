Using the examples
==================

This page explains how the downloadable example bundle is organised and how
each example is run.

Repository layout
^^^^^^^^^^^^^^^^^

The examples are organised as follows:

.. dropdown:: Show repository layout

   .. code-block:: text

      examples/
      ├── 01_Shift_Prediction
      │   ├── data/
      │   └── run.yml
      ├── 02_PCS_Isosurface
      │   ├── susceptibility.csv
      │   └── structure.xyz
      ├── 03_Shift_Prediction_With_Relaxation
      │   ├── data/
      │   └── run.yml
      ├── 04_Susceptibility_Fitting
      │   ├── data/
      │   └── run.yml
      ├── 05_VT_Susceptibility_Fitting
      │   ├── data/
      │   └── run.yml
      └── 06_Spin_Hamiltonian_Extraction
          └── isoaxrho_fit.csv

.. note::
    PCS Isosurface and Spin Hamiltonian Extraction are provided as CLI-driven
    examples and therefore do not follow the standard ``run.yml`` layout.

Running the examples
^^^^^^^^^^^^^^^^^^^^

Run each example from its own example directory.
The commands for the current examples are:

.. rubric:: 01. Shift Prediction

This example introduces the standard prediction workflow based on a ``run.yml``
configuration file and generated outputs.

.. code-block:: bash

   simpnmr predict run.yml

.. rubric:: 02. PCS Isosurface

This example shows how to generate a PCS isosurface directly from the command
line using susceptibility data, a structure file, and a selected paramagnetic
centre.

.. code-block:: bash

   simpnmr calc_pcs_iso susceptibility.csv 302.150 structure.xyz Dy1

.. note::

   In this command, ``susceptibility.csv`` provides the susceptibility data,
   ``302.150`` is the temperature, ``structure.xyz`` is the structure file,
   and ``Dy1`` selects the paramagnetic centre.

.. rubric:: 03. Shift Prediction with Relaxation

This example extends the prediction workflow to a case that also includes
relaxation-related outputs.

.. code-block:: bash

   simpnmr predict run.yml

.. rubric:: 04. Susceptibility Fitting

This example introduces the standard susceptibility fitting workflow using a
``run.yml`` configuration file.

.. code-block:: bash

   simpnmr fit_susc run.yml


.. rubric:: 05. Variable-Temperature Susceptibility Fitting

This example demonstrates the variable-temperature susceptibility fitting
workflow.

.. code-block:: bash

   simpnmr --hide fit_susc --susc_units 'cm3 mol-1' run.yml

.. note::

   The ``--hide`` option suppresses the interactive display of figures during
   execution, while still saving the generated outputs. The ``--susc_units``
   option allows the susceptibility units to be selected explicitly.

.. rubric:: 06. Spin Hamiltonian Extraction

This example shows how to extract a spin Hamiltonian directly from the
susceptibility fitting output.

.. code-block:: bash

   simpnmr get_sh --spin 2.0 isoaxrho_fit.csv

.. note::

   ``isoaxrho_fit.csv`` is the susceptibility fitting output used as input for
   spin Hamiltonian extraction.