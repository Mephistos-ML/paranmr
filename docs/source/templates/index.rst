Templates
=========


Downloadable YAML templates for common workflows. Edit the input file paths and run ``simpnmr``.
For parameter descriptions, see :doc:`Input Files <../user_guide/input_files>`.

.. toctree::
   :maxdepth: 1
   :hidden:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Prediction template

      Minimal configuration for pNMR prediction.

    :download:`Download predict.yml <../_downloads/templates/predict.yml>`

   .. grid-item-card:: Fit template

      Minimal configuration for susceptibility fitting.

    :download:`Download fit.yml <../_downloads/templates/fit.yml>`

Quick start
^^^^^^^^^^^

After downloading a template:

1. Update the file paths in the YAML.
2. Run the corresponding command:

.. code-block:: bash

   simpnmr predict predict.yml
   simpnmr fit_susc fit.yml