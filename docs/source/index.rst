Home
====

.. toctree::
   Theory <theory>
   Installation <installation>
   Workflows <workflows>
   Input files <input_files>
   Output files <output_files>
   Contributing <contributing>
   FAQ <faq> 
   Bugs <bugs>
   Authors and Citation <authors>
   License <https://gitlab.com/suturina-group/simpnmr/-/blob/main/LICENSE>
   Repository <https://gitlab.com/suturina-group/simpnmr>
   :maxdepth: 1
   :caption: Contents:
   :hidden:


``simpnmr`` is a Python package for analysing solution NMR data of paramagnetic metal complexes using computational chemistry.

It features two main subroutines for predicting 1D pNMR spectra and fitting pNMR data.

Each subroutine reads a YAML-based input file that specifies the input data and controls the workflow.

Additional functions that streamline workflows can also be called via the command-line interface (CLI).

.. note::
 ``simpnmr`` and this documentation are currently under active development. Please bear with us.

Quick start
^^^^^^^^^^^

After installing ``simpnmr`` (see :ref:`installation`), you can explore the command line interface:

.. code-block:: bash

    simpnmr -h
    simpnmr predict -h
    simpnmr fit_susc -h

As a minimal example using a YAML configuration file:

.. code-block:: bash

    simpnmr predict your_config.yml

For details of the available options and required input fields, see the :ref:`input_files` page.

Getting help
^^^^^^^^^^^^

If something does not work as expected, please:

- Check the :ref:`faq` for answers to common questions and error messages.
- See :doc:`Bugs <bugs>` for how to report issues or unexpected behaviour.
- If you would like to contribute fixes or new features, read :doc:`Contributing <contributing>`.

