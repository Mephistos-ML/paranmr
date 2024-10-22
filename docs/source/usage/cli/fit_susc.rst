Fitting Susceptibilities
========================

The ``fit_susc`` subprogram allows the user to obtain susceptibility tensor(s) from pNMR shifts and hyperfine coupling values for a given set of nuclei.

Input
-----

To run ``fit_susc`` simply type

.. code-block ::
    
    pnmr fit_susc <input_file>

where ``<input_file>`` is the name of your input file.

Input file format
^^^^^^^^^^^^^^^^^

The main input file for the ``fit_susc`` subprogram is a YAML (``.yml`` or ``.yaml``) file containing configuration options.

.. note::
    If you've never seen a YAML file before, then take a look at `this <https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started>`_ tutorial first.

The input file contains a series of keywords for which there are subkeywords and associated values - these are detailed in Table 1.

An example input file for ``fit_susc`` can be found in the ``simple_pnmr`` `repository <https://github.com/JonKragskow/simple_pnmr>`_, at ``examples/fit_susc/input.yml``.

.. table:: Table 1: ``fit_susc`` subprogram input file keywords and subkeywords.

    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | Keyword                     | Subkeyword              | Value                               | Description                                                             | Mandatory            |
    +=============================+=========================+=====================================+=========================================================================+======================+
    | ``project``                 | ``name``                | Directory Name                      | Name of directory to which all output files are written                 | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | If the named directory does not exist then it will be created           |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``method``              | ``dft``/``pdip``/``raw``            | Method for calculation of hyperfines                                    | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:dft`` specifies DFT calculated hyperfine values will be      |                      |
    |                             |                         |                                     |   used                                                                  |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:pdip`` specifies hyperfines will be calculated with the      |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |   point dipole approximation                                            |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:raw`` specifies ``.csv`` data entry - see Hyperfine Format   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``file``                | File Name                           | File containing hyperfine data and/or structure - Format depends on     | |:white_check_mark:| |
    |                             |                         |                                     | ``method``                                                              |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``            |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:pdip`` supports .xyz file                                    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``method:raw`` supports ``.csv`` file - see Hyperfine Format          |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``average``             | List of Chemical labels             | Specifies atoms for which hyperfines are averaged.                      | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | e.g. ``[Me1, tBu2]`` replaces hyperfines for atoms with chemical        |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | labels of ``Me1`` and ``tBu2`` with the average of all occurances of    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | each label.                                                             |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``hyperfine``               | ``pdip_centre``         | Atomic Label(s)                     | Atomic label(s) (including index) of atom on which unpaired electron    | |:x:|                |
    |                             |                         |                                     | resides.                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Used only when ``method:pdip``                                          |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``experiment``              | ``files``               | File Name(s)                        | File(s) containing experimental NMR peak data as ``.csv``.              | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <exp_csv>` for format information.                       |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``experiment``              | ``spectrum_files``      | File Name                           | File(s) containing experimental NMR spectrum as ``.csv``.               | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <exp_spectrum>` for format information.                  |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``chem_labels``             | ``file``                | File Name                           | File containing chemical labels, atom labels, and optionally            | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | and optionally chmeical math labels used for plotting.                  |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <chemlabels_csv>` Format for more information            |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``nuclei``                  | ``include``             | List of atom labels with indexing,  | Specifies nuclei for which shifts will be calculated                    | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | or just X where X is an atomic      |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | symbol and signifies all occurrences|                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | e.g. ``[all_C, all_H]``             |                                                                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic``             | ``method``              | ``dft``/``pdip``/``raw``            | Method for calculation of diamagnetic shifts                            | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` specifies DFT calculated shifts                       |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` specifies ``.csv`` data entry                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic``             | ``file``                | File Name                           | File containing hyperfine data - Format depends on ``method``           | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``           |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` supports ``.csv`` file - see :ref:`here <dia_csv>`.   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic``             | ``average``             | List of Chemical labels             | Specifies atoms for which diamagnetic shifts are averaged.              | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | e.g. ``[Me1, tBu2]`` replaces diamagnetic shifts for atoms with chemical|                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | labels of ``Me1`` and ``tBu2`` with the average of all occurrences of   |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | each label.                                                             |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic_reference``   | ``method``              | ``dft``/``pdip``/``raw``            | Method for calculation of reference diamagnetic shifts                  | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` specifies DFT calculated shifts                       |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` specifies ``.csv`` data entry                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``diamagnetic_reference``   | ``file``                | File Name                           | File containing reference diamagnetic data - Format depends on          | |:x:|                |
    |                             |                         |                                     | ``method``                                                              |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:dft`` supports Gaussian ``.log`` or Orca ``.out``           |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``method:raw`` supports ``.csv`` file - see :ref:`here <dia_csv>`.   |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susc_fit``                | ``type``                | ``split``/``isoaxrho``              | Form of susceptibility tensor to fit.                                   | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``split`` fits :math:`\chi_\mathrm{iso}`                             |                      |
    |                             |                         |                                     |    and :math:`\Delta\chi`                                               |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``isoaxrho`` fits :math:`\chi_\mathrm{iso}, \chi_\mathrm{ax}`,       |                      |
    |                             |                         |                                     |    and :math:`\chi_\mathrm{rho}`                                        |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See Fitters for more information                                        |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susc_fit``                | ``variables``           | List of                             | Specifies variables of model, whether each is fitted or fixed, and      | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | ``variable_name: [fit/fix, value]`` | the initial or fixed value of that variable in Å :sup:`3`               |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | e.g. ``iso [fit, 0.02]``            | See Fitters for more information                                        |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``assignment``              | ``method``              | ``fixed``/``permute``               | Specifies how to carry out assignment of experimental signals to        | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | nuclei of molecule.                                                     |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``fixed`` uses assignment given in experiment file                   |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |  - ``permute`` permutes labels using ``groups`` keyword information     |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``assignment``              | ``groups``              | ``- [label1, label2, ...]``         | Specifies a group of atoms whose assignments will be permuted           | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | One set per line                    | using chemlabels - Only required if assignment method is ``permute``.   |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | with no repeated labels             |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See Permutation for more information                                    |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+

Optional arguments
^^^^^^^^^^^^^^^^^^

Additionally, certain aspects of the program can be controlled on the command line, these are listed in Table 2.


.. table:: Table 2: Optional command line arguments to ``fit_susc``

    +------------------------------+--------------------------------------------------------------------------+
    | Optional Argument(s)         | Description                                                              |
    +==============================+==========================================================================+
    | ``-h``                       | Print help documentation                                                 |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--dry_run``                | Parses all files but exits prior to fitting                              |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--susc_units (string)``    | Controls units of susceptibility values in plots                         |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``A3`` :math:`\mathrm{Å}^3`                                            |
    |                              | - ``cm3 mol-1`` :math:`\mathrm{cm}^3 \ \mathrm{mol}^{-1}`                |
    |                              | - ``emu mol-1`` :math:`\mathrm{emu} \ \mathrm{mol}^{-1}`                 |
    |                              |                                                                          |
    |                              | Default ``A3``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--contrib_plots (string)`` | Plots theoretical shift broken down into Fermi contact, pseudocontact,   |
    |                              | and diamagnetic contributions                                            |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``off``                                                          |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--shift_plots (string)``   | Creates plots of experimental vs theoretical shift at each temperature   |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``on``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--spread_plots (string)``  | Plots theoretical shift broken down into Fermi contact, pseudocontact,   |
    |                              | and diamagnetic contributions as violin plots to illustrate spreads      |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``on``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--isoaxrho_plots (string)``| Creates plots isotropic, axial, and rhombic susceptibilities versus      |
    |                              | temperature                                                              |
    |                              |                                                                          |
    |                              | Options:                                                                 |
    |                              |                                                                          |
    |                              | - ``on`` shows and saves the plots                                       |
    |                              | - ``show`` shows the plots                                               |
    |                              | - ``save`` saves the plots                                               |
    |                              | - ``off`` neither shows nor saves                                        |
    |                              |                                                                          |
    |                              | Default ``on``                                                           |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--show_single``            | Show plots for first temperature and hide for the rest                   |
    +------------------------------+--------------------------------------------------------------------------+

Output Files
------------

All output files are added to the ``project`` directory specified in the input file.

Typically, ``fit_susc`` outputs the following files, and each filename is printed to screen.

1. ``assigned_experiment_<TEMPERATURE>_K.csv`` - If ``assignment: method: permute`` - Assigned experiment file at a given temperature with the same format as the input experiment.
2. ``dft_hyperfines.csv`` - If ``hyperfine: method: dft`` - Raw hperfine coupling constants from DFT output file.
3. ``hyperfines_and_shifts_<TEMPERATURE>_K.csv`` - Hyperfine coupling constants, chemical shifts, coordinates, and labels of each atom in system for the specified temperature.
4. ``pcs_isosurf_<TEMPERATURE>_K.cube`` - Pseudocontact shift isosurface cube file
5. ``susceptibility_components_chi.png`` - :math:`\chi` vs :math:`\mathrm{T}` - If more than one temperature is specified and ``--isoaxrho_plots on or save``
6. ``susceptibility_components_chiT.png`` - :math:`\chi \mathrm{T}` vs :math:`\mathrm{T}` - If more than one temperature is specified and ``--isoaxrho_plots on or save``
7. ``susceptibility_tensor.csv`` - Susceptibility tensors as a function of temperature, along with standard deviation of fitted parameters, :math:`r^2` and :math:`r^2_\mathrm{adj}`, mean absolute error (MAE), eigenvalues of susceptibility, and eigenvector representation using Euler angles

Example
-------

Here is an example of...

.. footbibliography::