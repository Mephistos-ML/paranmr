Predicting NMR Shifts
=====================

The ``predict`` subprogram allows the user to predict paramagnetic NMR spectra using the magnetic susceptibility tensor for a system and the hyperfine coupling constants for a set of nuclei.

Input
-----

Input file format
^^^^^^^^^^^^^^^^^

The main input file for the ``predict`` subprogram is a YAML (``.yml`` or ``.yaml``) file containing configuration options.

.. note::
    If you've never seen a YAML file before, then take a look at `this <https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started>`_ tutorial first.

The input file contains a series of keywords for which there are subkeywords and associated values - these are detailed in Table 1.

.. table:: Table 1: ``predict`` subprogram input file keywords and subkeywords.

    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | Keyword                     | Subkeyword              | Value                               | Description                                                             | Mandatory            |
    +=============================+=========================+=====================================+=========================================================================+======================+
    | ``project``                 | ``name``                | Directory Name                      | Name of directory to which all output files are written                 | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | If directory does not exist then it will be created                     |                      |
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
    | ``experiment``              | ``spectrum_files``      | File Name                           | File(s) containing experimental NMR spectrum ``.csv``.                  | |:x:|                |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <exp_spectrum>` for format information.                  |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susceptibility``          | ``file``                | File Name                           | File(s) containing susceptibility data                                  | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <exp_spectrum>` for format information.                  |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susceptibility``          | ``format``              | ``orca_nev``/``orca_cas``/``molcas``| Format of file(s) containing susceptibility data                        | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | /``txt``/``csv``                    |                                                                         |                      |
    |                             |                         |                                     | Options:                                                                |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:orca_nev`` uses data from NEVPT2 in an Orca ``.out`` file    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:orca_cas`` uses data from CASSCF in an Orca ``.out`` file    |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:molcas`` uses data from an OpenMOLCAS ``.out`` file          |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:csv`` uses data from a ``SimpNMR`` ``susceptibility.csv``|                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | - ``format:txt`` uses a tensor a ``.txt`` file                          |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <susceptibility_csv>` for format information.            |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``susceptibility``          | ``temperature``         | Float values                        | Temperature(s) to use from susceptibility file                          | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``chem_labels``             | ``file``                | File Name                           | File containing chemical labels, atom labels, and optionally            | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | and optionally chmeical math labels used for plotting.                  |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         |                                     | See :ref:`here <chemlabels_csv>` Format for more information            |                      |
    +-----------------------------+-------------------------+-------------------------------------+-------------------------------------------------------------------------+----------------------+
    | ``nuclei``                  | ``include``             | List of atom labels with indexing,  | Specifies which nuclei for which shifts will be calculated              | |:white_check_mark:| |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | or all_X where X is an atomic       |                                                                         |                      |
    |                             |                         |                                     |                                                                         |                      |
    |                             |                         | symbol                              |                                                                         |                      |
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
    |                             |                         |                                     | labels of ``Me1`` and ``tBu2`` with the average of all occurances of    |                      |
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

Optional arguments
^^^^^^^^^^^^^^^^^^

Additionally, certain aspects of the program can be controlled on the command line, these are listed in Table 2.


.. table:: Table 2: Optional command line arguments to ``predict``

    +------------------------------+--------------------------------------------------------------------------+
    | Optional Argument(s)         | Description                                                              |
    +==============================+==========================================================================+
    | ``-h``                       | Print help documentation                                                 |
    +------------------------------+--------------------------------------------------------------------------+
    | ``--contrib_plots (string)`` | Creates theoretical shift broken down by contribution at each temperature|
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

Example
-------


Here is an example of...

.. footbibliography::