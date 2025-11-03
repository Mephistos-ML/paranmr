.. _files:

Files
-----

This page outlines the file formats used in ``SimpNMR``.


Input Files
^^^^^^^^^^^

The command line interface to ``SimpNMR`` makes use of several input files.

Each subprogram uses its own input file with its own required keywords and subkeywords, though the format of a given
keyword is constant across different input files.

In general, the following rules apply to input files

1. All input files use the ``YAML`` format - see `here <https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started>`_ for a quick tutorial.
2. Any file names specified in an input file can contain absolute or relative path information.
3. Any file names specified in an input file can contain the ``*`` wildcard.
4. Additional keyword entries are ignored.
5. Comment lines (``#``) are ignored by the ``pyaml`` parser.

.. _general_csv:

General ``.csv`` file parsing rules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, any ``.csv`` file read by ``simpnmr`` follows the following set of rules

1. Column names are case-insensitive
2. Comment lines (``#``) are ignored (apart from metadata lines e.g. Experiment files)
3. No assumptions are made about delimiter
4. Additional columns are ignored


.. _exp_csv:

Experimental data ``.csv`` files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file contains the positions, tentative assignment (chem_labels), and shape information for the paramagnetic signals
of a single NMR spectrum.

This file must be formatted as ``.csv``, and must contain the columns specified in Table 2, along
with all of the following metadata comment lines.

.. code-block ::

    # temperature VALUE
    # isotope VALUE
    # larmor VALUE

where ``VALUE`` contains the temperature in Kelvin, the isotope measured in this experiment (e.g. ``1H``),
and the larmor frequency of the isotope in the spectrometer used in this experiment, respectively.

Table 2: Experimental data file ``.csv`` headers

+-------------------+----------------------+------------------------------------------------+
| Property          | Headers              | Headers                                        |
+===================+======================+================================================+
| Assignment        | assignment           | chem_label (tentative) assignment of signal    |
+-------------------+----------------------+------------------------------------------------+
| Shift             | shift (ppm)          | Experimental chemcial shift of signal          |
+-------------------+----------------------+------------------------------------------------+
| Area              | area                 | Area of signal                                 |
+-------------------+----------------------+------------------------------------------------+
| Linewidth         | width (Hz)           | Width of signal                                |
+-------------------+----------------------+------------------------------------------------+
| L/G               | L/G                  | Ratio of Lorentzian to Gaussian functions used |
|                   |                      | to model signal                                |
+-------------------+----------------------+------------------------------------------------+

.. _dia_csv:

Diamagnetic correction ``.csv`` files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This contains diamagnetic shifts for a given system.

+-------------------+----------------------+
| Property          | Headers              |
+===================+======================+
| Chemical label    | chem_label           |
+-------------------+----------------------+
| Shift             | shift (ppm)          |
+-------------------+----------------------+

It is assumed that diamagnetic corrections are independent of temperature.

.. _chemlabels_csv:

Chemical Label files
^^^^^^^^^^^^^^^^^^^^

This file contains the additional labels which allow the user to refer to the atoms of a molecule using a set of
chemically meaningful labels e.g. ``Me1`` rather than the atomic labels ``H1``, ``H2``, ``H3``. Chemical labels
can be repeated and serve to group atoms together for plotting and averaging purposes.

Optionally, the user may also specify `mathtext <https://matplotlib.org/stable/users/explain/text/mathtext.html>`_ labels to be used when plotting data in ``matplotlib`` - these must
be wrapped in ``$$``.

This must be a ``.csv`` file containing a series of columns, each denoting a specific property (see below).

+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+
| Property          | Headers              | Description                                                                  | Mandatory                |
+===================+======================+==============================================================================+==========================+
| Atom label        | atom_label           | Labels of each NMR active nucleus including indexing number                  | |:white_check_mark:|     |
+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+
| Chemical label    | chem_label           | Chemical label of each NMR active nucleus                                    | |:white_check_mark:|     |
+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+
| Math label        | chem_math_label      | Mathtext label of each NMR active nucleus e.g. ``$\mathregular{Me}_1$``      | |:x:|                    |
+-------------------+----------------------+------------------------------------------------------------------------------+--------------------------+

At least of each property must be present in the file (in any order), and a comma delimited header line
containing the headers in Table 4 must be present above the data columns.

Users can convert annotated chemcraft ``.xyz`` files into a Chemical label file using ``simpnmr``'s included ``xyz_to_chemlabel`` script. Run ``xyz_to_chemlabel -h`` for more information.
