.. _Configuration:

Configuration
-------------

.. _env_var:


Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Some default settings of ``SimpNMR`` can be configured using the environment variables listed in Table 1.

Note, these options only apply when ``SimpNMR`` is used through its command line interface (CLI),
and in the case of custom fonts, users must have the specified font installed on their system.

Additionally, changing some of the environment variables used for plotting may lead
to **unsupported** and unexpected behaviour - you have been warned!

+-------------------------+---------------------------------------------------------+
| Variable Name           | Description                                             |
+=========================+=========================================================+
| ``pnmr_csvdelimiter``   | Delimiter character used in csv creation                |
|                         |                                                         |
|                         | See :ref:`csv_files` for more information               |
+-------------------------+---------------------------------------------------------+
| ``pnmr_fontname``       | Name of font to use in plots                            |
|                         |                                                         |
|                         | Note this font must be installed on your system         |
+-------------------------+---------------------------------------------------------+
| ``pnmr_fontsize``       | Size of font to use in plots                            |
+-------------------------+---------------------------------------------------------+
| ``pnmr_plotformat``     | Extension/format to use for saved plots                 |
|                         |                                                         |
|                         | e.g. .png or .pdf                                       |
+-------------------------+---------------------------------------------------------+
| ``pnmr_termcolor``      | If set (to any value) enables colour-coded output in    |
|                         |                                                         |
|                         | windows terminal                                        |
+-------------------------+---------------------------------------------------------+

The process for setting these variables depends on your platform.

On macOS and Linux, you can set an environment variable in a shell with:

.. code-block:: bash

    export variablename=VALUE

For example:

.. code-block:: bash
    
    export pnmr_plotformat=.png


On Windows, ``CMD`` (Command Prompt) users can set variables temporarily using:

.. code-block:: doscon

    set variablename=VALUE

while Windows PowerShell (PS) users can use:

.. code-block:: doscon

    $env:variablename="VALUE"

These settings only persist for the current window. To set Windows environment variables permanently, use the *Environment Variables* dialog in the System Properties.

Terminal output
^^^^^^^^^^^^^^^

macOS and Linux users will see colour-coded terminal output by default. For Windows users this is disabled by default, since ``CMD`` does
not support ASCII colour codes. If you are on Windows and are using an ASCII-enabled terminal (e.g. `Windows Terminal <https://en.wikipedia.org/wiki/Windows_Terminal>`__) you can
enable colour-coded output by setting the ``pnmr_termcolor`` environment variable to any value in the *Environment Variables* dialog or in your terminal session.

.. _csv_files:

CSV Files
^^^^^^^^^

By default the comma ``,`` is used as the ``.csv`` output delimiter, but this can be changed using the ``pnmr_csvdelimiter`` environment variable.

Note that when specifying a semicolon delimiter on Linux and Mac, you must escape the semicolon character, i.e.

.. code-block:: bash

    # Semicolon ;
    export pnmr_csvdelimiter=\;
    # Comma ,
    export pnmr_csvdelimiter=,

Windows users can use the *Environment Variables* dialog, or the commands in the :ref:`env_var` section.
Note that unlike for Linux/macOS, no escape character is required for the semicolon – it is just ``;``.
