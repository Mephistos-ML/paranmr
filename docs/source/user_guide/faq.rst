.. _faq:

FAQ
---

This page contains some frequently asked questions. Please consult this page before opening a GitHub issue.

This documentation describes the `ParaNMR` fork of the original
`suturina-group/simpnmr <https://gitlab.com/suturina-group/simpnmr>`_ project.

Errors
^^^^^^

1. ``'paranmr' is not recognized as an internal or external command, operable program or batch file.``

    You are trying to run ``paranmr`` in a terminal where Python (and therefore ParaNMR) is not available.
    First check that Python is installed and works in this terminal (for example, by running ``python --version``),
    then reinstall or activate ParaNMR in that environment before running ``paranmr`` again.

2. ``ModuleNotFoundError: No module named 'paranmr'.``

    Python cannot find the ParaNMR package in the current environment.
    Activate the environment where you want to use ParaNMR (for example with ``conda activate ...`` or ``source venv/bin/activate``)
    and (re)install it:
    
    .. code-block:: bash

        pip install paranmr

3. ``no matches found: *yml``

    Your shell cannot find any files matching ``*yml`` in the current working directory.
    Check that you are in the folder that contains your YAML configuration file(s) before running the command.

4. ``Missing file error: [Errno 2] No such file or directory``

    ParaNMR cannot find one or more of the files specified.
    Double-check the paths to all required input files (see ``paranmr -h`` for a summary of the expected inputs and options).

5. ``yaml.scanner.ScannerError`` or ``yaml.parser.ParserError``

    ParaNMR could not read your YAML file because it is not valid YAML.
    Check the indentation, colons, and quotation marks in your configuration file
    (a YAML linter or editor with YAML support can also help you spot the issue).

6. ``PermissionError: [Errno 13] Permission denied``

    ParaNMR does not have permission to read or write one of the files or folders you selected.
    Make sure you have write access to the output directory and read access to all input files,
    or choose a different location for the outputs.
