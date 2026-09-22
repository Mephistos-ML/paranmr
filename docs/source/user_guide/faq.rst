.. _faq:

FAQ
---

This page contains some frequently asked questions. Please consult this page before opening a GitHub issue.

This documentation describes the `SimpNMR-X` fork of the original
`suturina-group/simpnmr <https://gitlab.com/suturina-group/simpnmr>`_ project.

Errors
^^^^^^

1. ``'simpnmr-x' is not recognized as an internal or external command, operable program or batch file.``

    You are trying to run ``simpnmr_x`` in a terminal where Python (and therefore SimpNMR-X) is not available.
    First check that Python is installed and works in this terminal (for example, by running ``python --version``),
    then reinstall or activate SimpNMR-X in that environment before running ``simpnmr_x`` again.

2. ``ModuleNotFoundError: No module named 'simpnmr-x'.``

    Python cannot find the SimpNMR-X package in the current environment.
    Activate the environment where you want to use SimpNMR-X (for example with ``conda activate ...`` or ``source venv/bin/activate``)
    and (re)install it:
    
    .. code-block:: bash

        pip install simpnmr-x

3. ``no matches found: *yml``

    Your shell cannot find any files matching ``*yml`` in the current working directory.
    Check that you are in the folder that contains your YAML configuration file(s) before running the command.

4. ``Missing file error: [Errno 2] No such file or directory``

    SimpNMR-X cannot find one or more of the files specified.
    Double-check the paths to all required input files (see ``simpnmr-x -h`` for a summary of the expected inputs and options).

5. ``yaml.scanner.ScannerError`` or ``yaml.parser.ParserError``

    SimpNMR-X could not read your YAML file because it is not valid YAML.
    Check the indentation, colons, and quotation marks in your configuration file
    (a YAML linter or editor with YAML support can also help you spot the issue).

6. ``PermissionError: [Errno 13] Permission denied``

    SimpNMR-X does not have permission to read or write one of the files or folders you selected.
    Make sure you have write access to the output directory and read access to all input files,
    or choose a different location for the outputs.
