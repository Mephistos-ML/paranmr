Usage
=====

.. toctree::
   Command Line Interface <cli/index>
   Scripting <scripting/index>
   :maxdepth: 3
   :caption: Contents:
   :hidden:

There are two main interfaces to ``SimpNMR``. Which one you use will depend on the complexity of the data
you are processing and your familiarity with the command line and Python.

Command line interface
----------------------

For most users, we recommend the command line interface (CLI). It:

- provides subprograms for common workflows (such as fitting susceptibility tensors and predicting pNMR shifts),
- exposes many advanced features without requiring you to write Python code,
- is well-suited to running repeatable calculations using YAML configuration files.

The CLI is installed together with the ``SimpNMR`` package. See :ref:`Installation` for instructions on how to install it,
and :ref:`guide` for an overview of the available subprograms and their options.

Python scripting interface
--------------------------

More advanced users may wish to work with the ``SimpNMR`` package directly from their own Python code. This is useful if you:

- need to integrate ``SimpNMR`` into larger data-processing or simulation workflows,
- want to automate or customise analyses beyond what the CLI offers,
- prefer to work interactively in Python (for example, in Jupyter notebooks).

The Python API is documented in this manual. See :ref:`guide_scripting` for guided examples,
and :ref:`modules` for a detailed reference of the available classes and functions.