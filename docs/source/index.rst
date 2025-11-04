Home
====

.. toctree::
   Installation <installation>
   Configuration <configuration>
   What's New <whatsnew>
   Usage <usage/index>
   Files <files>
   Theory <theory/index>
   Modules <modules/index>
   Contributing <contributing>
   FAQ <faq>
   Bugs <bugs>
   Authors and Citation <authors>
   License <https://gitlab.com/suturina-group/simpnmr/-/blob/main/LICENSE>
   Repository <https://gitlab.com/suturina-group/simpnmr>
   :maxdepth: 3
   :caption: Contents:
   :hidden:


``SimpNMR`` is a Python package for analysing and simulating paramagnetic nuclear magnetic resonance (pNMR) data.

It provides a command line interface and a Python API to help you organise input data, run common workflows, and collect results in a reproducible way.

Key features
^^^^^^^^^^^^

- Command line interface with subprograms for :doc:`fitting magnetic susceptibilities <fit_susc>` and :doc:`predicting pNMR shifts <predict>`.
- YAML-based configuration files for transparent and version-controlled workflows.
- Clear structure for input and output files (see :ref:`files`).
- Configurable behaviour via environment variables when using the CLI (see :ref:`Configuration`).
- Designed to integrate easily with scripting, automation, and larger data-processing pipelines in Python.

Quick start
^^^^^^^^^^^

After installing ``simpnmr`` (see :ref:`installation`), you can explore the command line interface:

.. code-block:: bash

    simpnmr -h
    simpnmr predict -h
    simpnmr fit_susc -h

As a minimal example, working with a YAML configuration file:

.. code-block:: bash

    simpnmr predict your_config.yml

For details of the available options and required input fields, see :ref:`guide` and the :ref:`files` page.

New users
^^^^^^^^^

If you are new to ``SimpNMR``, we recommend the following order:

1. :ref:`installation` – install ``simpnmr`` and verify it runs in your environment.
2. :ref:`guide` – learn the structure of the command line interface and its subprograms.
3. :ref:`files` – understand the input and output file formats used by the CLI.
4. :ref:`Configuration` – optionally customise default behaviour with environment variables.
5. Browse the :doc:`Modules <modules/index>` section if you want to use ``SimpNMR`` as a Python library in your own scripts.

Getting help
^^^^^^^^^^^^

If something does not work as expected:

- Check the :ref:`faq` for answers to common questions and error messages.
- See :doc:`Bugs <bugs>` for how to report issues or unexpected behaviour.
- If you would like to contribute fixes or new features, read :doc:`Contributing <contributing>`.
