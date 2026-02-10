Contributing
============

We welcome contributions to ``SimpNMR``. If you would like to add functionality or improve the code or the documentation,
please start by creating an `Issue <https://gitlab.com/suturina-group/simpnmr/-/issues>`_ on GitLab (using the relevant template)
to describe the change you propose.

When contributing, you **must** follow the rules below. These define the required development standards and help keep the project maintainable over time.

Source Code
-----------

1. All commits must conform to the `Angular/Conventional Commits style <https://www.conventionalcommits.org/en/v1.0.0-beta.4/>`_,
   using an imperative, present-tense subject line (for example: ``feat: add new fitting routine``).
   Commit scopes are encouraged and should reflect the affected subsystem (for example: ``feat(application): add new fitting routine`` or ``fix(io): handle malformed CSV input``).

   These commit messages are not just a style requirement: SimpNMR uses an automated semantic release workflow in CI.
   Commit types and optional breaking-change markers determine the version bump (major/minor/patch) and are used to
   generate release notes / changelog entries.

   Do not manually edit version numbers or release metadata unless explicitly instructed by the maintainers.

2. All code must comply with the rules enforced by the project linter (**Ruff**). Code that fails linting should not be committed.
3. Follow the line length and formatting rules defined in ``pyproject.toml`` (current maximum line length: **88 characters**).
4. Use **Google-style docstrings** and type hints for all public functions, methods, classes, and modules.
   Internal helper functions may omit docstrings if their intent is obvious.
5. Update the documentation to reflect your changes (for example, CLI usage pages or relevant user/developer guide sections).
6. If your changes add or modify dependencies, update the project configuration (``setup.py`` or ``pyproject.toml``) with explicit version constraints.

.. _local-development-setup:

Local development setup
-----------------------

For local development, it is recommended to install the project in editable mode with developer dependencies:

.. code-block:: bash

    pip install -e ".[dev]"

This installs all required development tools, including the project linter.

Do not
^^^^^^

1. Merge broken code.
2. Merge code with debug print statements.
3. Merge code with large amounts of commented-out code.
4. Change public APIs without prior discussion.
5. Silently modify YAML input contracts or configuration semantics.

These are simple requirements, and they make the code easier to use for everyone.

Documentation
-------------

Our documentation is written in `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and uses
the PyData Sphinx theme. The source code for the docs is available at ``simpnmr/docs/source``.

To build the documentation locally, ensure the project is installed with developer
dependencies (see :ref:`local development setup <local-development-setup>`), then
run the Sphinx build command from the ``docs`` directory.

To build the documentation locally, run:

.. code-block:: bash

    sphinx-build -b html source build/html

Alternatively, if ``make`` is available on your system, you may use the provided
Makefile wrapper:

.. code-block:: bash

    make clean html

The compiled HTML pages will be available at ``simpnmr/docs/build/html``.

Please ensure the documentation builds successfully prior to committing/merging.

Do not commit compiled pages to the repository.