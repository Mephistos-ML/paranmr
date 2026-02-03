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
2. All code must comply with the rules enforced by the project linter (**Ruff**). Code that fails linting should not be committed.
3. Follow the line length and formatting rules defined in ``pyproject.toml`` (current maximum line length: **88 characters**).
4. Use **Google-style docstrings** and type hints for all public functions, methods, classes, and modules.
   Internal helper functions may omit docstrings if their intent is obvious.
5. Update the documentation to reflect your changes (for example, CLI usage pages or relevant user/developer guide sections).
6. If your changes add or modify dependencies, update the project configuration (``setup.py`` or ``pyproject.toml``) with explicit version constraints.

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

To build the documentation **locally**, navigate to the ``docs`` directory from the repository root and install the Python dependencies with:

.. code-block:: bash

    pip install -r requirements.txt

You will also need to install ``make`` using your preferred system package manager.

To build the documentation, run:

.. code-block:: bash

    make clean html

The compiled HTML pages will be available at ``simpnmr/docs/build/html``.

Please ensure the documentation builds successfully prior to committing/merging.

Do not commit compiled pages to the repository.

To publish the documentation online, create a GitLab pipeline with the keyword ``force`` set to the value ``docs`` so that the documentation job is triggered.