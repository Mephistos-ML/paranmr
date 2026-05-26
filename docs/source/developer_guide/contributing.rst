Contributing
============

We welcome contributions to ``ParaNMR``. If you would like to add functionality or improve the code or the documentation,
please start by creating an `Issue <https://github.com/Mephistos-ML/paranmr/issues>`_ on GitHub (using the relevant template)
to describe the change you propose.

When contributing, you **must** follow the rules below. These define the required development standards and help keep the project maintainable over time.

Source Code
-----------

1. All commits must conform to the `Angular/Conventional Commits style <https://www.conventionalcommits.org/en/v1.0.0-beta.4/>`_,
   using an imperative, present-tense subject line (for example: ``feat: add new fitting routine``).
   Commit scopes are encouraged and should reflect the affected subsystem (for example: ``feat(application): add new fitting routine`` or ``fix(io): handle malformed CSV input``).

   These commit messages are not just a style requirement: ParaNMR uses an automated semantic release workflow in CI.
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

AI-assisted contributions
-------------------------

AI-assisted development is allowed, but it is subject to additional constraints.

All AI-generated or AI-assisted code **must** comply with the project-wide
AI contract defined in ``AI_CONTRACT.md`` (located at the root of the repository).
This contract takes precedence over general contribution guidelines when AI tools
are involved.

Requirements for AI-assisted contributions:

1. The contributor is responsible for the correctness, architecture, and scientific
   validity of the submitted code, regardless of AI usage.
2. The AI system **must** be provided with:
   - the ``AI_CONTRACT.md`` file,
   - the relevant directory tree,
   - the specific file(s) being modified.
3. AI-generated changes must not introduce:
   - cross-layer architectural violations,
   - silent changes to public contracts (CLI, YAML schemas, output formats),
   - speculative or assumed behaviour due to missing context.
4. If the AI system does not have sufficient context to make a correct change,
   it must stop and request clarification rather than guessing.

During review, maintainers may request confirmation that the AI contract was
provided to the agent. Contributions that violate ``AI_CONTRACT.md`` may be
rejected without further review.

Documentation
-------------

Our documentation is written in `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and uses
the PyData Sphinx theme. The source code for the docs is available at ``paranmr/docs/source``.

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

The compiled HTML pages will be available at ``paranmr/docs/build/html``.

Please ensure the documentation builds successfully prior to committing/merging.

Do not commit compiled pages to the repository.