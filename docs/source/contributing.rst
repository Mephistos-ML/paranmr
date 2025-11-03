Contributing
============

We welcome contributions to ``SimpNMR``, please create an `Issue <https://gitlab.com/suturina-group/simpnmr/-/issues>`_  on GitLab (with a template) if you wish to add functionality.

When contributing, you **must** follow the following rules which help to standardise development.

Source Code
-----------

1. All commits must conform to the `angular style <https://gist.github.com/brianclements/841ea7bffdb01346392c>`_ with imperative, present tense.
2. Use ``flake8`` to ensure compliance with ``pep8`` standards.
3. Use an 80-character line length limit, `CamelCase` for `Classes`, and `snake_case` for everything else.
4. Use numpy style docstrings, and type-hinting for all arguments and return values.
5. Update documentation accordingly with your changes - e.g. the relevant CLI 'usage' page, and the 'What's New' page.
6. If your changes add or modify dependencies, update ``setup.py`` with explicit version numbers.

Do not!
^^^^^^^

1. Merge broken code
2. Merge code with debug print statements
3. Merge code with large amounts of commented out code

These are simple requirements, and they make the code easier to use for everyone.

Documentation
-------------

Our documentation is written in `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and uses
the `Read the Docs` theme. The source-code for the docs is available at ``simpnmr/docs/source``.

To build the documentation **locally**, navigate to ``simpnmr/docs`` and install the python dependencies with

.. code-block:: bash

    pip install -r requirements.txt

You will also need to install ``make`` using your preferred system package manager.

To build the documentation run

.. code-block:: bash

    make clean html

The compiled html pages will be available at ``simpnmr/docs/build/html``.

Please ensure the documentation builds successfully prior to committing/merging.

Do not commit compiled pages to the repository.

To publish the documentation online, create a pipeline with the keyword ``force`` and the value ``docs``.