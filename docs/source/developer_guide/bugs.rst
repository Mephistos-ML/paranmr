Reporting Bugs
==============

If you encounter unexpected behaviour or believe you have found a bug in ``simpnmr``, we appreciate you letting us know – this helps us improve the code for everyone.

A bug report should describe **reproducible incorrect behaviour** of the software.
Questions about usage, configuration, or expected behaviour should be directed to
the documentation or discussion channels rather than reported as bugs.

Before opening an issue
-----------------------

Before reporting a bug, please ensure that:

1. Make sure you are using the latest version of ``simpnmr``:

   .. code-block:: bash

       pip install --upgrade simpnmr

2. Check the :ref:`faq` page to see whether the issue has already been described, is caused by an input configuration error, or has a known workaround.

If the problem persists and is not addressed in the FAQ, please open a `GitLab Issue <https://gitlab.com/suturina-group/simpnmr/-/work_items>`_
**instead of emailing individual developers**. Using the issue tracker makes it easier for us to keep track of problems and their fixes.

To make it easier to investigate and reproduce the issue, please **use the issue template** provided on the "Create new issue" page and include as much of the following information as possible:

1. Versions of ``Python`` and ``simpnmr`` (for example, from ``python --version`` and ``pip show simpnmr``).
2. Your operating system and how ``simpnmr`` was installed (e.g. system Python, virtual environment, conda environment).
3. The relevant YAML configuration file(s) and any associated input data, or a minimal working example that reproduces the problem.
4. The exact command(s) you ran, or a minimal code snippet if you are using the Python API.
5. The full error message and/or any unexpected output produced by ``simpnmr``.

Providing this information up front saves time on follow-up questions and helps us resolve issues more quickly.

What not to include
-------------------

Please avoid the following when reporting bugs:

- Screenshots of error messages instead of plain text.
- Partial logs that omit the original error traceback.
- Large private or unpublished datasets without a minimal reproducible example.
