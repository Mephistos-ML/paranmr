Home
====

``paranmr`` is a Python package for the analysis of solution NMR data of
paramagnetic metal complexes using computational chemistry methods.

The package provides reproducible, workflow-driven tools for:

- predicting one-dimensional paramagnetic NMR (pNMR) spectra
- fitting magnetic susceptibility tensors to experimental pNMR data
- analysing hyperfine and susceptibility tensors obtained from electronic
  structure calculations

Workflows are controlled via explicit YAML configuration files, which define
all input data, model assumptions, and numerical parameters. This design makes
calculations transparent, reproducible, and suitable for systematic studies.


.. grid:: 4
   :gutter: 3

   .. grid-item-card:: Get started
      :link: user_guide/index
      :link-type: doc

      Installation instructions, workflows, and practical guidance for running
      prediction and fitting calculations with ``paranmr``.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Downloadable example materials for the ParaNMR tutorials and workshop,
      provided as ready-to-use files from the repository.

   .. grid-item-card:: Developer Guide
      :link: developer_guide/index
      :link-type: doc

      Internal architecture, contribution guidelines, and maintenance notes
      for developers and advanced users.

   .. grid-item-card:: Templates
      :link: templates/index
      :link-type: doc

      Downloadable YAML templates for common prediction and fitting workflows.
      Modify the file to run your own calculations with minimal setup.

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   :hidden:

   developer_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Templates
   :hidden:

   templates/index