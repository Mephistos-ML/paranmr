Hyperfine Benchmarks
=====================

``paranmr`` provides dedicated benchmark workflows for comparing hyperfine
sources. The supported benchmark subcommands are:

- ``benchmark a_fc``: compare isotropic Fermi-contact values ``A_fc``.
- ``benchmark a_sd``: compare axial spin-dipolar values ``A_sd``.

Benchmark workflows use a standalone YAML configuration file with an explicit
list of hyperfine sources, and they save diagnostic plots into the configured
project directory.

Configuration
-------------

The minimal configuration for ``benchmark a_fc`` looks like this:

.. code-block:: yaml

   project:
     name: A_FC_Benchmark

   chem_labels:
     file: chem_labels.csv

   nuclei:
     include: H

   benchmark:
     max_label_tolerance: 0.05

   hyperfine:
     - functional: B3LYP
       method: dft
       file: hfc/B3LYP.out

     - functional: PBE0
       method: csv
       file: hfc/PBE0_hyperfine.csv


Required configuration blocks:

- ``project``: the output directory for benchmark results.
- ``chem_labels``: a chemical-label mapping file for grouping nuclei.
- ``nuclei``: the nuclei to include in the benchmark.
- ``hyperfine``: a list of hyperfine source blocks.

Each ``hyperfine`` entry contains:

- ``functional``: the functional name associated with the source.
- ``file``: the hyperfine source file path.
- ``method``: ``dft`` or ``csv``. Defaults to ``dft``.

The optional ``benchmark`` block supports:

- ``max_label_tolerance``: relative tolerance used when selecting a majority
  chemical label for the max benchmark curve.

Running benchmarks
------------------

Run the ``A_fc`` benchmark:

.. code-block:: console

   paranmr benchmark a_fc benchmark_hyperfine.yml

Run the ``A_sd`` benchmark:

.. code-block:: console

   paranmr benchmark a_sd benchmark_hyperfine.yml

Plots are always saved to the project directory. To suppress interactive display,
use the ``--hide`` flag:

.. code-block:: console

   paranmr --hide benchmark a_fc benchmark_hyperfine.yml

Outputs
-------

Both workflows create the directory specified in ``project:name``.

Files produced by ``benchmark a_fc``:

- ``A_FC_benchmark_max.csv``: maximum ``A_fc`` values by functional and nucleus.
- ``<FUNCTIONAL>_<NUCLEUS>_A_FC_benchmark_spread.pdf``: ``A_fc`` spread across
  chemical labels for one functional.
- ``<NUCLEUS>_A_FC_benchmark_max_curve.pdf``: sorted maximum ``A_fc`` values for
  each nucleus.

Files produced by ``benchmark a_sd``:

- ``<FUNCTIONAL>_<NUCLEUS>_A_SD_benchmark_spread.pdf``: ``A_sd`` spread across
  chemical labels for one functional.
- ``<NUCLEUS>_A_SD_benchmark_max_curve.pdf``: sorted maximum ``A_sd`` values for
  each nucleus.

Notes
-----

- The ``A_fc`` benchmark compares Fermi-contact hyperfine components.
- The ``A_sd`` benchmark compares axial spin-dipolar hyperfine components.
- Unknown top-level YAML keys cause a configuration validation error.
