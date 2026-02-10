Architecture
============

This page provides a high-level overview of the internal architecture of
``simpnmr``. It is intended for developers who want to understand how the
codebase is structured, how data flows through the system, and where new
functionality should be added.

The design prioritises explicit workflows, separation of concerns, and
reproducibility over maximal abstraction.

Design goals
------------

The architecture of ``simpnmr`` is guided by the following principles:

- **Explicit workflows**: all calculations are driven by user-supplied YAML
  configuration files and CLI entry points.
- **Separation of concerns**: domain logic, IO, orchestration, and visualisation
  are strictly separated.
- **Reproducibility**: no hidden defaults; all assumptions must be expressible
  in configuration or documented behaviour.
- **Extensibility**: new QC formats, models, or workflows can be added without
  modifying unrelated components.


High-level layering
-------------------

At a conceptual level, the codebase is organised into the following layers:

1. **CLI layer** (``simpnmr.cli``)
2. **Application layer** (``simpnmr.app``)
3. **Domain / core layer** (``simpnmr.core``)
4. **IO layer** (``simpnmr.io``)
5. **Visualisation layer** (``simpnmr.viz``)

The CLI layer is responsible for user interaction. The application layer
orchestrates workflows. The core layer contains all scientific and numerical
logic. IO and visualisation are treated as peripheral services.


Repository layout
-----------------

Top-level repository structure:

.. code-block:: text

   .
   ├── docs/                # Sphinx documentation
   ├── examples/            # End-to-end usage examples
   ├── simpnmr/             # Python package source
   └── tests/               # Test suite

The remainder of this page focuses on the ``simpnmr`` package itself.


Package overview
----------------

``simpnmr`` is organised as follows:

.. code-block:: text

   simpnmr/
   ├── app/                 # Workflow orchestration (pipelines, loaders, settings)
   ├── cfg/                 # Configuration loading and validation
   ├── cli/                 # Command-line entry points
   ├── core/                # Scientific and numerical domain logic
   ├── io/                  # File parsing and data import/export
   ├── tools/               # Standalone utilities
   └── viz/                 # Plotting and visualisation

Each top-level module has a clearly defined responsibility, described below.


CLI layer (``simpnmr.cli``)
---------------------------

The CLI layer defines user-facing commands such as ``predict`` and
``fit_susc``. Its responsibilities are deliberately minimal:

- parse command-line arguments
- locate and load YAML configuration files
- dispatch execution to the appropriate application-layer workflow

The CLI layer must not contain scientific logic, numerical routines, or file
parsing beyond trivial argument handling.


Application layer (``simpnmr.app``)
-----------------------------------

The application layer orchestrates complete workflows. It connects user
configuration, domain logic, IO, and visualisation into executable pipelines.

Key responsibilities:

- loading and validating configuration blocks
- constructing domain objects from inputs
- executing prediction or fitting workflows
- coordinating output generation

Submodules:

- ``loaders``: application-layer adapters that translate user configuration into
  fully initialised domain objects. Loaders select the appropriate IO backend
  (e.g. CSV, quantum chemistry outputs) based on configuration options and
  delegate file parsing to ``simpnmr.io`` and object construction to
  ``simpnmr.core.build``.

- ``pipelines``: end-to-end workflow orchestration (e.g. prediction, fitting).
  Pipelines wire together loaders, core computations, output writers, and
  visualisation.

- ``params``: typed application settings and plotting/runtime options used by
  pipelines and CLI wiring.

**Constraint:** numerical algorithms must not be implemented in ``simpnmr.app``.
Any scientific formulas, optimisation logic, scoring/metrics that affect results,
or numerical kernels belong in ``simpnmr.core``.


Core domain layer (``simpnmr.core``)
------------------------------------

The core layer contains all scientific and numerical logic. It is intentionally
independent of file formats, CLI concerns, and plotting.

Subpackages include:

- ``domain``: core data models (e.g. nuclei, tensors, electronic states)
- ``pcs``: paramagnetic chemical shift calculations
- ``fitting``: susceptibility fitting models and optimisation logic
- ``relaxation``: relaxation and broadening models
- ``sh``: spin Hamiltonian parameter extraction and related math
- ``spectrum``: spectrum construction and manipulation
- ``const`` and ``conv``: physical constants and unit transformations

All domain rules and mathematical assumptions must live in this layer.


IO layer (``simpnmr.io``)
-------------------------

The IO layer is responsible for reading and writing data in external formats.

It includes parsers for:

- CSV files
- quantum chemistry output files (e.g. ORCA, Gaussian, Molcas)
- XYZ structures
- plain text tensor formats

The IO layer may not implement scientific logic; it only translates external
representations into internal domain objects and vice versa.


Visualisation layer (``simpnmr.viz``)
-------------------------------------

The visualisation layer generates plots and figures for predicted and fitted
results.

The visualisation layer uses a unified plotting system that centralises styling,
layout, colour palettes, typography, and export behaviour.

New plots **must** be implemented using the existing visualisation
infrastructure (``simpnmr.viz``) rather than ad-hoc Matplotlib code.

Responsibilities:

- plotting spectra and tensor components
- consistent styling and layout
- exporting figures to files

Visualisation code must not modify domain objects or influence numerical
results.


Examples and tests
------------------

The ``examples`` directory contains end-to-end workflows demonstrating
prediction and fitting use cases. These examples serve as both user guidance
and informal integration tests.

The ``tests`` directory is structured by intent:

- ``unit``: isolated tests of individual functions and classes
- ``integration``: tests covering interactions between subsystems
- ``regression``: tests guarding against numerical regressions
- ``smoke``: fast checks ensuring workflows run without failure


Extension points
----------------

Common extension scenarios include:

- adding support for a new QC output format → ``simpnmr.io``
- introducing a new susceptibility model → ``simpnmr.core``
- adding a new workflow or CLI command → ``simpnmr.app`` and ``simpnmr.cli``

Developers should avoid cross-layer dependencies and keep extensions confined
to the appropriate layer.


Conventions and constraints
---------------------------

- Domain logic must not depend on IO or CLI modules.

- Public interfaces must be treated as stable contracts. Public APIs include:

  - CLI commands and flags
  - YAML configuration schemas and documented key semantics
  - Documented output files (file structure, column names, units, and meaning)

- YAML configuration schemas are part of the public contract. Renaming, removing,
  or changing the meaning of configuration keys requires explicit justification
  and must be treated as a breaking change unless explicitly documented as
  backward-compatible.

- Output semantics are part of the public interface. Changes to output file
  structure, column names, units, ordering, or interpretation must be
  backward-compatible or declared as breaking changes.

- Numerical algorithms must not be implemented in ``simpnmr.app``. Scientific
  formulas, optimisation logic, scoring/metrics that affect results, or
  numerical kernels belong in ``simpnmr.core``.

- All new plots must use the unified visualisation system in ``simpnmr.viz``.
  Ad-hoc plotting logic in pipelines or tools is not permitted.

- Experimental or exploratory code must live in ``simpnmr.tools`` (or in
  ``examples``) and must not be merged into core workflows without explicit
  review and justification.

- Workflows must be deterministic for the same inputs and configuration.
  Randomness, implicit defaults, or environment-dependent behaviour must be
  explicitly controlled and documented.

- Dependency boundaries must be respected. The core layer should not introduce
  convenience or plotting-related dependencies; heavyweight dependencies must be
  scientifically justified and kept out of ``simpnmr.core`` whenever possible.

- Backward-incompatible changes must be reflected in the changelog.

- Breaking-change releases (major version bumps) require maintainer alignment.
  Contributors must not introduce breaking changes or mark commits as breaking
  without discussing the change with maintainers first.


This separation is enforced by convention and code review rather than by a
formal framework.