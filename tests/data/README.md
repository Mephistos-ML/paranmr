# Canonical test inputs

Each complex keeps the same input layout used by its public example:

```text
tests/data/<complex>/DATA/...
```

Only immutable raw inputs belong here: geometry, experimental measurements,
diamagnetic data, hyperfine data, and quantum-chemistry outputs. Do not add
generated fit outputs, plots, or runnable YAML configurations.

Test-only configurations live under
`tests/integration/experimental/configs/<complex>/SIMULATIONS/`. Integration
tests materialize both trees into `tmp_path` through
`tests.helpers.fixtures.materialize_canonical_fixture`, so test runs never write
to canonical fixtures.
