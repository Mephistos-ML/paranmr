# Scientific correctness contracts

These tests verify mathematical and physical properties independently of the
implementation details of ParaNMR. They do not use canonical datasets, CLI
workflows, or unfinished GMM contracts.

Reference calculations in `oracles/` are intentionally small and explicit. A
test must not obtain its expected value by calling the production function that
it is verifying.
