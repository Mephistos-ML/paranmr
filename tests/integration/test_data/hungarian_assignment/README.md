# Hungarian Assignment Tests

This directory contains integration tests for the Hungarian assignment algorithm in the `fit_susc` workflow.

## Test Strategy

The tests verify that the Hungarian algorithm produces equivalent susceptibility tensors (χ) to the exhaustive permutation method. Since permutation tries every combination, it serves as ground truth.

## Test Cases

Four test scenarios cover the parameter space:

| Test Case     | χ Tensor | Peaks vs Nuclei | Status |
|---------------|----------|-----------------|--------|
| diag_equal    | Diagonal | Equal (3=3)     | ✅ Implemented |
| diag_fewer    | Diagonal | Fewer (2<3)     | 📝 TODO |
| full_equal    | Full     | Equal (3=3)     | 📝 TODO |
| full_fewer    | Full     | Fewer (2<3)     | 📝 TODO |

## Running Tests

### Run the basic test
```bash
pytest -v tests/integration/test_hungarian_assignment.py::test_hungarian_vs_permute_diag_equal
```

### Run all Hungarian tests (including skipped)
```bash
pytest -v tests/integration/test_hungarian_assignment.py
```

### Run all integration tests
```bash
pytest -m integration
```

### Run only unit tests (fast, default)
```bash
pytest
```

## Test Data Structure

Each test case has its own directory under `test_data/hungarian_assignment/`:

```
diag_equal/
├── input.yml           # Pipeline configuration
├── chem_labels.csv     # Atom to signal mapping
├── exp.csv             # Experimental shifts
├── hfc.csv             # Hyperfine coupling constants
└── dia.csv             # Diamagnetic contributions
```

The test runner:
1. Copies data to temporary directory
2. Runs with `method: permute` (ground truth)
3. Runs with `method: hungarian` 
4. Compares output χ tensors with tolerances:
   - Relative tolerance: `rtol=1e-4` (0.01%)
   - Absolute tolerance: `atol=1e-6`

## Adding New Test Cases

To add a test for `diag_fewer`:

1. **Create data files** in `test_data/hungarian_assignment/diag_fewer/`:
   ```csv
   # exp.csv - Only 2 peaks
   assignment,shift,width,area
   H1, 120.5, 1, 1
   H2, 80.3, 1, 1
   
   # hfc.csv - 3 nuclei available
   atom_label,Aiso,Axx,Ayy,Azz
   H1, 5.2, 0.8, 0.8, 0.8
   H2, -3.1, 0.5, 0.5, 0.5
   H3, 1.9, 0.3, 0.3, 0.3
   ```

2. **Remove `@pytest.mark.skip`** from `test_hungarian_vs_permute_diag_fewer` in [test_hungarian_assignment.py](../../test_hungarian_assignment.py#L160)

3. **Run the test**:
   ```bash
   pytest -v tests/integration/test_hungarian_assignment.py::test_hungarian_vs_permute_diag_fewer
   ```

For full χ tensor tests, modify `input.yml`:
```yaml
susc_fit:
  type: general
  variables:
    chi_11: [fit, 0.001]
    chi_22: [fit, 0.001]
    chi_33: [fit, 0.001]
    chi_12: [fit, 0.0]
    chi_13: [fit, 0.0]
    chi_23: [fit, 0.0]
```

## Understanding Test Output

### Success
```
test_hungarian_vs_permute_diag_equal PASSED
```

### Failure
```
AssertionError: Mismatch for chi_iso: hungarian=0.001234, permute=0.001456
```

### Common Issues

1. **Timeout**: Test hangs > 60s → Check for infinite loop or slow convergence
2. **File not found**: Output CSV missing → Check project name in YAML matches test
3. **Tolerance exceeded**: Values differ → Increase `n_attempts` or investigate convergence

## CI/CD Integration

The `@pytest.mark.integration` decorator excludes these tests from default runs since they're slow. To run in CI:

```yaml
# .gitlab-ci.yml
test:
  script:
    - pytest -m integration
```

## Performance Notes

- **diag_equal**: ~5-10 seconds (3 nuclei, small search space)
- **diag_fewer**: ~3-7 seconds (underdetermined, faster)
- **full_equal**: ~15-30 seconds (6 parameters, larger search)
- **full_fewer**: ~10-20 seconds (fewer constraints)

Keep `n_attempts: 20` in test configs for speed. Production use `n_attempts: 50`.
