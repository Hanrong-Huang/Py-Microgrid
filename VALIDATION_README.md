# Backwards Compatibility Validation

This directory contains validation scripts to ensure the py_microgrid refactoring maintains backwards compatibility.

## Files

- `validation_test.py` - Main validation test script
- `compare_validation_results.py` - Script to compare results between versions
- `VALIDATION_README.md` - This file

## How to Run Validation

### Step 1: Run on Current (New) Version

```bash
# Run validation on new version
python validation_test.py
# This creates: validation_results.json
```

### Step 2: Run on Baseline (Old) Version

```bash
# Save current results with version suffix
mv validation_results.json validation_results_new.json

# Go back to baseline version (before refactoring)
git checkout 0a30f46

# Run validation on old version
python validation_test.py
# This creates: validation_results.json

# Save with version suffix
mv validation_results.json validation_results_old.json

# Return to current version
git checkout main
```

### Step 3: Compare Results

```bash
# Compare the results
python compare_validation_results.py validation_results_old.json validation_results_new.json
```

## Expected Results

For backwards compatibility, the following metrics should be **identical** between old and new versions:

### Component Sizes
- PV Capacity (kW)
- Wind Capacity (kW) 
- Battery Energy (kWh)
- Battery Power (kW)
- Genset Capacity (kW)

### Financial Metrics
- System LCOE ($/kWh)
- System NPC ($)
- CO2 Emissions (tonnes)

### Performance Metrics
- Demand Met (%)
- Total Generation (kWh)
- Component-wise Generation (kWh)

## Key Configuration for Validation

The validation test uses this configuration to ensure backwards compatibility:

```yaml
technologies:
  pv:
    enabled: true
    # ... pv settings
  wind:
    enabled: true
    # ... wind settings
  battery:
    enabled: true
    # ... battery settings
  genset:
    enabled: true
    # ... genset settings (this was "grid" in old version)
  grid:
    enabled: false  # CRITICAL: Must be false for backwards compatibility
    # ... grid settings
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the right environment and py_microgrid is installed
2. **File Not Found**: Check that resource files exist in the expected locations
3. **Optimization Failed**: The bounds might be too restrictive for your system

### Debug Mode

To run with more verbose output:

```bash
# Add debug prints to validation_test.py
export PYTHONUNBUFFERED=1
python validation_test.py
```

## Interpretation of Results

### ✅ Success Case
```
✅ BACKWARDS COMPATIBILITY VALIDATED
All metrics match within acceptable tolerances!
Summary: 15/15 metrics match
```

### ❌ Failure Case
```
❌ BACKWARDS COMPATIBILITY ISSUES DETECTED
Failing metrics: 3
Summary: 12/15 metrics match
```

If you see failures, check:
1. Are the differences small (< 1%)?
2. Are only generation values different (this might be OK)?
3. Are the component sizes identical (this is critical)?

## Tolerance Levels

The comparison uses these tolerance levels:
- **Component Sizes**: Very strict (< 1e-6)
- **Financial Metrics**: Strict (< 1e-3)
- **Dollar Amounts**: Moderate ($1)
- **Percentages**: Moderate (0.1%)

## Contact

If you encounter issues with the validation, check:
1. Git commit versions match what you expect
2. All required files are present
3. Dependencies are installed correctly