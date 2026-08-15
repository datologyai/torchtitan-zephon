# Tests

This directory contains tests for the torchtitan project, including unit tests and integration tests.

## Test Structure

- `unit_tests/`: Contains unit tests for individual components. Tests marked
  `@pytest.mark.gpu` require one physical CUDA GPU and are excluded from CPU CI
- `integration_tests/`: Contains integration tests that test multiple components together
  - `features.py`: Tests for torchtitan features and composability, based on Llama3
  - `flux.py`: Tests for the FLUX model
  - `h100.py`: Tests cases for H100 GPUs
  - `models.py`: Tests for specific model architectures and configurations, other than Llama3 and FLUX
  - `run_numerics.py`: Loss and gradient norm guards against checked-in goldens
- `assets/`: Contains test assets and fixtures used by the tests
  - `losses/`: Golden loss and gradient norm curves for the numerics guards
  - `tokenizer/`: Tokenizer configuration and vocabulary files for testing
  - `custom_schedule.csv`: Custom PP schedule for testing

## TorchTitan CI Design

### Integration tests (Goal: E2E composability + Numerics)

- Feature tests: depth of infrastructure composability.
  - 1 GPU Fake PG tests verify that feature combinations configure, transform,
    and complete a training step. They do not validate distributed numerics or
    collective correctness.
  - 8 GPU Real PG tests cover features that require communication between real
    ranks.
- Model tests: width across supported models.
  - 1 GPU runs combine Fake PG composability coverage with real world-size-1
    numerics guards, comparing 10 steps of full-precision loss and gradient norm
    against checked-in A10G goldens.
  - 8 GPU runs retain the existing distributed loss comparisons and give each
    model a primary real distributed run whose loss can be inspected for
    convergence.

A test lands in the Fake PG or Real PG tier according to `requires_real_pg()` in
`integration_tests/__init__.py`: a test declares it directly with
`use_fake_pg=False`, and otherwise the tier is inferred from CLI overrides that
Fake PG cannot honor, namely checkpointing, pipeline parallelism, and an
explicit `--comm.mode` other than `fake_backend`.

### Unit tests (Goal: module functionality)

- CPU unit tests cover APIs and CPU functionality.
- GPU unit tests run GPU-marked module tests on one GPU and distributed module
  tests that require real collectives in a separate multi-GPU job.

## Running Tests

### Prerequisites

Ensure you have all development dependencies installed:

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### Running Integration Tests

To run the integration tests:

```bash
python -m tests.integration_tests.run_tests <output_dir> [--module MODULE] [--config CONFIG] [--test_suite TEST_SUITE] [--test_name TEST_NAME] [--ngpu NGPU]
```

Arguments:
- `output_dir`: (Required) Directory where test outputs will be stored
- `--module`: (Optional) Model module to use for training (default: "llama3"). Passed as `MODULE` env var to `run_train.sh`.
- `--config`: (Optional) Config function to use for training (default: "llama3_debugmodel"). Passed as `CONFIG` env var to `run_train.sh`.
- `--test_suite`: (Optional) Specific test suite to run by name (default: "features")
- `--fake_pg`: (Optional) Run the Fake PG tier of a feature or model suite. Without it, run the Real PG tier.
- `--test_name`: (Optional) Specific test to run by name (default: "all")
- `--ngpu`: (Optional) Number of GPUs to use for testing (default: 8)

Examples:
```bash
# Run fake-PG feature integration tests on one physical GPU
python -m tests.integration_tests.run_tests test_output --ngpu 1

# Run feature tests with a specific module and config
python -m tests.integration_tests.run_tests test_output --module llama3 --config llama3_8b

# Run feature tests that require real process groups
python -m tests.integration_tests.run_tests test_output --test_suite features

# Run a specific fake-PG feature test on one physical GPU
python -m tests.integration_tests.run_tests test_output --test_suite features --fake_pg --test_name gradient_accumulation_spmd_types --ngpu 1

# Run fake-PG model tests on one physical GPU
python -m tests.integration_tests.run_tests test_output --test_suite models --fake_pg --ngpu 1

# Run model tests that require real process groups
python -m tests.integration_tests.run_tests test_output --test_suite models --ngpu 8
```

### Running Unit Tests

To run only the unit tests:

```bash
pytest -s tests/unit_tests/
```

### Running Specific Unit Test Files

To run a specific test file:

```bash
pytest -s tests/unit_tests/test_config_manager.py
```

### Running Specific Test Functions in Unit Tests

To run a specific test function:

```bash
pytest -s tests/unit_tests/test_config_manager.py::TestConfigManager::test_cli_overrides
```
