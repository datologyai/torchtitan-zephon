# Zephon + TorchTitan

This example lets a TorchTitan config recipe use a Zephon pipeline instead of
the Grain dataloader for training or validation. Model construction,
optimization, distributed training, and checkpoint coordination remain in
TorchTitan. Zephon is opt-in; the stock TorchTitan path is unchanged.

For Zephon concepts and API details, use the
[Zephon User Guide](https://datologyai.github.io/zephon/). This page covers
only the TorchTitan adapter and its runnable examples.

## Install

Create a normal TorchTitan development environment, then install the current
Zephon `main` branch used by this integration:

```bash
uv pip install -r requirements-zephon.txt
```

The dependency currently resolves through the private `datologyai/zephon`
repository and requires GitHub access. For active Zephon development, install
a sibling checkout with `uv pip install -e /path/to/zephon`.

## CPU data-only elastic determinism demo

`elastic_resume_demo.py` tests that the dataloader stream remains deterministic
across checkpoint/resume with a changed DP degree, without constructing or
training a model:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
```

It checks both DP 2 -> 1 and DP 1 -> 2. Each direction compares an
uninterrupted stream with a stream restored after two global steps. Lane
assignment can change after a resize, so batches are compared without regard
to order inside each global step. Use `--work-dir PATH` to retain the generated
checkpoint and stream records.

## GPU trainer checkpoint smoke test

`run_training_smoke.sh` is the end-to-end example. It trains TorchTitan's
debug model through step 2, saves the model and Zephon dataloader state,
restores the checkpoint with a configurable GPU count, and completes step 3.

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

Use a new output path for each run. The default topology is two GPUs to one
GPU. A single-GPU checkpoint/resume run is:

```bash
FIRST_PHASE_GPUS=1 SECOND_PHASE_GPUS=1 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-1gpu
```

The trainer settings live in Python config recipes in
`torchtitan_recipes/zephon/recipes.py`: each phase runs `${RECIPE}_save` or
`${RECIPE}_resume`, and phase 2 loads the step-2 checkpoint with
`--resume-step 2`. The script passes the run's Zephon `aggregate_dir` and
`run_id` to the recipes as `ZEPHON_AGGREGATE_DIR` and `ZEPHON_RUN_ID`. For
example, fixed TP=2 can be tested with:

```bash
FIRST_PHASE_GPUS=2 SECOND_PHASE_GPUS=2 RECIPE=llama3_debugmodel_zephon_smoke_tp2 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-tp2
```

Additional `torchtitan.train` arguments can follow the output path.

## Configure data

Zephon data recipes are TOML files. Copy `local_jsonl.toml` and replace the
source entries:

```toml
text_field = "text"
cache_dir = "/local-ssd/zephon"
cache_limit_bytes = 536870912000
seed = 42
chunk_size = 16384
shuffle_block_size = "auto"
tokenize_parallelism = 8
pack_parallelism = 8

[[sources]]
name = "web"
path = "s3://example-bucket/web"
fmt = "parquet"
weight = 3.0

[[sources]]
name = "code"
path = "hf://organization/code/train"
weight = 1.0
```

Relative paths are resolved from the data recipe's directory. Source weights are
relative token proportions when `token_estimation = true`, which is the
training default. The shuffling controls are independent:

- `shuffle_shards` and `shuffle_within_shard` control source ordering.
- `shuffle_block_size` accepts `"auto"`, `"global"`, `"none"`, or a positive
  integer. `"none"` disables block shuffling.
- `shuffle_after_pack` controls the pipeline shuffle over packed sequences.

Shard prefetch is off by default. Set `prefetch_buffer_size` and optionally
`prefetch_parallelism` for remote datasets. `fetch_parallelism`,
`tokenize_parallelism`, `pack_parallelism`, and `shuffle_parallelism` tune the
individual stages. The default runner is `"process"`; MTP is enabled
automatically for this tokenize-and-pack pipeline unless `mtp_mode` is set
explicitly.

Validation should use a separate data recipe that explicitly selects its source,
repetition, shuffling, and token-estimation behavior, as shown by
`validation_local_jsonl.toml`.

## Use Zephon in a config recipe

A TorchTitan config recipe selects Zephon by setting its dataloader, for
training and optionally for validation:

```python
from torchtitan_recipes.models.llama3 import llama3_8b
from torchtitan_recipes.zephon import ZephonDataLoader


def llama3_8b_zephon():
    config = llama3_8b()
    config.dataloader = ZephonDataLoader.Config.from_toml(
        "/path/to/recipe.toml",
        aggregate_dir="/shared/zephon/run-42",
        run_id="run-42",
    )
    # For a recipe that configures a validator:
    # config.validator.dataloader = ZephonDataLoader.Config.from_toml(
    #     "/path/to/validation.toml"
    # )
    return config
```

```bash
python -m torchtitan.train --module my_package.recipes --config llama3_8b_zephon
```

`from_toml` reads a Zephon data recipe; its keyword arguments set config
fields on top of it. Elastic runs must keep the data recipe, tokenizer, seed,
logical global batch, `canonical_replicas`, `aggregate_dir`, and `run_id`
fixed, and the canonical lane count must be divisible by each supported DP
degree. For the order within a global window to survive a DP change, every
optimizer step should consume a whole number of canonical lane windows. A
checkpoint taken mid-window still resumes without skipped or repeated
samples; Zephon warns when it saves one, and a same-topology resume is exact.

The adapter stores Zephon's public checkpoint as opaque bytes in TorchTitan's
distributed checkpoint. It currently supports online raw text only.

Run the focused adapter tests with:

```bash
uv run --no-sync pytest -q tests/unit_tests/test_zephon_dataloader.py
```
