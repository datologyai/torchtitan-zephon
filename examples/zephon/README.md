# Zephon + TorchTitan

This example replaces TorchTitan's Grain training or validation dataloader
with a Zephon pipeline. Model construction, optimization, distributed
training, and checkpoint coordination remain in TorchTitan. Zephon is opt-in;
the stock TorchTitan path is unchanged.

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

## CPU data-only elastic demo

`elastic_resume_demo.py` tests the dataloader and its checkpoint without
constructing or training a model:

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

Additional TorchTitan arguments can follow the output path. For example,
fixed TP=2 can be tested with:

```bash
FIRST_PHASE_GPUS=2 SECOND_PHASE_GPUS=2 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-tp2 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 1
```

## Configure data

Copy `local_jsonl.toml` and replace the source entries:

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

Relative paths are resolved from the recipe directory. Source weights are
relative token proportions when `token_estimation = true`, which is the
training default. The shuffling controls are independent:

- `shuffle_shards` and `shuffle_within_shard` control source ordering.
- `shuffle_block_size` accepts `"auto"`, `"global"`, or a positive integer.
- `shuffle_after_pack` controls the pipeline shuffle over packed sequences.

Shard prefetch is off by default. Set `prefetch_buffer_size` and optionally
`prefetch_parallelism` for remote datasets. `fetch_parallelism`,
`tokenize_parallelism`, `pack_parallelism`, and `shuffle_parallelism` tune the
individual stages. The default runner is `"process"`; MTP is enabled
automatically for this tokenize-and-pack pipeline unless `mtp_mode` is set
explicitly.

Validation should use a separate recipe that explicitly selects its source,
repetition, shuffling, and token-estimation behavior, as shown by
`validation_local_jsonl.toml`.

## Enable the overrides

Training:

```text
--override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"/path/to/recipe.toml"}'
```

Validation:

```text
--override.imports 'torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader={"data_config":"/path/to/validation.toml"}'
```

The override requires an explicit Zephon recipe; it does not translate an
existing Grain dataset configuration. Elastic runs must keep the
recipe, tokenizer, seed, logical global batch, `canonical_replicas`,
`aggregate_dir`, and `run_id` fixed. The canonical lane count must be divisible
by each supported DP degree, and every optimizer step must consume a whole
number of canonical lane windows.

The adapter stores Zephon's public checkpoint as opaque bytes in TorchTitan's
distributed checkpoint. It currently supports online raw text only and
rejects `dataloader.max_num_documents`.

Run the focused adapter tests with:

```bash
uv run --no-sync pytest -q tests/unit_tests/test_zephon_dataloader_override.py
```
