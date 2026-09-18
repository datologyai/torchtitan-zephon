# Zephon + TorchTitan

This example replaces TorchTitan's Grain training and validation dataloaders
with Zephon while leaving model construction, tokenizer selection,
optimization, distributed training, and checkpoint coordination in
TorchTitan. It demonstrates token-aware source mixtures, deterministic
checkpoint/resume, and an elastic resume at a different data-parallel degree.

The stock TorchTitan path is unchanged. Zephon is enabled through TorchTitan's
opt-in configuration overrides.

## Prerequisites

- Linux for GPU training; the data-only elastic demo also runs on macOS.
- Python 3.11 or newer and `uv`.
- A working TorchTitan environment with a PyTorch build compatible with the
  installed NVIDIA driver. The full demonstration requires CUDA.
- GitHub access to the private `datologyai/zephon` repository until Zephon has
  a public distribution.
- Two CUDA GPUs for the complete topology-change demonstration, or one CUDA GPU
  for a checkpoint/resume smoke test without a physical DP resize.

## Install

Create the TorchTitan environment using the normal project instructions, then
install the Zephon release pinned for this integration:

```bash
uv pip install -r requirements-zephon.txt
```

The pin resolves through the private Zephon GitHub repository and uses your
normal Git credentials. To validate a local release candidate instead, set
`ZEPHON_WHEEL=/path/to/zephon.whl` when running the clean-environment validation
script. For active Zephon development, install a sibling checkout with
`uv pip install -e /path/to/zephon`.

## Quick verification: CPU elastic demo

The fastest end-to-end check exercises the real Zephon pipeline and checkpoint
state without training a model:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
```

By default the demo checks both DP 2 -> 1 and DP 1 -> 2. For each direction it
creates an uninterrupted reference stream, checkpoints a second stream after
two global steps, and compares every TorchTitan trainer-batch field after the
resume. Lane-to-worker assignment may reorder batches after the topology
change, so comparison is order-independent within each global step. A mismatch
exits nonzero.

Pass `--work-dir PATH` to retain the checkpoint and raw JSON stream records for
diagnosis.

## Run the full GPU demo

On a Linux machine with two CUDA GPUs, run:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

Use a new output path for every invocation. Additional TorchTitan arguments may
follow the output path. After initial compilation and tokenizer setup, this
bounded debug-model run should complete in a few minutes on a typical
development GPU. The script runs two phases:

1. Train through step 2 on the configured GPU count and save TorchTitan model
   and Zephon dataloader checkpoints.
2. Restore the latest completed checkpoint on the configured GPU count and
   train step 3 with the same canonical lanes and logical global batch.

The final line is:

```text
Zephon training checkpoint/resume smoke test passed: ./outputs/zephon-training-smoke
```

Checkpoints, logs, and Zephon aggregation state are written beneath the output
path. On a single-GPU development box, run:

```bash
FIRST_PHASE_GPUS=1 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-1gpu
```

That exercises model and dataloader checkpoint/resume but does not demonstrate
a physical data-parallel resize.

Fixed TP and PP examples use the same runner:

```bash
FIRST_PHASE_GPUS=2 SECOND_PHASE_GPUS=2 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-tp2 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 1

FIRST_PHASE_GPUS=2 SECOND_PHASE_GPUS=2 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-pp2 \
  --parallelism.pipeline_parallel_degree 2 \
  --parallelism.pipeline_parallel_schedule 1F1B \
  --parallelism.num_pp_microbatches 2 \
  --parallelism.data_parallel_shard_degree 1
```

## Acceptance status

On 2026-09-18, PR #1 commit `5169889676447b25948bc6142483067b47097dfe`
passed the focused 10-test suite, the clean-install validator, both CPU elastic
directions, GPU DP 1 -> 1, DP 2 -> 1, DP 1 -> 2, TP=2 with DP=1, and PP=2 with
DP=1. The GPU runs used a two-GPU `g7.12xlarge` and resumed completed step-2
DCP checkpoints through step 3. This matrix does not establish support for
multi-node runs, elastic changes combined with model parallelism, or arbitrary
mixed topologies.

## What the demo does

1. Loads two raw-text JSONL sources from the checked-in TOML recipe.
2. Passes the configured 3:1 weights to Zephon as token proportions.
3. Uses bare `TokenEstimation()` to calibrate online source allocation.
4. Tokenizes with TorchTitan's tokenizer, splits long records, and packs
   fixed-length sequences online.
5. Converts each Zephon batch into TorchTitan's flattened `TrainerBatch`.
6. Saves Zephon stream state inside the corresponding completed TorchTitan
   checkpoint.
7. Restores the logical stream after changing the physical worker count.

For the exact pipeline and checkpoint contract, see the
[integration reference](../../docs/zephon.md).

## Use your own data

Copy `local_jsonl.toml` and replace its sources:

```toml
text_field = "text"
cache_dir = "/local-ssd/zephon"
seed = 42
chunk_size = 4

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

Relative local paths are resolved from the recipe directory. `fmt` maps
directly to `Dataset.from_path(..., fmt=...)`; omit it to use Zephon's format
detection. `cache_dir` applies Zephon's file cache to every source, including
remote paths that do not use `hf://`.

Weights are relative token proportions: `3.0` and `1.0` request a 75/25 token
mixture. The integration passes them unchanged to `MixtureSpec`; Zephon
normalizes them. Training always uses bare `TokenEstimation()`. There are no
estimator tuning knobs in the recipe and no post-tokenization
`ensure_mixture()` operation.

TorchTitan supplies the tokenizer, context length, and batch size, so those
settings remain in TorchTitan's configuration. Set `training = false` only for
a finite, unshuffled validation recipe without token estimation.

For elastic training, start from `elastic_local_jsonl.toml`. Keep the recipe,
canonical replica count, aggregate directory, run ID, tokenizer, context
length, seed, and logical global batch unchanged across the resume. The
physical data-parallel degree may change.

The local and elastic recipes use the same portable settings and eight-record
prose and code fixtures as the Megatron-LM-Zephon reference integration. The
validation recipe uses the prose fixture as a separate finite source.

## Advanced: manual launch

The smoke script is the canonical complete launch. To adapt an existing
TorchTitan command, enable the training override with:

```text
--override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"/path/to/recipe.toml"}'
```

Enable Zephon validation with a separate recipe:

```text
--override.imports 'torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader={"data_config":"/path/to/validation.toml"}'
```

Elastic launches also require stable `canonical_replicas`, `aggregate_dir`,
and `run_id` override values. All model, optimizer, batch, tokenizer,
distributed, checkpoint, and output settings remain normal TorchTitan
configuration. Resume only from a completed checkpoint.

## Next steps

Read the [integration reference](../../docs/zephon.md) for the supported recipe
fields, framework adapter, and checkpoint contract.

Run the focused adapter tests:

```bash
uv run --no-sync pytest -q tests/unit_tests/test_zephon_dataloader_override.py
```

Validate the pinned private release in a fresh temporary environment:

```bash
scripts/validate_zephon_install.sh
```

The initial reference integration supports online raw text only. It does not
support a separate pretokenized/prepacked path, arbitrary Zephon options, or
TorchTitan's `dataloader.max_num_documents` setting.
