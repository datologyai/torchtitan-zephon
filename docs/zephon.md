# Zephon dataloader override

This opt-in reference integration replaces TorchTitan's training and
validation `GrainDataLoader` instances with Zephon. Models, tokenizers,
optimizers, trainer logic, and TorchTitan's distributed checkpointer remain
unchanged.

The integration demonstrates three Zephon capabilities in a small, adaptable
surface: token-aware weighted mixtures, deterministic resume through
TorchTitan checkpoints, and elastic resume at a different data-parallel
degree.

The training override replaces the `dataloader` node. The validation override
replaces `validator.dataloader`. Zephon reads each node's recipe, constructs a
deterministic mixture, tokenizes and packs text online, and stores training
state inside TorchTitan checkpoints.

## Install and run

Until Zephon has a suitable public PyPI release, `requirements-zephon.txt` pins
a release tag from the Zephon Git repository:

```bash
uv pip install --python .venv/bin/python -r requirements-zephon.txt
```

Run the checked-in local example on one CUDA GPU:

```bash
NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports \
  'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/local_jsonl.toml"}' \
  'torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader={"data_config":"examples/zephon/validation_local_jsonl.toml"}' \
  --training.steps 10 \
  --training.max_context_length 128 \
  --training.num_tokens_per_microbatch_per_dp_rank 256
```

The tiny checked-in data makes the command reproducible; it is not intended as
a representative training corpus.

Training shuffles and repeats its mixture. Validation is deterministic and
unshuffled. With `validator.steps = -1`, Zephon stops after every validation
source has completed a pass. With a positive validation step count,
TorchTitan repeats the validation stream until that bound, matching its stock
loader semantics. Keep validation in a separate recipe so training mixtures
cannot accidentally become evaluation data.

## Configure sources and mixtures

Zephon-specific configuration lives in a TOML recipe:

```toml
input_mode = "online"
text_field = "text"
cache_dir = "/local-ssd/zephon"
seed = 42

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

Each source has a stable name and path. `fmt` maps directly to
`Dataset.from_path(..., fmt=...)`; omit it to use Zephon's format detection.
Weights are relative token proportions and Zephon normalizes them, so `3.0`
and `1.0` request a 75/25 token mixture. Token-aware allocation is enabled by
default: Zephon calibrates each online source against TorchTitan's tokenizer,
schedules fewer records from sources with longer documents, and then uses
`ensure_mixture` after tokenization to smooth the remaining token-level drift.
The calibrated ratios are checkpointed and reused on resume.

The integration deliberately uses Zephon's `TokenEstimation()` defaults rather
than adding recipe knobs for calibration internals. `cache_dir` enables
Zephon's file cache for all sources, not only `hf://` paths.

The [example catalog](../examples/zephon/README.md) includes equal-weight,
weighted, Hugging Face, elastic, and pretokenized recipes. Zephon reads Hugging
Face Parquet shards directly; the pinned `hf` extra supplies the needed support.

## Choose the input contract

`input_mode = "online"` expects a text field. Zephon tokenizes with the
TorchTitan tokenizer, splits long samples, adds BOS/EOS boundaries, and packs
the result into complete training sequences.

`input_mode = "pretokenized"` skips tokenization and packing. Every record must
contain one `input_ids` sequence whose length is the configured per-rank
microbatch token budget plus one. Optional `positions` must have the same
length. The extra token becomes the final label, leaving exactly the requested
number of trainer input tokens. Token estimation measures `input_ids` length
directly in this mode.

## Checkpoint and elastic resume contract

`ZephonDataLoader.state_dict()` stores Zephon's complete public checkpoint
object as one opaque byte value. TorchTitan's distributed checkpointer saves
that value alongside model, optimizer, scheduler, and trainer state. Restore
must use the latest completed TorchTitan checkpoint; an interrupted or partial
checkpoint is not a valid resume point.

For elastic resume, keep these values stable:

- `canonical_replicas`: the fixed logical data-lane count; it must be at least
  the current DP world size.
- `aggregate_dir`: shared storage used to aggregate lane checkpoint state.
- `run_id`: the stable identity of this data stream.
- Recipe, tokenizer, token budgets, and seed.

The current DP world size may change. The canonical lane count must not.

Run the exact 2-worker to 1-worker proof on CPU-only macOS or Linux:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
```

It builds an uninterrupted two-worker reference, checkpoints after two steps,
resumes with one worker, and compares every TorchTitan trainer-batch field. A
mismatch exits nonzero.

Run the real TorchTitan checkpoint coordinator on one CUDA GPU:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

For the manual multi-GPU to single-GPU commands, see the catalog's
[elastic TorchTitan launch](../examples/zephon/README.md#elastic-torchtitan-launch).

## Validation and scope

Run the focused tests with:

```bash
uv run pytest tests/unit_tests/test_zephon_dataloader_override.py \
  tests/unit_tests/test_zephon_elastic_resume.py
```

`scripts/validate_zephon_install.sh` creates a clean environment, installs the
pinned release rather than a sibling checkout, and runs the release-facing
tests. Set `ZEPHON_WHEEL=/path/to/zephon.whl` to validate a local release wheel.

This example intentionally does not forward arbitrary Zephon options or
promise a general pretokenized schema.
`dataloader.max_num_documents` is not yet supported by Zephon; setting it fails
configuration validation instead of being ignored. The explicit choices and
temporary constraints are recorded in the
[defaults audit](zephon_defaults_audit.md).
