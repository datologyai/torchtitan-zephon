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
schedules fewer records from sources with longer documents, and checkpoints
the calibrated ratios for reuse on resume.

`training` defaults to `true`. Set `training = false` in a validation recipe
to disable training-time shuffling and token estimation. The training and
validation overrides supply those same defaults when the field is omitted.

The integration deliberately uses Zephon's `TokenEstimation()` defaults rather
than adding recipe knobs for calibration internals. `cache_dir` enables
Zephon's file cache for all sources, not only `hf://` paths.

The [example catalog](../examples/zephon/README.md) keeps only the local,
validation, and elastic recipes used by the demonstrations. Remote sources use
the same recipe structure shown above; the pinned `hf` extra lets Zephon read
Hugging Face Parquet shards directly.

Zephon tokenizes each source's text field with the TorchTitan tokenizer, splits
long samples, adds BOS/EOS boundaries, and packs the result into complete
training sequences. It batches those sequences at TorchTitan's configured
local batch size; the framework adapter flattens that batch only when producing
TorchTitan's trainer-batch structure.

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
resumes with one worker, and compares every TorchTitan trainer-batch field in
each global step. Lane-to-worker assignment may reorder those batches after a
topology change, so the comparison is order-independent within a global step.
A mismatch exits nonzero.

Run the real TorchTitan checkpoint coordinator on one CUDA GPU:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

For the manual multi-GPU to single-GPU commands, see the catalog's
[elastic TorchTitan launch](../examples/zephon/README.md#elastic-torchtitan-launch).

## Validation and scope

Run the focused tests with:

```bash
uv run pytest tests/unit_tests/test_zephon_dataloader_override.py
```

`scripts/validate_zephon_install.sh` creates a clean environment, installs the
pinned release rather than a sibling checkout, and runs the release-facing
tests. Set `ZEPHON_WHEEL=/path/to/zephon.whl` to validate a local release wheel.

This example intentionally does not forward arbitrary Zephon options or
support a separate pretokenized/prepacked input path.
`dataloader.max_num_documents` is not yet supported by Zephon; setting it fails
configuration validation instead of being ignored. The explicit choices and
temporary constraints are recorded in the
[defaults audit](zephon_defaults_audit.md).
