# Zephon data recipes

These recipes are the only Zephon-specific input to the TorchTitan override.
Each source has a stable `name`, a local directory or `hf://` URI, an optional
`fmt`, and an optional positive `weight`. `fmt` maps directly to Zephon's
`Dataset.from_path(..., fmt=...)`; omit it to use Zephon's format detection.
Zephon normalizes weights automatically and treats them as token proportions.
Its default `TokenEstimation()` calibrates online sources against the
TorchTitan tokenizer; no recipe flag is needed.

| Recipe | Purpose |
| --- | --- |
| `local_jsonl.toml` | Common 3:1 prose/code local JSONL smoke test. |
| `validation_local_jsonl.toml` | Separate deterministic validation source. |
| `weighted_local_jsonl.toml` | Additional explicit weighted-mixture example. |
| `elastic_local_jsonl.toml` | Local 3:1 mixture with two canonical data lanes. |
| `pretokenized_local_jsonl.toml` | Pretokenized, prepacked local records. |
| `hf_squad.toml` | Public Hugging Face Hub source. |

Use the recipe with the opt-in override described in
[`docs/zephon.md`](../../docs/zephon.md). Keep real source definitions in a
recipe rather than embedding them in a shell command; it makes their names,
paths, and mixture weights easy to review and adapt.

Add an optional `[token_estimation]` table only to tune Zephon's calibration
settings or pin known tokens-per-byte ratios. The supported keys map directly
to `TokenEstimation`: `primer`, `calibration_samples`,
`calibration_shards_min`, `calibration_shards_max`, and
`fallback_tokens_per_byte`.

TorchTitan supplies the tokenizer to Zephon, so these data recipes do not
contain a tokenizer path.

`local_jsonl.toml` and `elastic_local_jsonl.toml` intentionally use the same
portable data settings and fixtures as the Megatron-Zephon reference
integration. Framework-owned tokenizer, batch, and checkpoint settings stay
outside these reusable recipes.

## Elastic resume demo

Run the complete deterministic-resume demonstration on a CPU-only macOS or
Linux machine:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
```

The command creates an uninterrupted two-worker reference stream, checkpoints
the same stream after two steps, resumes it with one worker, and compares the
global token batches by fingerprint. It exits unsuccessfully if any token or
other trainer-batch field differs. Pass `--work-dir PATH` to keep each run's
checkpoint and JSON stream records for inspection.

## Training checkpoint smoke test

On a Linux machine with two CUDA GPUs, run the common bounded training and
elastic-resume workflow through TorchTitan's real checkpoint coordinator:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

The first launch trains through step 2 on two GPUs and saves full model and
dataloader checkpoints. The second launch restores the latest completed
checkpoint on one GPU and trains step 3 with the same logical global batch.
Use a new output path for each invocation. On a single-GPU development box,
set `FIRST_PHASE_GPUS=1`; this tests the complete checkpoint/resume path but
not the physical data-parallel resize.

## Pretokenized and prepacked records

Set `input_mode = "pretokenized"` when every source record already contains
one complete `input_ids` sequence. Its length must be the configured TorchTitan
token budget plus one: the loader shifts the final value into the labels and
therefore emits exactly `num_tokens_per_microbatch_per_dp_rank` input tokens.
An optional `positions` sequence must have the same original length.

The [`pretokenized_local_jsonl.toml`](pretokenized_local_jsonl.toml) recipe and
checked-in fixture demonstrate this path. Zephon skips both tokenization and
packing, measures token cost from `input_ids`, and retains the same mixing,
checkpointing, and elastic-resume behavior.

## Elastic TorchTitan launch

The elastic recipe fixes the data stream at two canonical lanes. On one host
with two GPUs, provide a shared aggregate directory and stable run ID:

```bash
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports \
  'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/elastic_local_jsonl.toml","aggregate_dir":"/mnt/zephon-aggregate","run_id":"local-elastic-demo"}' \
  'torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader={"data_config":"examples/zephon/validation_local_jsonl.toml"}' \
  --training.steps 4 \
  --training.max_context_length 128 \
  --training.num_tokens_per_microbatch_per_dp_rank 256 \
  --training.num_tokens_per_train_step 512 \
  --checkpoint.enable \
  --checkpoint.interval 2 \
  --dump_folder ./outputs/zephon-elastic
```

Keep the recipe, canonical lane count, aggregate directory, run ID, and train
step token budget unchanged when resuming with a different GPU count.

After the two-GPU run completes step 4, resume its latest completed checkpoint
with one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports \
  'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/elastic_local_jsonl.toml","aggregate_dir":"/mnt/zephon-aggregate","run_id":"local-elastic-demo"}' \
  'torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader={"data_config":"examples/zephon/validation_local_jsonl.toml"}' \
  --training.steps 6 \
  --training.max_context_length 128 \
  --training.num_tokens_per_microbatch_per_dp_rank 256 \
  --training.num_tokens_per_train_step 512 \
  --checkpoint.enable \
  --checkpoint.interval 2 \
  --dump_folder ./outputs/zephon-elastic
```

The per-rank microbatch is still 256 tokens, while each train step still
consumes 512 tokens globally. TorchTitan therefore accumulates two
microbatches on the single GPU. Resume only from a completed checkpoint and do
not change the recipe, canonical lane count, aggregate directory, run ID,
token budgets, tokenizer, or seed.
