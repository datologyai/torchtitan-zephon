# Zephon data recipes

These recipes are the only Zephon-specific input to the TorchTitan override.
Each source has a stable `name`, a local directory or `hf://` URI, an optional
`fmt`, and an optional positive `weight`. `fmt` maps directly to Zephon's
`Dataset.from_path(..., fmt=...)`; omit it to use Zephon's format detection.
Zephon normalizes weights automatically.

| Recipe | Purpose |
| --- | --- |
| `local_jsonl.toml` | Minimal 50/50 local JSONL smoke test. |
| `validation_local_jsonl.toml` | Separate deterministic validation source. |
| `weighted_local_jsonl.toml` | Local 3:1 prose/code mixture. |
| `elastic_local_jsonl.toml` | Local 3:1 mixture with two canonical data lanes. |
| `pretokenized_local_jsonl.toml` | Pretokenized, prepacked local records. |
| `hf_squad.toml` | Public Hugging Face Hub source. |

Use the recipe with the opt-in override described in
[`docs/zephon.md`](../../docs/zephon.md). Keep real source definitions in a
recipe rather than embedding them in a shell command; it makes their names,
paths, and mixture weights easy to review and adapt.

TorchTitan supplies the tokenizer to Zephon, so these data recipes do not
contain a tokenizer path.

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

On a Linux machine with one CUDA GPU, run a bounded TorchTitan training and
resume through the real checkpoint coordinator:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

The first launch trains through step 2 and saves full checkpoints. The second
launch restores the latest completed checkpoint and trains step 3 with a fresh
Zephon loader. Use a new output path for each invocation.

## Pretokenized and prepacked records

Set `input_mode = "pretokenized"` when every source record already contains
one complete `input_ids` sequence. Its length must be the configured TorchTitan
token budget plus one: the loader shifts the final value into the labels and
therefore emits exactly `num_tokens_per_microbatch_per_dp_rank` input tokens.
An optional `positions` sequence must have the same original length.

The [`pretokenized_local_jsonl.toml`](pretokenized_local_jsonl.toml) recipe and
checked-in fixture demonstrate this path. Zephon skips both tokenization and
packing; mixing, checkpointing, and elastic resume retain the same behavior.

## Elastic launch

The elastic recipe fixes the data stream at two canonical lanes. On one host
with two GPUs, provide a shared aggregate directory and stable run ID:

```bash
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/elastic_local_jsonl.toml","aggregate_dir":"/mnt/zephon-aggregate","run_id":"local-elastic-demo"}' \
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
