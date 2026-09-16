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
| `elastic_local_jsonl.toml` | Local mixture with two canonical data lanes. |
| `hf_squad.toml` | Public Hugging Face Hub source. |

Use the recipe with the opt-in override described in
[`docs/zephon.md`](../../docs/zephon.md). Keep real source definitions in a
recipe rather than embedding them in a shell command; it makes their names,
paths, and mixture weights easy to review and adapt.

TorchTitan supplies the tokenizer to Zephon, so these data recipes do not
contain a tokenizer path.

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
