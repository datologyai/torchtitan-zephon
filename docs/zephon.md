# Zephon dataloader override

This repository includes a small, opt-in example that replaces TorchTitan's
training text dataloader with a Zephon pipeline. It mixes two JSONL sources,
tokenizes and packs them online, and stores Zephon's full checkpoint state with
the TorchTitan checkpoint.

For local development with local files only, install Zephon from the sibling
checkout into this repository's uv environment:

```bash
uv pip install --python .venv/bin/python -e /path/to/zephon transformers
```

To read Hugging Face Hub datasets through Zephon's ``hf://`` backend, install
the optional Hugging Face dependencies instead:

```bash
uv pip install --python .venv/bin/python -e '/path/to/zephon[hf]' transformers
```

No ``datasets`` dependency is needed. Zephon reads the Hub dataset's Parquet
shards directly. Use ``HF_TOKEN`` (or ``huggingface-cli login``) for gated
datasets. ``transformers`` is required by Zephon's tokenizer stage.

Run the debug model with the demonstration mixture:

```bash
NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/local_jsonl.toml"}' \
  --training.steps 10 \
  --training.seq_len 128 \
  --training.local_batch_size 2
```

If you manually install a PyTorch nightly, activate `.venv` and run
`./run_train.sh` directly, or use `uv run --no-sync`; a normal `uv run` may
replace the nightly during dependency synchronization.

The data recipe contains the source list and all Zephon-specific data settings.
Relative local paths are resolved from the recipe file. Start from
``examples/zephon/local_jsonl.toml`` for local JSONL, or
``examples/zephon/hf_squad.toml`` for a public Hub source.

The checked-in local recipes are deliberately small, but their larger
``chunk_size`` reduces end-of-work packing warnings. A warning about dropping
a trailing partial pack is expected when the finite fixture does not fill the
final ``seq_len + 1`` token bin. It represents only the unfinished tail of a
work chunk, not a loss of complete training batches.

The override is deliberately scoped to the training `dataloader` node. It does
not replace validation, chat, interleaved, or multimodal loaders.

For distributed elastic checkpointing, keep `canonical_replicas` and the seed
constant across runs, supply a shared `aggregate_dir` and stable `run_id`, and
set `training.global_batch_size` to `training.local_batch_size *
canonical_replicas`. This lets every optimizer step consume a complete cycle of
canonical lanes even if the data-parallel degree changes.

The CPU integration test runs this exact scenario with Gloo: it checkpoints on
two ranks, resumes on one rank, and compares the global batches with an
uninterrupted two-rank baseline. It needs no GPU or Linux host:

```bash
HF_HOME="$PWD/.hf-cache" python -m pytest -q \
  tests/unit_tests/test_zephon_elastic_resume.py
```

Each source has a stable name, a local path or ``hf://`` URI, and an optional
relative weight. Weights default to ``1.0`` and Zephon normalizes them, so two
sources with omitted weights receive a 50/50 mixture. The URI includes the
Hugging Face split, and optionally a revision and config:

```text
hf://org/dataset/train
hf://org/dataset@revision/config/train
```

All sources in a mixture use the same text column. Set ``text_field`` when a
dataset uses a different name, such as SQuAD's ``context`` column.

## Mixtures

``local_jsonl.toml`` leaves both weights unspecified, so its two sources are
each sampled at 50%. ``weighted_local_jsonl.toml`` makes the same relationship
explicit: `prose = 3.0` and `code = 1.0`, which Zephon normalizes to a 75/25
requested mixture. The observed ratio converges to that target over a real
run; a short smoke run is intentionally too small to be a useful statistical
measurement.

## Elastic launch recipe

``elastic_local_jsonl.toml`` fixes the data stream at two canonical lanes. On
a single multi-GPU host, provide a stable shared aggregate directory and run
identifier at launch time:

```bash
CUDA_VISIBLE_DEVICES=0,1 NGPU=2 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/elastic_local_jsonl.toml","aggregate_dir":"/mnt/zephon-aggregate","run_id":"local-elastic-demo"}' \
  --training.steps 4 \
  --training.seq_len 128 \
  --training.global_batch_size 4 \
  --training.local_batch_size 2 \
  --checkpoint.enable \
  --checkpoint.interval 2 \
  --dump_folder ./outputs/zephon-elastic
```

The aggregate directory must be writable by every rank and persist across a
resume. Keep the recipe, `canonical_replicas`, aggregate directory, run ID,
and global batch size unchanged when changing GPU count.

``cache_dir`` is a writable, per-node cache for remote shards and decoded
Parquet data. Its default is ``./.zephon-cache``; set it to a suitably sized
local disk path for real runs. It is separate from the shared ``aggregate_dir``
used for elastic checkpoint aggregation.

To use a custom recipe, pass its path rather than embedding the source list in
the command line:

```bash
./run_train.sh \
  --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"/path/to/my_mixture.toml"}'
```

Explicit override kwargs still take precedence over the recipe when a one-off
experiment needs to adjust a setting.
