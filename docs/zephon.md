# Zephon dataloader override

This repository includes a small, opt-in example that replaces TorchTitan's
training text dataloader with a Zephon pipeline. It mixes two JSONL sources,
tokenizes and packs them online, and stores Zephon's full checkpoint state with
the TorchTitan checkpoint.

For local development, install Zephon from the sibling checkout into the same
virtual environment as this repository:

```bash
python -m pip install -e /path/to/zephon
```

Run the debug model with the demonstration mixture:

```bash
NGPU=1 ./run_train.sh \
  --override.imports torchtitan.overrides.zephon_dataloader.zephon_dataloader \
  --training.steps 10 \
  --training.seq_len 128
```

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

Override kwargs configure non-demo sources:

```bash
./run_train.sh \
  --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"sources":{"web":"/data/web","code":"/data/code"},"mixture":{"web":0.7,"code":0.3},"canonical_replicas":4,"aggregate_dir":"/shared/zephon","run_id":"example"}'
```
