# Zephon dataloader override

This repository includes a small, opt-in example that replaces TorchTitan's
training text dataloader with a Zephon pipeline. It mixes two JSONL sources,
tokenizes and packs them online, and stores Zephon's full checkpoint state with
the TorchTitan checkpoint.

For local development with local files only, install Zephon from the sibling
checkout into the same virtual environment as this repository:

```bash
python -m pip install -e /path/to/zephon
```

To read Hugging Face Hub datasets through Zephon's ``hf://`` backend, install
the optional Hugging Face dependencies instead:

```bash
python -m pip install -e '/path/to/zephon[hf]'
```

No ``datasets`` dependency is needed. Zephon reads the Hub dataset's Parquet
shards directly. Use ``HF_TOKEN`` (or ``huggingface-cli login``) for gated
datasets.

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

``cache_dir`` is a writable, per-node cache for remote shards and decoded
Parquet data. Its default is ``./.zephon-cache``; set it to a suitably sized
local disk path for real runs. It is separate from the shared ``aggregate_dir``
used for elastic checkpoint aggregation.

Override kwargs configure non-demo sources:

```bash
./run_train.sh \
  --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"sources":[{"name":"web","path":"hf://HuggingFaceFW/fineweb-edu/train","weight":3.0},{"name":"code","path":"/data/code"}],"text_field":"text","cache_dir":"/local/zephon-cache","canonical_replicas":4,"aggregate_dir":"/shared/zephon","run_id":"example"}'
```
