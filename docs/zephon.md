# Zephon dataloader override

This opt-in example replaces TorchTitan's training and validation dataloaders
with Zephon pipelines. It is intentionally small: models, optimizers, trainer
logic, and checkpoint coordination remain unchanged.

## What this demo changes

The training override replaces the `dataloader` node. The validation override
replaces `validator.dataloader`. Zephon reads each node's recipe, constructs a
deterministic mixture, tokenizes and packs text online, and stores training
state inside TorchTitan checkpoints.

The mechanism is an ordinary TorchTitan override; see the
[override reference](../torchtitan/overrides/README.md) for its general model.

## Quickstart

Install the pinned Zephon release into the environment used to run TorchTitan,
then launch the small local JSONL example:

```bash
uv pip install --python .venv/bin/python \
  "zephon @ git+ssh://git@github.com/datologyai/zephon.git@v0.0.1b21" \
  transformers

NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
  --override.imports \
  'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/local_jsonl.toml"}' \
  'torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader={"data_config":"examples/zephon/validation_local_jsonl.toml"}' \
  --training.steps 10 \
  --training.max_context_length 128 \
  --training.num_tokens_per_microbatch_per_dp_rank 256
```

The checked-in data is deliberately tiny and exists only to make the command
reproducible. Replace its recipe with real sources for training.

Training shuffles and repeats its mixture. Validation is deterministic and
unshuffled. With `validator.steps = -1`, Zephon stops after every validation
source has completed a pass. With a positive validation step count,
TorchTitan repeats the validation stream until that bound, matching its stock
loader semantics. Keep validation in a separate recipe so training mixtures
cannot accidentally become evaluation data.

## Data recipes

Keep Zephon settings and source definitions in a TOML recipe. A source has a
stable name, path, optional `fmt`, and optional relative weight:

```toml
text_field = "text"
cache_dir = "/local-ssd/zephon"

[[sources]]
name = "web"
path = "s3://example-bucket/web"
fmt = "parquet"
weight = 3.0

[[sources]]
name = "code"
path = "s3://example-bucket/code"
weight = 1.0
```

Zephon uses the tokenizer built by the selected TorchTitan configuration, so
the recipe deliberately does not contain a tokenizer path. Configure the
tokenizer with the usual TorchTitan model configuration.

`fmt` maps directly to `Dataset.from_path(..., fmt=...)`; omit it to use
Zephon's format detection. Zephon normalizes relative weights, so the example
requests a 75/25 web/code mixture. `cache_dir` is explicit: when supplied, it
enables a per-node cache for every file-backed source, including cloud URIs and
network-mounted paths.

For a Hugging Face Hub source, install Zephon's optional HF dependencies and
use an `hf://` URI. The included [`hf_squad.toml`](../examples/zephon/hf_squad.toml)
shows the full form:

```bash
uv pip install --python .venv/bin/python \
  "zephon[hf] @ git+ssh://git@github.com/datologyai/zephon.git@v0.0.1b21" \
  transformers
```

Zephon reads Hugging Face Parquet shards directly; the `datasets` package is
not required.

## Why Zephon?

- **Weighted mixtures:** sources are named and weighted in one small recipe;
  their raw weights need not sum to one.
- **Elastic deterministic resume:** Zephon checkpoints the data stream by
  canonical lanes rather than current worker count, so a run can resume with a
  different data-parallel degree while retaining deterministic global progress.

## More examples and development

The [recipe catalog](../examples/zephon/README.md) includes equal-weight,
weighted, Hugging Face, and elastic local examples. Its
[elastic launch section](../examples/zephon/README.md#elastic-launch) has the
two-GPU command.

For contribution and test guidance, see [CONTRIBUTING.md](../CONTRIBUTING.md)
and [tests/README.md](../tests/README.md).
