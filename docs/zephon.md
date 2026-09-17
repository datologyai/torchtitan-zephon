# Zephon dataloader integration

This opt-in reference integration replaces TorchTitan's training and
validation `GrainDataLoader` instances with Zephon. TorchTitan continues to own
model construction, tokenizer selection, optimization, distributed training,
and checkpoint coordination.

For installation and copy-paste demonstrations, start with the
[Zephon example guide](../examples/zephon/README.md).

## Integration boundary

The portable part of the integration ends at the Zephon batch:

```text
TOML recipe -> Zephon datasets and mixture -> tokenize -> pack -> batch
                                                              |
                                                              v
                        TorchTitan adapter -> TrainerBatch -> trainer
```

The recipes, fixtures, dataset construction, mixture, token estimation, and
pipeline operations match the Megatron-LM-Zephon reference integration. The
implementations diverge only when they derive framework runtime topology,
adapt a Zephon batch for the trainer, and attach Zephon state to the framework
checkpoint API.

TorchTitan's adapter flattens each batched sequence tensor into the
one-dimensional `input`, `labels`, and optional `positions` fields expected by
`TrainerBatch`, and computes `num_valid_tokens`. Megatron preserves its
framework-facing batch as two-dimensional tensors. This is an intentional
framework boundary, not a difference in the Zephon data stream.

`preflight_tokenizers()` is TorchTitan-specific pipeline initialization. It
does not alter the shared recipe or sequence of data transformations.

## Shared data contract

This reference path supports online raw text only. For training it:

1. Creates each source with `Dataset.from_path`, using the recipe's path and
   optional format.
2. Passes configured source weights unchanged to `MixtureSpec`. The weights are
   relative token proportions; Zephon performs normalization.
3. Constructs `StaticMixtureWorkSource` with the recipe's `chunk_size` and
   seed, `exhausted_policy="repeat"`, shard and within-shard shuffling enabled,
   and a bare `TokenEstimation()`.
4. Optionally applies recipe-controlled fetch parallelism.
5. Tokenizes the configured text field with TorchTitan's tokenizer, splits long
   samples, adds the shared special-token policy, and does not emit tokenizer
   attention masks.
6. Packs with `pack_flat(max_length=max_context_length + 1, algorithm="wrap",
   emit_positions=True)`.
7. Batches with TorchTitan's local batch size and `drop_last=True`.

There is no post-tokenization `ensure_mixture()` operation and there are no
recipe knobs for token-estimation internals. Pretokenized and prepacked inputs
are outside the scope of this example.

## Recipe reference

Zephon-specific data configuration lives in a TOML recipe. Relative paths are
resolved from the recipe's directory.

### Data semantics

| Field | Required | Meaning |
| --- | --- | --- |
| `sources[].name` | Yes | Stable source identity. |
| `sources[].path` | Yes | Local directory or Zephon-supported URI. |
| `sources[].fmt` | No | Explicit format passed to `Dataset.from_path`; omit for detection. |
| `sources[].weight` | No | Positive relative token proportion; defaults to `1.0`. |
| `text_field` | No | Raw-text field to tokenize; defaults to `text`. |
| `seed` | No | Deterministic mixture seed. |
| `chunk_size` | No | Number of work items allocated together. The shared demos use `4`. |
| `training` | No | Defaults to `true`; enables shuffling and token estimation. Set to `false` for validation. |

### Runtime and topology

| Field | Required | Meaning |
| --- | --- | --- |
| `cache_dir` | No | Cache directory applied to every source, including non-`hf://` paths. |
| `fetch_parallelism` | No | Optional parallelism for fetching records. |
| `canonical_replicas` | Elastic runs | Stable logical data-lane count across topology changes. |
| `aggregate_dir` | Distributed elastic runs | Shared directory for aggregating lane checkpoint state. Usually supplied at launch. |
| `run_id` | Distributed elastic runs | Stable identity for one data stream. Usually supplied at launch. |

The loader rejects unknown keys rather than forwarding arbitrary Zephon
options. Ordering, checkpointing, memory use, and distributed correctness can
all depend on those choices, so this example keeps its public surface small.

## Checkpoint and elastic-resume contract

`ZephonDataLoader.state_dict()` stores Zephon's complete public checkpoint
object as one opaque byte value. TorchTitan's distributed checkpointer saves
that value alongside model, optimizer, scheduler, and trainer state. On
resume, TorchTitan restores the dataloader state from the same completed
checkpoint. Interrupted or partial checkpoints are not valid resume points.

For an elastic resume, keep these values unchanged:

- recipe and source identities;
- tokenizer, context length, and logical global batch;
- seed and canonical replica count;
- aggregate directory and run ID.

The physical data-parallel world size may change. Zephon preserves the same
canonical-lane batch contents within each global training step, but lane
presentation order may change when lanes are reassigned to workers. Therefore,
elastic comparisons are order-independent within each global step.

## Behavior and defaults

| Choice | Training behavior | Validation behavior | Reason |
| --- | --- | --- | --- |
| Mixture unit | Tokens | Records | Training weights describe model-visible token proportions; validation is a finite source pass. |
| Source exhaustion | Repeat | Finite for `steps = -1`; repeat for bounded validation | Match TorchTitan's step policies. |
| Shard order | Shuffled | Stable | Train on mixed data while keeping evaluation inspectable. |
| Within-shard order | Shuffled | Stable | Avoid adjacent-record runs during training without reordering evaluation. |
| Token estimation | Bare `TokenEstimation()` | Disabled | Calibrate online training sources without estimator tuning knobs. |
| Long samples | Split | Split | Preserve usable tokens instead of truncating records. |
| Special tokens | BOS and EOS | BOS and EOS | Make sample-boundary behavior explicit. |
| Packing | Wrap, with positions | Wrap, with positions | Produce complete fixed-length sequences for TorchTitan. |
| Attention mask | Not emitted | Not emitted | TorchTitan's causal attention path does not consume tokenizer masks. |
| Checkpoint state | Opaque Zephon object | Opaque Zephon object | Preserve the complete public checkpoint through TorchTitan DCP. |

Runtime rank, world size, and TorchTitan options are framework concerns rather
than portable recipe settings. Zephon packs records at
`max_context_length + 1` so the adapter can shift inputs into labels.

## Validation

The validation override replaces `validator.dataloader` with a separate Zephon
loader. The checked-in validation recipe sets `training = false`, which
disables shuffling and token estimation. Keep validation in its own recipe so
training mixtures cannot accidentally become evaluation data.

With `validator.steps = -1`, the validation loader stops after every source has
completed one pass. With a positive step count, it repeats until TorchTitan
reaches that bound, matching the stock loader's validation semantics.

## Temporary constraints

- Online mode reads `tokenizer_path` from TorchTitan's tokenizer because
  Zephon's public tokenizer API currently accepts a tokenizer identifier.
- `dataloader.max_num_documents` is rejected until Zephon supports the
  corresponding contract; it is never silently ignored.
- Zephon checkpoint state is opaque bytes inside TorchTitan DCP.
- The integration intentionally exposes only the reviewed recipe fields above.
- Zephon is a private dependency pinned to a reviewed Git release tag.

## Run and validate

Use the [example guide](../examples/zephon/README.md) for the CPU elastic demo,
full GPU smoke test, installation, and data-recipe walkthrough.

Run the focused adapter tests with:

```bash
uv run --no-sync pytest -q tests/unit_tests/test_zephon_dataloader_override.py
```

With access to the private Zephon release, validate installation and runtime
behavior in a fresh temporary environment:

```bash
scripts/validate_zephon_install.sh
```

Set `ZEPHON_WHEEL=/path/to/zephon.whl` to validate a local release candidate.
