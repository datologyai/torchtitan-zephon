# Zephon defaults audit

This integration keeps the public surface deliberately small. The table below
records every non-default Zephon choice made by the override and why it is not
left implicit.

## Inherent integration choices

| Choice | Value | Reason |
| --- | --- | --- |
| Data parallel identity | TorchTitan DP degree and rank | Each TorchTitan DP rank owns one Zephon replica. |
| Process identity | Distributed world size and global rank, when initialized | Zephon coordinates all participating processes. |
| Determinism | Enabled | Checkpoint/resume and elastic demonstrations require a reproducible stream. |
| Tokenizer limit | `max_context_length + 1` | One extra token is needed to shift inputs into labels. |
| Packed record length | Microbatch token budget plus one | The trainer receives exactly its configured token budget after shifting. |
| Positions | Emitted for online packing | TorchTitan consumes packed-document positions when present. |
| Zephon batch size | One | One packed record already represents one TorchTitan token microbatch. |
| Attention mask | Disabled | TorchTitan's causal attention path does not consume tokenizer masks. |
| Checkpoint representation | One opaque pickled byte value | TorchTitan DCP can preserve the complete public Zephon checkpoint object. |

## Workload tuning exposed by the recipe

Source names, paths, formats, and weights describe the workload. `input_mode`,
`text_field`, `cache_dir`, `seed`, `chunk_size`, `fetch_parallelism`,
`canonical_replicas`, `aggregate_dir`, and `run_id` are also explicit because
they affect data semantics, throughput, or restart identity.

The override rejects unknown recipe keys. This is intentional: arbitrary
pipeline options are not forwarded because many interact with ordering,
checkpointing, memory use, or distributed correctness.

## Explicit Zephon choices for defaults review

| Choice | Override | Zephon default | Reason |
| --- | --- | --- | --- |
| Exhausted source policy | Repeat | Finite pass | TorchTitan training is step-based and expects an effectively unbounded stream. |
| Shard shuffle | Enabled | Enabled | Stated explicitly because it is part of the deterministic ordering contract. |
| Within-shard shuffle | Enabled | Disabled | Avoid long runs of adjacent records from large shards. |
| Long sample handling | Split | Do not split | Preserve all usable tokens instead of truncating long training records. |
| Special tokens | BOS and EOS | BOS and EOS | Stated explicitly because sample boundaries are part of the training contract. |
| Packing algorithm | Wrap | First fit | Keep the example stream simple and deterministic across canonical lanes. |

## Temporary constraints to revisit

- Online mode reads `tokenizer_path` from TorchTitan's tokenizer object because
  Zephon's public tokenizer API currently accepts a tokenizer identifier.
- Pretokenized mode validates records as they are fetched. It does not perform a
  separate declarative schema scan before iteration.
- Zephon checkpoint state is opaque bytes inside DCP. A future stable,
  structured serialization format could remove the pickle boundary.

These are integration constraints, not recommended Zephon defaults.
