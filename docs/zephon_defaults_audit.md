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
| Packed record length | Context length plus one | Both reference integrations pack one model sequence per Zephon record. |
| Positions | Emitted for online packing | TorchTitan consumes packed-document positions when present. |
| Zephon batch size | Local batch size | Zephon emits the same sequence batch shape in both reference integrations; TorchTitan flattens it in its adapter. |
| Attention mask | Disabled | TorchTitan's causal attention path does not consume tokenizer masks. |
| Mixture unit | Tokens | Source weights should describe model-visible token proportions, not document counts. |
| Checkpoint representation | One opaque pickled byte value | TorchTitan DCP can preserve the complete public Zephon checkpoint object. |

## Workload tuning exposed by the recipe

Source names, paths, formats, and weights describe the workload. `text_field`,
`cache_dir`, `seed`, `chunk_size`, `fetch_parallelism`,
`canonical_replicas`, `aggregate_dir`, and `run_id` are also explicit because
they affect data semantics, throughput, or restart identity.

The override rejects unknown recipe keys. This is intentional: arbitrary
pipeline options are not forwarded because many interact with ordering,
checkpointing, memory use, or distributed correctness.

## Explicit Zephon choices for defaults review

| Choice | Training | Validation | Reason |
| --- | --- | --- | --- |
| Exhausted source policy | Repeat | Finite pass for `steps = -1`; repeat for bounded validation | Training is step-based, while validation follows TorchTitan's configured step policy. |
| Shard shuffle | Enabled | Disabled | Evaluation order stays stable and easy to inspect. |
| Within-shard shuffle | Enabled | Disabled | Training avoids long runs of adjacent records; evaluation preserves source order. |
| Token estimation | Zephon defaults | Disabled | Training weights describe token proportions; validation preserves a simple finite source pass. |
| Long sample handling | Split | Split | Preserve all usable tokens instead of truncating long records. |
| Special tokens | BOS and EOS | BOS and EOS | Stated explicitly because sample boundaries are part of the training contract. |
| Packing algorithm | Wrap | Wrap | Keep the example stream and sequence shape identical across the reference integrations. |

## Temporary constraints to revisit

- Online mode reads `tokenizer_path` from TorchTitan's tokenizer object because
  Zephon's public tokenizer API currently accepts a tokenizer identifier.
- TorchTitan's `max_num_documents` loader option is rejected until Zephon can
  provide the corresponding fixed-shape document metadata contract.
- Zephon checkpoint state is opaque bytes inside DCP. A future stable,
  structured serialization format could remove the pickle boundary.

These are integration constraints, not recommended Zephon defaults.
