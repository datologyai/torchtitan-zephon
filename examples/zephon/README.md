# Zephon data recipes

These recipes are the only Zephon-specific input to the TorchTitan override.
Each source has a stable `name`, a local directory or `hf://` URI, and an
optional positive `weight`. Zephon normalizes weights automatically.

| Recipe | Purpose |
| --- | --- |
| `local_jsonl.toml` | Minimal 50/50 local JSONL smoke test. |
| `weighted_local_jsonl.toml` | Local 3:1 prose/code mixture. |
| `elastic_local_jsonl.toml` | Local mixture with two canonical data lanes. |
| `hf_squad.toml` | Public Hugging Face Hub source. |

Use the recipe with the opt-in override described in
[`docs/zephon.md`](../../docs/zephon.md). Keep real source definitions in a
recipe rather than embedding them in a shell command; it makes their names,
paths, and mixture weights easy to review and adapt.
