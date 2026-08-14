# Validation and Evaluation

`torchtitan` provides direct and indirect support for validation to support user's training goals. Direct support is provided by the `Validator` class which interacts directly with the training loop, and indirect support is provided through [HuggingFace checkpoint conversion](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md#huggingface) for users who want to do evaluation using external tools such as ELeutherAI's `lm_eval`.

## Validation
For users who want to perform validation directly during the training loop, we provide the `Validator` class which can be conveniently configured via `Validator.Config` in your config_registry function. The validator class has access to and reuses many of the trainer's functions such as its parallelization, including pipelining.

Below is an example validation config:

```python
validator=Validator.Config(
    freq=500,
    dataset="c4_validation",
    steps=-1,  # consumes the entire validation set
),
```

## Async Evaluation
Validation runs inside the training loop, so the training job waits for it. `AsyncEval` instead hands the step's weights to spare GPUs running *beside* training: it saves a model-only checkpoint, launches an eval runner as a subprocess, and keeps training. When the runner is done, the trainer logs its metrics to a separate `async_eval` TensorBoard run, at the training step the checkpoint was taken at.

```python
async_eval=AsyncEval.Config(
    enable=True,
    freq=1000,
    # Devices for the eval job. Without this it shares the training devices.
    cuda_visible_devices="6,7",
),
```

Async eval requires `checkpoint.enable=True`. The checkpoint the trainer writes is reused when the two cadences line up; otherwise a model-only checkpoint is written under `{dump_folder}/{checkpoint.folder}/async_eval/` and deleted once the eval job is done.

Metrics land in `{dump_folder}/{metrics.save_tb_folder}/async_eval/`, so pointing TensorBoard at `{dump_folder}` shows the eval curves next to the training curves:

```bash
tensorboard --logdir outputs
```

### Eval runners
By default the eval job runs `torchtitan.eval`, which loads the checkpoint and computes validation loss with the `Validator` above — the same numbers as in-loop validation, just off the critical path. It receives the training job's own arguments, so it evaluates the model that is being trained; use `async_eval.extra_args` for the eval-only bits:

```python
async_eval=AsyncEval.Config(
    enable=True,
    extra_args="--validator.dataloader.dataset c4_validation --validator.steps 50",
),
```

Any other runner works too — an `lm_eval` wrapper, a downstream task suite, a job submitted to a scheduler — as long as it honors the runner contract. `AsyncEval` appends four flags to the configured command:

| flag | meaning |
| --- | --- |
| `--checkpoint-dir` | model-only DCP checkpoint to evaluate |
| `--output-dir` | folder for this job's outputs |
| `--step` | training step the checkpoint was taken at |
| `--result-path` | file to write the metrics JSON to |

and the runner reports its metrics back by writing that JSON file:

```json
{"step": 1000, "metrics": {"loss": 2.31, "mmlu_acc": 0.62}}
```

Every numeric entry in `metrics` becomes a TensorBoard scalar. Point `async_eval.runner` at your own runner and turn off `forward_train_args` if it does not take torchtitan arguments:

```python
async_eval=AsyncEval.Config(
    enable=True,
    launcher="",
    runner="python ./my_eval/run_lm_eval.py",
    forward_train_args=False,
    extra_args="--tasks mmlu",
),
```

A failing eval job is logged and otherwise ignored, so evaluation cannot take down a training run. Set `async_eval.raise_on_failure=True` to make failures fatal instead.

## Third-Party Evaluation
With `./scripts/checkpoint_conversion/convert_to_hf.py`, `torchtitan` offers support for converting checkpoints from DCP to safetensors format. Using this script, users can perform efficient evaluation separate from their training using external libraries that support HuggingFace e.g. `lm_eval` with `vllm` backend.

### Example usage of `lm_eval` with `vllm`:
To use this specific setup make sure to include a HuggingFace `config.json` file which is not provided by conversion script or `last_save_in_hf` option. The HF config file can be downloaded by running `python ./scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets config`.

Note that pip installing `lm-eval` may result in breaking `torchtitan` dev environment so we recommend creating a separate env.
```bash
pip install "lm-eval[vllm]"
lm_eval --model vllm \
    --model_args pretrained=./outputs/checkpoint/step-1000,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8, \
    --tasks mmlu \
    --batch_size auto
```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6209|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5481|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.7045|±  |0.0078|
| - social sciences|      2|none  |      |acc   |↑  |0.7351|±  |0.0078|
| - stem           |      2|none  |      |acc   |↑  |0.5357|±  |0.0085|
