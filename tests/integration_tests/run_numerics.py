# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run single-GPU and distributed model numerics integration tests.

Both suites drive ``scripts/loss_compare.py``, which owns running training,
reading metrics out of TensorBoard, and comparing them against the checked-in
goldens under ``tests/assets/losses``.
"""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOSSES = REPO_ROOT / "tests/assets/losses"

# Unsharded runs pin every parallelism axis to 1 so the golden depends only on
# the model.
SINGLE_GPU_OPTIONS = (
    "--parallelism.data_parallel_replicate_degree=1 "
    "--parallelism.data_parallel_shard_degree=1 "
    "--parallelism.tensor_parallel_degree=1 "
    "--parallelism.context_parallel_degree=1 "
    "--parallelism.pipeline_parallel_degree=1 "
    "--parallelism.expert_parallel_degree=1"
)

# DeepSeek and GPT-OSS grouped MoE GEMMs pass CPU offsets that cannot be copied
# during CUDA graph capture. Other models retain their configured default.
SINGLE_GPU_MODEL_OPTIONS = {
    "deepseek_v3": "--training.disable_cuda_graphs",
    "gpt_oss": "--training.disable_cuda_graphs",
}


def build_1gpu_numerics_test_list() -> dict[str, str]:
    """Guard unsharded model numerics with exact 10-step loss and grad norm."""
    # Kimi and Muse use their text-only paths first. Their vision numerics are
    # separate follow-up coverage; Kimi's bicubic CUDA backward is currently
    # incompatible with deterministic mode. Flux is deferred because its
    # unsharded FP32 T5-XXL encoder does not fit on a 24 GB A10G.
    # Qwen3.5 is deferred until FLA chunked autotuning is bitwise stable across
    # fresh A10G runners.
    return {
        "deepseek_v3": "deepseek_v3_debugmodel",
        "gpt_oss": "gpt_oss_debugmodel_flex",
        "kimi_k2_7": "kimi_k2_5_debugmodel_text",
        "llama3": "llama3_debugmodel",
        "muse_glimmer": "muse_glimmer_debugmodel",
        "qwen3": "qwen3_debugmodel",
    }


def build_8gpu_numerics_test_list(output_dir: Path) -> dict[str, tuple[str, ...]]:
    """Guard selected distributed numerics and sharding parity with real collectives."""
    llama_fsdp = "--parallelism.data_parallel_replicate_degree=1"
    # The standard EP dispatcher synchronizes with the CPU, so EP=4 cannot use
    # CUDA graph capture even though the config's default EP=1 path can.
    qwen3_moe = (
        "--parallelism.tensor_parallel_degree 2 "
        "--parallelism.expert_parallel_degree 4 "
        "--training.disable_cuda_graphs"
    )
    return {
        "llama3_fsdp_hsdp": (
            f"--baseline-options={llama_fsdp}",
            "--test-options=--parallelism.data_parallel_replicate_degree=4",
            f"--job-dump-folder={output_dir / 'llama3_fsdp_hsdp'}",
            "--assert-equal",
            "--steps=1",
        ),
        "llama3_fsdp": (
            f"--baseline-options={llama_fsdp}",
            f"--job-dump-folder={output_dir / 'llama3_fsdp'}",
            f"--import-result={LOSSES / 'llama3_8gpu_a10g.txt'}",
            "--metrics=loss,grad_norm",
            "--assert-equal",
            "--steps=100",
        ),
        "qwen3_moe": (
            "--baseline-module=qwen3",
            "--baseline-config=qwen3_moe_debug",
            f"--baseline-options={qwen3_moe}",
            f"--test-options={qwen3_moe}",
            f"--job-dump-folder={output_dir / 'qwen3_moe'}",
            f"--import-result={LOSSES / 'qwen3_moe_8gpu_a10g.txt'}",
            "--metrics=loss,grad_norm",
            "--assert-equal",
            "--steps=100",
        ),
    }


def _run_loss_compare(test_name: str, options: tuple[str, ...], ngpus: int) -> None:
    print(f"[NUMERICS] Running {test_name}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/loss_compare.py",
            ".",
            ".",
            *options,
            f"--baseline-ngpus={ngpus}",
            f"--test-ngpus={ngpus}",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def run_1gpu_numerics(output_dir: Path) -> None:
    for model_name, config in build_1gpu_numerics_test_list().items():
        options = " ".join(
            (
                SINGLE_GPU_OPTIONS,
                SINGLE_GPU_MODEL_OPTIONS.get(model_name, ""),
            )
        ).strip()
        _run_loss_compare(
            model_name,
            (
                f"--baseline-module={model_name}",
                f"--baseline-config={config}",
                f"--baseline-options={options}",
                # Mirror the test settings so loss_compare stays in
                # baseline-only mode and runs the model exactly once.
                f"--test-module={model_name}",
                f"--test-config={config}",
                f"--test-options={options}",
                f"--job-dump-folder={output_dir / model_name}",
                f"--import-result={LOSSES / f'{model_name}_1gpu_a10g.txt'}",
                "--metrics=loss,grad_norm",
                "--no-seed-checkpoint",
                "--assert-equal",
                "--steps=10",
            ),
            ngpus=1,
        )


def run_8gpu_numerics(output_dir: Path) -> None:
    for test_name, options in build_8gpu_numerics_test_list(output_dir).items():
        _run_loss_compare(test_name, options, ngpus=8)


_TEST_SUITES = {
    "1gpu": run_1gpu_numerics,
    "8gpu": run_8gpu_numerics,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--test_suite", choices=_TEST_SUITES, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _TEST_SUITES[args.test_suite](output_dir)


if __name__ == "__main__":
    main()
