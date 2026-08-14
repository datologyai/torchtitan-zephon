# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run single-GPU and distributed model numerics integration tests."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.loss_compare import extract_scalar_from_tensorboard


REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_TRAINING_STEPS = 10
METRIC_TAGS = {
    "loss": "loss_metrics/global_avg_loss",
    "grad_norm": "grad_norm",
}
SINGLE_GPU_OPTIONS = (
    "--debug.seed=42",
    "--debug.deterministic",
    f"--training.steps={NUM_TRAINING_STEPS}",
    "--training.disable_cuda_graphs",
    "--metrics.enable_tensorboard",
    "--metrics.log_freq=1",
    "--metrics.save_tb_folder=tb",
    "--parallelism.data_parallel_replicate_degree=1",
    "--parallelism.data_parallel_shard_degree=1",
    "--parallelism.tensor_parallel_degree=1",
    "--parallelism.context_parallel_degree=1",
    "--parallelism.pipeline_parallel_degree=1",
    "--parallelism.expert_parallel_degree=1",
    "activation-checkpoint:none",
)


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
            f"--import-result={REPO_ROOT / 'tests/assets/losses/llama3_cuda.txt'}",
            "--assert-equal",
            "--steps=100",
        ),
        "qwen3_moe": (
            "--baseline-module=qwen3",
            "--baseline-config=qwen3_moe_debug",
            f"--baseline-options={qwen3_moe}",
            f"--test-options={qwen3_moe}",
            f"--job-dump-folder={output_dir / 'qwen3_moe'}",
            f"--import-result={REPO_ROOT / 'tests/assets/losses/qwen3_moe_cuda.txt'}",
            "--assert-equal",
            "--steps=100",
        ),
    }


def _write_json(result: dict[str, Any], path: Path) -> None:
    with path.open("w") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True, allow_nan=False)
        result_file.write("\n")


def _extract_metrics(dump_folder: Path, model_name: str) -> list[dict[str, Any]]:
    expected_steps = list(range(1, NUM_TRAINING_STEPS + 1))
    metrics = {
        name: extract_scalar_from_tensorboard(str(dump_folder), "tb", tag)
        for name, tag in METRIC_TAGS.items()
    }
    for metric_name, values in metrics.items():
        if sorted(values) != expected_steps:
            raise ValueError(
                f"Expected {metric_name} steps {expected_steps} for {model_name}, "
                f"found {sorted(values)}"
            )
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite {metric_name} for {model_name}")

    return [
        {
            "step": step,
            "loss": metrics["loss"][step],
            "grad_norm": metrics["grad_norm"][step],
        }
        for step in expected_steps
    ]


def run_1gpu_numerics(output_dir: Path, import_result: Path | None) -> None:
    actual: dict[str, Any] = {}
    actual_path = output_dir / "actual.json"
    for model_name, config in build_1gpu_numerics_test_list().items():
        dump_folder = output_dir / "runs" / model_name
        env = os.environ.copy()
        env.update(MODULE=model_name, CONFIG=config, NGPU="1", LOG_RANK="0")
        env.pop("COMM_MODE", None)
        subprocess.run(
            [
                "./run_train.sh",
                f"--dump_folder={dump_folder}",
                *SINGLE_GPU_OPTIONS,
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        actual[model_name] = {
            "config": config,
            "metrics": _extract_metrics(dump_folder, model_name),
        }
        _write_json(actual, actual_path)

    if import_result:
        with import_result.open() as result_file:
            expected = json.load(result_file)
        if expected != actual:
            raise AssertionError(
                f"Model numerics differ from {import_result}; inspect {actual_path}"
            )


def run_8gpu_numerics(output_dir: Path, import_result: Path | None) -> None:
    if import_result:
        raise ValueError("The 8-GPU suite uses its existing per-test loss files")

    env = os.environ.copy()
    env.pop("COMM_MODE", None)
    for test_name, options in build_8gpu_numerics_test_list(output_dir).items():
        print(f"[NUMERICS] Running {test_name}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/loss_compare.py",
                ".",
                ".",
                *options,
                "--baseline-ngpus=8",
                "--test-ngpus=8",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )


_TEST_SUITES = {
    "1gpu": run_1gpu_numerics,
    "8gpu": run_8gpu_numerics,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--test_suite", choices=_TEST_SUITES, required=True)
    parser.add_argument("--import-result", type=Path)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _TEST_SUITES[args.test_suite](output_dir, args.import_result)


if __name__ == "__main__":
    main()
