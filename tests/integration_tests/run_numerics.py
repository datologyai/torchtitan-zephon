# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run fake-process-group and real-distributed numerics integration tests.

Both suites drive ``scripts/loss_compare.py``, which runs training, reads
full-precision metrics from TensorBoard, and compares them against checked-in
goldens under ``tests/assets/losses``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOSSES = REPO_ROOT / "tests/assets/losses"


def build_fake_pg_numerics_test_list() -> dict[str, tuple[str, str, int, str]]:
    """Return Fake-PG golden configurations that fit A10G."""
    text_fixture = (
        "--training.local_batch_size 1 "
        "--training.seq_len 512 "
        "--hf_assets_path ./tests/assets/tokenizer "
        "--dataloader.dataset c4_test"
    )

    # CP is intentionally absent: FakeProcessGroup's local collective
    # approximations produced non-finite or pathological gradient norms for
    # every FSDP+TP+CP numerics probe.
    # TODO: add CP when Fake PG can produce stable CP numerics; current probes
    # produce non-finite or pathological gradient norms.
    deepseek_16b_fsdp_tp_ep = (
        "--parallelism.data_parallel_shard_degree 8 "
        "--parallelism.tensor_parallel_degree 2 "
        "--parallelism.expert_parallel_degree 2 "
        "--training.disable_cuda_graphs "
        f"{text_fixture}"
    )
    gpt_oss_20b_fsdp_tp_ep = (
        "--parallelism.data_parallel_shard_degree 16 "
        "--parallelism.tensor_parallel_degree 2 "
        "--parallelism.expert_parallel_degree 2 "
        "--parallelism.spmd_backend spmd_types "
        "--training.disable_cuda_graphs "
        f"{text_fixture}"
    )
    # TODO: replace the Kimi debug golden with Moonlight 16B-A3B once it fits the
    # A10G tier. Its stable world-size-8 FSDP+EP layout peaks at 29.44 GiB;
    # larger DistMuon layouts currently fail alignment or Fake-PG all-to-all.
    kimi_debug_fsdp_ep = (
        "--parallelism.data_parallel_shard_degree 8 "
        "--parallelism.expert_parallel_degree 8"
    )
    llama3_8b_fsdp_tp = (
        "--parallelism.data_parallel_shard_degree 8 "
        "--parallelism.tensor_parallel_degree 2 "
        f"{text_fixture}"
    )
    muse_glimmer_30b_fsdp_tp = (
        "--parallelism.data_parallel_shard_degree 32 "
        "--parallelism.tensor_parallel_degree 2 "
        f"{text_fixture}"
    )
    # Preserve the production scheduler horizon. Compressing its 600-step
    # warmup into ten test steps reaches the full LR and spikes the grad norm.
    qwen_30b_fsdp_ep = (
        "--parallelism.data_parallel_shard_degree 64 "
        "--parallelism.expert_parallel_degree 64 "
        "--parallelism.spmd_backend spmd_types "
        "--lr-scheduler.total-steps 3000 "
        "--training.local_batch_size 1 "
        "--training.seq_len 128 "
        "--hf_assets_path ./tests/assets/tokenizer "
        "--dataloader.dataset c4_test "
        "--training.disable_cuda_graphs"
    )
    return {
        "deepseek_v3_16b_fsdp_tp_ep_fake_pg": (
            "deepseek_v3",
            "deepseek_v3_16b",
            16,
            deepseek_16b_fsdp_tp_ep,
        ),
        "gpt_oss_20b_fsdp_tp_ep_fake_pg": (
            "gpt_oss",
            "gpt_oss_20b",
            32,
            gpt_oss_20b_fsdp_tp_ep,
        ),
        "kimi_k2_5_debugmodel_fsdp_ep_fake_pg": (
            "kimi_k2_7",
            "kimi_k2_5_debugmodel_text",
            8,
            kimi_debug_fsdp_ep,
        ),
        "llama3_8b_fsdp_tp_fake_pg": (
            "llama3",
            "llama3_8b",
            16,
            llama3_8b_fsdp_tp,
        ),
        "muse_glimmer_30b_fsdp_tp_fake_pg": (
            "muse_glimmer",
            "muse_glimmer_30b",
            64,
            muse_glimmer_30b_fsdp_tp,
        ),
        "qwen3_30b_a3b_fsdp_ep_fake_pg": (
            "qwen3",
            "qwen3_30b_a3b",
            64,
            qwen_30b_fsdp_ep,
        ),
    }


def build_8gpu_numerics_test_list(output_dir: Path) -> dict[str, tuple[str, ...]]:
    """Guard distributed numerics for parallelisms that need real collectives."""
    gpt_oss_pp_fsdp_cp_ep = (
        "--parallelism.spmd_backend spmd_types "
        "--parallelism.data_parallel_shard_degree 2 "
        "--parallelism.context_parallel_degree 2 "
        "--parallelism.context_parallel_load_balancer ptrr "
        "--parallelism.context_parallel_ptrr_mask_key basic_mask "
        "--parallelism.pipeline_parallel_degree 2 "
        "--parallelism.pipeline_parallel_schedule Interleaved1F1B "
        "--parallelism.expert_parallel_degree 4 "
        "--training.disable_cuda_graphs "
        "activation-checkpoint:selective"
    )
    llama3_fsdp_tp_cp = (
        "--parallelism.spmd_backend spmd_types "
        "--parallelism.data_parallel_shard_degree 2 "
        "--parallelism.tensor_parallel_degree 2 "
        "--parallelism.context_parallel_degree 2"
    )
    qwen3_moe_fsdp_tp_cp_ep = (
        "--parallelism.spmd_backend spmd_types "
        "--parallelism.data_parallel_shard_degree 2 "
        "--parallelism.tensor_parallel_degree 2 "
        "--parallelism.context_parallel_degree 2 "
        "--parallelism.expert_parallel_degree 4 "
        "--training.disable_cuda_graphs"
    )
    return {
        "gpt_oss_pp_fsdp_cp_ep": (
            "--baseline-module=gpt_oss",
            "--baseline-config=gpt_oss_debugmodel_flex",
            f"--baseline-options={gpt_oss_pp_fsdp_cp_ep}",
            f"--test-options={gpt_oss_pp_fsdp_cp_ep}",
            f"--job-dump-folder={output_dir / 'gpt_oss_pp_fsdp_cp_ep'}",
            f"--import-result={LOSSES / 'gpt_oss_pp_fsdp_cp_ep_8gpu_a10g.txt'}",
            "--metrics=loss,grad_norm",
            "--assert-equal",
            "--steps=100",
        ),
        "llama3_fsdp_tp_cp": (
            f"--baseline-options={llama3_fsdp_tp_cp}",
            f"--test-options={llama3_fsdp_tp_cp}",
            f"--job-dump-folder={output_dir / 'llama3_fsdp_tp_cp'}",
            f"--import-result={LOSSES / 'llama3_fsdp_tp_cp_8gpu_a10g.txt'}",
            "--metrics=loss,grad_norm",
            "--assert-equal",
            "--steps=100",
        ),
        "qwen3_moe_fsdp_tp_cp_ep": (
            "--baseline-module=qwen3",
            "--baseline-config=qwen3_moe_debug",
            f"--baseline-options={qwen3_moe_fsdp_tp_cp_ep}",
            f"--test-options={qwen3_moe_fsdp_tp_cp_ep}",
            f"--job-dump-folder={output_dir / 'qwen3_moe_fsdp_tp_cp_ep'}",
            f"--import-result={LOSSES / 'qwen3_moe_fsdp_tp_cp_ep_8gpu_a10g.txt'}",
            "--metrics=loss,grad_norm",
            "--assert-equal",
            "--steps=100",
        ),
    }


def _run_loss_compare(
    test_name: str,
    options: tuple[str, ...],
    ngpus: int,
    *,
    comm_mode: str = "",
) -> None:
    print(f"[NUMERICS] Running {test_name}", flush=True)
    env = os.environ.copy()
    env.pop("COMM_MODE", None)
    if comm_mode:
        env["COMM_MODE"] = comm_mode
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
        env=env,
        check=True,
    )


def run_1gpu_numerics(output_dir: Path) -> None:
    for test_name, test_spec in build_fake_pg_numerics_test_list().items():
        module, config, logical_world_size, options = test_spec
        _run_loss_compare(
            test_name,
            (
                f"--baseline-module={module}",
                f"--baseline-config={config}",
                f"--baseline-options={options}",
                f"--test-module={module}",
                f"--test-config={config}",
                f"--test-options={options}",
                f"--job-dump-folder={output_dir / test_name}",
                f"--import-result={LOSSES / f'{test_name}_a10g.txt'}",
                "--metrics=loss,grad_norm",
                "--no-seed-checkpoint",
                "--assert-equal",
                "--steps=10",
            ),
            ngpus=logical_world_size,
            comm_mode="fake_backend",
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
