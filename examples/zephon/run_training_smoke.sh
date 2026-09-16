#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dump_folder=${1:-"${repo_root}/outputs/zephon-training-smoke"}
first_phase_gpus=${FIRST_PHASE_GPUS:-2}
second_phase_gpus=${SECOND_PHASE_GPUS:-1}
first_phase_steps=${FIRST_PHASE_STEPS:-2}
total_steps=${TOTAL_STEPS:-3}
canonical_replicas=${CANONICAL_REPLICAS:-2}
run_id=${ZEPHON_RUN_ID:-zephon-training-smoke}

if [[ -e "${dump_folder}" ]]; then
    echo "Refusing to reuse existing output: ${dump_folder}" >&2
    echo "Pass a new output path or move the existing directory." >&2
    exit 2
fi

cd "${repo_root}"

common_args=(
    --override.imports
    "torchtitan.overrides.zephon_dataloader.zephon_dataloader={\"data_config\":\"examples/zephon/elastic_local_jsonl.toml\",\"canonical_replicas\":${canonical_replicas},\"aggregate_dir\":\"${dump_folder}/zephon-aggregate\",\"run_id\":\"${run_id}\"}"
    --training.max_context_length 128
    --training.num_tokens_per_microbatch_per_dp_rank 256
    --training.num_tokens_per_train_step 512
    --checkpoint.enable
    --checkpoint.interval 1
    --dump_folder "${dump_folder}"
)

echo "Phase 1: train ${first_phase_steps} steps with ${first_phase_gpus} GPU(s)"
NGPU="${first_phase_gpus}" MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
    "${common_args[@]}" \
    --training.steps "${first_phase_steps}"

echo "Phase 2: resume through step ${total_steps} with ${second_phase_gpus} GPU(s)"
NGPU="${second_phase_gpus}" MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
    "${common_args[@]}" \
    --training.steps "${total_steps}"

echo "Zephon training checkpoint/resume smoke test passed: ${dump_folder}"
