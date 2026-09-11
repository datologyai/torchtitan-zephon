#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dump_folder=${1:-"${repo_root}/outputs/zephon-training-smoke"}

if [[ -e "${dump_folder}" ]]; then
    echo "Refusing to reuse existing output: ${dump_folder}" >&2
    echo "Pass a new output path or move the existing directory." >&2
    exit 2
fi

cd "${repo_root}"

common_args=(
    --override.imports 'torchtitan.overrides.zephon_dataloader.zephon_dataloader={"data_config":"examples/zephon/local_jsonl.toml"}'
    --training.max_context_length 128
    --training.num_tokens_per_microbatch_per_dp_rank 256
    --training.num_tokens_per_train_step 256
    --checkpoint.enable
    --checkpoint.interval 1
    --dump_folder "${dump_folder}"
)

NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
    "${common_args[@]}" \
    --training.steps 2

NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh \
    "${common_args[@]}" \
    --training.steps 3

echo "Zephon training checkpoint/resume smoke test passed: ${dump_folder}"
