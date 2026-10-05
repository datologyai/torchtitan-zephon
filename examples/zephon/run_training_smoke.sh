#!/usr/bin/env bash
# Copyright (c) DatologyAI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# End-to-end GPU smoke test: train, checkpoint the model and Zephon stream,
# resume with a configurable GPU count, and complete one additional step.
#
# Step counts, parallelism and the Zephon data recipe live in the paired
# ${RECIPE}_save and ${RECIPE}_resume configs in
# torchtitan_recipes/zephon/recipes.py.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output_dir=${1:-"${repo_root}/outputs/zephon-training-smoke"}
if (( $# > 0 )); then
    shift
fi
extra_args=("$@")
first_phase_gpus=${FIRST_PHASE_GPUS:-2}
second_phase_gpus=${SECOND_PHASE_GPUS:-1}
recipe=${RECIPE:-llama3_debugmodel_zephon_smoke}
run_id=${ZEPHON_RUN_ID:-zephon-training-smoke}
# Must match training.steps in the save and resume recipes.
first_phase_steps=2
total_steps=3

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse existing output: ${output_dir}" >&2
    echo "Pass a new output path or move the existing directory." >&2
    exit 2
fi
# Resolve before cd so a relative path names the caller's directory.
output_dir=$(mkdir -p "${output_dir}" && cd "${output_dir}" && pwd)

cd "${repo_root}"

# The recipes read the per-run Zephon settings from these.
export ZEPHON_AGGREGATE_DIR="${output_dir}/zephon-aggregate"
export ZEPHON_RUN_ID="${run_id}"
module=torchtitan_recipes.zephon.recipes

echo "Phase 1: train ${first_phase_steps} steps with ${first_phase_gpus} GPU(s)"
NGPU="${first_phase_gpus}" MODULE="${module}" CONFIG="${recipe}_save" \
    ./run_train.sh \
    --output-dir "${output_dir}" \
    "${extra_args[@]}"

completed_checkpoint="${output_dir}/checkpoint/step-${first_phase_steps}"
if [[ ! -f "${completed_checkpoint}/.metadata" ]]; then
    echo "Phase 1 did not produce completed checkpoint ${completed_checkpoint}" >&2
    exit 1
fi

echo "Phase 2: resume through step ${total_steps} with ${second_phase_gpus} GPU(s)"
NGPU="${second_phase_gpus}" MODULE="${module}" CONFIG="${recipe}_resume" \
    ./run_train.sh \
    --output-dir "${output_dir}" \
    "${extra_args[@]}" \
    --resume-step "${first_phase_steps}"

final_checkpoint="${output_dir}/checkpoint/step-${total_steps}"
if [[ ! -f "${final_checkpoint}/.metadata" ]]; then
    echo "Phase 2 did not produce completed checkpoint ${final_checkpoint}" >&2
    exit 1
fi

echo "Zephon training checkpoint/resume smoke test passed: ${output_dir}"
echo "Restore evidence: phase 1 checkpoint ${completed_checkpoint}; phase 2 checkpoint ${final_checkpoint}"
