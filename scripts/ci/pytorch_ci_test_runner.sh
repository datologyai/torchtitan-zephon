#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Stable entrypoint for PyTorch CI to run torchtitan integration tests.
# PyTorch CI calls this script so that torchtitan maintainers can adjust
# test configuration without modifying the PyTorch repo.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts-to-be-uploaded}"
NGPU="${NGPU:-8}"

usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  feature_tests   Run feature integration tests"
    echo "  model_tests     Run model integration tests"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

COMMAND="$1"
shift

# Run both tiers so every test in the suite stays covered. Each tier needs its
# own subdirectory because run_tests.py requires an empty output directory.
case "$COMMAND" in
    feature_tests)
        python -m tests.integration_tests.run_tests \
            --test_suite features \
            --fake_pg \
            --exclude "cpu_offload+opt_in_bwd+TP+DP+CP" \
            --ngpu "$NGPU" \
            "$OUTPUT_DIR/features_fake_pg"
        python -m tests.integration_tests.run_tests \
            --test_suite features \
            --exclude "cpu_offload+opt_in_bwd+TP+DP+CP" \
            --ngpu "$NGPU" \
            "$OUTPUT_DIR/features_real_pg"
        ;;
    model_tests)
        python -m tests.integration_tests.run_tests \
            --test_suite models \
            --fake_pg \
            --exclude "qwen3_5_moe_fsdp+tp+ep+pp_spmd_types" \
            --ngpu "$NGPU" \
            "$OUTPUT_DIR/models_fake_pg"
        python -m tests.integration_tests.run_tests \
            --test_suite models \
            --exclude "qwen3_5_moe_fsdp+tp+ep+pp_spmd_types" \
            --ngpu "$NGPU" \
            "$OUTPUT_DIR/models_real_pg"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
