# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shlex
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "OverrideDefinitions",
    "requires_real_pg",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None

    def __repr__(self):
        return self.test_descr


# Tests that Fake PG cannot run correctly for reasons their CLI overrides do
# not express.
# TODO: spmd_types plus AC recompute is broken under Fake PG (issue #TBD).
_FAKE_PG_INCOMPATIBLE = frozenset({"2d_eager_spmd_types"})


def requires_real_pg(test: OverrideDefinitions) -> bool:
    """Return whether a test requires communication between real ranks."""
    if test.test_name in _FAKE_PG_INCOMPATIBLE:
        return True

    for variant in test.override_args:
        cli_args = [
            cli_arg for override in variant for cli_arg in shlex.split(override)
        ]

        # Checkpoint tests require rank-local shards, metadata coordination,
        # save/load, and resharding. Fake PG cannot validate these semantics.
        if (
            "--checkpoint.enable" in cli_args
            or "--checkpoint.create_seed_checkpoint" in cli_args
        ):
            return True

        for index, cli_arg in enumerate(cli_args):
            option, separator, value = cli_arg.partition("=")
            if option not in {
                "--parallelism.pipeline_parallel_degree",
                "--comm.mode",
            }:
                continue
            if not separator:
                value = cli_args[index + 1]

            # PP needs cross-rank send/recv, while Fake PG point-to-point
            # operations are no-ops and cannot transfer activations or grads.
            if option == "--parallelism.pipeline_parallel_degree" and int(value) > 1:
                return True

            # An explicit non-fake mode must exercise its requested backend,
            # instead of replacing the system under test with Fake PG.
            if option == "--comm.mode" and value != "fake_backend":
                return True
    return False
