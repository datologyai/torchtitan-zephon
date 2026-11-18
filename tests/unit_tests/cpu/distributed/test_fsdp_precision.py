# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP


class TestFSDPPrecision(unittest.TestCase):
    def test_fp64_retains_small_unsynchronized_gradients(self) -> None:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            config = PrecisionConfig(reduce_dtype="float64")
            model = torch.nn.Linear(1, 1, bias=False)
            fully_shard(
                model,
                mesh=init_device_mesh("cpu", (1,)),
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=TORCH_DTYPE_MAP[config.reduce_dtype],
                ),
            )
            # FP32 local accumulation loses the unit gradient before reduction.
            for index, value in enumerate((2**24, 1, -(2**24))):
                model.set_requires_gradient_sync(index == 2)
                model(torch.tensor([[value]], dtype=torch.float32)).sum().backward()

            torch.testing.assert_close(
                model.weight.grad.to_local(), torch.ones(1, 1), rtol=0, atol=0
            )
        finally:
            dist.destroy_process_group()
