# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.components.loss import sum_detached_losses


class TestLossReduction(unittest.TestCase):
    def test_fp64_cast_precedes_sum_and_preserves_backward(self) -> None:
        for dtype, expected in ((torch.float32, 0.0), (torch.float64, 1.0)):
            with self.subTest(dtype=dtype):
                parameter = torch.tensor(0.0, requires_grad=True)
                losses = [
                    parameter * (index + 1) + value
                    for index, value in enumerate((2**24, 1, -(2**24)))
                ]
                reporting_loss = sum_detached_losses(losses, dtype=dtype)
                self.assertEqual(reporting_loss.dtype, dtype)
                self.assertEqual(reporting_loss.item(), expected)
                self.assertFalse(reporting_loss.requires_grad)
                self.assertIsNone(parameter.grad)
                torch.stack(losses).sum().backward()
                self.assertEqual(parameter.grad.item(), 6.0)
