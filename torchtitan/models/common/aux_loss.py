# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Auxiliary-loss gradient injection and distributed metric collection.

Auxiliary objectives computed inside the model (MoE load-balance, DSA KL, ...)
cannot reach the trainer's loss function under pipeline parallelism: the model
forward runs on every stage but the scalar loss is only formed on the last
stage. ``LoggedAuxLoss`` decouples the gradient from the scalar output via an
identity autograd function, and defers per-step metric readout to
``collect_aux_loss_metrics`` with PP-safe collection.

Framework contract
------------------

- The normalization is the framework's job: ``inject`` scales every value by
  ``1 / per_step_denominator``, so both the injected gradient and the logged
  metric follow the loss's ``normalize`` mode. The denominator is set per
  loss by ``LoggedAuxLoss.update_from_config`` (called by the decoder's
  ``update_from_config``) from the training batch size and sequence length:
  - ``"tokens"`` (default): denominator = ``global_batch_size * seq_len`` --
    a mean over the global batch's tokens. Loss forwards produce raw sums
    and must not normalize by sequence length of any kind.
  - ``"sequences"``: denominator = ``global_batch_size`` -- a mean over the
    batch's sequences (e.g. the DS-V3 sequence-wise load-balance loss, whose
    per-sequence value is already token-count-normalized inside Eqs 18/20).
  - ``"batch"``: denominator = 1 -- the loss is an O(1) global-batch
    statistic. NOTE: such losses are not additive over microbatches (their
    counts must cover the whole batch), so they are incompatible with
    gradient accumulation and pipeline parallelism; any future consumer must
    validate against those configs (see PR #3000).
- The metric accumulates in the autograd backward (once per microbatch by
  autograd semantics; activation-checkpoint recompute never double counts),
  so it is available after ``backward``; ``zero_all`` snapshots it at the
  optimizer step pre-hook.
- ``reduce_mesh`` names the mesh the metric is reduced over:
  sequence-statistics losses (seqwise load balance) are identical across CP
  and choose ``batch``; slice-additive losses (per-token penalties, e.g.
  DSA indexer KL) are partials over DP and CP and choose ``"loss"``.
- Model specs must register ``register_aux_loss_zero_hook`` after building
  the optimizer (see e.g. the DeepSeek-V3 model registry).
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, Literal

import spmd_types as spmd
import torch
import torch.nn as nn
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.tensor import DTensor

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type

__all__ = [
    "LoggedAuxLoss",
    "collect_aux_loss_metrics",
    "register_aux_loss_zero_hook",
]


@spmd.register_autograd_function
class _AuxLossInjection(torch.autograd.Function):
    """Identity-forward autograd that injects an aux-loss gradient on backward.

    The metric accumulates here, in the backward: autograd invokes a node's
    backward exactly once per microbatch, so activation-checkpoint recompute
    (which re-executes the forward but discards its nodes) can never double
    count, and no forward-time graph break is needed.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        carrier: torch.Tensor,
        aux_loss: torch.Tensor,
        acc_value: torch.Tensor,
        loss_module: "LoggedAuxLoss",
    ) -> torch.Tensor:
        ctx.save_for_backward(aux_loss, acc_value)
        ctx.loss_module = loss_module
        return carrier

    @staticmethod
    def typecheck_forward(
        carrier: torch.Tensor,
        aux_loss: torch.Tensor,
        acc_value: torch.Tensor,
        loss_module: "LoggedAuxLoss",
    ) -> torch.Tensor:
        # The autograd function is a blackbox to typechecking: re-annotate the
        # output from the carrier, which this identity function passes through.
        out = _AuxLossInjection.apply(carrier, aux_loss, acc_value, loss_module)
        spmd.assert_type(
            out,
            spmd.get_local_type(carrier),
            spmd.runtime.get_partition_spec(carrier),
        )
        return out

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_carrier: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        aux_loss, acc_value = ctx.saved_tensors
        with spmd.no_typecheck():
            ctx.loss_module._acc_sum.add_(acc_value)
        return grad_carrier, torch.ones_like(aux_loss), None, None


class LoggedAuxLoss(Module):
    """Inject auxiliary gradients and defer metric readout to the step hook.

    Subclasses call ``inject()`` each microbatch to inject the gradient; the
    metric value accumulates in the autograd backward. ``zero_all`` snapshots
    the accumulators into ``_step_snapshots`` and clears them, at the
    optimizer step pre-hook; metric collection later reduces the snapshots.
    """

    # Precompute *identical* metric groups on all pipeline stages. Populated
    # during model build, before PP splitting, so every stage participates
    # in the collectives with pre-allocated zero accumulators.
    _group_counts: ClassVar[dict[tuple[str, str], int]] = defaultdict(int)

    # Step snapshots keyed by ``(reduce_mesh, metric_name)``, filled by
    # ``zero_all`` at each optimizer step pre-hook.
    _step_snapshots: ClassVar[dict[tuple[str, str], torch.Tensor]] = {}

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        coeff: float
        """Aux loss coefficient. Scales the gradient contribution."""
        reduce_mesh: str = "loss"
        """Mesh the metric is ``sum``-reduced over (see module docstring for
        the ownership contract). ``"batch"`` for full-sequence-statistics
        losses, ``"loss"`` for slice-additive ones. PP is reduced separately.
        """
        normalize: Literal["tokens", "sequences", "batch"] = "tokens"
        """Normalization mode for the injected gradient and logged metric;
        see the module docstring's framework contract. The matching
        ``per_step_denominator`` is set by ``update_from_config``.
        """
        per_step_denominator: int = -1
        """Denominator of the per-step normalization (tokens per step for
        ``"tokens"``, sequences per step for ``"sequences"``, 1 for
        ``"batch"``), set by ``LoggedAuxLoss.update_from_config`` before the
        modules are built.
        """

    @classmethod
    def update_from_config(
        cls,
        aux_loss_cfg: "Config",
        *,
        global_batch_size: int,
        seq_len: int,
    ) -> None:
        """Set the per-step normalization denominator from the training config.

        Maps the ``normalize`` mode to its denominator:
        ``"tokens"`` -> ``global_batch_size * seq_len`` (per-token mean),
        ``"sequences"`` -> ``global_batch_size`` (per-sequence mean),
        ``"batch"`` -> 1 (global-batch statistic). Called by the decoder's
        ``update_from_config`` for every aux-loss config in the model tree.
        """
        aux_loss_cfg.per_step_denominator = {
            "tokens": global_batch_size * seq_len,
            "sequences": global_batch_size,
            "batch": 1,
        }[aux_loss_cfg.normalize]

    @property
    def metric_name(self) -> str:
        """Convert the class name from PascalCase to snake_case."""
        return re.sub(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            "_",
            type(self).__name__,
        ).lower()

    def __init__(self, config: Config):
        super().__init__()
        self.coeff = config.coeff
        self.reduce_mesh = config.reduce_mesh
        self.per_step_denominator = config.per_step_denominator
        # Running sum of the scaled per-microbatch values, covering one step.
        self.register_buffer(
            "_acc_sum", torch.zeros((), dtype=torch.float32), persistent=False
        )
        LoggedAuxLoss._group_counts[(config.reduce_mesh, self.metric_name)] += 1

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            buffer_device = self._acc_sum.device
        with torch.device(buffer_device):
            self._acc_sum = torch.zeros((), dtype=torch.float32)

    def inject(self, raw_sum: torch.Tensor, *, carrier: torch.Tensor) -> torch.Tensor:
        """Inject aux loss gradient into the forward graph.

        ``carrier`` is the model activation the aux loss rides on: the
        identity forward returns it unchanged, and its graph carries the
        aux-loss gradient on backward. ``raw_sum`` is the loss's raw sum over
        the local batch; the framework applies the normalization (see
        ``normalize`` / ``per_step_denominator``) and accumulates the metric
        in the backward (see ``_AuxLossInjection``).
        """
        assert self.per_step_denominator > 0
        scale = 1.0 / self.per_step_denominator
        acc_value = raw_sum.detach() * scale
        # Remove once the default/full_dtensor backends are deleted.
        if isinstance(acc_value, DTensor):
            acc_value = acc_value.to_local()
        return _AuxLossInjection.apply(
            carrier, raw_sum * (self.coeff * scale), acc_value, self
        )

    @classmethod
    def zero_all(cls, model_parts) -> None:
        """Snapshot all accumulators for the current step, then clear them.

        Runs at the optimizer step pre-hook, so the snapshot covers every
        microbatch of the step. ``clear()`` resets the snapshot per step;
        ``add_`` accumulates every module of the group exactly once. Walking
        ``part.modules()`` (instead of only ``part.layers``) also covers aux
        losses living outside the main decoder layers, e.g. MTP modules.
        """
        cls._step_snapshots.clear()
        for part in model_parts:
            for module in part.modules():
                if isinstance(module, LoggedAuxLoss):
                    key = (module.reduce_mesh, module.metric_name)
                    if key not in cls._step_snapshots:
                        cls._step_snapshots[key] = torch.zeros_like(module._acc_sum)
                    cls._step_snapshots[key] += module._acc_sum
                    module._acc_sum.zero_()


def collect_aux_loss_metrics(model_parts, parallel_dims) -> dict[str, float]:
    """Reduce auxiliary-loss metrics across configured meshes.

    Returns ``{metric_name}/mean`` per group (mean over the global batch's
    tokens, sequences, or batch depending on the loss's ``normalize`` mode,
    per module); ``{}`` when no aux losses are configured. Called on all
    ranks at log time, so the collectives are safe.

    Note: the snapshots are taken per optimizer step (see ``zero_all``), so
    the reported value is the last completed step, not an average over the
    ``log_freq`` window.
    """
    if not LoggedAuxLoss._group_counts:
        return {}

    pp_mesh = parallel_dims.get_optional_mesh("pp")
    local_sums = {
        key: LoggedAuxLoss._step_snapshots.get(
            key, torch.zeros((), dtype=torch.float32, device=device_type)
        )
        for key in LoggedAuxLoss._group_counts
    }
    metrics = {}
    for key, total in sorted(local_sums.items()):
        mesh_name, tag = key
        for mesh in (parallel_dims.get_optional_mesh(mesh_name), pp_mesh):
            if mesh is not None:
                total = all_reduce(total, reduceOp="sum", group=mesh)
        metrics[f"{tag}/mean"] = float(total.item()) / LoggedAuxLoss._group_counts[key]
    return metrics


def register_aux_loss_zero_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register an optimizer step pre-hook that snapshots and zeroes aux losses.

    This follows the same pattern as ``register_moe_load_balancing_hook``:
    accumulator management happens outside the model, once per training step.
    """
    optimizers.register_step_pre_hook(
        lambda *args, **kwargs: LoggedAuxLoss.zero_all(model_parts)
    )
