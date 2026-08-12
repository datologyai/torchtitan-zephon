# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.aux_loss import LoggedAuxLoss
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

from .token_dispatcher import LocalTokenDispatcher

# Shape suffix legend
# (https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
#   B = batch, L = sequence length, D = model dimension,
#   F = hidden (FFN intermediate) dimension, E = num experts,
#   e = num local experts (E / EP, used in token dispatcher for
#       per-local-expert token counts after EP dispatch /_permute),
#   K = top-k, T = num tokens (B*L flattened),
#   N = routed tokens (T*K), R = routed tokens assigned to local experts


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine.

        Shape suffixes here describe logical grouped-mm inputs, not physical
        sharding. Under EP, E may be a local shard of experts; under TP,
        expert weights shard hidden dimensions instead; under SP, R may be a
        local token shard. Keep logical capital suffixes here to avoid encoding
        a specific parallel layout in these local tensor names.
        """
        if isinstance(self.w1_EFD, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if (
            get_spmd_backend() == "spmd_types"
            and spmd.is_type_checking()
            and spmd_mesh_size("ep") == 1
        ):
            for axis in ("dp", "cp"):
                # if no EP, convert to V for grouped_mm, which would otherwise see
                # x:R, w1:V, offsets:P in local SPMD typechecking.
                # spmd.P is not currently allowed to mix with spmd.V.
                # TODO(pianpwk): likely relax this in spmd_types.
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        h_RF = F.silu(
            self._grouped_mm(
                A=x_RD.bfloat16(),
                B_t=w1_EFD.bfloat16().transpose(-2, -1),
                offs=offsets_E,
            )
        )
        h_RF = h_RF * self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        return self._grouped_mm(
            A=h_RF, B_t=w2_EDF.bfloat16().transpose(-2, -1), offs=offsets_E
        ).type_as(x_RD)

    def _grouped_mm(
        self, *, A: torch.Tensor, B_t: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        """Grouped matmul of ``A @ B_t`` with per-expert token offsets.

        Overridable seam for low-precision variants (e.g. the MXFP8 converter
        swaps this for a dynamically-quantized scaled grouped GEMM). Keeping the
        op here -- rather than behind a tensor-subclass ``__torch_function__`` --
        means it is captured by FX tracers such as graph_trainer's make_fx path.
        """
        return torch._grouped_mm(A, B_t, offs=offs)


class RoutedExperts(Module):
    """Routed-expert ``local_map`` region: composes token_dispatcher + inner_experts
    as sibling nodes so each can be overridden independently."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_experts: GroupedExperts.Config
        token_dispatcher: LocalTokenDispatcher.Config

    def __init__(self, config: Config):
        super().__init__()
        self.inner_experts = config.inner_experts.build()
        self.token_dispatcher = config.token_dispatcher.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add.

        When parallelized, ``local_map`` (from ``sharding_config``) handles
        DTensor→local conversion on entry and local→DTensor(Partial) wrapping
        on exit. The forward body operates on plain local tensors.
        """
        B, L, D = x_BLD.shape
        K = topk_scores_BLK.size(-1)
        T = B * L
        x_TD = x_BLD.view(T, D)

        topk_scores_TK = topk_scores_BLK.view(T, K)
        topk_expert_ids_TK = topk_expert_ids_BLK.view(T, K)
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )
        out_TD = self.token_dispatcher.combine(
            routed_output_RD,
            metadata,
            x_TD,
        )
        # Un-flatten back to 3-D (B, *, D) so the local_map output sharding
        # won't cause _StridedShard in the downstream view (e.g., CP is used).
        return out_TD.view(B, -1, D)

    def parallelize(self, parallel_dims) -> None:
        """Parallelize the grouped experts, then wire the EP mesh on the
        dispatcher so dispatch/combine see the right mesh at runtime."""
        super().parallelize(parallel_dims)
        # TODO(@pianpwk): With spmd_types and set_current_spmd_mesh, replace wire_meshes
        # with current_spmd_mesh calls inside AllToAllTokenDispatcher and
        # DeepEPTokenDispatcher.
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
        )


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(
        self, scores_BLE: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (topk_expert_ids_BLK ``(B, L, K)`` LongTensor, topk_scores_BLK ``(B, L, K)`` FloatTensor).
        """
        bs, slen, _ = scores_BLE.shape
        # Round-robin indices with exact balance
        topk_expert_ids_BLK = (
            torch.arange(
                bs * slen * self.top_k, device=scores_BLE.device, dtype=torch.int64
            ).reshape(bs, slen, self.top_k)
            % self.num_experts
        )
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)
        return topk_expert_ids_BLK, topk_scores_BLK

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice_BLE: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf.

        Args:
            scores_for_choice_BLE: Router scores with expert_bias (if any), shape ``(B, L, E)``.

        Returns:
            scores_for_choice_BLE: shape ``(B, L, E)``.
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice_BLE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(-1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice_BLE = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).flatten(-2)

        return scores_for_choice_BLE

    def forward(
        self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_BLD: Input ``(B, L, D)``.
            expert_bias_E: Optional load-balancing bias ``(E,)``.

        Returns:
            topk_scores_BLK: Routing scores ``(B, L, K)``.
            topk_expert_ids_BLK: Expert indices ``(B, L, K)``.
            scores_BLE: Full routing scores ``(B, L, E)``.
        """
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
            scores_BLE = self.gate(x_BLD)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion.
        # scores_BLE is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores_BLE = torch.sigmoid(scores_BLE)
        elif self.score_func == "softmax":
            scores_BLE = F.softmax(scores_BLE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice_BLE = (
            scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
        )
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice_BLE = self._get_node_limited_routing_scores(
                scores_for_choice_BLE
            )
        _, topk_expert_ids_BLK = torch.topk(
            scores_for_choice_BLE, k=self.top_k, dim=-1, sorted=False
        )

        # NOTE: The expert_bias is only used for routing. The gating value
        #       topk_scores_BLK is still derived from the original scores.
        topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                topk_expert_ids_BLK,
                topk_scores_BLK,
            ) = self._debug_force_load_balance_routing(scores_BLE)

        if self.route_norm:
            denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
            topk_scores_BLK = topk_scores_BLK / denominator
        topk_scores_BLK = topk_scores_BLK * self.route_scale

        return (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        )


class _SeqwiseCounts(Module):
    """Per-sequence routing counts and prob sums (DeepSeek-V3 Sec A.2 Eqs 18-20).

    Eq 18: ``f_i = N / (K_r * T) * count_i``.
    Eq 19: ``s'_i,t = s_i,t / sum_j(s_j,t)``.
    Eq 20: ``P_i = 1/T * sum_t(s'_i,t)``.

    ``counts_BE`` counts, per sequence, how often each expert is selected
    (``routing_map_BLE`` scatters the top-k ids into a one-hot map); its
    column sum equals ``K_r * T`` tokens. ``prob_sums_BE`` sums the
    normalized scores per sequence. Both are partial over the token dim and
    are all-reduced at this module's output boundary: the boundary's
    ``out_src -> out_dst`` redistributes Partial -> Invariant over the
    token-partition axes (CP, and TP when EP shards tokens over the TP axis).
    The output concatenates the two ``(B, E)`` tensors into one ``(B, 2E)``
    tensor because the config-based redistribution layer only supports
    single-tensor outputs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        top_k: int

    def __init__(self, config: Config):
        super().__init__()
        self.top_k = config.top_k

    def forward(
        self,
        scores_BLE: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
    ) -> torch.Tensor:
        # The one-hot map is float (not bool) so the counts sum needs no
        # dtype cast: casting a Partial tensor is non-linear and rejected by
        # spmd_types typechecking.
        routing_map_BLE = torch.zeros_like(scores_BLE).scatter_(
            -1, topk_expert_ids_BLK, 1.0
        )
        counts_BE = routing_map_BLE.sum(dim=1)
        probs_BLE = scores_BLE / scores_BLE.sum(dim=-1, keepdim=True)
        prob_sums_BE = probs_BLE.sum(dim=1)
        # Single tensor avoids _redistribute_outputs tuple limitation.
        return torch.cat([counts_BE, prob_sums_BE], dim=-1)


class SeqwiseLoadBalanceLoss(LoggedAuxLoss):
    """Per-sequence MoE load-balance gradient (DeepSeek-V3 Sec A.2 Eqs 17-20).

    Eq 17: ``L = sum_i f_i * P_i`` per sequence, with ``f_i`` (Eq 18) and
    ``P_i`` (Eq 20) from ``_SeqwiseCounts``. The gradient reaches the router
    scores through both the normalized probs (Eq 19) and the top-k score
    carrier; the one-hot counts are non-differentiable, so ``f_i`` receives no
    gradient, matching the paper.

    Formula-internal ``1/T`` (Eqs 18/20) is per-sequence and derived from the
    all-reduced counts (``num_tokens_B``), so it stays CP-correct and
    varlen-safe regardless of the local tensor shape. The framework
    normalizes by ``per_step_denominator = global_batch_size`` (``normalize =
    "sequences"``), i.e. the loss is a mean over the batch's sequences -- the
    DS-V3 Eq 17 convention -- so ``coeff`` directly sets the aux:CE gradient
    ratio (a per-token normalization would weaken it by the sequence length).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LoggedAuxLoss.Config):
        top_k: int
        # "batch" is this loss's disjoint-ownership mesh: ``_SeqwiseCounts``
        # all-reduces across the token-partition group, and the metric is
        # summed over the DP mesh at collection time.
        reduce_mesh: str = "batch"
        # Mean over sequences, not tokens: each sequence's loss is already
        # token-count-normalized inside Eqs 18/20.
        normalize: Literal["tokens", "sequences", "batch"] = "sequences"
        # Boundary sharding config for ``_SeqwiseCounts``: the Partial ->
        # Invariant all-reduce over the token-partition axes (CP, and TP when
        # EP shards tokens over the TP axis) lands at the child's output
        # boundary instead of an explicit collective in forward code. Set by
        # ``set_moe_sharding_config`` before the modules are built; stored
        # here rather than nested in a counts config so
        # ``dataclasses.replace`` in ``Config.build`` does not drop it.
        counts_sharding_config: ShardingConfig | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.top_k = config.top_k
        if get_spmd_backend() != "spmd_types":
            raise ValueError(
                "SeqwiseLoadBalanceLoss requires --parallelism.spmd_backend "
                f"spmd_types, got {get_spmd_backend()!r}. The per-sequence "
                "counts all-reduce relies on spmd_types mesh semantics."
            )
        self._seqwise_counts = _SeqwiseCounts.Config(
            top_k=self.top_k, sharding_config=config.counts_sharding_config
        ).build()

    def forward(
        self,
        topk_scores_BLK: torch.Tensor,
        scores_BLE: torch.Tensor,
        topk_expert_ids_BLK: torch.Tensor,
    ) -> torch.Tensor:
        combined = self._seqwise_counts(scores_BLE, topk_expert_ids_BLK)
        # The child boundary already all-reduced Partial -> Invariant on the
        # token-partition axes (CP, and TP when EP shards tokens over TP).
        # Split the concatenated (counts, prob_sums) tensor.
        E = scores_BLE.size(-1)
        counts_BE, prob_sums_BE = combined[..., :E], combined[..., E:]
        num_tokens_B = counts_BE.sum(dim=1) / self.top_k
        f_BE = counts_BE * (E / (self.top_k * num_tokens_B.unsqueeze(1)))
        p_BE = prob_sums_BE / num_tokens_B.unsqueeze(1)
        loss_per_seq_B = (f_BE * p_BE).sum(dim=1)
        return self.inject(loss_per_seq_B.sum(), carrier=topk_scores_BLK)


class MoE(Module):
    """Mixture of Experts layer.

    The forward pass proceeds as:
    1. Router computes expert assignments (stays on DTensor)
    2. RoutedExperts.forward() converts DTensor to local, then handles:
       a. dispatch (TokenDispatcher) — reorder tokens by expert assignment.
          With EP, also performs all-to-all communication to send tokens
          to expert-owning ranks.
       b. expert computation (GroupedExperts, local tensors)
       c. combine (TokenDispatcher) — reverse the dispatch reordering.
          - LocalTokenDispatcher (no EP): scatter_add only.
          - AllToAll: all-to-all communication, then scatter_add.
          - DeepEP: combine_tokens followed by backend synchronization.
          - HybridEP: synchronous combine_tokens.
    3. Shared experts compute their output.
    4. Routed and shared expert outputs are summed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        routed_experts: RoutedExperts.Config
        router: TokenChoiceTopKRouter.Config
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None
        aux_loss: SeqwiseLoadBalanceLoss.Config | None = None

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.routed_experts = config.routed_experts.build()
        self.router = config.router.build()
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert_E is accumulated in the model forward pass.
        #       expert_bias_E is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias_E",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias_E = None
        # tokens_per_expert_E will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert_E",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_BLD: Input ``(B, L, D)``.

        Returns:
            Output ``(B, L, D)``.

        Under TP, the MoE wrapper's ``sharding_config`` (set by
        ``set_moe_sharding_config``) handles input/output redistribution:
        input is redistributed from sp_layout to desired_input_layouts;
        output is redistributed to sp_layout. MoE.forward() operates on
        DTensors; the DTensor->local conversion happens at the GroupedExperts
        boundary. GroupedExperts operates on local tensors. When EP internally
        sequence-shards tokens across TP, the caller must provide a TP-divisible
        sequence length.
        """
        # topk_scores_BLK and topk_expert_ids_BLK shape (B, L, K)
        # scores_BLE shape (B, L, E)
        (
            topk_scores_BLK,
            topk_expert_ids_BLK,
            scores_BLE,
        ) = self.router(x_BLD, self.expert_bias_E)

        # The aux-loss gradient must reach the router scores, which produce
        # the per-expert gating weights.  Injection before the experts consume
        # the scores means the gradient propagates into the gating network
        # while the MoE output tensor stays unchanged (PP contract).
        if self.training and self.aux_loss is not None:
            topk_scores_BLK = self.aux_loss(
                topk_scores_BLK, scores_BLE, topk_expert_ids_BLK
            )

        # Build a one-hot routing map (B, L, E) marking the experts each token
        # is routed to. Under TP/SP the router outputs are DTensors sharded on
        # the token dim; scatter_ writes along the (replicated) expert dim, so
        # DTensor runs it as a local op with no redistribution.
        routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_BLK,
            True,
        )
        num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))

        # tokens_per_expert_E will be used to update the expert bias for load balancing,
        # and also to count the expert usage.
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert_E --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_BLD = self.routed_experts(
            x_BLD,
            topk_scores_BLK,
            topk_expert_ids_BLK,
            num_local_tokens_per_expert_E,
        )

        shared_out_BLD = (
            self.shared_experts(x_BLD) if self.shared_experts is not None else None
        )

        if shared_out_BLD is not None:
            out_BLD = out_BLD + shared_out_BLD
        return out_BLD

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            # After ``to_empty()``, the existing buffer records the target device.
            # Reinitialize MoE counters there when no explicit buffer device is passed.
            buffer_device = self.tokens_per_expert_E.device

        with torch.device(buffer_device):
            self.tokens_per_expert_E = torch.zeros(
                self.routed_experts.inner_experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias_E = torch.zeros(
                    self.routed_experts.inner_experts.num_experts, dtype=torch.float32
                )
