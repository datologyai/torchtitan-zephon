# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Worker process for the Zephon elastic-resume integration test."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch.distributed as dist

from torchtitan.overrides.zephon_dataloader import ZephonDataLoader, ZephonSource


def _batch_state(input_dict, labels):
    return {
        "input": input_dict["input"].tolist(),
        "labels": labels.tolist(),
    }


def _collect_global_steps(
    loader: ZephonDataLoader, *, num_steps: int, canonical_replicas: int
) -> list[list[dict[str, list]]]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batches_per_rank, remainder = divmod(canonical_replicas, world_size)
    if remainder:
        raise ValueError("canonical_replicas must divide the test world size")

    data_iter = iter(loader)
    global_steps: list[list[dict[str, list]]] = []
    for _ in range(num_steps):
        local_batches = [
            _batch_state(*next(data_iter)) for _ in range(batches_per_rank)
        ]
        gathered: list[list[dict[str, list]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_batches)
        if rank == 0:
            global_steps.append(
                [batch for rank_batches in gathered for batch in rank_batches or []]
            )
    return global_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "save", "resume"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--canonical-replicas", type=int, default=2)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_root = (
        Path(__file__).resolve().parents[2] / "tests" / "assets" / "zephon_mixture"
    )
    aggregate_dir = args.output_dir / "aggregate"
    config = ZephonDataLoader.Config(
        sources=[
            ZephonSource(name="prose", path=str(source_root / "prose")),
            ZephonSource(name="code", path=str(source_root / "code")),
        ],
        tokenizer_path=str(
            Path(__file__).resolve().parents[2] / "tests" / "assets" / "tokenizer"
        ),
        chunk_size=2,
        canonical_replicas=args.canonical_replicas,
        aggregate_dir=str(aggregate_dir),
        run_id="elastic-resume-test",
    )
    loader = ZephonDataLoader(
        config,
        dp_world_size=world_size,
        dp_rank=rank,
        seq_len=16,
        local_batch_size=2,
    )

    if args.mode == "resume":
        with (args.output_dir / "checkpoint.pkl").open("rb") as checkpoint_file:
            loader.load_state_dict(pickle.load(checkpoint_file))

    global_steps = _collect_global_steps(
        loader,
        num_steps=args.num_steps,
        canonical_replicas=args.canonical_replicas,
    )

    if args.mode == "save":
        checkpoint = loader.state_dict()
        if rank == 0:
            with (args.output_dir / "checkpoint.pkl").open("wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)
        dist.barrier()

    if rank == 0:
        with (args.output_dir / f"{args.mode}.pkl").open("wb") as output_file:
            pickle.dump(global_steps, output_file)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
