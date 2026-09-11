# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Demonstrate exact Zephon data-stream resume from two workers to one."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = REPO_ROOT / "examples" / "zephon" / "elastic_local_jsonl.toml"
DEFAULT_TOKENIZER = REPO_ROOT / "tests" / "assets" / "tokenizer"
INITIAL_NUM_WORKERS = 2
RESUME_NUM_WORKERS = 1


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as local_socket:
        local_socket.bind(("127.0.0.1", 0))
        return local_socket.getsockname()[1]


def _batch_state(input_dict: dict[str, Any], labels: Any) -> dict[str, Any]:
    return {
        "input": input_dict["input"].tolist(),
        "positions": input_dict["positions"].tolist(),
        "num_valid_tokens": input_dict["num_valid_tokens"],
        "labels": labels.tolist(),
    }


def _collect_global_steps(
    loader: ZephonDataLoader, *, num_steps: int, canonical_replicas: int
) -> list[list[dict[str, Any]]]:
    import torch.distributed as dist

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batches_per_rank, remainder = divmod(canonical_replicas, world_size)
    if remainder:
        raise ValueError(
            f"canonical_replicas={canonical_replicas} must be divisible by "
            f"world_size={world_size}"
        )

    data_iter = iter(loader)
    global_steps: list[list[dict[str, Any]]] = []
    for _ in range(num_steps):
        local_batches = [
            _batch_state(*next(data_iter)) for _ in range(batches_per_rank)
        ]
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_batches)
        if rank == 0:
            global_steps.append(
                [batch for rank_batches in gathered for batch in rank_batches or []]
            )
    return global_steps


def _worker(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.overrides.zephon_dataloader import (
        _load_data_config,
        ZephonDataLoader,
    )

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_values = _load_data_config(str(args.recipe))
    canonical_replicas = config_values.get("canonical_replicas")
    if canonical_replicas is None:
        raise ValueError("The elastic demo recipe must set canonical_replicas")
    config_values.update(
        aggregate_dir=str(args.output_dir / "aggregate"),
        run_id=args.run_id,
    )
    loader = ZephonDataLoader(
        ZephonDataLoader.Config(**config_values),
        dp_world_size=world_size,
        dp_rank=rank,
        tokenizer=HuggingFaceTokenizer(
            HuggingFaceTokenizer.Config(), tokenizer_path=str(args.tokenizer_path)
        ),
        max_context_length=args.max_context_length,
        num_tokens_per_batch=args.num_tokens_per_batch,
    )

    if args.phase == "resume":
        with (args.output_dir / "checkpoint.pkl").open("rb") as checkpoint_file:
            loader.load_state_dict(pickle.load(checkpoint_file))

    global_steps = _collect_global_steps(
        loader,
        num_steps=args.num_steps,
        canonical_replicas=canonical_replicas,
    )

    if args.phase == "save":
        checkpoint = loader.state_dict()
        if rank == 0:
            with (args.output_dir / "checkpoint.pkl").open("wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)
        dist.barrier()

    if rank == 0:
        with (args.output_dir / f"{args.phase}.json").open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(global_steps, output_file)
    dist.barrier()
    dist.destroy_process_group()


def _run_phase(
    *,
    phase: str,
    num_workers: int,
    num_steps: int,
    output_dir: Path,
    recipe: Path,
    tokenizer_path: Path,
    run_id: str,
    max_context_length: int,
    num_tokens_per_batch: int,
) -> None:
    torchrun = Path(sys.executable).with_name("torchrun")
    master_port = _find_free_local_port()
    command = [
        str(torchrun),
        "--nnodes=1",
        f"--nproc-per-node={num_workers}",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        phase,
        "--num-steps",
        str(num_steps),
        "--output-dir",
        str(output_dir),
        "--recipe",
        str(recipe),
        "--tokenizer-path",
        str(tokenizer_path),
        "--run-id",
        run_id,
        "--max-context-length",
        str(max_context_length),
        "--num-tokens-per-batch",
        str(num_tokens_per_batch),
    ]
    env = os.environ | {
        "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        "HF_HOME": str(output_dir / "hf-cache"),
    }
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()


def _load_steps(path: Path) -> list[list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _fingerprints(steps: list[list[dict[str, Any]]]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(step, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()[:8]
        for step in steps
    ]


def _format_fingerprints(fingerprints: list[str]) -> str:
    return "  ".join(fingerprints)


def _validate_worker_topology(canonical_replicas: int) -> None:
    for phase, num_workers in (
        ("initial", INITIAL_NUM_WORKERS),
        ("resume", RESUME_NUM_WORKERS),
    ):
        if canonical_replicas % num_workers:
            raise ValueError(
                f"canonical_replicas={canonical_replicas} must be divisible by "
                f"the {phase} num_workers={num_workers}"
            )


def _run_demo(args: argparse.Namespace, work_dir: Path) -> bool:
    if args.checkpoint_after <= 0 or args.checkpoint_after >= args.total_steps:
        raise ValueError("checkpoint-after must be between 1 and total-steps - 1")

    with args.recipe.open("rb") as recipe_file:
        recipe_values = tomllib.load(recipe_file)
    canonical_replicas = recipe_values.get("canonical_replicas")
    if not isinstance(canonical_replicas, int) or canonical_replicas <= 0:
        raise ValueError("The elastic demo requires positive canonical_replicas")
    _validate_worker_topology(canonical_replicas)
    source_weights = ", ".join(
        f"{source['name']}={source.get('weight', 1.0):g}"
        for source in recipe_values["sources"]
    )

    reference_dir = work_dir / "reference"
    elastic_dir = work_dir / "elastic"
    shared = {
        "recipe": args.recipe,
        "tokenizer_path": args.tokenizer_path,
        "max_context_length": args.max_context_length,
        "num_tokens_per_batch": args.num_tokens_per_batch,
    }
    _run_phase(
        phase="reference",
        num_workers=INITIAL_NUM_WORKERS,
        num_steps=args.total_steps,
        output_dir=reference_dir,
        run_id="zephon-elastic-reference",
        **shared,
    )
    _run_phase(
        phase="save",
        num_workers=INITIAL_NUM_WORKERS,
        num_steps=args.checkpoint_after,
        output_dir=elastic_dir,
        run_id="zephon-elastic-resume",
        **shared,
    )
    _run_phase(
        phase="resume",
        num_workers=RESUME_NUM_WORKERS,
        num_steps=args.total_steps - args.checkpoint_after,
        output_dir=elastic_dir,
        run_id="zephon-elastic-resume",
        **shared,
    )

    reference = _load_steps(reference_dir / "reference.json")
    before_checkpoint = _load_steps(elastic_dir / "save.json")
    after_resume = _load_steps(elastic_dir / "resume.json")
    actual = before_checkpoint + after_resume
    matches = actual == reference

    reference_hashes = _fingerprints(reference)
    before_hashes = _fingerprints(before_checkpoint)
    after_hashes = _fingerprints(after_resume)
    reference_before_text = _format_fingerprints(
        reference_hashes[: args.checkpoint_after]
    )
    reference_after_text = _format_fingerprints(
        reference_hashes[args.checkpoint_after :]
    )
    before_text = _format_fingerprints(before_hashes)
    after_text = _format_fingerprints(after_hashes)
    print("Zephon deterministic elastic resume")
    print(f"Mixture weights:       {source_weights}")
    print("Data-parallel workers: 2 -> 1")
    print(f"{'Reference:':<23}{reference_before_text} | {reference_after_text}")
    print(f"{'2-worker stream:':<23}{before_text} | checkpoint")
    print(f"{'1-worker resume:':<23}{' ' * len(before_text)} | {after_text}")
    print(f"Exact stream match:    {'YES' if matches else 'NO'}")
    return matches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that a Zephon stream checkpointed with two workers resumes "
            "exactly with one worker."
        )
    )
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--checkpoint-after", type=int, default=2)
    parser.add_argument("--max-context-length", type=int, default=16)
    parser.add_argument("--num-tokens-per-batch", type=int, default=32)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Keep checkpoints and stream records in this directory.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase",
        choices=["reference", "save", "resume"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.recipe = args.recipe.resolve()
    args.tokenizer_path = args.tokenizer_path.resolve()
    if args.worker:
        _worker(args)
        return

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.work_dir))
        matches = _run_demo(args, run_dir)
        print(f"Artifacts:             {run_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="zephon-elastic-demo-") as temp_dir:
            matches = _run_demo(args, Path(temp_dir))
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
