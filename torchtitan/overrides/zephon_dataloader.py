# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Example Zephon dataloader override for TorchTitan.

The override replaces only the training ``HuggingFaceTextDataLoader`` with a
Zephon pipeline. It deliberately keeps the integration small: two local JSONL
sources, a deterministic mixture, online tokenization and packing, and Zephon
checkpoint state. Production-specific features belong in an integration
package, not in this illustrative override.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import derive, override
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader


@dataclass(frozen=True, slots=True)
class ZephonSource:
    """One named Zephon dataset and its relative sampling weight."""

    name: str
    path: str
    weight: float = 1.0


def _demo_sources() -> list[ZephonSource]:
    root = Path(__file__).resolve().parents[2] / "tests" / "assets" / "zephon_mixture"
    return [
        ZephonSource(name="prose", path=str(root / "prose")),
        ZephonSource(name="code", path=str(root / "code")),
    ]


def _parse_sources(sources: Sequence[Mapping[str, Any]]) -> list[ZephonSource]:
    parsed_sources = []
    for source in sources:
        try:
            parsed_sources.append(
                ZephonSource(
                    name=source["name"],
                    path=source["path"],
                    weight=source.get("weight", 1.0),
                )
            )
        except KeyError as exc:
            raise ValueError(
                "Each Zephon source must contain 'name' and 'path' fields"
            ) from exc
    return parsed_sources


def _require_zephon() -> tuple[Any, Any, Any, Any]:
    try:
        from zephon import Pipeline
        from zephon.io import Dataset
        from zephon.work import MixtureSpec, StaticMixtureWorkSource
    except ImportError as exc:
        raise ImportError(
            "The Zephon dataloader override requires Zephon. For local development, "
            "install the sibling checkout with `python -m pip install -e "
            "/path/to/zephon`."
        ) from exc
    return Pipeline, Dataset, MixtureSpec, StaticMixtureWorkSource


class ZephonDataLoader(BaseDataLoader):
    """A TorchTitan ``BaseDataLoader`` backed directly by a Zephon pipeline."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        sources: list[ZephonSource] = field(default_factory=_demo_sources)
        """Named local paths or ``hf://`` URIs with optional mixture weights."""

        tokenizer_path: str = "./tests/assets/tokenizer"
        """Local Hugging Face tokenizer path used by Zephon's tokenize stage."""

        text_field: str = "text"
        """Name of the text column shared by all configured sources."""

        cache_dir: str = "./.zephon-cache"
        """Writable local cache for remote shards and decoded Parquet data."""

        seed: int = 42
        chunk_size: int = 64
        canonical_replicas: int | None = None
        aggregate_dir: str | None = None
        run_id: str | None = None
        fetch_parallelism: int | None = None

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        **_: Any,
    ) -> None:
        Pipeline, Dataset, MixtureSpec, StaticMixtureWorkSource = _require_zephon()
        self._validate_config(config, dp_world_size)

        datasets = [
            Dataset.from_path(name=source.name, path=source.path)
            for source in config.sources
        ]
        work_source = StaticMixtureWorkSource(
            datasets=datasets,
            mixture=MixtureSpec(
                {source.name: source.weight for source in config.sources}
            ),
            chunk_size=config.chunk_size,
            seed=config.seed,
            exhausted_policy="repeat",
            shuffle_shards=True,
            shuffle_within_shard=True,
        )

        pipeline = Pipeline(work_source)
        if config.fetch_parallelism is not None:
            pipeline = pipeline.fetch_parallelism(config.fetch_parallelism)
        self._pipeline = (
            pipeline.tokenize(
                tokenizer_id=config.tokenizer_path,
                field=config.text_field,
                max_length=seq_len + 1,
                split_long_samples=True,
                special_tokens="bos_eos",
            )
            .pack_flat(
                max_length=seq_len + 1,
                algorithm="wrap",
                emit_positions=True,
            )
            .batch(local_batch_size, drop_last=True)
            .options(**self._runtime_options(config, dp_world_size, dp_rank))
        )
        self._pipeline.preflight_tokenizers()
        self._tokens_field = "input_ids"

    @staticmethod
    def _validate_config(config: Config, dp_world_size: int) -> None:
        if not config.sources:
            raise ValueError("dataloader.sources must not be empty")
        source_names = [source.name for source in config.sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError("dataloader.sources names must be unique")
        if any(not source.path for source in config.sources):
            raise ValueError("dataloader.sources paths must not be empty")
        if any(source.weight <= 0 for source in config.sources):
            raise ValueError("dataloader.sources weights must all be positive")
        if not config.cache_dir:
            raise ValueError("dataloader.cache_dir must not be empty")
        canonical_replicas = config.canonical_replicas or dp_world_size
        if canonical_replicas < dp_world_size:
            raise ValueError(
                "dataloader.canonical_replicas must be at least the current "
                "data-parallel world size"
            )
        if dp_world_size > 1 and (not config.aggregate_dir or not config.run_id):
            raise ValueError(
                "Distributed Zephon runs require dataloader.aggregate_dir and "
                "dataloader.run_id so checkpoint() can aggregate lane state."
            )

    @staticmethod
    def _runtime_options(
        config: Config, dp_world_size: int, dp_rank: int
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "deterministic": True,
            "dp_degree": dp_world_size,
            "dp_group_id": dp_rank,
            "canonical_replicas": config.canonical_replicas or dp_world_size,
        }
        if any(source.path.startswith("hf://") for source in config.sources):
            options["io_options"] = {
                "cache": {
                    "enabled": True,
                    "root": config.cache_dir,
                }
            }
        if dist.is_initialized():
            options["world_size"] = dist.get_world_size()
            options["global_rank"] = dist.get_rank()
        if config.aggregate_dir is not None:
            options["aggregate_dir"] = config.aggregate_dir
        if config.run_id is not None:
            options["run_id"] = config.run_id
        return options

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        for sample_batch in self._pipeline:
            training_batch = sample_batch.to_training(
                tokens_field=self._tokens_field,
                return_labels=True,
                dtype=torch.long,
                ignore_index=IGNORE_INDEX,
                rename_fields={"input_ids": "input"},
            )
            input_dict = {"input": training_batch["input"]}
            if "positions" in training_batch:
                input_dict["positions"] = training_batch["positions"]
            yield input_dict, training_batch["labels"]

    def state_dict(self) -> dict[str, bytes]:
        """Store the complete Zephon checkpoint as one DCP-safe opaque value."""
        return {"zephon": pickle.dumps(self._pipeline.checkpoint())}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not state_dict or "zephon" not in state_dict:
            return
        checkpoint = state_dict["zephon"]
        if not isinstance(checkpoint, bytes):
            raise ValueError("Expected Zephon checkpoint state to be bytes")
        self._pipeline.restore(pickle.loads(checkpoint))


@override(
    target=HuggingFaceTextDataLoader.Config,
    fqns=["dataloader"],
    exact=True,
    description="Zephon deterministic mixed-data dataloader",
)
def zephon_dataloader(
    config: HuggingFaceTextDataLoader.Config,
    *,
    sources: list[Mapping[str, Any]] | None = None,
    tokenizer_path: str = "./tests/assets/tokenizer",
    text_field: str = "text",
    cache_dir: str = "./.zephon-cache",
    seed: int = 42,
    chunk_size: int = 64,
    canonical_replicas: int | None = None,
    aggregate_dir: str | None = None,
    run_id: str | None = None,
    fetch_parallelism: int | None = None,
) -> ZephonDataLoader.Config:
    """Replace the stock text loader with the compact Zephon demonstration."""
    return derive(
        config,
        ZephonDataLoader.Config,
        sources=_demo_sources() if sources is None else _parse_sources(sources),
        tokenizer_path=tokenizer_path,
        text_field=text_field,
        cache_dir=cache_dir,
        seed=seed,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        aggregate_dir=aggregate_dir,
        run_id=run_id,
        fetch_parallelism=fetch_parallelism,
    )
