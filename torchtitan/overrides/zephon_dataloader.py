# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Example Zephon dataloader override for TorchTitan.

The override replaces only the training ``GrainDataLoader`` with a
Zephon pipeline. It deliberately keeps the integration small: two local JSONL
sources, a deterministic mixture, online tokenization and packing, and Zephon
checkpoint state. Production-specific features belong in an integration
package, not in this illustrative override.
"""

from __future__ import annotations

import pickle
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.loader import BaseDataLoader, GrainDataLoader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import derive, override


@dataclass(frozen=True, slots=True)
class ZephonSource:
    """One named Zephon dataset, optional format, and relative sampling weight."""

    name: str
    path: str
    fmt: str | None = None
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
                    fmt=source.get("fmt"),
                    weight=source.get("weight", 1.0),
                )
            )
        except KeyError as exc:
            raise ValueError(
                "Each Zephon source must contain 'name' and 'path' fields"
            ) from exc
    return parsed_sources


def _resolve_recipe_path(path: str, recipe_dir: Path) -> str:
    if "://" in path or Path(path).is_absolute():
        return path
    return str((recipe_dir / path).resolve())


def _load_data_config(data_config: str) -> dict[str, Any]:
    recipe_path = Path(data_config)
    with recipe_path.open("rb") as recipe_file:
        values = tomllib.load(recipe_file)

    allowed_keys = {
        "sources",
        "text_field",
        "cache_dir",
        "seed",
        "chunk_size",
        "canonical_replicas",
        "aggregate_dir",
        "run_id",
        "fetch_parallelism",
    }
    unknown_keys = set(values) - allowed_keys
    if unknown_keys:
        raise ValueError(
            "Unknown Zephon data recipe keys: " + ", ".join(sorted(unknown_keys))
        )
    raw_sources = values.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Zephon data recipes must contain a 'sources' list")

    recipe_dir = recipe_path.parent
    sources = _parse_sources(raw_sources)
    values["sources"] = [
        ZephonSource(
            name=source.name,
            path=_resolve_recipe_path(source.path, recipe_dir),
            fmt=source.fmt,
            weight=source.weight,
        )
        for source in sources
    ]
    for key in ("cache_dir", "aggregate_dir"):
        if key in values and values[key] is not None:
            values[key] = _resolve_recipe_path(values[key], recipe_dir)
    return values


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

        text_field: str = "text"
        """Name of the text column shared by all configured sources."""

        cache_dir: str | None = None
        """Optional writable cache for all file-backed sources."""

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
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **_: Any,
    ) -> None:
        Pipeline, Dataset, MixtureSpec, StaticMixtureWorkSource = _require_zephon()
        self._validate_config(config, dp_world_size)
        tokenizer_path = getattr(tokenizer, "tokenizer_path", None)
        if not isinstance(tokenizer_path, str) or not tokenizer_path:
            raise ValueError(
                "ZephonDataLoader requires a tokenizer with a tokenizer_path"
            )

        datasets = [
            Dataset.from_path(name=source.name, path=source.path, fmt=source.fmt)
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
                tokenizer_id=tokenizer_path,
                field=config.text_field,
                max_length=max_context_length + 1,
                split_long_samples=True,
                special_tokens="bos_eos",
            )
            .pack_flat(
                max_length=num_tokens_per_batch + 1,
                algorithm="wrap",
                emit_positions=True,
            )
            .batch(1, drop_last=True)
            .options(**self._runtime_options(config, dp_world_size, dp_rank))
        )
        self._pipeline.preflight_tokenizers()
        self._tokens_field = "input_ids"

    @staticmethod
    def _validate_config(config: Config, dp_world_size: int) -> None:
        if config.max_num_documents is not None:
            raise ValueError(
                "ZephonDataLoader does not yet support dataloader.max_num_documents"
            )
        if not config.sources:
            raise ValueError("dataloader.sources must not be empty")
        source_names = [source.name for source in config.sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError("dataloader.sources names must be unique")
        if any(not source.path for source in config.sources):
            raise ValueError("dataloader.sources paths must not be empty")
        if any(source.weight <= 0 for source in config.sources):
            raise ValueError("dataloader.sources weights must all be positive")
        if config.cache_dir is not None and not config.cache_dir:
            raise ValueError("dataloader.cache_dir must not be empty when set")
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
        if config.cache_dir is not None:
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

    def __iter__(self) -> Iterator[TrainerBatch]:
        for sample_batch in self._pipeline:
            converted_batch = sample_batch.to_training(
                tokens_field=self._tokens_field,
                return_labels=True,
                dtype=torch.long,
                ignore_index=IGNORE_INDEX,
                rename_fields={"input_ids": "input"},
            )
            labels = converted_batch["labels"].squeeze(0)
            training_batch = {
                "input": converted_batch["input"].squeeze(0),
                "labels": labels,
                "num_valid_tokens": int((labels != IGNORE_INDEX).sum()),
            }
            if "positions" in converted_batch:
                training_batch["positions"] = converted_batch["positions"].squeeze(0)
            yield training_batch

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
    target=GrainDataLoader.Config,
    fqns=["dataloader"],
    exact=True,
    description="Zephon deterministic mixed-data dataloader",
)
def zephon_dataloader(
    config: GrainDataLoader.Config,
    *,
    data_config: str | None = None,
    sources: list[Mapping[str, Any]] | None = None,
    text_field: str | None = None,
    cache_dir: str | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
    canonical_replicas: int | None = None,
    aggregate_dir: str | None = None,
    run_id: str | None = None,
    fetch_parallelism: int | None = None,
) -> ZephonDataLoader.Config:
    """Replace the stock text loader with the compact Zephon demonstration."""
    deltas = {} if data_config is None else _load_data_config(data_config)
    if sources is not None:
        deltas["sources"] = _parse_sources(sources)
    elif "sources" not in deltas:
        deltas["sources"] = _demo_sources()
    for name, value in {
        "text_field": text_field,
        "cache_dir": cache_dir,
        "seed": seed,
        "chunk_size": chunk_size,
        "canonical_replicas": canonical_replicas,
        "aggregate_dir": aggregate_dir,
        "run_id": run_id,
        "fetch_parallelism": fetch_parallelism,
    }.items():
        if value is not None:
            deltas[name] = value
    return derive(config, ZephonDataLoader.Config, **deltas)
