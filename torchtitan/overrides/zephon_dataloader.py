# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Example Zephon training and validation dataloaders for TorchTitan.

The overrides replace TorchTitan's training and validation ``GrainDataLoader``
instances with Zephon pipelines. They deliberately keep the integration small:
named data sources, deterministic mixtures, online tokenization and packing,
and Zephon checkpoint state. Production-specific features belong in an
integration package, not in this illustrative override.
"""

from __future__ import annotations

import pickle
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
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
        "training",
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
        raise ValueError(  # noqa: TRY004
            "Zephon data recipes must contain a 'sources' list"
        )

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


def _require_zephon() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from zephon import Pipeline
        from zephon.io import Dataset
        from zephon.work import MixtureSpec, StaticMixtureWorkSource, TokenEstimation
    except ImportError as exc:
        raise ImportError(
            "The Zephon dataloader override requires Zephon. Install the pinned "
            "distribution from requirements-zephon.txt."
        ) from exc
    return Pipeline, Dataset, MixtureSpec, StaticMixtureWorkSource, TokenEstimation


class ZephonDataLoader(BaseDataLoader):
    """A TorchTitan ``BaseDataLoader`` backed directly by a Zephon pipeline."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        sources: list[ZephonSource] = field(default_factory=list)
        """Named local paths or ``hf://`` URIs with optional mixture weights."""

        text_field: str = "text"
        """Name of the text column shared by all configured sources."""

        cache_dir: str | None = None
        """Optional writable cache for all file-backed sources."""

        seed: int = 42
        chunk_size: int = 64
        training: bool = True
        """Enable training-time shuffling and token-aware mixture scheduling."""

        repeat: bool = True
        """Repeat indefinitely, or stop after every source completes a pass."""

        canonical_replicas: int | None = None
        aggregate_dir: str | None = None
        run_id: str | None = None
        fetch_parallelism: int | None = None

        def __post_init__(self) -> None:
            if self.max_num_documents is not None:
                raise ValueError(
                    "ZephonDataLoader does not yet support dataloader.max_num_documents"
                )
            BaseDataLoader.Config.__post_init__(self)

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        num_tokens_per_train_step: int | None = None,
        **_: Any,
    ) -> None:
        (
            Pipeline,
            Dataset,
            MixtureSpec,
            StaticMixtureWorkSource,
            TokenEstimation,
        ) = _require_zephon()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._validate_config(
            config,
            dp_world_size,
            world_size,
            num_tokens_per_batch=num_tokens_per_batch,
            num_tokens_per_train_step=num_tokens_per_train_step,
        )
        if num_tokens_per_batch % max_context_length:
            raise ValueError(
                "num_tokens_per_batch must be divisible by max_context_length"
            )
        local_batch_size = num_tokens_per_batch // max_context_length

        datasets = [
            Dataset.from_path(name=source.name, path=source.path, fmt=source.fmt)
            for source in config.sources
        ]
        work_source_options: dict[str, Any] = {
            "datasets": datasets,
            "mixture": MixtureSpec(
                {source.name: source.weight for source in config.sources}
            ),
            "chunk_size": config.chunk_size,
            "seed": config.seed,
            "shuffle_shards": config.training,
            "shuffle_within_shard": config.training,
        }
        if config.training:
            work_source_options["token_estimation"] = TokenEstimation()
        if config.repeat:
            work_source_options["exhausted_policy"] = "repeat"
        work_source = StaticMixtureWorkSource(**work_source_options)

        pipeline = Pipeline(work_source)
        if config.fetch_parallelism is not None:
            pipeline = pipeline.fetch_parallelism(config.fetch_parallelism)

        tokenizer_path = getattr(tokenizer, "tokenizer_path", None)
        if not isinstance(tokenizer_path, str) or not tokenizer_path:
            raise ValueError("Zephon input requires a tokenizer with a tokenizer_path")
        pipeline = pipeline.tokenize(
            tokenizer_id=tokenizer_path,
            field=config.text_field,
            add_attention_mask=False,
            max_length=max_context_length + 1,
            split_long_samples=True,
            special_tokens="bos_eos",
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
        )
        pipeline = pipeline.pack_flat(
            max_length=max_context_length + 1,
            algorithm="wrap",
            emit_positions=True,
        )
        pipeline.preflight_tokenizers()

        self._pipeline = pipeline.batch(local_batch_size, drop_last=True).options(
            **self._runtime_options(config, dp_world_size, dp_rank)
        )
        self._tokens_field = "input_ids"

    @staticmethod
    def _validate_config(
        config: Config,
        dp_world_size: int,
        world_size: int,
        *,
        num_tokens_per_batch: int | None = None,
        num_tokens_per_train_step: int | None = None,
    ) -> None:
        if not isinstance(config.training, bool):
            raise ValueError(  # noqa: TRY004
                "dataloader.training must be true or false"
            )
        if not config.sources:
            raise ValueError("dataloader.sources must not be empty")
        source_names = [source.name for source in config.sources]
        if any(not name for name in source_names):
            raise ValueError("dataloader.sources names must not be empty")
        if len(set(source_names)) != len(source_names):
            raise ValueError("dataloader.sources names must be unique")
        if any(not source.path for source in config.sources):
            raise ValueError("dataloader.sources paths must not be empty")
        if any(
            not isinstance(source.weight, (int, float))
            or not isfinite(source.weight)
            or source.weight <= 0
            for source in config.sources
        ):
            raise ValueError("dataloader.sources weights must all be positive")
        if not config.text_field:
            raise ValueError("dataloader.text_field must not be empty")
        if config.cache_dir is not None and not config.cache_dir:
            raise ValueError("dataloader.cache_dir must not be empty when set")
        if config.chunk_size <= 0:
            raise ValueError("dataloader.chunk_size must be positive")
        if config.fetch_parallelism is not None and config.fetch_parallelism <= 0:
            raise ValueError("dataloader.fetch_parallelism must be positive when set")
        if config.canonical_replicas is not None and config.canonical_replicas <= 0:
            raise ValueError("dataloader.canonical_replicas must be positive when set")
        if config.aggregate_dir is not None and not config.aggregate_dir:
            raise ValueError("dataloader.aggregate_dir must not be empty when set")
        if config.run_id is not None and not config.run_id:
            raise ValueError("dataloader.run_id must not be empty when set")
        canonical_replicas = (
            config.canonical_replicas
            if config.canonical_replicas is not None
            else dp_world_size
        )
        if canonical_replicas < dp_world_size:
            raise ValueError(
                "dataloader.canonical_replicas must be at least the current "
                "data-parallel world size"
            )
        if canonical_replicas % dp_world_size:
            raise ValueError(
                "dataloader.canonical_replicas must be divisible by the current "
                "data-parallel world size so every rank owns the same number of lanes"
            )
        if (
            num_tokens_per_train_step is not None
            and num_tokens_per_batch is not None
        ):
            num_batches_per_step, remainder = divmod(
                num_tokens_per_train_step, num_tokens_per_batch
            )
            if remainder:
                raise ValueError(
                    "num_tokens_per_train_step must be divisible by "
                    "num_tokens_per_batch"
                )
            if num_batches_per_step % canonical_replicas:
                raise ValueError(
                    "The number of batches consumed per training step must be a "
                    "multiple of dataloader.canonical_replicas so checkpoints land "
                    "on complete lane-window boundaries"
                )
        if world_size > 1 and (not config.aggregate_dir or not config.run_id):
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
            "canonical_replicas": (
                config.canonical_replicas
                if config.canonical_replicas is not None
                else dp_world_size
            ),
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
            yield sample_batch.to_training(
                tokens_field=self._tokens_field,
                return_labels=True,
                dtype=torch.long,
                ignore_index=IGNORE_INDEX,
                rename_fields={"input_ids": "input"},
                flatten=True,
                exclude_fields=("ids", "texts"),
                return_num_valid_tokens=True,
            )

    def state_dict(self) -> dict[str, bytes]:
        """Store the complete Zephon checkpoint as one DCP-safe opaque value."""
        return {"zephon": pickle.dumps(self._pipeline.checkpoint())}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not state_dict:
            raise ValueError(
                "Zephon dataloader checkpoint state is empty during resume; "
                "fresh training must not call load_state_dict()"
            )
        if "zephon" not in state_dict:
            raise ValueError("Zephon dataloader checkpoint is missing 'zephon' state")
        checkpoint = state_dict["zephon"]
        if not isinstance(checkpoint, bytes):
            raise ValueError(  # noqa: TRY004
                "Expected Zephon checkpoint state to be bytes"
            )
        try:
            zephon_state = pickle.loads(checkpoint)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
            raise ValueError(
                "Zephon dataloader checkpoint contains corrupt pickle"
            ) from exc
        try:
            self._pipeline.restore(zephon_state)
        except Exception as exc:
            raise ValueError(
                "Zephon dataloader checkpoint is malformed or incompatible"
            ) from exc


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
    """Replace the training loader with a shuffled, repeating Zephon stream."""
    return _derive_zephon_config(
        config,
        data_config=data_config,
        sources=sources,
        text_field=text_field,
        cache_dir=cache_dir,
        seed=seed,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        aggregate_dir=aggregate_dir,
        run_id=run_id,
        fetch_parallelism=fetch_parallelism,
        default_training=True,
        repeat=True,
    )


@override(
    target=GrainDataLoader.Config,
    fqns=["validator.dataloader"],
    exact=True,
    description="Zephon deterministic validation dataloader",
)
def zephon_validation_dataloader(
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
    """Replace the validation loader with an unshuffled Zephon stream."""
    return _derive_zephon_config(
        config,
        data_config=data_config,
        sources=sources,
        text_field=text_field,
        cache_dir=cache_dir,
        seed=seed,
        chunk_size=chunk_size,
        canonical_replicas=canonical_replicas,
        aggregate_dir=aggregate_dir,
        run_id=run_id,
        fetch_parallelism=fetch_parallelism,
        default_training=False,
        repeat=config.repeat,
    )


def _derive_zephon_config(
    config: GrainDataLoader.Config,
    *,
    data_config: str | None,
    sources: list[Mapping[str, Any]] | None,
    text_field: str | None,
    cache_dir: str | None,
    seed: int | None,
    chunk_size: int | None,
    canonical_replicas: int | None,
    aggregate_dir: str | None,
    run_id: str | None,
    fetch_parallelism: int | None,
    default_training: bool,
    repeat: bool,
) -> ZephonDataLoader.Config:
    deltas = {} if data_config is None else _load_data_config(data_config)
    if sources is not None:
        deltas["sources"] = _parse_sources(sources)
    elif "sources" not in deltas:
        raise ValueError(
            "The Zephon override requires data_config or explicit sources; it does "
            "not translate the existing Grain dataset configuration"
        )
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
    deltas.setdefault("training", default_training)
    deltas["repeat"] = repeat
    return derive(config, ZephonDataLoader.Config, **deltas)
