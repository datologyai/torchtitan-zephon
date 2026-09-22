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
        "text_field",
        "cache_dir",
        "cache_limit_bytes",
        "seed",
        "chunk_size",
        "shuffle_shards",
        "shuffle_within_shard",
        "shuffle_block_size",
        "token_estimation",
        "shuffle_after_pack",
        "shuffle_buffer_size",
        "shuffle_parallelism",
        "repeat",
        "canonical_replicas",
        "aggregate_dir",
        "run_id",
        "fetch_parallelism",
        "prefetch_buffer_size",
        "prefetch_parallelism",
        "tokenize_parallelism",
        "pack_parallelism",
        "runner",
        "mtp_mode",
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
            "The Zephon dataloader override requires Zephon. Install the configured "
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
        """Optional writable cache for local and cloud-backed sources."""

        cache_limit_bytes: int | None = None
        """Maximum bytes retained in the on-disk shard cache."""

        seed: int = 42
        chunk_size: int = 16_384
        shuffle_shards: bool = True
        shuffle_within_shard: bool = True
        shuffle_block_size: int | str | None = "auto"
        token_estimation: bool = True
        shuffle_after_pack: bool = True
        shuffle_buffer_size: int | None = None
        shuffle_parallelism: int | None = None

        repeat: bool = True
        """Repeat indefinitely, or stop after every source completes a pass."""

        canonical_replicas: int | None = None
        aggregate_dir: str | None = None
        run_id: str | None = None
        fetch_parallelism: int | None = None
        prefetch_buffer_size: int = 0
        prefetch_parallelism: int | None = None
        tokenize_parallelism: int | None = None
        pack_parallelism: int | None = None
        runner: str = "process"
        mtp_mode: bool | None = None

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
        work_source = StaticMixtureWorkSource(
            datasets=datasets,
            mixture=MixtureSpec(
                {source.name: source.weight for source in config.sources}
            ),
            chunk_size=config.chunk_size,
            seed=config.seed,
            shuffle_shards=config.shuffle_shards,
            shuffle_within_shard=config.shuffle_within_shard,
            shuffle_block_size=config.shuffle_block_size,
            token_estimation=(TokenEstimation() if config.token_estimation else None),
            exhausted_policy="repeat" if config.repeat else None,
        )

        pipeline = Pipeline(work_source)
        if config.prefetch_buffer_size:
            pipeline = pipeline.prefetch(
                buffer_size=config.prefetch_buffer_size,
                parallelism=config.prefetch_parallelism,
            )
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
            parallelism=config.tokenize_parallelism,
        )
        pipeline = pipeline.pack_flat(
            max_length=max_context_length + 1,
            algorithm="wrap",
            emit_positions=True,
            parallelism=config.pack_parallelism,
        )
        if config.shuffle_after_pack:
            pipeline = pipeline.shuffle(
                seed=config.seed,
                buffer_size=config.shuffle_buffer_size,
                parallelism=config.shuffle_parallelism,
            )
        pipeline.preflight_tokenizers()

        pipeline = pipeline.batch(local_batch_size, drop_last=True)
        self._pipeline = self._apply_runtime_options(
            pipeline,
            config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
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
        canonical_replicas = (
            config.canonical_replicas
            if config.canonical_replicas is not None
            else dp_world_size
        )
        if canonical_replicas > 0 and canonical_replicas % dp_world_size:
            raise ValueError(
                "dataloader.canonical_replicas must be divisible by the current "
                "data-parallel world size so every rank owns the same number of lanes"
            )
        if (
            canonical_replicas > 0
            and num_tokens_per_train_step is not None
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
    def _apply_runtime_options(
        pipeline: Any,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
    ) -> Any:
        from zephon.io import CacheOptions, StoreOptions

        pipeline = pipeline.options(
            runner=config.runner,
            deterministic=True,
            dp_degree=dp_world_size,
            dp_group_id=dp_rank,
            canonical_replicas=(
                config.canonical_replicas
                if config.canonical_replicas is not None
                else dp_world_size
            ),
            mtp_mode=config.mtp_mode if config.mtp_mode is not None else True,
        )
        if config.cache_dir is not None:
            pipeline = pipeline.options(
                io_options=StoreOptions(
                    cache=CacheOptions(
                        enabled=True,
                        root=config.cache_dir,
                        limit_bytes=config.cache_limit_bytes,
                    )
                )
            )
        if dist.is_initialized():
            pipeline = pipeline.options(
                world_size=dist.get_world_size(),
                global_rank=dist.get_rank(),
            )
        if config.aggregate_dir is not None:
            pipeline = pipeline.options(aggregate_dir=config.aggregate_dir)
        if config.run_id is not None:
            pipeline = pipeline.options(run_id=config.run_id)
        return pipeline

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
    data_config: str,
    canonical_replicas: int | None = None,
    aggregate_dir: str | None = None,
    run_id: str | None = None,
) -> ZephonDataLoader.Config:
    """Replace the training loader with a recipe-defined Zephon stream."""
    return _derive_zephon_config(
        config,
        data_config=data_config,
        canonical_replicas=canonical_replicas,
        aggregate_dir=aggregate_dir,
        run_id=run_id,
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
    data_config: str,
    canonical_replicas: int | None = None,
    aggregate_dir: str | None = None,
    run_id: str | None = None,
) -> ZephonDataLoader.Config:
    """Replace the validation loader with a recipe-defined Zephon stream."""
    return _derive_zephon_config(
        config,
        data_config=data_config,
        canonical_replicas=canonical_replicas,
        aggregate_dir=aggregate_dir,
        run_id=run_id,
    )


def _derive_zephon_config(
    config: GrainDataLoader.Config,
    *,
    data_config: str,
    canonical_replicas: int | None,
    aggregate_dir: str | None,
    run_id: str | None,
) -> ZephonDataLoader.Config:
    deltas = _load_data_config(data_config)
    for name, value in {
        "canonical_replicas": canonical_replicas,
        "aggregate_dir": aggregate_dir,
        "run_id": run_id,
    }.items():
        if value is not None:
            deltas[name] = value
    return derive(config, ZephonDataLoader.Config, **deltas)
