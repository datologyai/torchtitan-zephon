# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
import torch

from torchtitan.components.data.loader import BaseDataLoader, GrainDataLoader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import (
    apply_overrides,
    clear_overrides,
    Configurable,
    OverrideConfig,
)
from torchtitan.hf_datasets.text_datasets import DATASETS


def _grain_config() -> GrainDataLoader.Config:
    return GrainDataLoader.Config(dataset=DATASETS["c4_test"])


class OverrideRoot(Configurable):
    @dataclass(kw_only=True, slots=True)
    class ValidatorConfig(Configurable.Config):
        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: GrainDataLoader.Config(
                dataset=DATASETS["c4_validation"],
                repeat=False,
            )
        )

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dataloader: BaseDataLoader.Config = field(default_factory=_grain_config)
        validator: "OverrideRoot.ValidatorConfig" = field(
            default_factory=lambda: OverrideRoot.ValidatorConfig()
        )


@pytest.fixture(autouse=True)
def clear_override_registry():
    clear_overrides()
    sys.modules.pop("torchtitan.overrides.zephon_dataloader", None)
    yield
    clear_overrides()
    sys.modules.pop("torchtitan.overrides.zephon_dataloader", None)


def test_zephon_override_replaces_only_training_dataloader() -> None:
    config = OverrideRoot.Config()
    override_config = OverrideConfig(
        imports=["torchtitan.overrides.zephon_dataloader.zephon_dataloader"]
    )

    importlib.import_module("torchtitan.overrides.zephon_dataloader")
    apply_overrides(override_config, config)

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    assert isinstance(config.dataloader, ZephonDataLoader.Config)
    assert [source.weight for source in config.dataloader.sources] == [1.0, 1.0]
    assert not isinstance(config.validator.dataloader, ZephonDataLoader.Config)


def test_zephon_overrides_replace_training_and_validation_dataloaders() -> None:
    config = OverrideRoot.Config()
    override_config = OverrideConfig(
        imports=[
            "torchtitan.overrides.zephon_dataloader.zephon_dataloader",
            "torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader",
        ]
    )

    importlib.import_module("torchtitan.overrides.zephon_dataloader")
    apply_overrides(override_config, config)

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    assert isinstance(config.dataloader, ZephonDataLoader.Config)
    assert config.dataloader.shuffle
    assert config.dataloader.repeat
    assert isinstance(config.validator.dataloader, ZephonDataLoader.Config)
    assert not config.validator.dataloader.shuffle
    assert not config.validator.dataloader.repeat


def test_zephon_dataloader_validates_sources() -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader, ZephonSource

    config = ZephonDataLoader.Config(
        sources=[
            ZephonSource(name="prose", path="/data/prose"),
            ZephonSource(name="prose", path="/data/code"),
        ],
    )

    with pytest.raises(ValueError, match="names must be unique"):
        ZephonDataLoader._validate_config(config, dp_world_size=1)

    with pytest.raises(ValueError, match="cache_dir must not be empty"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(cache_dir=""), dp_world_size=1
        )

    with pytest.raises(ValueError, match="does not yet support.*max_num_documents"):
        ZephonDataLoader.Config(max_num_documents=4)

    with pytest.raises(ValueError, match="weights must all be positive"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(
                sources=[ZephonSource(name="prose", path="/data", weight=0)]
            ),
            dp_world_size=1,
        )


def test_zephon_dataloader_validates_distributed_configuration() -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    with pytest.raises(ValueError, match="must be at least"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(canonical_replicas=1), dp_world_size=2
        )

    with pytest.raises(ValueError, match="aggregate_dir and dataloader.run_id"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(canonical_replicas=2), dp_world_size=2
        )

    with pytest.raises(ValueError, match="aggregate_dir and dataloader.run_id"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(
                canonical_replicas=2,
                aggregate_dir="/aggregate",
            ),
            dp_world_size=2,
        )


def test_zephon_override_parses_weighted_local_and_hf_sources() -> None:
    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
        ZephonSource,
    )

    config = zephon_dataloader(
        _grain_config(),
        sources=[
            {
                "name": "web",
                "path": "hf://org/dataset/train",
                "fmt": "parquet",
                "weight": 3.0,
            },
            {"name": "code", "path": "/data/code"},
        ],
        text_field="content",
    )

    assert [
        (source.name, source.path, source.fmt, source.weight)
        for source in config.sources
    ] == [
        ("web", "hf://org/dataset/train", "parquet", 3.0),
        ("code", "/data/code", None, 1.0),
    ]
    assert config.text_field == "content"

    assert "io_options" not in ZephonDataLoader._runtime_options(
        ZephonDataLoader.Config(
            sources=[ZephonSource(name="remote", path="hf://org/dataset/train")]
        ),
        dp_world_size=1,
        dp_rank=0,
    )
    assert ZephonDataLoader._runtime_options(
        ZephonDataLoader.Config(
            sources=[ZephonSource(name="local", path="/data/local")],
            cache_dir="/data/cache",
        ),
        dp_world_size=1,
        dp_rank=0,
    )["io_options"] == {"cache": {"enabled": True, "root": "/data/cache"}}


def test_zephon_override_loads_local_data_recipe() -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    repo_root = Path(__file__).resolve().parents[2]
    config = zephon_dataloader(
        _grain_config(),
        data_config=str(repo_root / "examples" / "zephon" / "local_jsonl.toml"),
    )

    assert [(source.name, source.weight) for source in config.sources] == [
        ("prose", 1.0),
        ("code", 1.0),
    ]
    assert config.sources[0].path == str(
        repo_root / "tests" / "assets" / "zephon_mixture" / "prose"
    )
    assert config.chunk_size == 8


def test_zephon_override_loads_weighted_and_elastic_recipes() -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    repo_root = Path(__file__).resolve().parents[2]
    weighted_config = zephon_dataloader(
        _grain_config(),
        data_config=str(
            repo_root / "examples" / "zephon" / "weighted_local_jsonl.toml"
        ),
    )
    elastic_config = zephon_dataloader(
        _grain_config(),
        data_config=str(repo_root / "examples" / "zephon" / "elastic_local_jsonl.toml"),
    )

    assert [source.fmt for source in weighted_config.sources] == ["jsonl", "jsonl"]
    assert [source.weight for source in weighted_config.sources] == [3.0, 1.0]
    assert elastic_config.canonical_replicas == 2


def test_zephon_dataloader_yields_torchtitan_batches_and_checkpoints() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = zephon_dataloader(
        _grain_config(),
        data_config=str(
            repo_root / "examples" / "zephon" / "weighted_local_jsonl.toml"
        ),
    )

    def make_loader() -> ZephonDataLoader:
        return ZephonDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=HuggingFaceTokenizer(
                HuggingFaceTokenizer.Config(),
                tokenizer_path=str(repo_root / "tests" / "assets" / "tokenizer"),
            ),
            max_context_length=16,
            num_tokens_per_batch=32,
        )

    baseline_iter = iter(make_loader())
    baseline = [next(baseline_iter) for _ in range(3)]

    loader = make_loader()
    loader_iter = iter(loader)
    batch = next(loader_iter)

    assert batch["input"].shape == (32,)
    assert batch["labels"].shape == (32,)
    assert batch["input"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert "positions" in batch
    assert batch["num_valid_tokens"] == int((batch["labels"] != IGNORE_INDEX).sum())

    checkpoint = loader.state_dict()
    assert isinstance(checkpoint["zephon"], bytes)

    loader.load_state_dict({})
    with pytest.raises(ValueError, match="missing 'zephon' state"):
        loader.load_state_dict({"unexpected": b"state"})
    with pytest.raises(ValueError, match="checkpoint state to be bytes"):
        loader.load_state_dict({"zephon": "not-bytes"})

    restored_loader = make_loader()
    restored_loader.load_state_dict(checkpoint)
    restored_iter = iter(restored_loader)
    restored = [next(restored_iter) for _ in range(2)]

    for actual, expected in zip([batch, *restored], baseline):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                assert torch.equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]


def test_zephon_validation_dataloader_is_unshuffled_and_finite() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import (
        zephon_validation_dataloader,
        ZephonDataLoader,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = zephon_validation_dataloader(
        GrainDataLoader.Config(
            dataset=DATASETS["c4_validation"],
            repeat=False,
        ),
        data_config=str(
            repo_root / "examples" / "zephon" / "validation_local_jsonl.toml"
        ),
    )

    assert not config.shuffle
    assert not config.repeat
    assert replace(config, repeat=True).repeat

    loader = ZephonDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(
            HuggingFaceTokenizer.Config(),
            tokenizer_path=str(repo_root / "tests" / "assets" / "tokenizer"),
        ),
        max_context_length=16,
        num_tokens_per_batch=32,
    )
    batches = list(loader)

    assert batches
    assert len(batches) < 10
