# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config import (
    apply_overrides,
    clear_overrides,
    Configurable,
    OverrideConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader


class OverrideRoot(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dataloader: BaseDataLoader.Config = field(
            default_factory=HuggingFaceTextDataLoader.Config
        )
        validator_dataloader: BaseDataLoader.Config = field(
            default_factory=HuggingFaceTextDataLoader.Config
        )


@pytest.fixture(autouse=True)
def clear_override_registry():
    clear_overrides()
    yield
    clear_overrides()


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
    assert not isinstance(config.validator_dataloader, ZephonDataLoader.Config)


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


def test_zephon_override_parses_weighted_local_and_hf_sources() -> None:
    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
        ZephonSource,
    )

    config = zephon_dataloader(
        HuggingFaceTextDataLoader.Config(),
        sources=[
            {"name": "web", "path": "hf://org/dataset/train", "weight": 3.0},
            {"name": "code", "path": "/data/code"},
        ],
        text_field="content",
    )

    assert [(source.name, source.path, source.weight) for source in config.sources] == [
        ("web", "hf://org/dataset/train", 3.0),
        ("code", "/data/code", 1.0),
    ]
    assert config.text_field == "content"

    assert "io_options" not in ZephonDataLoader._runtime_options(
        ZephonDataLoader.Config(
            sources=[ZephonSource(name="local", path="/data/local")]
        ),
        dp_world_size=1,
        dp_rank=0,
    )
    assert ZephonDataLoader._runtime_options(
        ZephonDataLoader.Config(
            sources=[ZephonSource(name="remote", path="hf://org/dataset/train")],
            cache_dir="/data/cache",
        ),
        dp_world_size=1,
        dp_rank=0,
    )["io_options"] == {"cache": {"enabled": True, "root": "/data/cache"}}


def test_zephon_override_loads_local_data_recipe() -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    repo_root = Path(__file__).resolve().parents[2]
    config = zephon_dataloader(
        HuggingFaceTextDataLoader.Config(),
        data_config=str(repo_root / "examples" / "zephon" / "local_jsonl.toml"),
    )

    assert [(source.name, source.weight) for source in config.sources] == [
        ("prose", 1.0),
        ("code", 1.0),
    ]
    assert config.sources[0].path == str(
        repo_root / "tests" / "assets" / "zephon_mixture" / "prose"
    )
    assert config.tokenizer_path == str(repo_root / "tests" / "assets" / "tokenizer")
    assert config.chunk_size == 8


def test_zephon_override_loads_weighted_and_elastic_recipes() -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    repo_root = Path(__file__).resolve().parents[2]
    weighted_config = zephon_dataloader(
        HuggingFaceTextDataLoader.Config(),
        data_config=str(
            repo_root / "examples" / "zephon" / "weighted_local_jsonl.toml"
        ),
    )
    elastic_config = zephon_dataloader(
        HuggingFaceTextDataLoader.Config(),
        data_config=str(
            repo_root / "examples" / "zephon" / "elastic_local_jsonl.toml"
        ),
    )

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
        HuggingFaceTextDataLoader.Config(),
        data_config=str(repo_root / "examples" / "zephon" / "local_jsonl.toml"),
    )

    def make_loader() -> ZephonDataLoader:
        return ZephonDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            seq_len=16,
            local_batch_size=2,
        )

    baseline_iter = iter(make_loader())
    baseline = [next(baseline_iter) for _ in range(3)]

    loader = make_loader()
    loader_iter = iter(loader)
    inputs, labels = next(loader_iter)

    assert inputs["input"].shape == (2, 16)
    assert labels.shape == (2, 16)
    assert inputs["input"].dtype == torch.long
    assert labels.dtype == torch.long
    assert "positions" in inputs

    checkpoint = loader.state_dict()
    assert isinstance(checkpoint["zephon"], bytes)

    restored_loader = make_loader()
    restored_loader.load_state_dict(checkpoint)
    restored_iter = iter(restored_loader)
    restored = [next(restored_iter) for _ in range(2)]

    for (actual_inputs, actual_labels), (expected_inputs, expected_labels) in zip(
        [(inputs, labels), *restored], baseline
    ):
        assert torch.equal(actual_inputs["input"], expected_inputs["input"])
        assert torch.equal(actual_labels, expected_labels)
