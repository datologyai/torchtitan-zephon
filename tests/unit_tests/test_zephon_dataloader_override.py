# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from dataclasses import dataclass, field

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
    assert config.dataloader.mixture == {"prose": 0.7, "code": 0.3}
    assert not isinstance(config.validator_dataloader, ZephonDataLoader.Config)


def test_zephon_dataloader_requires_matching_mixture_components() -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    config = ZephonDataLoader.Config(
        sources={"prose": "tests/assets/zephon_mixture/prose"},
        mixture={"code": 1.0},
    )

    with pytest.raises(ValueError, match="must exactly match"):
        ZephonDataLoader._validate_config(config, dp_world_size=1)


def test_zephon_dataloader_yields_torchtitan_batches_and_checkpoints() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    config = ZephonDataLoader.Config(chunk_size=2)

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
