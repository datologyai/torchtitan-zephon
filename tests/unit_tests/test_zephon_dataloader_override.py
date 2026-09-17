# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

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


def _tokenizer(repo_root: Path) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(
        HuggingFaceTokenizer.Config(),
        tokenizer_path=str(repo_root / "tests" / "assets" / "tokenizer"),
    )


def test_zephon_overrides_load_local_training_and_validation_recipes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = OverrideRoot.Config()
    override_config = OverrideConfig(
        imports=[
            (
                "torchtitan.overrides.zephon_dataloader.zephon_dataloader",
                {"data_config": str(repo_root / "examples/zephon/local_jsonl.toml")},
            ),
            (
                "torchtitan.overrides.zephon_dataloader.zephon_validation_dataloader",
                {
                    "data_config": str(
                        repo_root / "examples/zephon/validation_local_jsonl.toml"
                    )
                },
            ),
        ]
    )

    importlib.import_module("torchtitan.overrides.zephon_dataloader")
    apply_overrides(override_config, config)

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    assert isinstance(config.dataloader, ZephonDataLoader.Config)
    assert config.dataloader.shuffle
    assert config.dataloader.repeat
    assert config.dataloader.chunk_size == 4
    assert [source.name for source in config.dataloader.sources] == ["prose", "code"]
    assert [source.fmt for source in config.dataloader.sources] == ["jsonl", "jsonl"]
    assert [source.weight for source in config.dataloader.sources] == [3.0, 1.0]

    validation = config.validator.dataloader
    assert isinstance(validation, ZephonDataLoader.Config)
    assert not validation.shuffle
    assert not validation.repeat
    assert [source.name for source in validation.sources] == ["validation"]


def test_zephon_configuration_rejects_unsupported_contracts(tmp_path: Path) -> None:
    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
        ZephonSource,
    )

    with pytest.raises(ValueError, match="does not yet support.*max_num_documents"):
        ZephonDataLoader.Config(max_num_documents=4)

    duplicate_sources = ZephonDataLoader.Config(
        sources=[
            ZephonSource(name="same", path="/data/one"),
            ZephonSource(name="same", path="/data/two"),
        ]
    )
    with pytest.raises(ValueError, match="names must be unique"):
        ZephonDataLoader._validate_config(duplicate_sources, dp_world_size=1)

    recipe = tmp_path / "unknown.toml"
    recipe.write_text(
        """
unsupported_option = true
sources = []
""".strip()
    )
    with pytest.raises(ValueError, match="unsupported_option"):
        zephon_dataloader(_grain_config(), data_config=str(recipe))


def test_zephon_training_batches_and_checkpoint_continuation() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = zephon_dataloader(
        _grain_config(),
        data_config=str(repo_root / "examples" / "zephon" / "local_jsonl.toml"),
    )

    def build_loader() -> ZephonDataLoader:
        return ZephonDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=_tokenizer(repo_root),
            max_context_length=16,
            num_tokens_per_batch=32,
        )

    reference_iterator = iter(build_loader())
    reference = [next(reference_iterator) for _ in range(3)]

    loader = build_loader()
    assert loader._pipeline.ws.requires_token_priming
    loader_iterator = iter(loader)
    first = next(loader_iterator)
    assert not loader._pipeline.ws.requires_token_priming
    assert first["input"].shape == (32,)
    assert first["labels"].shape == (32,)
    assert first["positions"].shape == (32,)
    assert first["input"].dtype == torch.long
    assert first["labels"].dtype == torch.long
    assert first["num_valid_tokens"] == int((first["labels"] != IGNORE_INDEX).sum())

    restored_loader = build_loader()
    restored_loader.load_state_dict(loader.state_dict())
    restored_iterator = iter(restored_loader)
    actual = [first, next(restored_iterator), next(restored_iterator)]

    for actual_batch, expected_batch in zip(actual, reference):
        assert actual_batch.keys() == expected_batch.keys()
        for key in actual_batch:
            if isinstance(actual_batch[key], torch.Tensor):
                assert torch.equal(actual_batch[key], expected_batch[key])
            else:
                assert actual_batch[key] == expected_batch[key]


def test_zephon_validation_is_finite_without_token_estimation() -> None:
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
    loader = ZephonDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=_tokenizer(repo_root),
        max_context_length=16,
        num_tokens_per_batch=32,
    )

    assert not config.shuffle
    assert not config.repeat
    assert not loader._pipeline.ws.requires_token_priming
    assert list(loader)


def test_zephon_override_state_round_trips_through_dcp(tmp_path: Path) -> None:
    pytest.importorskip("zephon")

    repo_root = Path(__file__).resolve().parents[2]
    data_config = repo_root / "examples" / "zephon" / "local_jsonl.toml"
    root_config = OverrideRoot.Config()
    override_config = OverrideConfig(
        imports=[
            (
                "torchtitan.overrides.zephon_dataloader.zephon_dataloader",
                {"data_config": str(data_config)},
            )
        ]
    )
    importlib.import_module("torchtitan.overrides.zephon_dataloader")
    apply_overrides(override_config, root_config)
    tokenizer = _tokenizer(repo_root)

    def build_loader() -> BaseDataLoader:
        return root_config.dataloader.build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            max_context_length=16,
            num_tokens_per_batch=32,
        )

    reference_iterator = iter(build_loader())
    next(reference_iterator)
    expected = next(reference_iterator)

    loader = build_loader()
    next(iter(loader))
    checkpoint_dir = tmp_path / "checkpoint"
    with pytest.warns(UserWarning, match="assuming the intent is to save"):
        dcp.save({"dataloader": loader}, checkpoint_id=checkpoint_dir)

    restored_loader = build_loader()
    with pytest.warns(UserWarning, match="assuming the intent is to load"):
        dcp.load({"dataloader": restored_loader}, checkpoint_id=checkpoint_dir)
    actual = next(iter(restored_loader))

    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]
