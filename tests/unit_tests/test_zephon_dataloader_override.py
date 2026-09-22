# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import pickle
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.components.data.loader import BaseDataLoader, GrainDataLoader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import (
    Configurable,
    OverrideConfig,
    apply_overrides,
    clear_overrides,
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
    assert config.dataloader.shuffle_shards
    assert config.dataloader.shuffle_within_shard
    assert config.dataloader.shuffle_block_size == "auto"
    assert config.dataloader.token_estimation
    assert config.dataloader.shuffle_after_pack
    assert config.dataloader.runner == "process"
    assert config.dataloader.mtp_mode is None
    assert config.dataloader.repeat
    assert config.dataloader.chunk_size == 4
    assert [source.name for source in config.dataloader.sources] == ["prose", "code"]
    assert [source.fmt for source in config.dataloader.sources] == ["jsonl", "jsonl"]
    assert [source.weight for source in config.dataloader.sources] == [3.0, 1.0]

    validation = config.validator.dataloader
    assert isinstance(validation, ZephonDataLoader.Config)
    assert not validation.shuffle_shards
    assert not validation.shuffle_within_shard
    assert validation.shuffle_block_size == "auto"
    assert not validation.token_estimation
    assert not validation.shuffle_after_pack
    assert not validation.repeat
    assert [source.name for source in validation.sources] == ["validation"]


def test_zephon_configuration_rejects_unsupported_contracts(tmp_path: Path) -> None:
    from torchtitan.overrides.zephon_dataloader import (
        ZephonDataLoader,
        ZephonSource,
        zephon_dataloader,
        zephon_validation_dataloader,
    )

    with pytest.raises(ValueError, match="does not yet support.*max_num_documents"):
        ZephonDataLoader.Config(max_num_documents=4)

    with pytest.raises(TypeError, match="data_config"):
        zephon_dataloader(_grain_config())

    source = ZephonSource(name="source", path="/data/source")
    # Zephon owns validation of its scalar options. The adapter only checks
    # cross-system invariants after their Zephon prerequisites are valid.
    ZephonDataLoader._validate_config(
        ZephonDataLoader.Config(sources=[source], canonical_replicas=0),
        dp_world_size=1,
        world_size=1,
        num_tokens_per_batch=32,
        num_tokens_per_train_step=32,
    )

    with pytest.raises(ValueError, match="divisible by.*data-parallel"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(
                sources=[source],
                canonical_replicas=3,
                aggregate_dir="/tmp/aggregate",
                run_id="run",
            ),
            dp_world_size=2,
            world_size=2,
        )

    with pytest.raises(ValueError, match="complete lane-window boundaries"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(sources=[source], canonical_replicas=2),
            dp_world_size=1,
            world_size=1,
            num_tokens_per_batch=32,
            num_tokens_per_train_step=32,
        )

    with pytest.raises(ValueError, match="Distributed Zephon runs require"):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(sources=[source]),
            dp_world_size=1,
            world_size=2,
        )

    tuned = ZephonDataLoader.Config(
        sources=[source],
        cache_dir="/cache",
        cache_limit_bytes=1024,
        shuffle_block_size="global",
        prefetch_buffer_size=8,
        prefetch_parallelism=2,
        fetch_parallelism=3,
        tokenize_parallelism=4,
        pack_parallelism=5,
        shuffle_buffer_size=128,
        shuffle_parallelism=6,
        runner="threads",
        mtp_mode=False,
    )
    ZephonDataLoader._validate_config(tuned, dp_world_size=1, world_size=1)
    pytest.importorskip("zephon")
    pipeline = mock.Mock()
    pipeline.options.return_value = pipeline
    ZephonDataLoader._apply_runtime_options(
        pipeline, tuned, dp_world_size=1, dp_rank=0
    )
    base_options = pipeline.options.call_args_list[0].kwargs
    cache_options = pipeline.options.call_args_list[1].kwargs["io_options"]
    assert base_options["runner"] == "threads"
    assert not base_options["mtp_mode"]
    assert cache_options.cache.limit_bytes == 1024

    independent_recipe = tmp_path / "independent.toml"
    independent_recipe.write_text(
        """
shuffle_shards = false
shuffle_block_size = "global"
token_estimation = true
repeat = true

[[sources]]
name = "source"
path = "/data/source"
""".strip()
    )
    for recipe_override in (zephon_dataloader, zephon_validation_dataloader):
        recipe_config = recipe_override(
            _grain_config(), data_config=str(independent_recipe)
        )
        assert not recipe_config.shuffle_shards
        assert recipe_config.shuffle_within_shard
        assert recipe_config.shuffle_block_size == "global"
        assert recipe_config.token_estimation
        assert recipe_config.repeat

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
        ZephonDataLoader,
        zephon_dataloader,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = replace(
        zephon_dataloader(
            _grain_config(),
            data_config=str(
                repo_root / "examples" / "zephon" / "local_jsonl.toml"
            ),
        ),
        runner="inline",
        mtp_mode=False,
    )

    def build_loader() -> ZephonDataLoader:
        return ZephonDataLoader(
            config,
            dp_world_size=1,
            dp_rank=0,
            tokenizer=_tokenizer(repo_root),
            max_context_length=16,
            num_tokens_per_batch=32,
            num_tokens_per_train_step=32,
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
        ZephonDataLoader,
        zephon_validation_dataloader,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = replace(
        zephon_validation_dataloader(
            GrainDataLoader.Config(
                dataset=DATASETS["c4_validation"],
                repeat=False,
            ),
            data_config=str(
                repo_root / "examples" / "zephon" / "validation_local_jsonl.toml"
            ),
        ),
        runner="inline",
        mtp_mode=False,
    )
    loader = ZephonDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=_tokenizer(repo_root),
        max_context_length=16,
        num_tokens_per_batch=32,
    )

    assert not config.shuffle_shards
    assert not config.token_estimation
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
    root_config.dataloader = replace(
        root_config.dataloader, runner="inline", mtp_mode=False
    )
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


def test_zephon_checkpoint_rejects_missing_corrupt_and_incompatible_state() -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    loader = object.__new__(ZephonDataLoader)
    loader._pipeline = mock.Mock()

    with pytest.raises(ValueError, match="empty during resume"):
        loader.load_state_dict({})
    with pytest.raises(ValueError, match="missing 'zephon'"):
        loader.load_state_dict({"other": b"value"})
    with pytest.raises(ValueError, match="Expected.*bytes"):
        loader.load_state_dict({"zephon": "not-bytes"})
    with pytest.raises(ValueError, match="corrupt pickle"):
        loader.load_state_dict({"zephon": b"not-a-pickle"})

    loader._pipeline.restore.side_effect = RuntimeError("bad Zephon state")
    with pytest.raises(ValueError, match="malformed or incompatible"):
        loader.load_state_dict({"zephon": pickle.dumps({"wrong": "shape"})})


def test_runtime_coordination_uses_all_checkpointing_ranks() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader, ZephonSource

    zephon_module = importlib.import_module("torchtitan.overrides.zephon_dataloader")
    config = ZephonDataLoader.Config(
        sources=[ZephonSource(name="source", path="/data/source")],
        canonical_replicas=2,
    )
    pipeline = mock.Mock()
    pipeline.options.return_value = pipeline
    with (
        mock.patch.object(zephon_module.dist, "is_initialized", return_value=True),
        mock.patch.object(zephon_module.dist, "get_world_size", return_value=4),
        mock.patch.object(zephon_module.dist, "get_rank", return_value=3),
    ):
        ZephonDataLoader._apply_runtime_options(
            pipeline, config, dp_world_size=2, dp_rank=1
        )

    base_options = pipeline.options.call_args_list[0].kwargs
    distributed_options = pipeline.options.call_args_list[1].kwargs
    assert base_options["dp_degree"] == 2
    assert base_options["dp_group_id"] == 1
    assert distributed_options["world_size"] == 4
    assert distributed_options["global_rank"] == 3

    pipeline.reset_mock()
    pipeline.options.return_value = pipeline
    with mock.patch.object(zephon_module.dist, "is_initialized", return_value=False):
        ZephonDataLoader._apply_runtime_options(
            pipeline, config, dp_world_size=2, dp_rank=1
        )
    assert pipeline.options.call_count == 1
    pure_dp_options = pipeline.options.call_args.kwargs
    assert pure_dp_options["dp_degree"] == 2
    assert pure_dp_options["dp_group_id"] == 1
    assert "world_size" not in pure_dp_options
    assert "global_rank" not in pure_dp_options


@pytest.mark.parametrize("topology", ["2-to-1", "1-to-2"])
def test_cpu_elastic_resume_directions(tmp_path: Path, topology: str) -> None:
    pytest.importorskip("zephon")
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples/zephon/elastic_resume_demo.py"),
            "--topology",
            topology,
            "--work-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    initial_workers, resume_workers = topology.split("-to-")
    assert (
        f"Data-parallel workers: {initial_workers} -> {resume_workers}" in result.stdout
    )
    assert f"{initial_workers}-worker stream:" in result.stdout
    assert f"{resume_workers}-worker resume:" in result.stdout
    assert "Exact global-step match: YES" in result.stdout
    assert "Checkpoint SHA-256:" in result.stdout
