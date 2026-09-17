# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import importlib
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    ("config_values", "message"),
    [
        ({"text_field": ""}, "text_field"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"fetch_parallelism": 0}, "fetch_parallelism"),
        ({"canonical_replicas": 0}, "canonical_replicas"),
        ({"aggregate_dir": ""}, "aggregate_dir"),
        ({"run_id": ""}, "run_id"),
    ],
)
def test_zephon_dataloader_rejects_invalid_pipeline_settings(
    config_values: dict[str, object], message: str
) -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    with pytest.raises(ValueError, match=message):
        ZephonDataLoader._validate_config(
            ZephonDataLoader.Config(**config_values), dp_world_size=1
        )


def test_zephon_override_parses_weighted_local_and_hf_sources() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
        ZephonSource,
    )
    from zephon import MixtureSpec

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
    assert MixtureSpec(
        {source.name: source.weight for source in config.sources}
    ).normalized == {"web": 0.75, "code": 0.25}

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

    assert [(source.name, source.fmt, source.weight) for source in config.sources] == [
        ("prose", "jsonl", 3.0),
        ("code", "jsonl", 1.0),
    ]
    assert config.sources[0].path == str(
        repo_root / "tests" / "assets" / "zephon_mixture" / "prose"
    )
    assert config.chunk_size == 4


def test_zephon_shared_artifact_hashes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected_hashes = {
        Path("tests/assets/zephon_mixture/prose/data.jsonl"): (
            "849b787063a051d8e4247b9006d7f16a4f4824692b2d14fae846b8ac7a0fbc1e"
        ),
        Path("tests/assets/zephon_mixture/code/data.jsonl"): (
            "c81b60a1d641b40db0226112ea89e1bdaff42c6efb393e632bfe44c6f5d1b42a"
        ),
        Path("examples/zephon/local_jsonl.toml"): (
            "18f7f2aeb2960a3f2a8527dcafb795a6eabdf8cd488510fd85ad558767beac98"
        ),
        Path("examples/zephon/elastic_local_jsonl.toml"): (
            "17e7657a445598b5434a459848c4d8de0370814397cbc575e43717dd90d76ed9"
        ),
    }

    for path, expected_hash in expected_hashes.items():
        assert (
            hashlib.sha256((repo_root / path).read_bytes()).hexdigest() == expected_hash
        )


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


def test_zephon_recipe_resolves_paths_and_preserves_pipeline_settings(
    tmp_path: Path,
) -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    recipe = tmp_path / "recipe.toml"
    recipe.write_text(
        """
text_field = "content"
cache_dir = "cache"
seed = 7
chunk_size = 12
canonical_replicas = 4
aggregate_dir = "aggregate"
run_id = "stable-run"
fetch_parallelism = 3

[[sources]]
name = "local"
path = "records"
fmt = "jsonl"
weight = 2.0

[[sources]]
name = "remote"
path = "s3://bucket/records"
""".strip()
    )

    config = zephon_dataloader(_grain_config(), data_config=str(recipe))

    assert config.sources[0].path == str(tmp_path / "records")
    assert config.sources[0].fmt == "jsonl"
    assert config.sources[1].path == "s3://bucket/records"
    assert config.sources[1].fmt is None
    assert config.cache_dir == str(tmp_path / "cache")
    assert config.aggregate_dir == str(tmp_path / "aggregate")
    assert config.text_field == "content"
    assert config.seed == 7
    assert config.chunk_size == 12
    assert config.canonical_replicas == 4
    assert config.run_id == "stable-run"
    assert config.fetch_parallelism == 3


def test_zephon_recipe_rejects_unknown_settings(tmp_path: Path) -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    recipe = tmp_path / "unknown.toml"
    recipe.write_text(
        """
unsupported_option = true
sources = []
""".strip()
    )

    with pytest.raises(ValueError, match="unsupported_option"):
        zephon_dataloader(_grain_config(), data_config=str(recipe))


def test_zephon_training_and_validation_pipeline_contract(monkeypatch) -> None:
    module = importlib.import_module("torchtitan.overrides.zephon_dataloader")
    work_source_options = []

    class FakeDataset:
        @staticmethod
        def from_path(**options):
            return options

    class FakeMixtureSpec:
        __slots__ = ("weights",)

        def __init__(self, weights):
            self.weights = weights

    class FakeTokenEstimation:
        pass

    class FakeWorkSource:
        def __init__(self, **options):
            work_source_options.append(options)

    class FakePipeline:
        def __init__(self, _work_source):
            self.calls = []

        def fetch_parallelism(self, value):
            self.calls.append(("fetch_parallelism", (value,), {}))
            return self

        def tokenize(self, **options):
            self.calls.append(("tokenize", (), options))
            return self

        def pack_flat(self, **options):
            self.calls.append(("pack_flat", (), options))
            return self

        def preflight_tokenizers(self):
            self.calls.append(("preflight_tokenizers", (), {}))

        def batch(self, value, **options):
            self.calls.append(("batch", (value,), options))
            return self

        def options(self, **options):
            self.calls.append(("options", (), options))
            return self

    monkeypatch.setattr(
        module,
        "_require_zephon",
        lambda: (
            FakePipeline,
            FakeDataset,
            FakeMixtureSpec,
            FakeWorkSource,
            FakeTokenEstimation,
        ),
    )
    sources = [
        module.ZephonSource(name="prose", path="/prose", weight=3.0),
        module.ZephonSource(name="code", path="/code", weight=1.0),
    ]

    training_loader = module.ZephonDataLoader(
        module.ZephonDataLoader.Config(
            sources=sources,
            chunk_size=4,
            fetch_parallelism=3,
        ),
        dp_world_size=1,
        dp_rank=0,
        tokenizer=SimpleNamespace(tokenizer_path="tokenizer"),
        max_context_length=16,
        num_tokens_per_batch=32,
    )
    training_options = work_source_options[0]
    assert training_options["mixture"].weights == {"prose": 3.0, "code": 1.0}
    assert training_options["chunk_size"] == 4
    assert isinstance(training_options["token_estimation"], FakeTokenEstimation)
    assert training_loader._pipeline.calls[:-1] == [
        ("fetch_parallelism", (3,), {}),
        (
            "tokenize",
            (),
            {
                "tokenizer_id": "tokenizer",
                "field": "text",
                "add_attention_mask": False,
                "max_length": 17,
                "split_long_samples": True,
                "special_tokens": "bos_eos",
            },
        ),
        (
            "pack_flat",
            (),
            {"max_length": 17, "algorithm": "wrap", "emit_positions": True},
        ),
        ("preflight_tokenizers", (), {}),
        ("batch", (2,), {"drop_last": True}),
    ]

    module.ZephonDataLoader(
        module.ZephonDataLoader.Config(
            sources=sources,
            chunk_size=4,
            shuffle=False,
            repeat=False,
        ),
        dp_world_size=1,
        dp_rank=0,
        tokenizer=SimpleNamespace(tokenizer_path="tokenizer"),
        max_context_length=16,
        num_tokens_per_batch=32,
    )
    validation_options = work_source_options[1]
    assert "token_estimation" not in validation_options
    assert "exhausted_policy" not in validation_options


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
    assert loader._pipeline.ws.requires_token_priming
    loader_iter = iter(loader)
    batch = next(loader_iter)
    assert not loader._pipeline.ws.requires_token_priming

    assert batch["input"].shape == (32,)
    assert batch["labels"].shape == (32,)
    assert batch["positions"].shape == (32,)
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
    assert not loader._pipeline.ws.requires_token_priming
    batches = list(loader)

    assert batches


def test_zephon_override_state_round_trips_through_dcp(tmp_path: Path) -> None:
    pytest.importorskip("zephon")

    repo_root = Path(__file__).resolve().parents[2]
    data_config = repo_root / "examples" / "zephon" / "weighted_local_jsonl.toml"
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

    tokenizer = HuggingFaceTokenizer(
        HuggingFaceTokenizer.Config(),
        tokenizer_path=str(repo_root / "tests" / "assets" / "tokenizer"),
    )

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
    loader_iterator = iter(loader)
    next(loader_iterator)
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
