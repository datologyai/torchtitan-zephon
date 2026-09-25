# Copyright (c) DatologyAI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint as dcp
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

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
    assert validation.shuffle_block_size is None
    assert not validation.token_estimation
    assert not validation.shuffle_after_pack
    assert not validation.repeat
    assert [source.name for source in validation.sources] == ["validation"]


@pytest.mark.parametrize("name", ["max_num_documents", "pack_num_bins", "pad_token_id"])
@pytest.mark.parametrize("value", [True, False, 2.0, 2.5, "2"])
def test_zephon_numeric_config_rejects_non_integers(name: str, value: object) -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    with pytest.raises(ValueError, match=f"{name} must be an integer or None"):
        ZephonDataLoader.Config(**{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [("max_num_documents", "2.5"), ("pack_num_bins", "true"), ("pad_token_id", '"2"')],
)
def test_zephon_recipe_rejects_non_integer_options(
    tmp_path: Path, name: str, value: str
) -> None:
    from torchtitan.overrides.zephon_dataloader import zephon_dataloader

    recipe = tmp_path / "invalid.toml"
    recipe.write_text(f"{name} = {value}\nsources = []\n")
    with pytest.raises(ValueError, match=f"{name} must be an integer or None"):
        zephon_dataloader(_grain_config(), data_config=str(recipe))


def test_zephon_numeric_config_accepts_integers_and_none() -> None:
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    defaults = ZephonDataLoader.Config()
    assert defaults.max_num_documents is None
    assert defaults.pack_num_bins is None
    assert defaults.pad_token_id is None
    config = ZephonDataLoader.Config(
        max_num_documents=2, pack_algorithm="first_fit", pack_num_bins=1, pad_token_id=0
    )
    assert (config.max_num_documents, config.pack_num_bins, config.pad_token_id) == (
        2,
        1,
        0,
    )


def test_zephon_configuration_rejects_unsupported_contracts(tmp_path: Path) -> None:
    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        zephon_validation_dataloader,
        ZephonDataLoader,
        ZephonSource,
    )

    with pytest.raises(ValueError, match="max_num_documents must be positive"):
        ZephonDataLoader.Config(max_num_documents=0)
    with pytest.raises(ValueError, match="max_num_documents_scope"):
        ZephonDataLoader.Config(max_num_documents_scope="global")

    with pytest.raises(TypeError, match="data_config"):
        zephon_dataloader(_grain_config())

    source = ZephonSource(name="source", path="/data/source")
    # Zephon owns range and algorithm compatibility checks. The adapter checks
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
    ZephonDataLoader._apply_runtime_options(pipeline, tuned, dp_world_size=1, dp_rank=0)
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
max_num_documents = 5
max_num_documents_scope = "bin"
pad_token_id = 7
tokenize_special_tokens = "eos"
pack_algorithm = "best_fit"
pack_num_bins = 4

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
        assert recipe_config.max_num_documents == 5
        assert recipe_config.max_num_documents_scope == "bin"
        assert recipe_config.pad_token_id == 7
        assert recipe_config.tokenize_special_tokens == "eos"
        assert recipe_config.pack_algorithm == "best_fit"
        assert recipe_config.pack_num_bins == 4

    recipe = tmp_path / "unknown.toml"
    recipe.write_text(
        """
unsupported_option = true
sources = []
""".strip()
    )
    with pytest.raises(ValueError, match="unsupported_option"):
        zephon_dataloader(_grain_config(), data_config=str(recipe))


# Deliberately misspelled "weight" to exercise unknown-field validation.
_MISSPELLED_WEIGHT = "wieght"  # codespell:ignore


@pytest.mark.parametrize(
    ("sources", "error"),
    [
        (
            [{"name": "source", "path": "/data/source", _MISSPELLED_WEIGHT: 9.0}],
            f"Unknown Zephon source fields: {_MISSPELLED_WEIGHT}",
        ),
        (
            [
                {"name": "source", "path": "/data/one"},
                {"name": "source", "path": "/data/two"},
            ],
            "Zephon source names must be unique",
        ),
        (
            [{"name": "source", "path": "/data/source", "weight": float("nan")}],
            "must have a finite weight",
        ),
        (
            [{"name": "source", "path": "/data/source", "weight": float("inf")}],
            "must have a finite weight",
        ),
    ],
)
def test_zephon_source_validation(sources: list[dict[str, object]], error: str) -> None:
    from torchtitan.overrides.zephon_dataloader import _parse_sources

    with pytest.raises(ValueError, match=error):
        _parse_sources(sources)


@pytest.mark.parametrize(
    ("max_num_documents", "scope"), [(None, "batch"), (3, "batch"), (1, "bin")]
)
@pytest.mark.parametrize(
    "algorithm", ["wrap", "first_fit", "best_fit", "best_fit_wrap"]
)
def test_zephon_training_batches_and_checkpoint_continuation(
    max_num_documents: int | None, scope: str, algorithm: str
) -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import (
        zephon_dataloader,
        ZephonDataLoader,
    )

    repo_root = Path(__file__).resolve().parents[2]
    config = replace(
        zephon_dataloader(
            _grain_config(),
            data_config=str(repo_root / "examples" / "zephon" / "local_jsonl.toml"),
        ),
        runner="inline",
        mtp_mode=False,
        max_num_documents=max_num_documents,
        max_num_documents_scope=scope,
        pack_algorithm=algorithm,
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
    assert first["padding_mask"].shape == (32,)
    assert first["padding_mask"].dtype == torch.bool
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


@pytest.mark.parametrize(
    ("algorithm", "scope", "limit", "per_bin", "per_batch"),
    [
        ("wrap", "batch", 5, 2, 5),
        ("wrap", "batch", 2, 1, 2),
        ("wrap", "bin", 2, 2, 4),
        ("wrap", "bin", 1, 1, 2),
        ("first_fit", "batch", 5, 2, 5),
        ("best_fit", "batch", 5, 2, 5),
        ("best_fit_wrap", "batch", 5, 2, 5),
    ],
)
@pytest.mark.parametrize("pad_with_eos", [False, True])
@pytest.mark.parametrize("special_tokens", ["bos_eos", "eos"])
def test_zephon_document_limit_and_attention_metadata(
    tmp_path: Path,
    algorithm: str,
    scope: str,
    limit: int,
    per_bin: int,
    per_batch: int,
    pad_with_eos: bool,
    special_tokens: str,
) -> None:
    pytest.importorskip("zephon")

    from torchtitan.models.common.attention import create_varlen_metadata_for_document
    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader, ZephonSource

    repo_root = Path(__file__).resolve().parents[2]
    tokenizer = _tokenizer(repo_root)
    if special_tokens == "eos":
        tokenizer.bos_id = None
    if pad_with_eos:
        tokenizer._hf_config["pad_token"] = tokenizer.eos_token
    pad_id = tokenizer.tokenizer.token_to_id(tokenizer._hf_config["pad_token"])
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": text}) + "\n"
            for text in ["hello"] * 12 + ["hello " * 30]
        )
    )
    loader = ZephonDataLoader(
        ZephonDataLoader.Config(
            sources=[ZephonSource(name="documents", path=str(tmp_path), fmt="jsonl")],
            max_num_documents=limit,
            max_num_documents_scope=scope,
            pack_algorithm=algorithm,
            tokenize_special_tokens=special_tokens,
            chunk_size=1,
            shuffle_shards=False,
            shuffle_within_shard=False,
            shuffle_block_size=None,
            shuffle_after_pack=False,
            token_estimation=False,
            repeat=False,
            runner="inline",
            mtp_mode=False,
        ),
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        max_context_length=16,
        num_tokens_per_batch=32,
    )
    assert loader.max_num_documents == per_batch
    batches = list(loader)
    assert len(batches) > 1
    first = next(batch for batch in batches if batch["padding_mask"].any())
    assert first["padding_mask"].any()
    assert (
        first["input"] == tokenizer.tokenizer.token_to_id(tokenizer.bos_token)
    ).any() == (special_tokens == "bos_eos")
    assert ((first["input"] == tokenizer.eos_id) & ~first["padding_mask"]).any()
    assert (first["labels"] == tokenizer.eos_id).any()
    # The final real input predicts padding: its label is ignored but it is
    # still a real token for attention.
    assert ((first["labels"] == IGNORE_INDEX) & ~first["padding_mask"]).any()
    for batch in batches:
        positions = batch["positions"]
        padding = batch["padding_mask"]
        assert positions.shape == padding.shape == (32,)
        real_starts = ((positions == 0) & ~padding).reshape(2, 16).sum(dim=1)
        assert (real_starts <= per_bin).all()
        assert (batch["input"][padding] == pad_id).all()
        assert (batch["labels"][padding] == IGNORE_INDEX).all()
        assert (
            batch["num_valid_tokens"] == (batch["labels"] != IGNORE_INDEX).sum().item()
        )
        metadata = create_varlen_metadata_for_document(
            positions,
            padding_mask=padding,
            max_num_documents=loader.max_num_documents,
            max_context_length=16,
        )
        starts = (positions == 0).nonzero().flatten().tolist()
        expected = starts + [32] * (per_batch + 3 - len(starts))
        assert metadata.cu_seq_q.tolist() == expected
        assert metadata.max_q == metadata.max_k == 16
        assert torch.diff(metadata.cu_seq_q).max() <= 16
    assert (
        max(
            ((batch["positions"] == 0) & ~batch["padding_mask"]).sum().item()
            for batch in batches
        )
        == 2 * per_bin
    )


def test_zephon_document_limit_rejects_less_than_one_document_per_bin() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader

    repo_root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="at least the local batch size.*2"):
        ZephonDataLoader(
            ZephonDataLoader.Config(max_num_documents=1),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=_tokenizer(repo_root),
            max_context_length=16,
            num_tokens_per_batch=32,
        )


@pytest.fixture
def short_document_loader(tmp_path: Path):
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import ZephonDataLoader, ZephonSource

    (tmp_path / "documents.jsonl").write_text(json.dumps({"text": "hello"}) + "\n")
    tokenizer = _tokenizer(Path(__file__).resolve().parents[2])

    def build(**packing_options):
        return ZephonDataLoader(
            ZephonDataLoader.Config(
                sources=[
                    ZephonSource(name="documents", path=str(tmp_path), fmt="jsonl")
                ],
                chunk_size=1,
                shuffle_shards=False,
                shuffle_within_shard=False,
                shuffle_block_size=None,
                shuffle_after_pack=False,
                token_estimation=False,
                repeat=False,
                runner="inline",
                mtp_mode=False,
                **packing_options,
            ),
            dp_world_size=1,
            dp_rank=0,
            tokenizer=tokenizer,
            max_context_length=16,
            num_tokens_per_batch=16,
        )

    return build, tokenizer


@pytest.mark.parametrize(
    "algorithm", ["wrap", "first_fit", "best_fit", "best_fit_wrap"]
)
def test_zephon_uncapped_packing_tail(short_document_loader, algorithm: str) -> None:
    build, tokenizer = short_document_loader
    batches = list(build(pack_algorithm=algorithm))
    if algorithm == "wrap":
        assert batches == []
        return

    assert len(batches) == 1
    batch = batches[0]
    tokens = tokenizer.encode("hello", add_bos=True, add_eos=True)
    num_tokens = len(tokens)
    pad_id = tokenizer.tokenizer.token_to_id(tokenizer._hf_config["pad_token"])
    assert batch["input"].tolist() == tokens + [pad_id] * (16 - num_tokens)
    assert batch["labels"].tolist() == tokens[1:] + [IGNORE_INDEX] * (17 - num_tokens)
    assert batch["positions"].tolist() == list(range(num_tokens)) + list(
        range(16 - num_tokens)
    )
    assert batch["padding_mask"].tolist() == [False] * num_tokens + [True] * (
        16 - num_tokens
    )
    assert batch["num_valid_tokens"] == num_tokens - 1


def test_zephon_uncapped_wrap_does_not_resolve_padding(short_document_loader) -> None:
    build, _ = short_document_loader
    with mock.patch(
        "torchtitan.overrides.zephon_dataloader._resolve_pad_token_id",
        side_effect=AssertionError("uncapped wrap does not need padding"),
    ):
        assert list(build()) == []


@pytest.mark.parametrize(
    ("algorithm", "num_bins", "error"),
    [
        ("first_fit", 0, "num_bins must be positive"),
        ("best_fit", -1, "num_bins must be positive"),
        ("wrap", 2, "num_bins does not apply"),
        ("best_fit_wrap", 2, "num_bins does not apply"),
        ("invalid", None, "Unknown algorithm"),
    ],
)
def test_zephon_packing_options_are_validated(
    short_document_loader, algorithm: str, num_bins: int | None, error: str
) -> None:
    build, _ = short_document_loader
    with pytest.raises(ValueError, match=error):
        build(pack_algorithm=algorithm, pack_num_bins=num_bins)


def _padding_tokenizer(vocab: list[str], config: dict) -> HuggingFaceTokenizer:
    tokenizer = object.__new__(HuggingFaceTokenizer)
    tokenizer.tokenizer = Tokenizer(
        WordLevel({token: i for i, token in enumerate(vocab)})
    )
    tokenizer._hf_config = config
    tokenizer.eos_id = tokenizer.tokenizer.token_to_id("<eos>")
    return tokenizer


@pytest.mark.parametrize(
    ("vocab", "config", "explicit", "expected"),
    [
        (["word", "<eos>"], {}, 0, 0),
        (["<pad>", "<custom>"], {"pad_token": "<custom>"}, None, 1),
        (["<pad>", "<custom>"], {"pad_token": {"content": "<custom>"}}, 1, 1),
        (["<eos>", "<pad>"], {"pad_token": "<eos>"}, None, 0),
        (["<eos>", "<|finetune_right_pad_id|>", "<pad>"], {}, None, 1),
        (["<eos>", "[PAD]"], {}, None, 1),
        (
            ["<eos>", "<reserved_1>", "<reserved_2>", "<unk>"],
            {
                "unk_token": "<unk>",
                "added_tokens_decoder": {
                    "2": {"content": "<reserved_2>", "special": True},
                    "1": {"content": "<reserved_1>", "special": True},
                },
            },
            None,
            1,
        ),
        (["<eos>", "<unk>"], {"unk_token": {"content": "<unk>"}}, None, 1),
        (["word", "<eos>"], {}, None, 1),
    ],
)
def test_zephon_pad_token_resolution(
    vocab: list[str], config: dict, explicit: int | None, expected: int
) -> None:
    from torchtitan.overrides.zephon_dataloader import _resolve_pad_token_id

    tokenizer = _padding_tokenizer(vocab, config)
    assert _resolve_pad_token_id(tokenizer, explicit) == expected
    assert tokenizer.get_vocab_size() == len(vocab)


@pytest.mark.parametrize("explicit", [-1, 2])
def test_zephon_pad_token_rejects_out_of_vocab(explicit: int) -> None:
    from torchtitan.overrides.zephon_dataloader import _resolve_pad_token_id

    with pytest.raises(ValueError, match="out of vocab range"):
        _resolve_pad_token_id(_padding_tokenizer(["word", "<eos>"], {}), explicit)


def test_zephon_pad_token_rejects_conflicting_or_missing_ids() -> None:
    from torchtitan.overrides.zephon_dataloader import _resolve_pad_token_id

    tokenizer = _padding_tokenizer(["word", "<pad>"], {"pad_token": "<pad>"})
    with pytest.raises(ValueError, match="contradicts"):
        _resolve_pad_token_id(tokenizer, 0)
    tokenizer.pad_id = 3
    with pytest.raises(ValueError, match="out of vocab range"):
        _resolve_pad_token_id(tokenizer, None)
    tokenizer = _padding_tokenizer(["word"], {})
    with pytest.raises(ValueError, match="Could not resolve a pad token"):
        _resolve_pad_token_id(tokenizer, None)


def test_zephon_validation_is_finite_without_token_estimation() -> None:
    pytest.importorskip("zephon")

    from torchtitan.overrides.zephon_dataloader import (
        zephon_validation_dataloader,
        ZephonDataLoader,
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
