# Copyright (c) DatologyAI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan config recipes for the Zephon checkpoint/resume smoke test.

``aggregate_dir`` and ``run_id`` identify one run rather than the recipe, so
they come from the ``ZEPHON_AGGREGATE_DIR`` and ``ZEPHON_RUN_ID`` environment
variables; ``examples/zephon/run_training_smoke.sh`` sets both.

The save and resume recipes differ only in ``training.steps``, following the
paired ``*_save``/``*_load`` recipes in ``torchtitan_recipes.tests.suites``.
"""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel
from torchtitan_recipes.zephon.dataloader import ZephonDataLoader

_MAX_CONTEXT_LENGTH = 128
_DATA_CONFIG = "examples/zephon/elastic_local_jsonl.toml"


def llama3_debugmodel_zephon_smoke_save() -> Trainer.Config:
    """Train two steps and checkpoint every step, including the full last step."""
    config = llama3_debugmodel(seq_len=_MAX_CONTEXT_LENGTH)
    config.dataloader = ZephonDataLoader.Config.from_toml(
        _DATA_CONFIG,
        aggregate_dir=os.environ.get("ZEPHON_AGGREGATE_DIR"),
        run_id=os.environ.get("ZEPHON_RUN_ID"),
    )
    config.training.max_context_length = _MAX_CONTEXT_LENGTH
    # Two sequences per microbatch and four per optimizer step: gradient
    # accumulation is 1 on 2 DP ranks and 2 on 1 DP rank, so each step reads a
    # whole window of the recipe's two canonical Zephon lanes.
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * _MAX_CONTEXT_LENGTH
    config.training.num_tokens_per_train_step = 4 * _MAX_CONTEXT_LENGTH
    config.training.steps = 2
    config.checkpointer = CheckpointManager.Config(
        interval=1,
        # The resume phase needs optimizer and dataloader state from the last
        # step of the save phase, not a model-only export.
        last_save_model_only=False,
    )
    return config


def llama3_debugmodel_zephon_smoke_resume() -> Trainer.Config:
    """Resume the save recipe's checkpoint and train one more step."""
    config = llama3_debugmodel_zephon_smoke_save()
    config.training.steps = 3
    return config


def llama3_debugmodel_zephon_smoke_tp2_save() -> Trainer.Config:
    """The save recipe with fixed TP=2 and no data parallelism."""
    config = llama3_debugmodel_zephon_smoke_save()
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.data_parallel_shard_degree = 1
    return config


def llama3_debugmodel_zephon_smoke_tp2_resume() -> Trainer.Config:
    """The resume recipe with fixed TP=2 and no data parallelism."""
    config = llama3_debugmodel_zephon_smoke_tp2_save()
    config.training.steps = 3
    return config
