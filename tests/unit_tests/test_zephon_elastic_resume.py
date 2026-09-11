# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_elastic_checkpoint_resume_preserves_global_batch_order(tmp_path: Path) -> None:
    pytest.importorskip("zephon")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "zephon" / "elastic_resume_demo.py"),
            "--total-steps",
            "2",
            "--checkpoint-after",
            "1",
            "--work-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Data-parallel workers: 2 -> 1" in result.stdout
    assert "Exact stream match:    YES" in result.stdout

    [run_dir] = tmp_path.glob("run-*")
    with (run_dir / "reference" / "reference.json").open() as input_file:
        reference = json.load(input_file)
    with (run_dir / "elastic" / "save.json").open() as input_file:
        before_checkpoint = json.load(input_file)
    with (run_dir / "elastic" / "resume.json").open() as input_file:
        after_resume = json.load(input_file)
    assert before_checkpoint + after_resume == reference


def test_elastic_demo_rejects_indivisible_canonical_replicas(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    recipe = tmp_path / "indivisible.toml"
    recipe.write_text(
        """
canonical_replicas = 3

[[sources]]
name = "records"
path = "unused"
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "zephon" / "elastic_resume_demo.py"),
            "--recipe",
            str(recipe),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must be divisible by the initial num_workers=2" in result.stderr
