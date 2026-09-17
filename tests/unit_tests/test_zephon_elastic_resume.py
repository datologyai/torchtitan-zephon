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


def test_elastic_checkpoint_resume_preserves_global_step_contents(
    tmp_path: Path,
) -> None:
    pytest.importorskip("zephon")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "zephon" / "elastic_resume_demo.py"),
            "--total-steps",
            "4",
            "--checkpoint-after",
            "2",
            "--work-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Data-parallel workers: 2 -> 1" in result.stdout
    assert "Exact global-step match: YES" in result.stdout

    [run_dir] = tmp_path.glob("run-*")
    with (run_dir / "reference" / "reference.json").open() as input_file:
        reference = json.load(input_file)
    with (run_dir / "elastic" / "save.json").open() as input_file:
        before_checkpoint = json.load(input_file)
    with (run_dir / "elastic" / "resume.json").open() as input_file:
        after_resume = json.load(input_file)

    def canonicalize(steps):
        return [
            sorted(step, key=lambda batch: json.dumps(batch, sort_keys=True))
            for step in steps
        ]

    assert canonicalize(before_checkpoint + after_resume) == canonicalize(reference)
