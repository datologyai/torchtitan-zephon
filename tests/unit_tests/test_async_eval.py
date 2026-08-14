# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from torchtitan.components.async_eval import AsyncEval
from torchtitan.components.metrics import BaseLogger, MetricsProcessor

# Stands in for a real eval runner: reports a loss and echoes the arguments it
# was launched with, so tests can check the runner contract.
FAKE_RUNNER = """
import json, sys

args = sys.argv[1:]
metrics = {"loss": 1.5, "task": "made-up"}
with open(args[args.index("--result-path") + 1], "w") as f:
    json.dump({"step": int(args[args.index("--step") + 1]),
               "metrics": metrics, "args": args}, f)
"""

FAILING_RUNNER = "import sys; sys.exit(1)"

SILENT_RUNNER = "pass"


class FakeCheckpointer:
    """Minimal CheckpointManager stand-in that only records eval saves."""

    def __init__(self, folder: str, enable: bool = True, temporary: bool = True):
        self.enable = enable
        self.folder = folder
        self.temporary = temporary
        self.saved_steps: list[int] = []

    def save_for_async_eval(self, step: int) -> tuple[str, bool]:
        self.saved_steps.append(step)
        checkpoint_dir = os.path.join(self.folder, f"step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir, self.temporary


class RecordingLogger(BaseLogger):
    def __init__(self):
        self.records: list[tuple[int, dict]] = []
        self.closed = False

    def log(self, metrics: dict, step: int) -> None:
        self.records.append((step, metrics))

    def close(self) -> None:
        self.closed = True


class AsyncEvalTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.dump_folder = self.tmp_dir.name
        self.runner_path = os.path.join(self.dump_folder, "runner.py")

    def build_async_eval(self, runner_source: str = FAKE_RUNNER, **overrides):
        """An AsyncEval that runs a local script instead of a training job."""
        with open(self.runner_path, "w") as f:
            f.write(runner_source)

        config = AsyncEval.Config(
            enable=True,
            freq=overrides.pop("freq", 10),
            launcher="",
            runner=f"{sys.executable} {self.runner_path}",
            forward_train_args=overrides.pop("forward_train_args", False),
            exit_timeout=overrides.pop("exit_timeout", 30.0),
            **overrides,
        )
        async_eval = config.build(dump_folder=self.dump_folder)
        async_eval.logger = RecordingLogger()
        return async_eval

    def run_to_completion(self, async_eval: AsyncEval) -> RecordingLogger:
        async_eval.close()
        self.assertEqual(async_eval.jobs, [], "eval job did not finish in time")
        return async_eval.logger

    def read_result(self, step: int) -> dict:
        result_path = os.path.join(
            self.dump_folder, "async_eval", f"step-{step}", "result.json"
        )
        with open(result_path) as f:
            return json.load(f)


class TestAsyncEvalConfig(AsyncEvalTestCase):
    def test_disabled_by_default(self):
        config = AsyncEval.Config()
        self.assertFalse(config.enable)
        self.assertFalse(config.build().should_eval(10))

    def test_should_eval_on_multiples_of_freq(self):
        async_eval = AsyncEval.Config(enable=True, freq=10).build()
        self.assertTrue(async_eval.should_eval(10))
        self.assertTrue(async_eval.should_eval(20))
        self.assertFalse(async_eval.should_eval(15))

    def test_invalid_freq(self):
        with self.assertRaisesRegex(ValueError, "at least 1 step"):
            AsyncEval.Config(freq=0)

    def test_empty_runner_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "runner cannot be empty"):
            AsyncEval.Config(enable=True, runner=" ")

    def test_tensorboard_logger_follows_metrics_config(self):
        config = AsyncEval.Config(enable=True)
        metrics_config = MetricsProcessor.Config(enable_tensorboard=True)
        with mock.patch(
            "torchtitan.components.async_eval.TensorBoardLogger"
        ) as tb_logger:
            config.build(dump_folder=self.dump_folder, metrics_config=metrics_config)
        log_dir = tb_logger.call_args.args[0]
        self.assertTrue(
            log_dir.startswith(os.path.join(self.dump_folder, "tb", "async_eval")),
            log_dir,
        )

    def test_no_tensorboard_logger_without_metrics_config(self):
        async_eval = AsyncEval.Config(enable=True).build(dump_folder=self.dump_folder)
        self.assertIsInstance(async_eval.logger, BaseLogger)


class TestAsyncEvalLaunch(AsyncEvalTestCase):
    def test_metrics_are_logged_at_the_launch_step(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder)

        async_eval.launch(step=10, checkpointer=checkpointer)
        self.assertEqual(checkpointer.saved_steps, [10])

        logger = self.run_to_completion(async_eval)
        # The non-numeric metric the runner reported is dropped, not fatal.
        self.assertEqual(logger.records, [(10, {"loss": 1.5})])
        self.assertTrue(logger.closed)

    def test_runner_receives_the_contract_arguments(self):
        async_eval = self.build_async_eval(extra_args="--tasks mmlu")
        checkpointer = FakeCheckpointer(self.dump_folder)

        async_eval.launch(step=20, checkpointer=checkpointer)
        self.run_to_completion(async_eval)

        args = self.read_result(20)["args"]
        step_dir = os.path.join(self.dump_folder, "async_eval", "step-20")
        self.assertEqual(
            args,
            [
                "--tasks",
                "mmlu",
                "--checkpoint-dir",
                os.path.join(self.dump_folder, "step-20"),
                "--output-dir",
                step_dir,
                "--step",
                "20",
                "--result-path",
                os.path.join(step_dir, "result.json"),
            ],
        )

    def test_train_args_are_forwarded(self):
        with mock.patch.object(
            sys, "argv", ["train.py", "--module", "llama3", "--config", "debugmodel"]
        ):
            async_eval = self.build_async_eval(forward_train_args=True)

        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))
        self.run_to_completion(async_eval)

        args = self.read_result(10)["args"]
        self.assertEqual(args[:4], ["--module", "llama3", "--config", "debugmodel"])

    def test_extra_args_precede_train_args(self):
        # tyro rejects options that follow a subcommand, and a training command
        # line can end with one.
        with mock.patch.object(sys, "argv", ["train.py", "activation-checkpoint:none"]):
            async_eval = self.build_async_eval(
                forward_train_args=True, extra_args="--validator.steps 2"
            )

        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))
        self.run_to_completion(async_eval)

        args = self.read_result(10)["args"]
        self.assertEqual(
            args[:3], ["--validator.steps", "2", "activation-checkpoint:none"]
        )

    def test_temporary_checkpoint_is_deleted_once_eval_is_done(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder, temporary=True)

        async_eval.launch(step=10, checkpointer=checkpointer)
        checkpoint_dir = os.path.join(self.dump_folder, "step-10")
        self.assertTrue(os.path.isdir(checkpoint_dir))

        self.run_to_completion(async_eval)
        self.assertFalse(os.path.isdir(checkpoint_dir))

    def test_training_checkpoint_is_kept(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder, temporary=False)

        async_eval.launch(step=10, checkpointer=checkpointer)
        self.run_to_completion(async_eval)
        self.assertTrue(os.path.isdir(os.path.join(self.dump_folder, "step-10")))

    def test_keep_checkpoint(self):
        async_eval = self.build_async_eval(keep_checkpoint=True)
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))

        self.run_to_completion(async_eval)
        self.assertTrue(os.path.isdir(os.path.join(self.dump_folder, "step-10")))

    def test_launch_requires_checkpointing(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder, enable=False)

        with self.assertRaisesRegex(ValueError, "checkpoint.enable=True"):
            async_eval.launch(step=10, checkpointer=checkpointer)

    def test_disabled_async_eval_does_not_launch(self):
        async_eval = self.build_async_eval()
        async_eval.config.enable = False

        checkpointer = FakeCheckpointer(self.dump_folder)
        async_eval.launch(step=10, checkpointer=checkpointer)
        self.assertEqual(checkpointer.saved_steps, [])
        self.assertEqual(async_eval.jobs, [])


class TestAsyncEvalFailures(AsyncEvalTestCase):
    def test_failing_runner_does_not_stop_training(self):
        async_eval = self.build_async_eval(FAILING_RUNNER)
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))

        logger = self.run_to_completion(async_eval)
        self.assertEqual(logger.records, [])

    def test_failing_runner_can_be_made_fatal(self):
        async_eval = self.build_async_eval(FAILING_RUNNER, raise_on_failure=True)
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))

        with self.assertRaisesRegex(RuntimeError, "failed with exit code 1"):
            async_eval.close()
        # The failed job is not reported twice.
        async_eval.close()

    def test_runner_reporting_no_metrics_can_be_made_fatal(self):
        async_eval = self.build_async_eval(SILENT_RUNNER, raise_on_failure=True)
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))

        with self.assertRaisesRegex(RuntimeError, "reported no metrics"):
            async_eval.close()

    def test_missing_result_file(self):
        async_eval = self.build_async_eval(SILENT_RUNNER)
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))

        logger = self.run_to_completion(async_eval)
        self.assertEqual(logger.records, [])

    def test_malformed_result_file(self):
        async_eval = self.build_async_eval()
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))
        self.run_to_completion(async_eval)

        # Same path, but this time the runner leaves behind an unreadable result.
        job = mock.Mock(
            step=10,
            result_path=os.path.join(
                self.dump_folder, "async_eval", "step-10", "result.json"
            ),
        )
        with open(job.result_path, "w") as f:
            f.write("not json")
        self.assertEqual(async_eval._read_metrics(job), {})

        with open(job.result_path, "w") as f:
            json.dump({"loss": 1.0}, f)
        self.assertEqual(async_eval._read_metrics(job), {})

    def test_unfinished_jobs_are_not_waited_for_forever(self):
        async_eval = self.build_async_eval(
            "import time; time.sleep(60)", exit_timeout=0.0
        )
        async_eval.launch(step=10, checkpointer=FakeCheckpointer(self.dump_folder))

        async_eval.close()
        self.assertEqual(len(async_eval.jobs), 1)
        async_eval.jobs[0].process.kill()
        async_eval.jobs[0].process.wait()


if __name__ == "__main__":
    unittest.main()
