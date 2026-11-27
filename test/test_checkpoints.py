"""Unit tests for RT step-checkpoint CLI handles and file layout."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from rt import experiment


class CheckpointHelpersTest(unittest.TestCase):
    def test_checkpoint_filename_contains_step(self):
        self.assertEqual(experiment.checkpoint_filename(1), "step-000001.pt")
        self.assertEqual(experiment.checkpoint_filename(2000), "step-002000.pt")
        self.assertEqual(experiment.checkpoint_filename(1234567), "step-1234567.pt")

    def test_parse_save_steps(self):
        self.assertEqual(experiment.parse_save_steps(""), set())
        self.assertEqual(experiment.parse_save_steps(" , "), set())
        self.assertEqual(experiment.parse_save_steps("500, 2000, 1908"), {500, 1908, 2000})
        with self.assertRaises(ValueError):
            experiment.parse_save_steps("0")
        with self.assertRaises(ValueError):
            experiment.parse_save_steps("-10")
        with self.assertRaises(ValueError):
            experiment.parse_save_steps("abc")

    def test_should_save_checkpoint(self):
        self.assertTrue(experiment.should_save_checkpoint(2000, 2000, set()))
        self.assertTrue(experiment.should_save_checkpoint(4000, 2000, set()))
        self.assertFalse(experiment.should_save_checkpoint(1999, 2000, set()))
        self.assertFalse(experiment.should_save_checkpoint(2000, 0, set()))
        self.assertTrue(experiment.should_save_checkpoint(1500, 0, {1500}))
        self.assertTrue(experiment.should_save_checkpoint(1500, 2000, {1500}))

    def test_save_step_checkpoint_and_index_merge(self):
        model = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "checkpoints"
            path_4000 = experiment.save_step_checkpoint(
                ckpt_dir, 4000, model, loss=0.4, tokens=400, slots=512
            )
            path_2000 = experiment.save_step_checkpoint(
                ckpt_dir, 2000, model, loss=0.8, tokens=200, slots=256
            )
            self.assertEqual(path_2000.name, "step-002000.pt")
            self.assertEqual(path_4000.name, "step-004000.pt")

            loaded = torch.nn.Linear(4, 2)
            experiment.load_weights(
                loaded, torch.load(path_2000, weights_only=True, map_location="cpu")
            )
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[key]))

            index = json.loads((ckpt_dir / "index.json").read_text())
            self.assertEqual([item["step"] for item in index], [2000, 4000])
            self.assertEqual(index[0]["file"], "step-002000.pt")
            self.assertEqual(index[1]["file"], "step-004000.pt")

    def test_cli_defaults_and_validation(self):
        help_out = subprocess.check_output(
            [sys.executable, "-m", "rt.experiment", "--help"],
            text=True,
        )
        self.assertIn("--save-every", help_out)
        self.assertIn("--save-steps", help_out)
        self.assertIn("--save-dir", help_out)
        self.assertIn("--step", help_out)

        # Default SAVE_EVERY in Makefile and CLI must be 2000.
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
        self.assertIn("SAVE_EVERY ?= 2000", makefile)

        for bad_args in (
            ["train", "--save-every", "-1"],
            ["train", "--save-steps", "0"],
            ["evaluate", "--step", "0"],
            ["evaluate", "--step", "2000", "--checkpoint", "/tmp/foo.pt"],
        ):
            proc = subprocess.run(
                [sys.executable, "-m", "rt.experiment", *bad_args],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0, msg=f"Expected failure for {bad_args}")


if __name__ == "__main__":
    unittest.main()
