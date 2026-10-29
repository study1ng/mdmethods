"""CPU/filesystem regression checks; run manually in the project container.

No CUDA, real checkpoints, medical images or inference jobs are needed.
The existing project's compiled dependencies must still be importable.
"""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from monai.data import MetaTensor

from experiments.munet import MUNetCustom
from experiments.munet.observe import (
    Observation,
    _cached_digest,
    _checkpoints,
    _compatible_settings,
    _parser,
    _sample_indices,
    _select_checkpoint,
    _sha256,
    _signature,
    _stable_candidates,
    _training_roots,
    _worker_environment,
)


class SamplingTests(unittest.TestCase):
    def test_large_tensor_boundaries_without_allocating_the_tensor(self):
        for elements in (2**24 + 1, 189464576, 2**31 + 1, 2**63 - 1):
            with self.subTest(elements=elements):
                indices = _sample_indices(elements, 4096)
                self.assertEqual(indices.dtype, torch.long)
                self.assertEqual(indices.numel(), 4096)
                self.assertEqual(indices[0].item(), 0)
                self.assertEqual(indices[-1].item(), elements - 1)
                self.assertTrue(torch.all(indices[1:] > indices[:-1]).item())
                self.assertTrue(torch.all((indices >= 0) & (indices < elements)).item())

    def test_small_empty_single_and_oversized_limits(self):
        for elements, limit, expected in (
            (0, 4, []), (1, 4, [0]), (8, 1, [0]),
            (6, 4, [0, 1, 3, 5]), (3, 8, [0, 1, 2]),
        ):
            with self.subTest(elements=elements, limit=limit):
                self.assertEqual(_sample_indices(elements, limit).tolist(), expected)
        for elements, limit in ((-1, 4), (8, 0)):
            with self.assertRaises(ValueError):
                _sample_indices(elements, limit)

    def test_meta_tensor_sampling_has_plain_tensor_storage(self):
        source = MetaTensor(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        observer = SimpleNamespace(args=SimpleNamespace(feature_samples=4))
        sample = Observation._sample(observer, source)
        self.assertIs(type(sample), torch.Tensor)
        self.assertEqual(sample.tolist(), [0, 1, 3, 5])
        source.fill_(-1)
        self.assertEqual(sample.tolist(), [0, 1, 3, 5])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="munet-observe-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.first = self.root / "first"
        self.second = self.root / "second"
        self.first.mkdir()
        self.second.mkdir()

    def checkpoint(self, root, epoch, content=b"checkpoint"):
        path = root / f"ckpt_epoch={epoch - 1}.pth.ckpt"
        path.write_bytes(content)
        return path

    def test_comma_parser_normalizes_and_deduplicates_roots(self):
        roots = _training_roots(f" {self.first}, {self.second},{self.first} ")
        self.assertEqual(roots, tuple(sorted((self.first, self.second))))
        self.assertEqual(_training_roots(str(self.first)), (self.first,))
        for value in ("", ",", f"{self.first},", f",{self.first}"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _training_roots(value)

    def test_sparse_epochs_sort_globally_and_overlapping_roots_deduplicate(self):
        ten = self.checkpoint(self.second, 10)
        forty = self.checkpoint(self.first, 40)
        self.checkpoint(self.first, 15)
        (self.first / "last.ckpt").write_bytes(b"last")
        actual = _checkpoints((self.root, self.first, self.second))
        self.assertEqual(actual, [(10, (ten,)), (40, (forty,))])

    def test_identical_copies_count_once_and_conflicting_copies_fail(self):
        original = self.checkpoint(self.first, 10)
        duplicate = self.checkpoint(self.second, 10)
        epoch, paths = _checkpoints((self.first, self.second))[0]
        cache = {}
        self.assertEqual(_select_checkpoint(epoch, paths, cache), min(original, duplicate))
        duplicate.write_bytes(b"different contents")
        with self.assertRaisesRegex(ValueError, "Conflicting checkpoints"):
            _select_checkpoint(epoch, paths, cache)

    def test_completed_epoch_accepts_a_relocated_identical_copy(self):
        original = self.checkpoint(self.first, 10)
        duplicate = self.checkpoint(self.second, 10)
        manifest = {"checkpoint": _signature(original), "checkpoint_sha256": _sha256(original)}
        self.assertEqual(_select_checkpoint(10, (duplicate,), {}, manifest), duplicate)
        duplicate.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "Completed checkpoint was replaced"):
            _select_checkpoint(10, (duplicate,), {}, manifest)

    def test_partial_copy_must_stabilize_before_content_comparison(self):
        original = self.checkpoint(self.first, 10)
        duplicate = self.checkpoint(self.second, 10, b"partial")
        paths, seen = (original, duplicate), {}
        with patch("experiments.munet.observe.time.monotonic", return_value=0):
            self.assertFalse(_stable_candidates(paths, seen, 60))
        duplicate.write_bytes(original.read_bytes())
        with patch("experiments.munet.observe.time.monotonic", return_value=60):
            self.assertFalse(_stable_candidates(paths, seen, 60))
        with patch("experiments.munet.observe.time.monotonic", return_value=119):
            self.assertFalse(_stable_candidates(paths, seen, 60))
        with patch("experiments.munet.observe.time.monotonic", return_value=120):
            self.assertTrue(_stable_candidates(paths, seen, 60))
        self.assertIn(_select_checkpoint(10, paths, {}), paths)

    def test_digest_rejects_changes_during_hashing(self):
        checkpoint = self.checkpoint(self.first, 10)

        def changing_hash(path):
            path.write_bytes(b"changed during hash computation")
            return "not-committed"

        cache = {}
        with patch("experiments.munet.observe._sha256", side_effect=changing_hash):
            with self.assertRaisesRegex(OSError, "changed while hashing"):
                _cached_digest(checkpoint, cache)
        self.assertFalse(cache)


class SettingsTests(unittest.TestCase):
    def test_legacy_settings_accept_added_roots_but_not_changed_inputs(self):
        previous = {"schema": 1, "trained": "/run/first", "data": "/data", "dice": True}
        current = {**previous, "trained": ["/run/first", "/run/second"]}
        self.assertTrue(_compatible_settings(previous, current))
        self.assertTrue(_compatible_settings(previous, {**current, "trained": ["/run/first"]}))
        self.assertFalse(_compatible_settings(previous, {**current, "trained": ["/run/second"]}))
        self.assertFalse(_compatible_settings(previous, {**current, "dice": False}))
        self.assertFalse(_compatible_settings(previous, {**current, "data": "/another-dataset"}))

    def test_multiple_roots_can_be_reordered_or_extended(self):
        previous = {"schema": 1, "trained": ["/run/first", "/run/second"]}
        self.assertTrue(_compatible_settings(previous, {**previous, "trained": ["/run/second", "/run/first"]}))
        self.assertTrue(_compatible_settings(previous, {**previous, "trained": ["/run/first", "/run/second", "/run/third"]}))
        self.assertFalse(_compatible_settings(previous, {**previous, "trained": ["/run/first"]}))

    def test_worker_temp_environment_replaces_shared_paths(self):
        args = SimpleNamespace(temp_dir=Path("/local/tmp"))
        with patch.dict(os.environ, {"TMPDIR": "/shared/tmp", "TEMP": "/shared", "TMP": "/shared"}):
            environment = _worker_environment(args, {"uuid": "GPU-test"})
            for name in ("TMPDIR", "TEMP", "TMP"):
                self.assertEqual(environment[name], str(args.temp_dir))
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-test")
            self.assertEqual(os.environ["TMPDIR"], "/shared/tmp")

    def test_loader_workers_zero_is_independent_of_metric_workers(self):
        parser = _parser(MUNetCustom)
        args = parser.parse_args(["data", "first,second", "plan.json", "output",
                                  "--loader-workers", "0", "-w", "3", "--temp-dir", "/local/tmp"])
        self.assertEqual(args.loader_workers, 0)
        self.assertEqual(args.workers, 3)
        self.assertEqual(args.temp_dir, Path("/local/tmp").resolve())


if __name__ == "__main__":
    unittest.main()
