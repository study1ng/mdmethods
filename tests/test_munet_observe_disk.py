"""Synthetic CPU IPC checks; run manually in the project container.

No CUDA, dataset, checkpoint, training, or inference job is required.
The project's compiled dependencies must still be importable.
"""

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
import io
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from monai.data import MetaTensor
import torch

from experiments.munet.observe import Observation
from experiments.munet.observe_cpu import initialize_worker, run_disk_job


def summarize_payload(value):
    tensor = value["tensor"]
    return {"records": [], "values": tensor.float().tolist(),
            "dtype": str(tensor.dtype), "device": tensor.device.type,
            "shared": tensor.is_shared(), "meta": tensor.meta["case"],
            "history": tensor.applied_operations,
            "nested_values": value["nested"][0].tolist()}


def failing_job(value):
    raise ValueError("synthetic CPU job failure")


class DiskTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="munet-observe-disk-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def payload(self):
        tensor = MetaTensor(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T,
                            meta={"case": "synthetic"})
        tensor.applied_operations = [{"class": "synthetic", "orig_size": (3, 4)}]
        return {"tensor": tensor, "nested": [torch.tensor([2, 5, 8])]}

    def test_submit_sends_only_function_and_path_without_tensor_reducers(self):
        observer = object.__new__(Observation)
        observer._make_room = Mock()
        observer.pool = Mock()
        observer.pending = deque()
        observer.ipc_directory = self.temporary
        observer.disk_write_seconds = 0.0
        future = Future()
        observer.pool.submit.return_value = future
        payload = self.payload()
        with patch.object(torch.Tensor, "share_memory_", side_effect=AssertionError("shared memory")):
            observer._submit(summarize_payload, payload)
        runner, function, path = observer.pool.submit.call_args.args
        self.assertIs(runner, run_disk_job)
        self.assertIs(function, summarize_payload)
        self.assertIsInstance(path, Path)
        self.assertTrue(path.is_file())
        self.assertEqual(observer.pending, deque([future]))
        expected = payload["tensor"].float().tolist()
        payload["tensor"].fill_(-1)
        result = runner(function, path)
        self.assertEqual(result["values"], expected)
        self.assertFalse(result["shared"])
        self.assertEqual(result["nested_values"], [2, 5, 8])
        self.assertFalse(path.exists())

    def test_spawn_preserves_dtype_metadata_and_history_without_shared_storage(self):
        path = self.root / "payload.pt"
        payload = self.payload()
        torch.save((payload,), path)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"),
                                 initializer=initialize_worker) as pool:
            result = pool.submit(run_disk_job, summarize_payload, path).result(timeout=120)
        self.assertEqual(result["values"], payload["tensor"].float().tolist())
        self.assertEqual(result["dtype"], "torch.bfloat16")
        self.assertEqual(result["device"], "cpu")
        self.assertFalse(result["shared"])
        self.assertEqual(result["meta"], "synthetic")
        self.assertEqual(result["history"], payload["tensor"].applied_operations)
        self.assertGreaterEqual(result["disk_read_seconds"], 0.0)
        self.assertFalse(path.exists())

    def test_failed_job_removes_its_input_and_propagates_error(self):
        path = self.root / "failure.pt"
        torch.save((torch.ones(4),), path)
        with self.assertRaisesRegex(ValueError, "synthetic CPU job failure"):
            run_disk_job(failing_job, path)
        self.assertFalse(path.exists())

    def test_failed_submission_removes_serialized_payload(self):
        observer = object.__new__(Observation)
        observer._make_room = Mock()
        observer.pool = Mock()
        observer.pool.submit.side_effect = RuntimeError("pool stopped")
        observer.pending = deque()
        observer.ipc_directory = self.temporary
        observer.disk_write_seconds = 0.0
        with self.assertRaisesRegex(RuntimeError, "pool stopped"):
            observer._submit(summarize_payload, self.payload())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(len(observer.pending), 0)

    def test_cancelled_job_cleanup_happens_after_pool_shutdown(self):
        observer = object.__new__(Observation)
        observer.ipc_directory = self.temporary
        path = self.root / "cancelled.pt"
        path.write_bytes(b"synthetic")
        future = Future()
        observer.pending = deque([future])
        observer.stream = io.StringIO()
        observer.pool = Mock()
        observer.pool.shutdown.side_effect = lambda **kwargs: self.assertTrue(path.exists())
        observer.close()
        self.assertTrue(future.cancelled())
        observer.pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        self.assertFalse(self.root.exists())
        self.assertTrue(observer.stream.closed)


if __name__ == "__main__":
    unittest.main()
