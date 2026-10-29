"""Run manually: CPU/mock shutdown checks and a Linux synthetic process tree.

No inference, CUDA, medical data or real experiment processes are used.
"""

from collections import deque
from concurrent.futures import Future
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

from experiments.munet.observe import (
    Observation, _cleanup_epoch_ipc, _group_running, _stop_epoch, _stop_processes,
    _worker_environment,
)


class WorkerShutdownTests(unittest.TestCase):
    def test_worker_helpers_use_monitor_owned_temporary_root(self):
        args = Mock(temp_dir=Path("/local/tmp"))
        root = Path("/local/tmp/munet-observe-epoch-synthetic")
        environment = _worker_environment(args, {"uuid": "GPU-synthetic"}, root)
        for key in ("TMPDIR", "TEMP", "TMP"):
            self.assertEqual(environment[key], str(root))

    def test_abort_stops_running_children_before_pool_and_file_cleanup(self):
        events = []
        observer = object.__new__(Observation)
        future = Future()
        future.set_running_or_notify_cancel()
        observer.pending = deque([future])
        observer.pool = Mock()
        observer.pool.shutdown.side_effect = lambda **kwargs: events.append("pool")
        observer.ipc_directory = Mock()
        observer.ipc_directory.cleanup.side_effect = lambda: events.append("files")
        observer.stream = io.StringIO()
        processes = [Mock()]
        with patch("experiments.munet.observe.active_children", return_value=processes), \
             patch("experiments.munet.observe._stop_processes",
                   side_effect=lambda children: events.append("stop")) as stop:
            observer.close(abort=True)
        stop.assert_called_once_with(processes)
        self.assertEqual(events, ["stop", "pool", "files"])
        self.assertFalse(future.cancelled())  # A running Future cannot be cancelled.
        self.assertTrue(observer.stream.closed)

    def test_live_reader_prevents_cleanup(self):
        observer = object.__new__(Observation)
        observer.pending = deque()
        observer.pool = Mock()
        observer.ipc_directory = Mock()
        observer.stream = io.StringIO()
        with patch("experiments.munet.observe.active_children", return_value=[]), \
             patch("experiments.munet.observe._stop_processes",
                   side_effect=RuntimeError("still running")):
            with self.assertRaisesRegex(RuntimeError, "still running"):
                observer.close(abort=True)
        observer.ipc_directory.cleanup.assert_not_called()
        self.assertTrue(observer.stream.closed)

    def test_child_ignoring_terminate_is_killed_and_joined(self):
        alive = [True]
        process = Mock(pid=12345)
        process.is_alive.side_effect = lambda: alive[0]
        process.kill.side_effect = lambda: alive.__setitem__(0, False)
        _stop_processes([process], timeout=0)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.join.call_count, 2)


@unittest.skipUnless(os.name == "posix", "POSIX process groups required")
class ProcessGroupTests(unittest.TestCase):
    def test_exited_leader_does_not_skip_surviving_workers(self):
        child = Mock(pid=12345)
        child.poll.return_value = 0
        with patch("experiments.munet.observe.os.killpg") as killpg, \
             patch("experiments.munet.observe.os.getpgrp", return_value=54321), \
             patch("experiments.munet.observe._wait_epoch_group", side_effect=[False, True]):
            _stop_epoch(child, child.pid, grace=0, timeout=0)
        self.assertEqual(killpg.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args, (12345, signal.SIGKILL))
        child.wait.assert_called_once()

    def test_live_leader_gets_graceful_interrupt_first(self):
        child = Mock(pid=12345)
        child.poll.return_value = None
        with patch("experiments.munet.observe.os.kill") as kill, \
             patch("experiments.munet.observe.os.killpg") as killpg, \
             patch("experiments.munet.observe.os.getpgrp", return_value=54321), \
             patch("experiments.munet.observe._wait_epoch_group", return_value=True):
            _stop_epoch(child, child.pid, grace=0, timeout=0)
        kill.assert_called_once_with(12345, signal.SIGINT)
        killpg.assert_not_called()

    def test_own_process_group_is_never_signalled(self):
        child = Mock(pid=os.getpid())
        with patch("experiments.munet.observe.os.killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "non-isolated"):
                _stop_epoch(child, os.getpgrp())
        killpg.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc check required")
    def test_zombies_do_not_count_as_file_holding_workers(self):
        entry = Mock()
        entry.name = "12345"
        stat = Mock()
        stat.read_text.return_value = "12345 (synthetic worker) Z 1 12345 12345 0"
        entry.__truediv__ = Mock(return_value=stat)
        with patch("experiments.munet.observe.os.killpg"), \
             patch("experiments.munet.observe.Path") as path:
            path.return_value.iterdir.return_value = [entry]
            self.assertFalse(_group_running(12345))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux integration check required")
    def test_orphan_worker_holding_file_is_stopped_before_directory_removal(self):
        with TemporaryDirectory(prefix="munet-shutdown-test-") as temporary:
            root = Path(temporary)
            ready = root / "worker.pid"
            held = root / "held.bin"
            held.write_bytes(b"synthetic")
            worker = (
                "import os, signal, sys, time; from pathlib import Path; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "handle = open(sys.argv[1], 'rb'); "
                "Path(sys.argv[2]).write_text(str(os.getpid())); time.sleep(120)"
            )
            leader = (
                "import subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], *sys.argv[2:]])"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", leader, worker, str(held), str(ready)],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(ready.exists(), "Synthetic worker did not start")
                pid = int(ready.read_text())
                child.wait(timeout=10)  # Leader is already gone; worker is still alive.
                self.assertTrue(_group_running(child.pid))
                _stop_epoch(child, child.pid, grace=.2, timeout=1)
                self.assertFalse(_group_running(child.pid))
                descriptors = Path(f"/proc/{pid}/fd")
                if descriptors.exists():
                    self.assertEqual(list(descriptors.iterdir()), [])
                _cleanup_epoch_ipc(root)
                self.assertFalse(root.exists())
            finally:
                # Only the dedicated synthetic group created by this test.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
