"""A library call in a child process that a window can poll every frame and cancel: the viewer's long steps.

The Polyscope viewer's window runs one frame at a time; a reconstruction or a seed ensemble takes seconds to
minutes, and a library call cannot be interrupted inside a thread. So `ChildJob` starts the call in a child process
(`spawn` start method, in a process group of its own) and returns at once; `poll` reads, without waiting, what the
child has sent (progress, then the result or the refusal); `cancel` stops the child's process group (SIGTERM, then
SIGKILL after a grace period), seed-ensemble workers included. The child sends its result by pickling, with the
warnings it raised (`foambryo.results.WarningCollector`). Nothing partial is ever kept: a cancelled or failed job
has no value.

The call is named `"module:function"` (for example `"foambryo.io:reconstruct_mask"`), so the child imports exactly
the library function the command and plain Python call: nothing is computed here. No Polyscope, no display.
"""

from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import os
import signal
import time
import traceback
from typing import Any

__all__ = ["ChildJob"]

#: Seconds between SIGTERM and SIGKILL when a job is cancelled.
TERMINATE_GRACE_SECONDS = 5.0

RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"


def _child_main(connection: Any, target: str, kwargs: dict, with_progress: bool) -> None:  # noqa: ANN401
    """The child: a process group of its own, the call with its warnings counted, the outcome sent back."""
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    from foambryo.results import WarningCollector

    collector = WarningCollector()
    try:
        module, _, name = target.partition(":")
        function = getattr(importlib.import_module(module), name)
        if with_progress:
            kwargs = {**kwargs, "progress": lambda *args: connection.send(("progress", args))}
        with collector:
            value = function(**kwargs)
        connection.send(("ok", value, collector.summary))
    except BaseException as error:  # everything is reported to the parent, which decides
        connection.send(("error", type(error).__name__, str(error), traceback.format_exc(), collector.summary))
    finally:
        connection.close()


class ChildJob:
    """One library call running in a child process.

    Attributes:
        target: `"module:function"`.
        state: `"running"`, `"done"`, `"failed"` or `"cancelled"`.
        value: the call's return value once `"done"`, else `None`.
        warnings_summary: the warnings the call raised, each once with its count.
        error: `"<ExceptionType>: <message>"` once `"failed"`.
        progress: the arguments of the call's last `progress(...)` call (with `with_progress`), else `None`.
        seconds: wall time from the start to the end (or until now while running).
    """

    def __init__(self, target: str, kwargs: dict[str, Any], *, with_progress: bool = False) -> None:
        """Start `target(**kwargs)` in a child process; the call receives a `progress` callback with `with_progress`."""
        self.target = target
        self.state = RUNNING
        self.value: Any = None
        self.warnings_summary: list = []
        self.error = ""
        self.progress: tuple | None = None
        context = multiprocessing.get_context("spawn")
        self._receive, send = context.Pipe(duplex=False)
        # Not a daemon: a seed ensemble's child starts worker processes of its own, which a daemon may not. Whoever
        # starts a job cancels it when it no longer wants it (the viewer does when its window closes).
        self._process = context.Process(target=_child_main, args=(send, target, kwargs, with_progress))
        self._started = time.perf_counter()
        self._ended: float | None = None
        self._process.start()
        send.close()

    @property
    def seconds(self) -> float:
        """Seconds since the start, or the job's duration once it ended."""
        return (self._ended if self._ended is not None else time.perf_counter()) - self._started

    def poll(self) -> str:
        """Read what the child sent, without waiting for more; returns the state.

        Progress messages update `progress`; the outcome sets `value` or `error`. A child that ended without an
        outcome (for example killed by the system for lack of memory) is `"failed"`, with its exit code.
        """
        if self.state != RUNNING:
            return self.state
        while self._receive.poll(0):
            try:
                message = self._receive.recv()
            except EOFError:
                break
            if message[0] == "progress":
                self.progress = message[1]
                continue
            self._finish()
            if message[0] == "ok":
                self.value, self.warnings_summary, self.state = message[1], message[2], DONE
            else:
                self.error, self.warnings_summary = f"{message[1]}: {message[2]}", message[4]
                self.state = FAILED
            return self.state
        if not self._process.is_alive() and not self._receive.poll(0):
            self._finish()
            self.error = (f"the child process ended without a result (exit code {self._process.exitcode}; a negative "
                          "code is the signal that stopped it, for example -9 when the system ran out of memory)")
            self.state = FAILED
        return self.state

    def wait(self, timeout: float | None = None, interval: float = 0.05) -> str:
        """Poll until the job is no longer running, or `timeout` seconds have passed; returns the state."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while self.poll() == RUNNING and (deadline is None or time.perf_counter() < deadline):
            time.sleep(interval)
        return self.state

    def cancel(self) -> None:
        """Stop the child's process group; the job is `"cancelled"` and keeps no value."""
        if self.state != RUNNING:
            return
        _stop_group(self._process)
        self._finish()
        self.state = CANCELLED

    def _finish(self) -> None:
        self._ended = time.perf_counter()
        with contextlib.suppress(OSError):
            self._receive.close()
        self._process.join(TERMINATE_GRACE_SECONDS)
        if self._process.is_alive():
            _stop_group(self._process)


def _stop_group(process: multiprocessing.process.BaseProcess) -> None:
    """SIGTERM to the child's process group, then SIGKILL after the grace period."""
    pid = process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError, TypeError):
        process.terminate()  # the child had not made its group yet
    process.join(TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError, TypeError):
            process.kill()
        process.join()
