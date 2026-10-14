"""Completed tasks cannot starve the service's idle waiter."""

import subprocess
import sys


def test_wait_idle_returns_before_done_callback_cleanup():
    script = """
import asyncio
import tempfile
from pathlib import Path
from idea2hypothesis.api.service import RunService
from tests.conftest import make_config

async def finished():
    return

async def main():
    service = RunService(make_config(Path(tempfile.mkdtemp())))
    task = service.spawn(finished())
    await asyncio.sleep(0)
    assert task.done() and task in service._tasks
    await service.wait_idle()

asyncio.run(main())
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10, capture_output=True)
