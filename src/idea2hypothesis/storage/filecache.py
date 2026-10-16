"""Stat-validated, write-through file cache.

An ``open`` costs milliseconds on some systems (antivirus scans on open) while ``os.stat`` is
cheap. Files that this process just wrote, or that have not changed, are served from memory when
``(size, mtime_ns, inode)`` still matches. A file changed by anyone else shows a new signature and
is read again, so the cache never hides a change. Durability is the caller's job and is unchanged.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_FILE_BYTES = 4 * 1024 * 1024

Signature = tuple[int, int, int]


def signature(path: str | os.PathLike[str]) -> Signature | None:
    """``(size, mtime_ns, inode)`` or ``None`` when the file cannot be stat-ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns, st.st_ino)


def _key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(path))


class FileCache:
    def __init__(self, max_total: int = MAX_TOTAL_BYTES, max_file: int = MAX_FILE_BYTES) -> None:
        self._max_total = max_total
        self._max_file = max_file
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[Signature, bytes]] = OrderedDict()
        self._total = 0

    # -- internals (call with the lock held) ----------------------------------

    def _drop(self, key: str) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._total -= len(old[1])

    def _store(self, key: str, sig: Signature, data: bytes) -> None:
        self._drop(key)
        if len(data) > self._max_file:
            return
        self._entries[key] = (sig, data)
        self._total += len(data)
        while self._total > self._max_total and self._entries:
            _, (_, evicted) = self._entries.popitem(last=False)
            self._total -= len(evicted)

    # -- reads -----------------------------------------------------------------

    def read_bytes(self, path: str | os.PathLike[str]) -> bytes:
        key = _key(path)
        before = os.stat(path)  # raises exactly what the open would
        sig = (before.st_size, before.st_mtime_ns, before.st_ino)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] == sig:
                self._entries.move_to_end(key)
                return entry[1]
        data = Path(path).read_bytes()
        after = signature(path)
        with self._lock:
            if after == sig:
                self._store(key, sig, data)
            else:
                self._drop(key)
        return data

    def read_text(self, path: str | os.PathLike[str], encoding: str = "utf-8") -> str:
        text = self.read_bytes(path).decode(encoding)
        if "\r" in text:  # same newline translation as ``Path.read_text``
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    def read_json(self, path: str | os.PathLike[str]) -> Any:
        return json.loads(self.read_text(path))

    def last_byte(self, path: str | os.PathLike[str]) -> bytes | None:
        """The final byte if the file is cached and unchanged, else ``None`` (caller opens)."""
        sig = signature(path)
        with self._lock:
            entry = self._entries.get(_key(path))
        if entry is None or sig is None or entry[0] != sig or not entry[1]:
            return None
        return entry[1][-1:]

    # -- writes ----------------------------------------------------------------

    def wrote(self, path: str | os.PathLike[str], data: bytes) -> None:
        """Call after a complete write or atomic replace succeeded."""
        key = _key(path)
        sig = signature(path)
        with self._lock:
            if sig is None:
                self._drop(key)
            else:
                self._store(key, sig, data)

    def appended(
        self, path: str | os.PathLike[str], before_sig: Signature | None, data: bytes
    ) -> None:
        """Call after an append succeeded; ``before_sig`` is ``signature(path)`` taken before it."""
        key = _key(path)
        sig = signature(path)
        with self._lock:
            entry = self._entries.get(key)
            if sig is None:
                self._drop(key)
            elif before_sig is None:
                self._store(key, sig, data)  # the append created the file
            elif entry is not None and entry[0] == before_sig:
                self._store(key, sig, entry[1] + data)
            else:
                self._drop(key)

    def forget(self, path: str | os.PathLike[str]) -> None:
        """Call after a truncate, delete or any write that does not go through ``wrote``."""
        with self._lock:
            self._drop(_key(path))


CACHE = FileCache()
read_bytes = CACHE.read_bytes
read_text = CACHE.read_text
read_json = CACHE.read_json
last_byte = CACHE.last_byte
wrote = CACHE.wrote
appended = CACHE.appended
forget = CACHE.forget
