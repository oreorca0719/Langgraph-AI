from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

_buffer: deque = deque(maxlen=500)
_subscribers: List[asyncio.Queue] = []
_lock = threading.Lock()


def _push(record: Dict[str, Any]) -> None:
    with _lock:
        _buffer.append(record)
        for q in list(_subscribers):
            try:
                q.put_nowait(record)
            except Exception:
                pass


class _BufHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _push({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "msg": self.format(record),
            })
        except Exception:
            pass


class _StdoutCapture:
    def __init__(self, orig):
        self._orig = orig

    def write(self, text: str) -> None:
        self._orig.write(text)
        text = text.strip()
        if text:
            _push({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "level": "PRINT",
                "msg": text,
            })

    def flush(self) -> None:
        self._orig.flush()

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


def setup() -> None:
    handler = _BufHandler()
    handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    logging.root.addHandler(handler)
    sys.stdout = _StdoutCapture(sys.stdout)


def get_recent(n: int = 300) -> List[Dict[str, Any]]:
    with _lock:
        return list(_buffer)[-n:]


def subscribe(queue: asyncio.Queue) -> None:
    with _lock:
        _subscribers.append(queue)


def unsubscribe(queue: asyncio.Queue) -> None:
    with _lock:
        try:
            _subscribers.remove(queue)
        except ValueError:
            pass