from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List

_buffer: deque = deque(maxlen=300)
_lock = threading.Lock()


def push(
    trace_id: str,
    node: str,
    event: str,
    label: str = "execute",
    data: Dict[str, Any] | None = None,
) -> None:
    """트레이스 이벤트 1건을 버퍼에 기록합니다. 예외는 무시합니다."""
    try:
        record = {
            "trace_id": trace_id or "",
            "node": node,
            "event": event,   # "enter" | "exit" | "call"
            "label": label,   # "preview" | "execute"
            "data": data or {},
            "ts": time.time(),
        }
        with _lock:
            _buffer.append(record)
    except Exception:
        pass


def get_recent(n: int = 300) -> List[Dict[str, Any]]:
    """최근 n개 raw 이벤트를 반환합니다."""
    with _lock:
        return list(_buffer)[-n:]


def get_recent_traces(n: int = 50) -> List[Dict[str, Any]]:
    """
    최근 n개 trace_id 기준으로 그룹핑된 트레이스 목록을 반환합니다.
    각 트레이스는 { trace_id, ts_start, events: [...] } 구조입니다.
    """
    with _lock:
        records = list(_buffer)

    # trace_id 순서 보존 (최초 등장 순)
    seen: dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in records:
        tid = r["trace_id"]
        if tid not in seen:
            seen[tid] = []
            order.append(tid)
        seen[tid].append(r)

    # 최근 n개 trace_id만 (역순으로 n개 선택 후 시간순 복원)
    recent_ids = order[-n:]
    result = []
    for tid in reversed(recent_ids):
        events = seen[tid]
        result.append({
            "trace_id": tid,
            "ts_start": events[0]["ts"] if events else 0,
            "events": events,
        })
    return result