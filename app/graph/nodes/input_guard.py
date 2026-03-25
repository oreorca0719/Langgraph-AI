from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage

from app.core import trace_buffer
from app.graph.states.state import GraphState
from app.security.injection_detector import check as injection_check


def input_guard_node(state: GraphState) -> Dict[str, Any]:
    """
    그래프 진입점 보안 필터.
    - 임베딩 기반 injection 감지
    - 감지 시 task_type = 'injection' 설정 → rejection_node로 라우팅
    """
    print("--- [NODE] Input Guard ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()

    trace_buffer.push(trace_id, node="input_guard", event="enter", label="execute",
                      data={"input": user_input[:200]})

    if not user_input:
        trace_buffer.push(trace_id, node="input_guard", event="exit", label="execute",
                          data={"result": "empty"})
        return {"task_type": "knowledge_search"}

    # 이전 HumanMessage 턴 수집 (슬라이딩 윈도우용)
    recent_turns = [
        msg.content for msg in (state.get("messages") or [])
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
    ][-3:]

    if injection_check(user_input, recent_turns):
        trace_buffer.push(trace_id, node="input_guard", event="exit", label="execute",
                          data={"result": "injection_blocked"})
        return {"task_type": "injection"}

    trace_buffer.push(trace_id, node="input_guard", event="exit", label="execute",
                      data={"result": "pass"})
    return {}
