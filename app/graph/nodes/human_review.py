from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.core import trace_buffer
from app.graph.states.state import GraphState
from app.graph.nodes.task_router import _semantic_route, _contains_any

# 승인 키워드
_APPROVE_HINTS = [
    "완료", "확인", "좋아", "좋아요", "괜찮아", "괜찮아요", "ok", "okay",
    "승인", "통과", "보내줘", "전송", "발송", "이대로", "그대로",
    "완성", "완성됐어", "완성됐습니다", "이걸로 할게", "이걸로 할게요",
    "이걸로", "그걸로", "보내도 돼", "보내도돼",
]

# 수정 키워드
_REVISE_HINTS = [
    "수정", "변경", "바꿔", "고쳐", "다시", "재작성", "바꿔줘",
    "틀렸어", "틀렸습니다", "잘못", "추가해줘", "빼줘", "넣어줘",
]


def _format_draft_preview(state: GraphState) -> str:
    """현재 State의 초안을 사용자에게 보여줄 텍스트로 포맷합니다."""
    draft_email = state.get("draft_email")
    draft_rfp = (state.get("draft_rfp") or "").strip()

    if draft_email:
        d = draft_email
        return (
            f"[이메일 초안]\n"
            f"- To: {d.get('to', '')}\n"
            f"- CC: {d.get('cc', '')}\n"
            f"- Subject: {d.get('subject', '')}\n\n"
            f"{d.get('body', '')}"
        )

    if draft_rfp:
        return f"[RFP 초안]\n\n{draft_rfp[:3000]}"

    return ""


def human_review_node(state: GraphState) -> Dict[str, Any]:
    """
    email_draft / rfp_draft 완료 후 사용자 검토를 받는 노드.

    흐름:
    1. 초안을 interrupt()로 사용자에게 전달 + 대기
    2. 재개 후 응답 분류:
       - approve : 키워드 기반 → END
       - revise  : 키워드 기반 → email/rfp_draft 루프백
       - switch  : _semantic_route()로 현재 task와 다른 task 감지 → task_router 루프백
    """
    print("--- [NODE] Human Review ---")

    trace_id = (state.get("trace_id") or "")
    current_task = (state.get("current_task") or "").strip()
    draft_preview = _format_draft_preview(state)

    trace_buffer.push(trace_id, node="human_review", event="enter", label="execute",
                      data={"current_task": current_task})

    # interrupt: 초안 전달 및 사용자 응답 대기
    user_response = interrupt({
        "type": "human_review",
        "message": draft_preview,
        "hint": "수정이 필요하시면 내용을 말씀해 주세요. 완료하시려면 '완료' 또는 '확인'을 입력해 주세요.",
    })

    user_response_str = (user_response or "").strip()
    trace_buffer.push(trace_id, node="human_review", event="call", label="execute",
                      data={"response": user_response_str[:100]})

    # 1) 승인 판단 (키워드)
    if _contains_any(user_response_str, _APPROVE_HINTS):
        trace_buffer.push(trace_id, node="human_review", event="exit", label="execute",
                          data={"action": "approve"})
        from langchain_core.messages import AIMessage
        return {
            "review_action": "approve",
            "input_data": user_response_str,
            "draft_email": None,
            "draft_rfp": "",
            "messages": [
                HumanMessage(content=user_response_str),
                AIMessage(content="확인되었습니다. 작업이 완료되었습니다."),
            ],
        }

    # 2) switch 판단: 수정 키워드가 없을 때만 semantic routing으로 task 전환 감지
    # 수정 힌트가 있는 경우 현재 초안 수정 요청으로 간주 (switch 건너뜀)
    if not _contains_any(user_response_str, _REVISE_HINTS):
        try:
            routed, _, _ = _semantic_route(user_response_str)
            if routed not in ("unknown", current_task) and routed != "":
                trace_buffer.push(trace_id, node="human_review", event="exit", label="execute",
                                  data={"action": "switch", "new_task": routed})
                return {
                    "review_action": "switch",
                    "input_data": user_response_str,
                    "task_type": "",   # task_router가 재분류
                    "messages": [HumanMessage(content=user_response_str)],
                }
        except Exception:
            pass

    # 3) 수정 요청 (기본)
    trace_buffer.push(trace_id, node="human_review", event="exit", label="execute",
                      data={"action": "revise"})
    return {
        "review_action": "revise",
        "input_data": user_response_str,
        "messages": [HumanMessage(content=user_response_str)],
    }


def route_after_review(state: GraphState) -> str:
    """human_review_node 이후 분기."""
    action = (state.get("review_action") or "").strip()
    current_task = (state.get("current_task") or "").strip()

    if action == "approve":
        return "end"
    if action == "switch":
        return "task_router"
    # revise: 현재 task로 루프백
    if current_task in ("email_draft", "rfp_draft"):
        return current_task
    return "end"
