from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from app.graph.states.state import GraphState
from app.graph.nodes.task_router import _semantic_route, _contains_any

# 승인 키워드
_APPROVE_HINTS = [
    "완료", "확인", "좋아", "좋아요", "괜찮아", "괜찮아요", "ok", "okay",
    "승인", "통과", "보내줘", "전송", "발송", "이대로", "그대로",
    "완성", "완성됐어", "완성됐습니다", "이걸로 할게", "이걸로 할게요",
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

