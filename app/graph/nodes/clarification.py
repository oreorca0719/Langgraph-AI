from __future__ import annotations

import re

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.core.config import get_llm
from app.core.history_utils import extract_text_content
from app.graph.states.state import GraphState


_FILE_REFERENCE_RE = re.compile(r"(?:^|[\/\s])[^\/\s]+\.(pdf|docx|pptx|xlsx|xlsm|txt|md|csv|log)\b", re.I)


def _looks_like_file_reference(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if _FILE_REFERENCE_RE.search(text):
        return True
    if any(token in text for token in ("./", ".\\", "/", "\\")):
        return True
    return False


# ── Priority 3: 슬롯별 고정 질문 ───────────────────────────────
_SLOT_QUESTIONS: dict[str, str] = {
    "file_path":     "분석할 파일 경로 또는 파일명을 알려주세요.",
    "file_context":  "첨부 파일이 필요합니다. 먼저 파일을 업로드해 주세요.",
    "to":            "이메일 수신자 또는 받는 부서를 알려주세요. (예: hr@example.com 또는 인사팀)",
    "project_scope": "RFP 대상 프로젝트명이나 제안 범위를 알려주세요.",
}


def clarification_node(state: GraphState) -> Dict[str, Any]:
    """
    라우팅 신뢰도가 낮을 때 사용자 의도를 확인하는 노드.

    흐름:
    1. clarification_count >= 2 → 무한 루프 방어, knowledge_search 강제 fallback
    2. LLM이 사내 기능 연관 여부 판단
       - 범위 외 → rejection_node로 직행
       - 연관 가능 → 의도 확인 질문 생성
    3. interrupt()로 질문을 사용자에게 전달하고 응답 대기
    4. 재개 후 original + 응답을 결합 → task_router_node 루프백
    """
    print("--- [NODE] Clarification ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    clarification_count = (state.get("clarification_count") or 0)
    task_type = (state.get("task_type") or "").strip()
    task_args = state.get("task_args") or {}
    missing_slots: list[str] = task_args.get("missing_slots") or []


    response = get_llm().invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_input),
    ])
    response_text = extract_text_content(response.content)
    is_rejection = "지원 범위에 포함되지 않아" in response_text

