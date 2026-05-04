"""
Router (Planner) node — 질문 분류 + 검색 도구 선택 (FR-101, FR-102, FR-103).

분류 (structured output, FR-104):
  - "no_retrieval": 인사·기능 안내·메타 질의 (검색 불필요)
  - "single_retrieval": 단일 검색으로 충분
  - "multi_hop_retrieval": sub-question 분해 후 다단계
  - "ai_guide": v1과 호환 (안내 응답)
  - "file_chat": file_context 있을 때
  - "rejected": injection 등 차단

추가:
  - question_type 동시 분류 (Lever 5 통합): exact_phrase / numerical / list_n / fill_blank / reasoning / comparison
  - sub_questions 생성 (multi_hop 시 최대 5개, FR-103)
"""
from __future__ import annotations

from app.graph_v2.states.state import GraphState


def router_node(state: GraphState) -> dict:
    """LLM structured output으로 (routing_decision, question_type, sub_questions) 결정.

    TODO:
    - LLM 호출 + structured JSON output (Pydantic schema enforced, FR-104)
    - file 키워드 명시 검증 (Phase A-1 학습 — file_chat false-positive 차단)
    - 짧은 사실 질의 → no_retrieval로 가지 않음 (Phase A-2 학습)
    - intent_samples 기반 semantic routing 보조 (v1 자산 재활용)
    """
    return {
        "routing_decision": "single_retrieval",  # placeholder
        "question_type": "reasoning",
        "sub_questions": [],
        "decision_path": ["router:single_retrieval"],
    }


def route_by_decision(state: GraphState) -> str:
    """routing_decision → 다음 노드."""
    d = state.routing_decision
    if d == "rejected":
        return "rejected"
    if d == "no_retrieval":
        return "no_retrieval_answer"
    if d == "ai_guide":
        return "ai_guide"
    if d == "file_chat":
        return "file_chat"
    # single / multi 모두 retrieval subgraph로
    return "retrieve"
