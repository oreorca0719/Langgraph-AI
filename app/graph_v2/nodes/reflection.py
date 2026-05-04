"""
Reflection node — 답변의 자기검증 (FR-403, FR-404, FR-405).

검증 항목:
  - groundedness: 답변이 검색 근거에서 도출됐는가
  - relevance: 답변이 원 질문에 답하는가
  - hallucination_risk: 환각 의심 여부
  - 추가 (Phase D 학습):
    * 수치/entity in chunks
    * no_info_misclaim 검출

Reflection 실패 시 router로 회귀하여 재계획. 재계획 최대 1회 (FR-404).
"""
from __future__ import annotations

from app.graph_v2.states.state import GraphState, VerificationResult


def reflection_node(state: GraphState) -> dict:
    """답변 검증 → VerificationResult 산출.

    TODO:
    - LLM-as-judge로 groundedness/relevance/hallucination 평가
    - 수치/entity 매칭 검증 (regex + cited chunks 체크)
    - 결과를 verification 필드에 저장
    """
    return {
        "verification": VerificationResult(passed=True),
        "decision_path": ["reflect:pass"],
    }


def route_after_reflection(state: GraphState) -> str:
    """passed → END, 실패 + replan_iterations < 1 → router 회귀."""
    if not state.verification:
        return "end"
    if state.verification.needs_replan() and state.replan_iterations < 1:
        return "replan"
    return "end"
