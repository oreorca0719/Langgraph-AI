"""
Security node — v1의 4계층(Layer 1~4)을 2계층으로 통합.

기존 (v1):
  Layer 1: input_guard (임베딩 + 슬라이딩 윈도우)
  Layer 2: task_router → rejection (실제 미구현)
  Layer 3: content_sanitizer (RAG 문서 sanitize)
  Layer 4: output_validator (응답 sensitive 패턴)

신규 (v2):
  pre_input_check: Layer 1 + 4 통합 — 단일 패턴 사전, 임베딩 1회
  content_sanitize: Layer 3 (Retriever 내부에서 호출)

이 파일은 pre_input_check만 담당. content_sanitize는 retrievers/ 안에서.
"""
from __future__ import annotations

from app.graph_v2.states.state import GraphState


def security_gate_node(state: GraphState) -> dict:
    """입력 단계 security check.

    TODO: v1 input_guard + output_validator 패턴 통합 후 단일 임베딩 호출.
    """
    # TODO: v1의 injection_detector + output_validator 패턴 통합
    # TODO: 임베딩 1회 → input_embedding state에 캐시
    # TODO: 차단 시 security_blocked=True, security_reason 설정 → END로 라우팅
    return {
        "decision_path": ["security:pass"],
        "security_blocked": False,
    }


def route_after_security(state: GraphState) -> str:
    return "rejected" if state.security_blocked else "router"
