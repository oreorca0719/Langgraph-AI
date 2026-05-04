"""
Security gate node — v1의 4계층(Layer 1~4)을 입력 단계 통합 구현.

기존 (v1):
  Layer 1: input_guard (임베딩 + 슬라이딩 윈도우)
  Layer 2: task_router → rejection (실제 미구현)
  Layer 3: content_sanitizer (Retriever 내부에서 호출 — graph_v2에서도 동일)
  Layer 4: output_validator (응답 sensitive 패턴)

신규 (v2):
  pre_input_check: Layer 1 통합 — 임베딩 1회 + 슬라이딩 윈도우
  output_validate: Layer 4를 별도 함수로 (generator/reflection 후 호출 가능)

이 파일은 pre_input_check만 담당. content_sanitize는 retrievers/에서.
"""
from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage

from app.core.config import get_embeddings
from app.graph_v2.states.state import GraphState
from app.security.injection_detector import check as injection_check


def security_gate_node(state: GraphState) -> dict:
    """입력 단계 보안 검사. v1 Layer 1과 동일 메커니즘.

    - 임베딩 1회 계산 → state.input_embedding 캐시
    - 임베딩 기반 injection 패턴 매칭 + 슬라이딩 윈도우
    - 차단 시 security_blocked=True
    """
    user_input = (state.input_data or "").strip()
    if not user_input:
        return {
            "security_blocked": False,
            "decision_path": ["security:empty_input"],
        }

    # 임베딩 1회 (v1과 동일)
    input_embedding: list[float] = []
    try:
        input_embedding = get_embeddings().embed_query(user_input)
    except Exception as e:
        print(f"[SECURITY] embedding failed (non-fatal): {e}")
        input_embedding = []

    # 이전 HumanMessage 턴 (슬라이딩 윈도우용)
    recent_turns: List[str] = [
        msg.content for msg in (state.messages or [])
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
    ][-3:]

    blocked = False
    try:
        blocked = injection_check(user_input, recent_turns, input_embedding=input_embedding)
    except Exception as e:
        print(f"[SECURITY] injection_check failed (non-fatal): {e}")

    if blocked:
        return {
            "security_blocked": True,
            "security_reason": "injection_detected",
            "input_embedding": input_embedding,
            "decision_path": ["security:blocked(injection)"],
        }

    return {
        "security_blocked": False,
        "input_embedding": input_embedding,
        "decision_path": ["security:pass"],
    }


def route_after_security(state: GraphState) -> str:
    return "rejected" if state.security_blocked else "router"
