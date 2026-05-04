"""
Generator node — 검색 결과 기반 답변 생성 + Citation 부착 (FR-401, FR-402).

설계:
  - relevant + partial 문서만 컨텍스트로 사용 (FR-401)
  - question_type별 프롬프트 분기 (Lever 5 학습 회복):
    * exact_phrase / numerical / fill_blank: 단답, verbatim 강제
    * list_n: 정확 N개 항목 (Phase B 학습 — list_n 누락 방지)
    * reasoning: CoT 허용
  - 모든 사실 진술에 [N] citation 강제 (FR-402)
"""
from __future__ import annotations

from app.graph_v2.states.state import GraphState


def generator_node(state: GraphState) -> dict:
    """answer 생성 + citations 부착.

    TODO:
    - question_type 분기로 프롬프트 선택
    - LLM 호출 → answer + citation IDs
    - state.citations 채움
    """
    return {
        "answer": "",  # placeholder
        "citations": [],
        "decision_path": ["generate"],
    }
