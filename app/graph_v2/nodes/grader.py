"""
Grader node — 검색 결과의 관련성 평가 (FR-301, FR-302, FR-303, FR-304).

라벨:
  - "relevant" / "partially_relevant" / "irrelevant"

`relevant` 비율이 임계값(0.5) 미만 시:
  1. 쿼리 재작성 후 동일 retriever 재시도 (max_retrieval_iterations)
  2. (FR-302의 폴백·정보 없음 거절은 router/reflection이 담당)

추가 평가 (Phase 회복 학습):
  - premise_coverage: 검색 결과가 질문의 모든 entity·전제를 커버하는가
"""
from __future__ import annotations

from app.graph_v2.states.state import GraphState


def grader_node(state: GraphState) -> dict:
    """LLM이 각 chunk를 relevant/partial/irrelevant 라벨링 + premise_coverage 평가.

    TODO:
    - LLM 호출로 chunk별 라벨
    - relevant + partial chunk만 retrieved_docs에 유지 (FR-401)
    - 모두 irrelevant이고 retrieval_iterations < max → rewrite trigger
    """
    return {
        "decision_path": ["grade:pass"],  # placeholder
    }


def route_after_grade(state: GraphState) -> str:
    """grade 결과 → next node."""
    # TODO: relevant 비율 + retrieval_iterations 한도 체크
    return "generate"
