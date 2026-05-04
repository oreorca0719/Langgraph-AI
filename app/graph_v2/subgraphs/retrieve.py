"""
Retrieval subgraph — multi-query → retrieve → grade.

Subgraph 패턴 (LangGraph 권장):
  - 내부 state는 main GraphState의 부분집합
  - 외부에서 보면 single node로 동작
  - 내부 변경(multi-query 알고리즘 교체 등)이 main graph에 영향 0

흐름:
  query_planner (variants 생성, optional)
    → fan-out (Send API, 병렬 retrieve)
    → merge (dedupe, entity-aware boost)
    → grade
    → (rewrite 필요 시 query_planner로 회귀, 한도까지)
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.graph_v2.states.state import GraphState
from app.graph_v2.nodes.grader import grader_node, route_after_grade


def _query_planner_node(state: GraphState) -> dict:
    """질문 변형 생성 (optional). question_type 기반 분기:
    - exact_phrase / numerical / fill_blank: 변형 안 함 (정확 매칭이 중요)
    - reasoning / comparison: 변형 3개 생성
    """
    # TODO: question_type별 분기 구현
    return {"decision_path": ["query_plan:single"]}


def _retrieve_node(state: GraphState) -> dict:
    """RetrieverRegistry에서 적합한 retriever 선택 + 호출.
    현재는 chroma_hybrid 단일. 추후 SQL/Web retriever 추가 시 router가 선택.
    """
    # TODO: registry.get(...).retrieve(query, top_k)
    # TODO: 결과를 state.retrieved_docs에 저장
    return {"retrieved_docs": [], "decision_path": ["retrieve:hybrid"]}


def _rewrite_node(state: GraphState) -> dict:
    """Grader가 fail 판정 시 query 재작성 (FR-302, FR-304)."""
    # TODO: LLM으로 query 재작성, retrieval_iterations 증가
    return {"retrieval_iterations": state.retrieval_iterations + 1, "decision_path": [f"rewrite:{state.retrieval_iterations + 1}"]}


def build_retrieve_subgraph():
    """Retrieval subgraph 컴파일."""
    g = StateGraph(GraphState)
    g.add_node("query_planner", _query_planner_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("grade", grader_node)
    g.add_node("rewrite", _rewrite_node)

    g.set_entry_point("query_planner")
    g.add_edge("query_planner", "retrieve")
    g.add_edge("retrieve", "grade")

    def _route(state: GraphState) -> str:
        decision = route_after_grade(state)
        if decision == "rewrite" and state.retrieval_iterations < 3:
            return "rewrite"
        return "end"

    g.add_conditional_edges("grade", _route, {"rewrite": "rewrite", "end": END})
    g.add_edge("rewrite", "retrieve")

    return g.compile()
