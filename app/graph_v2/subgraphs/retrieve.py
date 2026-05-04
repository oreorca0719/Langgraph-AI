"""
Retrieval subgraph — query_planner → retrieve → grade → (rewrite if needed).

Phase C 학습 반영:
  - multi-query는 question_type 분기로만 (verbatim 질문엔 끔 — K 카테고리 -41pp 회귀 방지)
  - reasoning / comparison 질문에만 query 변형 생성

흐름:
  query_planner → retrieve → grade → END (또는 rewrite로 회귀)
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from app.core.config import get_llm
from app.graph_v2.states.state import GraphState
from app.graph_v2.retrievers.base import Document, RetrieverRegistry
from app.graph_v2.retrievers.chroma_hybrid import ChromaHybridRetriever
from app.graph_v2.nodes.grader import grader_node, route_after_grade


# ────────────────────────────────────────────────────────────
# Retriever Registry — 현재는 chroma_hybrid 단일.
# 추후 sql_retriever / web_retriever 추가 시 여기 등록.
# ────────────────────────────────────────────────────────────

_registry: Optional[RetrieverRegistry] = None


def _get_registry() -> RetrieverRegistry:
    global _registry
    if _registry is None:
        r = RetrieverRegistry()
        r.register(ChromaHybridRetriever())
        _registry = r
    return _registry


# ────────────────────────────────────────────────────────────
# Query planner (Phase C 학습 — question_type 분기로 multi-query 적용)
# ────────────────────────────────────────────────────────────

# multi-query를 적용할 question_type (검색 표현 다양화가 도움되는 case)
_MULTI_QUERY_TYPES = {"reasoning", "comparison", "list_n"}

_VARIANT_PROMPT = """사용자 질문을 사내 문서 검색에 적합한 검색 쿼리 변형 2개로 만들어주세요.

원칙:
- 원본 질문의 핵심 단어(고유명사·수치·시간)는 변형에도 포함
- 같은 의미를 다른 표현·키워드 조합으로

출력: JSON 배열만, 다른 텍스트 금지
["변형1", "변형2"]

원본 질문: {q}"""


def _generate_variants(query: str, n: int = 2) -> list[str]:
    """LLM으로 query 변형 생성. 실패 시 원본만 반환."""
    try:
        resp = get_llm().invoke([
            HumanMessage(content=_VARIANT_PROMPT.format(q=query)),
        ])
        raw = resp.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        raw = str(raw).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("`").strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            out = [str(v).strip() for v in variants if str(v).strip()][:n]
            return [query] + out
    except Exception as e:
        print(f"[QUERY_PLANNER] failed (non-fatal): {e}")
    return [query]


def query_planner_node(state: GraphState) -> dict:
    """question_type 기반 multi-query 분기.

    - exact_phrase / numerical / fill_blank: 변형 안 함 (정확 매칭이 중요)
    - reasoning / comparison / list_n: 변형 2개 추가
    """
    qtype = state.question_type or "reasoning"
    user_input = (state.input_data or "").strip()

    if qtype not in _MULTI_QUERY_TYPES:
        # 단일 query (변형 없음)
        return {
            "decision_path": [f"query_plan:single({qtype})"],
        }

    variants = _generate_variants(user_input, n=2)
    # 변형은 sub_questions 필드에 임시 저장 (retrieve 노드가 사용)
    return {
        "sub_questions": variants,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [f"query_plan:multi({len(variants)},{qtype})"],
    }


# ────────────────────────────────────────────────────────────
# Retrieve node
# ────────────────────────────────────────────────────────────

def retrieve_node(state: GraphState) -> dict:
    """Retriever Protocol 통해 검색. 단일 또는 multi-query."""
    queries = state.sub_questions or [state.input_data]
    queries = [q for q in queries if q]

    # 현재는 chroma_hybrid 단일 retriever 사용
    retriever = _get_registry().get("chroma_hybrid")

    pool: dict[str, Document] = {}
    for q in queries:
        try:
            for d in retriever.retrieve(q, top_k=5):
                key = d.content
                if key in pool:
                    # 동일 chunk가 여러 query에서 나오면 score 누적 (max + 0.1 boost)
                    pool[key] = Document(
                        content=d.content,
                        source=d.source,
                        score=min(1.0, max(pool[key].score, d.score) + 0.05),
                        metadata=d.metadata,
                    )
                else:
                    pool[key] = d
        except Exception as e:
            print(f"[RETRIEVE] '{q[:40]}' failed (non-fatal): {e}")

    docs = sorted(pool.values(), key=lambda d: d.score, reverse=True)[:5]

    return {
        "retrieved_docs": docs,
        "decision_path": [f"retrieve:{len(queries)}q→{len(docs)}docs"],
    }


# ────────────────────────────────────────────────────────────
# Rewrite node (FR-302, FR-304)
# ────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """사용자의 검색 쿼리가 사내 문서를 충분히 찾지 못했습니다.
같은 의미를 더 일반적·핵심적인 키워드로 재작성해주세요.

원칙:
- 핵심 entity(고유명사·수치·시간)는 그대로 유지
- 동의어·상위어 활용
- 부수 단어 제거

출력: 재작성된 쿼리만 한 줄. 다른 텍스트 금지.

원본 쿼리: {q}"""


def rewrite_node(state: GraphState) -> dict:
    """Grader fail 시 query 재작성."""
    user_input = (state.input_data or "").strip()
    try:
        resp = get_llm().invoke([HumanMessage(content=_REWRITE_PROMPT.format(q=user_input))])
        raw = resp.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        new_query = str(raw).strip().splitlines()[0] if raw else user_input
        if not new_query or new_query == user_input:
            new_query = user_input
    except Exception as e:
        print(f"[REWRITE] failed (non-fatal): {e}")
        new_query = user_input

    return {
        "input_data": new_query,
        "sub_questions": [],  # 변형 reset (다음 retrieve가 새 쿼리로 단일 검색)
        "retrieval_iterations": state.retrieval_iterations + 1,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [f"rewrite:{state.retrieval_iterations + 1}"],
    }


# ────────────────────────────────────────────────────────────
# Subgraph builder
# ────────────────────────────────────────────────────────────

def _route_after_grade_with_limit(state: GraphState) -> str:
    decision = route_after_grade(state)
    if decision == "rewrite" and state.retrieval_iterations < 3:
        return "rewrite"
    return "end"


def build_retrieve_subgraph():
    g = StateGraph(GraphState)
    g.add_node("query_planner", query_planner_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grader_node)
    g.add_node("rewrite", rewrite_node)

    g.set_entry_point("query_planner")
    g.add_edge("query_planner", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade_with_limit, {"rewrite": "rewrite", "end": END})
    g.add_edge("rewrite", "retrieve")

    return g.compile()
