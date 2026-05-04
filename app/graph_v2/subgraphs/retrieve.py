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
from app.knowledge.chunking.tagger import extract_entities


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
    """Retriever Protocol 통해 검색. 단일 또는 multi-query.

    Phase G 추가:
      1. 기본 hybrid 검색 후
      2. expand_with_same_page: 같은 page_unit의 다른 chunks 함께 fetch
      3. boost_by_query_context: query qtype 일치 chunks score boost
         (doc_topic boost는 router가 query_doc_topic 분류 시 활성화)
    """
    queries = state.sub_questions or [state.input_data]
    queries = [q for q in queries if q]

    retriever = _get_registry().get("chroma_hybrid")

    pool: dict[str, Document] = {}
    for q in queries:
        try:
            for d in retriever.retrieve(q, top_k=5):
                key = d.content
                if key in pool:
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

    # Phase G: 같은 page의 다른 chunks 함께 fetch (표·본문 함께 보기)
    if hasattr(retriever, "expand_with_same_page"):
        try:
            docs = retriever.expand_with_same_page(docs)
        except Exception as e:
            print(f"[RETRIEVE] co-retrieval failed (non-fatal): {e}")

    # Phase G: doc_topic boost (router가 query_doc_topic 분류 시 활성화)
    if hasattr(retriever, "boost_by_query_context"):
        try:
            docs = retriever.boost_by_query_context(docs)
        except Exception as e:
            print(f"[RETRIEVE] boost failed (non-fatal): {e}")

    # 최종 top 7 (co-retrieval로 늘었으니 약간 더)
    docs = docs[:7]

    return {
        "retrieved_docs": docs,
        "decision_path": [f"retrieve:{len(queries)}q→{len(docs)}docs(co+boost)"],
    }


# ────────────────────────────────────────────────────────────
# Rewrite node (FR-302, FR-304) — Negative Feedback 주입
# ────────────────────────────────────────────────────────────
#
# 두 케이스 분기:
#  Case A — 직전 retrieve가 빈 결과 (또는 entity 추출 실패)
#           → 일반화 prompt
#  Case B — 직전 chunks 있었으나 grader가 모두 irrelevant 판정
#           → extraneous entities (chunk_entities − query_entities) 를 회피 키워드로 명시
#
# anchor: state.original_input (router에서 1회 초기화, 이후 불변)
# rewrite 결과: state.input_data 만 갱신, original_input은 보존

_PROMPT_NEGATIVE_FEEDBACK = """당신은 사내 RAG 시스템의 query rewriter입니다. 직전 검색이 정답을 못 찾았으니 다른 방향으로 재작성합니다.

[원본 사용자 질문]
{original}

[직전 시도 쿼리]
{current}

[직전 retrieve 결과 요약 — grader가 모두 무관 판정 ({iter}회차)]
- 회피해야 할 키워드: {extraneous}
- 가져온 문서 주제: {topics}

【재작성 원칙】
1. 원본 질문의 핵심 entity ({query_entities}) 는 그대로 유지
2. 위 "회피 키워드" 방향으로 가는 표현은 사용 금지
3. 같은 의도를 다른 어휘 조합으로 — 동의어·구체화·상위어
4. 군더더기 제거, 검색에 효과적인 핵심어만

【출력】 재작성된 쿼리만 한 줄. 다른 텍스트 금지.
"""

_PROMPT_EMPTY = """직전 retrieve가 결과를 반환하지 못했습니다.
- 너무 좁거나 구체적인 표현일 가능성
- 동의어·상위어로 확장 필요

[원본 사용자 질문]
{original}

[직전 시도 쿼리]
{current}

【재작성 원칙】
1. 핵심 entity 유지
2. 좁은 표현을 더 일반적인 키워드로 (예: "당사 슬로건 문구" → "슬로건")
3. 군더더기 제거

【출력】 재작성된 쿼리만 한 줄. 다른 텍스트 금지.
"""


def _collect_chunk_entities(docs: list[Document]) -> set[str]:
    """retrieved_docs의 metadata.entities (공백 join 문자열) 합집합."""
    out: set[str] = set()
    for d in docs:
        ent_str = (d.metadata or {}).get("entities", "") or ""
        for e in ent_str.split():
            e = e.strip().lower()
            if e:
                out.add(e)
    return out


def _collect_chunk_topics(docs: list[Document]) -> set[str]:
    out: set[str] = set()
    for d in docs:
        t = (d.metadata or {}).get("doc_topic", "") or ""
        if t and t != "general":
            out.add(t)
    return out


def _parse_rewrite_response(raw, original_input: str) -> str:
    """LLM 응답에서 첫 줄 추출 + fallback 처리."""
    if isinstance(raw, list):
        raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
    text = str(raw or "").strip()
    if not text:
        return original_input
    first_line = text.splitlines()[0].strip()
    if not first_line:
        return original_input
    # 따옴표 / 마크다운 fence 제거
    first_line = first_line.strip("`\"'“”").strip()
    if not first_line or first_line == original_input.strip():
        return original_input
    return first_line


def rewrite_node(state: GraphState) -> dict:
    """Grader가 모두 irrelevant 판정 시 query 재작성.

    Negative feedback:
      - 직전 chunks의 entity union − query entity = extraneous (회피 대상)
      - LLM에 "이 방향은 비껴갔으니 다른 표현으로" 명시
    """
    original = (state.original_input or state.input_data or "").strip()
    current = (state.input_data or "").strip()
    iter_next = state.retrieval_iterations + 1
    docs: list[Document] = state.retrieved_docs or []

    # 입력 정보 수집
    query_entities = set(extract_entities(original))
    chunk_entities = _collect_chunk_entities(docs)
    chunk_topics = _collect_chunk_topics(docs)
    extraneous = sorted(chunk_entities - {e.lower() for e in query_entities})

    # Case 분기
    use_neg_feedback = bool(docs) and bool(extraneous)
    if use_neg_feedback:
        prompt = _PROMPT_NEGATIVE_FEEDBACK.format(
            original=original,
            current=current,
            iter=iter_next,
            extraneous=", ".join(extraneous[:8]) or "(없음)",
            topics=", ".join(sorted(chunk_topics)[:3]) or "(미분류)",
            query_entities=", ".join(sorted(query_entities)[:5]) or "(추출 실패)",
        )
        case_label = "B"
    else:
        prompt = _PROMPT_EMPTY.format(original=original, current=current)
        case_label = "A"

    # LLM 호출
    fallback = False
    try:
        resp = get_llm().invoke([HumanMessage(content=prompt)])
        new_query = _parse_rewrite_response(resp.content, original)
        if new_query == original or new_query == current:
            fallback = True
            new_query = original
    except Exception as e:
        print(f"[REWRITE] failed (non-fatal): {e}")
        new_query = original
        fallback = True

    label_suffix = (
        f"rewrite:{case_label}({iter_next},avoid={len(extraneous)})"
        if not fallback else
        f"rewrite:fallback({iter_next})"
    )

    return {
        "input_data": new_query,
        "sub_questions": [],  # 변형 reset (다음 retrieve가 새 쿼리로 단일 검색)
        "retrieval_iterations": iter_next,
        "llm_call_count": state.llm_call_count + 1,
        "decision_path": [label_suffix],
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
