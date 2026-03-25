from __future__ import annotations

import os
import threading
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.config import (
    get_embeddings, get_llm,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    RETRIEVAL_MIN_RELEVANCE, RETRIEVAL_MAX_DISTANCE, RETRIEVAL_TOP_K,
)
from app.core import trace_buffer
from app.core.history_utils import HISTORY_MAX_MESSAGES, cosine as _cosine, filter_history_by_relevance as _filter_history_by_relevance
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize_docs
from app.security.output_validator import validate as validate_output

# ── Chroma 싱글톤 ──
_chroma_instance: Chroma | None = None
_chroma_lock = threading.Lock()


def _get_chroma() -> Chroma:
    global _chroma_instance
    if _chroma_instance is None:
        with _chroma_lock:
            if _chroma_instance is None:
                _chroma_instance = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=get_embeddings(),
                    collection_name=CHROMA_COLLECTION,
                )
    return _chroma_instance


def _search_chroma(query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
    vectorstore = _get_chroma()
    min_relevance = RETRIEVAL_MIN_RELEVANCE
    max_distance = RETRIEVAL_MAX_DISTANCE

    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        pairs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [doc for doc, score in pairs if score >= min_relevance]

    if hasattr(vectorstore, "similarity_search_with_score"):
        pairs = vectorstore.similarity_search_with_score(query, k=k)
        return [doc for doc, score in pairs if score <= max_distance]

    return vectorstore.similarity_search(query, k=k)


def _format_search_result(docs: List[Document]) -> str:
    if not docs:
        return ""
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip()
        md = getattr(d, "metadata", {}) or {}
        title = md.get("title") or md.get("file_name") or "문서"
        page_num = md.get("page_number")
        location = f" (p.{page_num})" if page_num else ""
        blocks.append(f"[{i}] {title}{location}\n{text[:1200]}")
    return "\n\n".join(blocks)




def _rewrite_query(original_query: str) -> str:
    """검색 결과가 없을 때 LLM으로 쿼리를 재작성합니다."""
    try:
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM
        response = get_llm().invoke([
            _SM(content=(
                "사용자의 검색 쿼리를 사내 문서 검색에 더 적합하게 재작성하세요.\n"
                "더 일반적인 용어를 사용하고, 핵심 키워드만 남기세요.\n"
                "재작성된 쿼리만 출력하세요."
            )),
            _HM(content=original_query),
        ])
        rewritten = str(response.content).strip()
        return rewritten if rewritten and rewritten != original_query else ""
    except Exception:
        return ""


def build_knowledge_search_subgraph():
    """
    사내 문서 검색 전용 서브그래프.
    LLM이 도구 선택을 판단하지 않고 코드에서 직접 RAG 검색 후 LLM 요약.
    검색 결과 없으면 쿼리 재작성 후 1회 재시도.
    """

    def knowledge_search_node(state: GraphState) -> Dict[str, Any]:
        print("--- [NODE] Knowledge Search ---")

        trace_id   = (state.get("trace_id") or "")
        user_input = (state.get("input_data") or "").strip()
        raw_history  = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
        chat_history = _filter_history_by_relevance(raw_history, user_input)
        k = RETRIEVAL_TOP_K

        trace_buffer.push(trace_id, node="knowledge_search", event="enter", label="execute",
                          data={"input": user_input[:200],
                                "history_raw": len(raw_history),
                                "history_filtered": len(chat_history)})

        # 1차 RAG 검색
        docs = _search_chroma(user_input, k=k)
        docs = sanitize_docs(docs, source="rag")

        # 결과 없으면 쿼리 재작성 후 1회 재시도
        if not docs:
            rewritten_query = _rewrite_query(user_input)
            if rewritten_query:
                print(f"DEBUG: [Knowledge Search] 쿼리 재작성: {rewritten_query[:80]}")
                trace_buffer.push(trace_id, node="knowledge_search", event="call", label="rewrite",
                                  data={"rewritten_query": rewritten_query[:80]})
                docs = _search_chroma(rewritten_query, k=k)
                docs = sanitize_docs(docs, source="rag_retry")

        search_result = _format_search_result(docs)

        trace_buffer.push(trace_id, node="knowledge_search", event="call", label="execute",
                          data={"docs_found": len(docs)})

        if not docs:
            final_message = AIMessage(
                content="관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."
            )
        else:
            system_content = (
                "당신은 Kaiper AI입니다. 아래 사내 문서 검색 결과를 바탕으로 사용자 질문에 답하세요.\n\n"
                "【응답 원칙】\n"
                "• 검색 결과가 사용자 질문과 직접 관련 없으면 '관련 사내 문서를 찾을 수 없습니다'라고만 답하세요.\n"
                "• 사용자 질문 중심으로 핵심만 요약하세요 (원문 나열 금지)\n"
                "• 수치·날짜·고유명사는 원문 그대로, 출처는 [1][2] 형식으로 문장 끝에 표시\n"
                "• 정중한 비즈니스 어투, 불렛(•) 활용\n\n"
                f"【검색 결과】\n{search_result}"
            )
            messages = (
                [SystemMessage(content=system_content)]
                + chat_history
                + [HumanMessage(content=user_input)]
            )
            response = get_llm().invoke(messages)

            is_valid, safe_content = validate_output(str(response.content))
            final_message = response if is_valid else AIMessage(content=safe_content)

        response_len = len(str(final_message.content))
        print(f"DEBUG: [Knowledge Search] 응답 생성 완료 ({response_len}자)")
        trace_buffer.push(trace_id, node="knowledge_search", event="exit", label="execute",
                          data={"response_len": response_len})

        return {
            **state,
            "messages": [HumanMessage(content=user_input), final_message],
            "citations_used": [],
        }

    g = StateGraph(GraphState)
    g.add_node("knowledge_search", knowledge_search_node)
    g.set_entry_point("knowledge_search")
    g.add_edge("knowledge_search", END)
    return g.compile()
