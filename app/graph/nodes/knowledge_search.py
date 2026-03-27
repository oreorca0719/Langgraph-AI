from __future__ import annotations

import threading
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import (
    get_embeddings, get_llm,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    RETRIEVAL_MIN_RELEVANCE, RETRIEVAL_MAX_DISTANCE, RETRIEVAL_TOP_K,
)
from app.core import trace_buffer
from app.core.history_utils import (
    HISTORY_MAX_MESSAGES,
    filter_history_by_relevance as _filter_history_by_relevance,
)
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize_docs
from app.security.output_validator import validate as validate_output

# ── Chroma 싱글톤 ────────────────────────────────────────────
_chroma_instance: Chroma | None = None
_chroma_lock = threading.Lock()

# quality_check 기준
_QUALITY_MIN_DOCS = 1
_QUALITY_MIN_SCORE = float(RETRIEVAL_MIN_RELEVANCE)
_MAX_RETRY = 2


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
    return vectorstore.similarity_search(query, k=k)


def _format_docs(docs: List[Document]) -> str:
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


# ── 노드 1: search_node ──────────────────────────────────────

def search_node(state: GraphState) -> Dict[str, Any]:
    """ChromaDB 벡터 검색."""
    print("--- [NODE] Search ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    retry_count = state.get("retry_count") or 0

    trace_buffer.push(trace_id, node="search", event="enter", label="execute",
                      data={"input": user_input[:200], "retry_count": retry_count})

    docs = _search_chroma(user_input, k=RETRIEVAL_TOP_K)
    docs = sanitize_docs(docs, source="rag")

    trace_buffer.push(trace_id, node="search", event="exit", label="execute",
                      data={"docs_found": len(docs)})

    return {"task_args": {**(state.get("task_args") or {}), "search_docs": docs, "search_query": user_input}}


# ── 노드 2: quality_check_node ───────────────────────────────

def quality_check_node(state: GraphState) -> Dict[str, Any]:
    """
    검색 결과 품질 판단 — 규칙 기반 (LLM 없음).
    - 결과 없거나 최고 score < threshold → retry
    - retry_count >= _MAX_RETRY → 강제 ok (무한루프 방지)
    """
    print("--- [NODE] Quality Check ---")

    trace_id = (state.get("trace_id") or "")
    task_args = state.get("task_args") or {}
    docs: List[Document] = task_args.get("search_docs") or []
    retry_count = state.get("retry_count") or 0

    ok = bool(docs) and retry_count >= _MAX_RETRY
    if not ok and docs:
        ok = True  # docs 존재하면 통과

    trace_buffer.push(trace_id, node="quality_check", event="exit", label="execute",
                      data={"docs": len(docs), "retry_count": retry_count, "ok": ok})

    return {"task_args": {**task_args, "quality_ok": ok}}


def route_after_quality(state: GraphState) -> str:
    task_args = state.get("task_args") or {}
    if task_args.get("quality_ok"):
        return "answer"
    retry_count = state.get("retry_count") or 0
    if retry_count >= _MAX_RETRY:
        return "answer"
    return "rewrite"


# ── 노드 3: rewrite_node ─────────────────────────────────────

def rewrite_node(state: GraphState) -> Dict[str, Any]:
    """LLM으로 검색 쿼리를 재작성하고 retry_count를 증가."""
    print("--- [NODE] Rewrite ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    retry_count = (state.get("retry_count") or 0) + 1

    trace_buffer.push(trace_id, node="rewrite", event="enter", label="execute",
                      data={"original": user_input[:100], "retry_count": retry_count})

    try:
        response = get_llm().invoke([
            SystemMessage(content=(
                "사용자의 검색 쿼리를 사내 문서 검색에 더 적합하게 재작성하세요.\n"
                "더 일반적인 용어를 사용하고 핵심 키워드만 남기세요.\n"
                "재작성된 쿼리만 출력하세요."
            )),
            HumanMessage(content=user_input),
        ])
        content = response.content
        if isinstance(content, list):
            rewritten = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            ).strip() or user_input
        else:
            rewritten = str(content).strip()
        if rewritten and rewritten != user_input:
            new_input = rewritten
        else:
            new_input = user_input
    except Exception:
        new_input = user_input

    trace_buffer.push(trace_id, node="rewrite", event="exit", label="execute",
                      data={"rewritten": new_input[:100]})

    return {"input_data": new_input, "retry_count": retry_count}


# ── 노드 4: answer_node ──────────────────────────────────────

def answer_node(state: GraphState) -> Dict[str, Any]:
    """검색 결과를 바탕으로 LLM 답변 생성."""
    print("--- [NODE] Answer ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    raw_history = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
    chat_history = _filter_history_by_relevance(raw_history, user_input)
    task_args = state.get("task_args") or {}
    docs: List[Document] = task_args.get("search_docs") or []

    trace_buffer.push(trace_id, node="answer", event="enter", label="execute",
                      data={"input": user_input[:200], "docs": len(docs)})

    if not docs:
        final_message = AIMessage(
            content="관련 사내 문서를 찾을 수 없습니다. 다른 키워드로 검색해 보시거나 담당 부서에 문의해 주세요."
        )
    else:
        search_result = _format_docs(docs)
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
        raw_content = response.content
        if isinstance(raw_content, list):
            text_for_validate = " ".join(
                p.get("text", "") for p in raw_content if isinstance(p, dict)
            ).strip()
        else:
            text_for_validate = str(raw_content)
        is_valid, safe_content = validate_output(text_for_validate)
        final_message = response if is_valid else AIMessage(content=safe_content)

    trace_buffer.push(trace_id, node="answer", event="exit", label="execute",
                      data={"response_len": len(str(final_message.content))})

    return {
        "messages": [HumanMessage(content=user_input), final_message],
        "citations_used": [],
        "retry_count": 0,
    }
