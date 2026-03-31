from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import get_embeddings, ROUTER_TOP1_MIN, ROUTER_MARGIN_MIN
from app.core import trace_buffer
from app.core.history_utils import cosine as _cosine
from app.graph.states.state import GraphState
from app.graph.nodes.llm_intent_fallback import llm_intent_fallback
from app.auth.intent_samples import load_all_samples, add_sample

_ALLOWED = {"knowledge_search", "ai_guide", "file_chat", "file_extract", "email_draft", "rfp_draft", "detail_search", "planner"}

_DETAIL_HINTS = ["자세히", "자세하게", "더 알려줘", "좀 더", "구체적", "상세히", "상세하게", "더 설명", "추가로 설명", "자세한 내용", "더 자세한"]

_EMAIL_STRUCT_RE = re.compile(r"(수신자\s*:|제목\s*:|내용\s*:|to\s*:|subject\s*:|body\s*:)", re.I)
_FILE_EXT_RE = re.compile(r"\.(pdf|docx|pptx|xlsx|txt|md)\b", re.I)

_EMAIL_WRITE_HINTS = ["작성", "초안", "문안", "써줘", "draft", "compose", "작성해", "작성해줘"]
_EMAIL_EDIT_HINTS = ["수정", "변경", "바꿔", "고쳐", "초안에서", "이 초안", "그 초안", "방금 메일", "메일에서"]
_FILE_EXTRACT_HINTS = ["추출", "파싱", "텍스트", "본문", "extract", "parse"]

# (선행 작업 패턴, 후행 작업 패턴): 두 조건이 동시에 존재하면 복합 요청으로 판단
_COMPOUND_SIGNALS: List[Tuple[List[str], List[str]]] = [
    (["검색", "조회", "찾아", "알아봐", "확인해"],  ["rfp", "제안요청서", "이메일", "메일", "초안", "작성해"]),
    (["분석", "읽어", "요약"],                      ["rfp", "이메일", "초안", "작성해"]),
]

_SAMPLE_VECTORS: Dict[str, List[List[float]]] | None = None
_SAMPLE_LOCK = threading.Lock()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _is_compound_request(text: str) -> bool:
    """선행 검색 + 후행 작성 패턴이 동시에 존재하면 복합 요청으로 판단."""
    for lead, follow in _COMPOUND_SIGNALS:
        if _contains_any(text, lead) and _contains_any(text, follow):
            return True
    return False


def _has_email_struct(text: str) -> bool:
    return bool(_EMAIL_STRUCT_RE.search(text or ""))


def _has_file_path_hint(text: str) -> bool:
    t = text or ""
    if _FILE_EXT_RE.search(t):
        return True
    if any(x in t for x in ["./", ".\\", "/", "\\"]):
        return True
    if _contains_any(t, ["경로", "file_path", "filepath", "파일경로"]):
        return True
    return False


def invalidate_sample_cache() -> None:
    global _SAMPLE_VECTORS
    with _SAMPLE_LOCK:
        _SAMPLE_VECTORS = None


def _load_sample_vectors() -> Dict[str, List[List[float]]]:
    global _SAMPLE_VECTORS
    if _SAMPLE_VECTORS is not None:
        return _SAMPLE_VECTORS
    with _SAMPLE_LOCK:
        if _SAMPLE_VECTORS is not None:
            return _SAMPLE_VECTORS
        samples = load_all_samples()
        embeddings = get_embeddings()
        vectors: Dict[str, List[List[float]]] = {}
        for task, texts in samples.items():
            if task not in _ALLOWED:
                continue
            vectors[task] = embeddings.embed_documents(texts)
        _SAMPLE_VECTORS = vectors
        return _SAMPLE_VECTORS


def _rule_based_route(user_input: str) -> str:
    if _contains_any(user_input, _EMAIL_EDIT_HINTS):
        return "email_draft"
    if _has_email_struct(user_input) or _contains_any(user_input, ["이메일", "메일", "email"]):
        return "email_draft"
    if _contains_any(user_input, ["rfp", "제안요청서", "요구사항", "요구사항 정의서"]):
        return "rfp_draft"
    has_extract_intent = _contains_any(user_input, ["추출", "텍스트 추출", "내용 추출", "파싱", "읽어줘", "extract"])
    if has_extract_intent and _has_file_path_hint(user_input):
        return "file_extract"
    return "knowledge_search"


def _semantic_route(user_input: str) -> Tuple[str, Dict[str, Any], List[float]]:
    if _contains_any(user_input, _EMAIL_EDIT_HINTS):
        debug = {
            "mode": "semantic", "top1_task": "email_draft", "top1_score": 1.0,
            "top2_score": 0.0, "margin": 1.0, "decision": "email_draft",
            "reason": "email_edit_hint", "ranked": [{"task": "email_draft", "score": 1.0}],
        }
        return "email_draft", debug, []

    if (
        _has_email_struct(user_input)
        or (
            _contains_any(user_input, ["이메일", "메일", "email"])
            and _contains_any(user_input, _EMAIL_WRITE_HINTS)
        )
    ):
        debug = {
            "mode": "semantic", "top1_task": "email_draft", "top1_score": 1.0,
            "top2_score": 0.0, "margin": 1.0, "decision": "email_draft",
            "reason": "email_write_hint", "ranked": [{"task": "email_draft", "score": 1.0}],
        }
        return "email_draft", debug, []

    vectors = _load_sample_vectors()
    embeddings = get_embeddings()
    query_vec = embeddings.embed_query(user_input)

    ranked: List[Tuple[str, float]] = []
    for task, task_vectors in vectors.items():
        if not task_vectors:
            continue
        score = max(_cosine(query_vec, svec) for svec in task_vectors)
        ranked.append((task, score))

    if not ranked:
        return "unknown", {"reason": "no_samples"}, []

    ranked.sort(key=lambda x: x[1], reverse=True)
    top1_task, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = top1_score - top2_score

    top1_min = ROUTER_TOP1_MIN
    margin_min = ROUTER_MARGIN_MIN
    decision = top1_task

    if top1_score < top1_min or margin < margin_min:
        decision = "unknown"

    if decision == "email_draft":
        if not (
            _has_email_struct(user_input)
            or _contains_any(user_input, _EMAIL_WRITE_HINTS)
            or _contains_any(user_input, _EMAIL_EDIT_HINTS)
        ):
            decision = "unknown"

    if decision == "file_extract":
        if not (_has_file_path_hint(user_input) and _contains_any(user_input, _FILE_EXTRACT_HINTS)):
            decision = "unknown"

    debug = {
        "mode": "semantic",
        "top1_task": top1_task,
        "top1_score": round(top1_score, 4),
        "top2_score": round(top2_score, 4),
        "margin": round(margin, 4),
        "decision": decision,
        "threshold_top1": top1_min,
        "threshold_margin": margin_min,
        "ranked": [{"task": t, "score": round(s, 4)} for t, s in ranked[:4]],
    }
    return decision, debug, query_vec


def task_router_node(state: GraphState) -> GraphState:
    print("--- [NODE] Task Router ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    task_args = state.get("task_args") or {}

    trace_buffer.push(trace_id, node="task_router", event="enter", label="execute",
                      data={"input": user_input[:200]})

    if not user_input:
        trace_buffer.push(trace_id, node="task_router", event="exit", label="execute",
                          data={"task_type": "knowledge_search", "mode": "empty_input"})
        return {"task_type": "knowledge_search", "task_args": task_args}

    try:
        routed, debug, query_vec = _semantic_route(user_input)

        if routed == "unknown":
            llm_task, llm_debug = llm_intent_fallback(user_input)
            debug["llm_fallback"] = llm_debug
            if llm_task != "unknown":
                routed = llm_task
                debug["decision"] = llm_task
                debug["final_source"] = "llm_fallback"
                added = add_sample(llm_task, user_input, source="llm_fallback")
                if added:
                    invalidate_sample_cache()
            else:
                debug["final_source"] = "unknown"
        else:
            debug["final_source"] = "semantic"

        # 복합 요청 감지: 검색 + 작성 패턴이 동시 존재 (injection/file 계열 제외)
        if routed not in ("injection", "file_chat", "file_extract") and _is_compound_request(user_input):
            routed = "planner"
            debug["decision"] = "planner"
            debug["final_source"] = "compound_request_detected"

        # 상세 검색 감지: 후속 심화 패턴 + 직전 task가 knowledge_search/detail_search + 메시지 존재
        if routed in ("unknown", "knowledge_search"):
            prev_task = (state.get("task_type") or "").strip()
            prev_messages = state.get("messages") or []
            if (
                prev_task in ("knowledge_search", "detail_search")
                and prev_messages
                and _contains_any(user_input, _DETAIL_HINTS)
            ):
                routed = "detail_search"
                debug["decision"] = "detail_search"
                debug["final_source"] = "detail_search_detected"

        # file_context 있는데 unknown이면 file_chat으로 fallback
        file_context_present = bool((state.get("file_context") or "").strip())
        if routed == "unknown" and file_context_present:
            routed = "file_chat"
            debug["decision"] = "file_chat"
            debug["final_source"] = "file_context_fallback"

        merged_args = {**task_args, "routing_debug": debug}

        trace_buffer.push(trace_id, node="task_router", event="exit", label="execute",
                          data={
                              "task_type": routed,
                              "mode": debug.get("mode", ""),
                              "final_source": debug.get("final_source", ""),
                              "top1_score": debug.get("top1_score", ""),
                              "margin": debug.get("margin", ""),
                          })
        return {"task_type": routed, "task_args": merged_args}

    except Exception as e:
        fallback = _rule_based_route(user_input)
        merged_args = {
            **task_args,
            "routing_debug": {
                "mode": "rule_fallback", "decision": fallback,
                "final_source": "rule_fallback", "reason": str(e),
            },
        }
        trace_buffer.push(trace_id, node="task_router", event="exit", label="execute",
                          data={"task_type": fallback, "mode": "rule_fallback", "error": str(e)})
        return {"task_type": fallback, "task_args": merged_args}


def rejection_node(state: GraphState) -> Dict[str, Any]:
    print("--- [NODE] Rejection ---")
    user_input = (state.get("input_data") or "").strip()
    trace_id = (state.get("trace_id") or "")
    trace_buffer.push(trace_id, node="rejection", event="exit", label="execute",
                      data={"input": user_input[:200]})
    return {
        **state,
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content="해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요."),
        ],
        "citations_used": [],
    }


def route_by_task(state: GraphState) -> str:
    task = (state.get("task_type") or "knowledge_search").strip()
    if task == "chat":
        return "knowledge_search"
    if task in ("unknown", ""):
        return "clarification"
    if task == "injection":
        return "rejection"
    if task == "detail_search":
        return "detail_search"
    if task == "planner":
        return "planner"
    if task in _ALLOWED:
        return task
    return "knowledge_search"


def route_after_input_guard(state: GraphState) -> str:
    task = (state.get("task_type") or "").strip()
    if task == "injection":
        return "rejection"
    return "task_router"
