from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import get_embeddings, ROUTER_TOP1_MIN, ROUTER_MARGIN_MIN
from app.core.config import EMAIL_DRAFT_HIGH_CONF_THRESHOLD
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

# ── Priority 3: 슬롯 감지용 상수 ──────────────────────────────
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_DEPT_NAMES = ["인사팀", "hr", "총무팀", "재무팀", "회계팀", "it지원팀", "it 지원팀",
               "마케팅팀", "영업팀", "개발팀", "법무팀", "기획팀", "경영지원팀"]

_RFP_SCOPE_HINTS = ["프로젝트", "시스템", "솔루션", "플랫폼", "구축", "개발", "도입",
                    "고도화", "전환", "운영", "서비스", "인프라", "제안", "설계", "구현",
                    "마이그레이션", "통합", "개선", "최적화", "리팩토링", "업그레이드"]

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


def _has_file_slot_reference(value: Any) -> bool:
    text = (value or "").strip() if isinstance(value, str) else ""
    return _has_file_path_hint(text)


def _has_recipient_hint(text: str) -> bool:
    """이메일 수신자 힌트: 이메일 주소 or 부서명 포함 여부."""
    if _EMAIL_RE.search(text or ""):
        return True
    if _contains_any(text, _DEPT_NAMES):
        return True
    return False


def _has_rfp_scope_hint(text: str) -> bool:
    """RFP 범위 신호: _RFP_SCOPE_HINTS 중 하나라도 있으면 통과."""
    value = (text or "").strip()
    lowered = value.lower()
    if _contains_any(lowered, _RFP_SCOPE_HINTS):
        return True

    english_scope_patterns = [
        r"\brfp\s+for\s+(?:a|an|the|new)?\s*[a-z0-9][a-z0-9\s\-]{2,}\b",
        r"\bfor\s+(?:vendor selection|cloud migration|erp|crm|system|platform|implementation|upgrade|integration|proposal)\b",
        r"\b(new|existing)\s+(erp|crm|system|platform|portal|service|solution)\b",
    ]
    return any(re.search(pattern, lowered) for pattern in english_scope_patterns)


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


def _quick_route(user_input: str) -> str:
    """
    Rule-based 빠른 라우팅 (Phase 3: 단순화).
    명확한 힌트가 있을 때만 확정 반환, 아니면 "unknown".
    """
    # 1. 이메일 편집 (우선도 최고: 기존 초안 수정)
    if _contains_any(user_input, _EMAIL_EDIT_HINTS):
        return "email_draft"

    # 2. 이메일 구조 감지 (to:/subject: 등 명시적 구조)
    if _has_email_struct(user_input):
        return "email_draft"

    # 3. 이메일 키워드 + 작성 힌트 (둘 다 있어야 확정)
    if (
        _contains_any(user_input, ["이메일", "메일", "email"])
        and _contains_any(user_input, _EMAIL_WRITE_HINTS)
    ):
        return "email_draft"

    # 4. RFP 명시적 키워드
    if _contains_any(user_input, ["rfp", "제안요청서", "요구사항", "요구사항 정의서"]):
        return "rfp_draft"

    # 5. RFP 범위 힌트 (프로젝트, 구축 등)
    if _has_rfp_scope_hint(user_input):
        return "rfp_draft"

    # 6. 파일 추출 (파일 경로 + 추출 힌트 동시 존재)
    if _has_file_path_hint(user_input) and _contains_any(user_input, _FILE_EXTRACT_HINTS):
        return "file_extract"

    # 명확하지 않음 → semantic으로 넘김
    return "unknown"


def _semantic_route(
    user_input: str,
    input_embedding: List[float] | None = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Semantic 라우팅 (Phase 3: 단순화).
    _quick_route()에서 "unknown"인 경우만 실행.
    """
    vectors = _load_sample_vectors()
    embeddings = get_embeddings()

    # Phase 2: 캐시 임베딩 사용
    if input_embedding:
        query_vec = input_embedding
    else:
        query_vec = embeddings.embed_query(user_input)

    ranked: List[Tuple[str, float]] = []
    for task, task_vectors in vectors.items():
        if not task_vectors:
            continue
        score = max(_cosine(query_vec, svec) for svec in task_vectors)
        ranked.append((task, score))

    if not ranked:
        return "unknown", {"mode": "semantic", "reason": "no_samples", "decision": "unknown"}

    ranked.sort(key=lambda x: x[1], reverse=True)
    top1_task, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = top1_score - top2_score

    top1_min = ROUTER_TOP1_MIN
    margin_min = ROUTER_MARGIN_MIN
    decision = top1_task

    if top1_score < top1_min or margin < margin_min:
        decision = "unknown"

    # email_draft 신뢰도 재검증 (semantic 결과가 email_draft일 때만)
    if decision == "email_draft":
        hint_present = (
            _has_email_struct(user_input)
            or _contains_any(user_input, _EMAIL_WRITE_HINTS)
            or _contains_any(user_input, _EMAIL_EDIT_HINTS)
        )
        if not hint_present and top1_score <= EMAIL_DRAFT_HIGH_CONF_THRESHOLD:
            decision = "unknown"

    # file_extract 신뢰도 재검증 (semantic 결과가 file_extract일 때만)
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
    return decision, debug


def task_router_node(state: GraphState) -> GraphState:
    print("--- [NODE] Task Router ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    task_args = state.get("task_args") or {}
    input_embedding = state.get("input_embedding")  # Phase 2: 캐시 임베딩

    if not user_input:
        return {"task_type": "knowledge_search", "task_args": task_args}

    try:
        # Step 1: Rule-based 빠른 라우팅
        routed = _quick_route(user_input)
        debug: Dict[str, Any] = {"mode": "rule_based", "final_source": "quick_route", "decision": routed}

        # Step 2: Semantic (quick_route 결과가 unknown이면만)
        if routed == "unknown":
            routed, debug = _semantic_route(user_input, input_embedding)

            # LLM Fallback (semantic도 unknown이면)
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
        file_path_present = _has_file_slot_reference(task_args.get("file_path"))
        pending_slots = [slot for slot in (task_args.get("missing_slots") or []) if slot in ("file_path", "file_context")]

        if routed == "unknown" and file_context_present:
            routed = "file_chat"
            debug["decision"] = "file_chat"
            debug["final_source"] = "file_context_fallback"

        if routed == "unknown" and pending_slots:
            if file_context_present:
                routed = "file_chat"
                debug["decision"] = "file_chat"
                debug["final_source"] = "pending_file_slot_resolved"
            elif file_path_present or _has_file_path_hint(user_input):
                routed = "file_extract"
                debug["decision"] = "file_extract"
                debug["final_source"] = "pending_file_slot_resolved"
            else:
                routed = "clarification"
                debug["decision"] = "clarification"
                debug["final_source"] = "pending_file_slot"
                debug["missing_slots"] = pending_slots

        if routed == "unknown" and file_path_present:
            routed = "file_extract"
            debug["decision"] = "file_extract"
            debug["final_source"] = "file_path_resume"

        if routed == "file_chat" and not file_context_present and file_path_present:
            routed = "file_extract"
            debug["decision"] = "file_extract"
            debug["final_source"] = "file_path_resume"

        # ?? Priority 3: ?? ?? ?? ???????????????????????????????
        missing_slots: list[str] = []

        # file_path: user_input?? ?????, task_args? ?? ???? ??
        if routed == "file_extract" and not _has_file_path_hint(user_input) and not file_path_present:
            missing_slots = ["file_path"]
        # file_context: state?? ?? ????, file_path? ??? file_extract? ?? ??
        elif routed == "file_chat" and not file_context_present and not file_path_present:
            missing_slots = ["file_context"]
        # to: user_input?? ?????, task_args? ?? ???? ??
        elif routed == "email_draft" and not _has_recipient_hint(user_input) and not task_args.get("to"):
            # email_edit ??? ??? ??? ?? draft ????? ??
            if not _contains_any(user_input, _EMAIL_EDIT_HINTS):
                missing_slots = ["to"]
        # project_scope: user_input?? ?????, task_args? ?? ???? ??
        elif routed == "rfp_draft" and not _has_rfp_scope_hint(user_input) and not task_args.get("project_scope"):
            missing_slots = ["project_scope"]

        if missing_slots:
            routed = "clarification"
            debug["decision"] = "clarification"
            debug["final_source"] = "slot_missing"
            debug["missing_slots"] = missing_slots

        merged_args = {**task_args, "routing_debug": debug}
        if missing_slots:
            merged_args["missing_slots"] = missing_slots

        return {"task_type": routed, "task_args": merged_args}

    except Exception as e:
        fallback = _quick_route(user_input)
        if fallback == "unknown":
            fallback = "knowledge_search"
        merged_args = {
            **task_args,
            "routing_debug": {
                "mode": "rule_fallback", "decision": fallback,
                "final_source": "rule_fallback", "reason": str(e),
            },
        }
        return {"task_type": fallback, "task_args": merged_args}


def rejection_node(state: GraphState) -> Dict[str, Any]:
    print("--- [NODE] Rejection ---")
    user_input = (state.get("input_data") or "").strip()
    trace_id = (state.get("trace_id") or "")
    return {
        **state,
        "messages": [
            HumanMessage(content=user_input),
            AIMessage(content="해당 질문은 사내 AI 어시스턴트의 지원 범위에 포함되지 않아 답변을 제공하지 않습니다. 사내 업무 관련 질문을 입력해 주세요."),
        ],
        "citations_used": [],
    }


def _unknown_fallback_route(state: GraphState) -> str:
    """Unknown task를 AI 안내 또는 지식 검색으로 우선 실행."""
    user_input = (state.get("input_data") or "").strip()
    _AI_GUIDE_TRIGGERS = [
        "안녕", "반가워", "뭐 할 수 있어", "무슨 기능", "도움말", "help",
        "소개해", "뭐야", "누구야", "어떤 ai", "기능 안내", "사용법",
    ]
    if len(user_input) <= 15 or _contains_any(user_input, _AI_GUIDE_TRIGGERS):
        return "ai_guide"
    return "knowledge_search"


def route_by_task(state: GraphState) -> str:
    task = (state.get("task_type") or "knowledge_search").strip()
    if task == "chat":
        return "knowledge_search"
    if task in ("unknown", ""):
        return _unknown_fallback_route(state)
    if task == "injection":
        return "rejection"
    if task == "clarification":
        return "clarification"
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
