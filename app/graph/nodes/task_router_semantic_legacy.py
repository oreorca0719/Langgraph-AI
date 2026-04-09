"""
[LEGACY / 고도화 참고용]

시맨틱 라우팅 방식 (v1) — 현재 프로덕션에서 사용하지 않습니다.

현재 라우터(task_router.py)는 LLM 구조화 출력 방식으로 교체되었습니다.
이 파일은 향후 다음과 같은 고도화 작업 시 참고 자료로 보존합니다:

  - 대용량 트래픽 환경에서 LLM 라우팅 비용 절감이 필요한 경우
  - 라우팅 경로가 크게 늘어나 임베딩 분류가 더 효율적인 경우
  - 오프라인/저지연 환경에서 LLM 호출 없이 라우팅이 필요한 경우

포함 내용:
  - 키워드 기반 규칙 라우터 (_rule_based_route)
  - 임베딩 코사인 유사도 라우터 (_semantic_route)
  - LLM 폴백 분류기 (llm_intent_fallback — 시맨틱 unknown 시 2차 분류)
  - 샘플 벡터 캐시 관리 (_load_sample_vectors, invalidate_sample_cache)
  - 슬롯 감지 헬퍼 (_has_recipient_hint, _has_rfp_scope_hint 등)

주의: 이 파일의 함수들은 그래프에 연결되어 있지 않습니다.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import get_llm, get_embeddings
from app.core.config import ROUTER_TOP1_MIN, ROUTER_MARGIN_MIN, EMAIL_DRAFT_HIGH_CONF_THRESHOLD
from app.core.config import LLM_FALLBACK_ENABLED, LLM_FALLBACK_TIMEOUT_SEC
from app.core.history_utils import cosine as _cosine, extract_text_content
from app.auth.intent_samples import load_all_samples, add_sample


_ALLOWED = {"knowledge_search", "ai_guide", "file_chat", "file_extract", "email_draft", "rfp_draft", "detail_search", "planner"}

_DETAIL_HINTS = ["자세히", "자세하게", "더 알려줘", "좀 더", "구체적", "상세히", "상세하게", "더 설명", "추가로 설명", "자세한 내용", "더 자세한"]

_EMAIL_STRUCT_RE = re.compile(r"(수신자\s*:|제목\s*:|내용\s*:|to\s*:|subject\s*:|body\s*:)", re.I)
_FILE_EXT_RE = re.compile(r"\.(pdf|docx|pptx|xlsx|txt|md)\b", re.I)

_EMAIL_WRITE_HINTS = ["작성", "초안", "문안", "써줘", "draft", "compose", "작성해", "작성해줘"]
_EMAIL_EDIT_HINTS  = ["수정", "변경", "바꿔", "고쳐", "초안에서", "이 초안", "그 초안", "방금 메일", "메일에서"]
_FILE_EXTRACT_HINTS = ["추출", "파싱", "텍스트", "본문", "extract", "parse"]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_PERSON_TITLE_RE = re.compile(
    r"[가-힣A-Za-z0-9]+\s*(?:에게|께|님께)|"
    r"[가-힣A-Za-z0-9]+\s*(?:담당자|팀장|부장|과장|차장|대리|이사|대표|사원|매니저)",
    re.I,
)

_DEPT_NAMES = ["인사팀", "hr", "총무팀", "재무팀", "회계팀", "it지원팀", "it 지원팀",
               "마케팅팀", "영업팀", "개발팀", "법무팀", "기획팀", "경영지원팀"]

_RFP_SCOPE_HINTS = ["프로젝트", "시스템", "솔루션", "플랫폼", "구축", "개발", "도입",
                    "고도화", "전환", "운영", "서비스", "인프라", "제안", "설계", "구현",
                    "마이그레이션", "통합", "개선", "최적화", "리팩토링", "업그레이드",
                    "프로그램", "교육", "과정", "앱", "애플리케이션", "기획", "컨설팅", "분석"]

_COMPOUND_SIGNALS: List[Tuple[List[str], List[str]]] = [
    (["검색", "조회", "찾아", "알아봐", "확인해"],  ["rfp", "제안요청서", "이메일", "메일", "초안", "작성해"]),
    (["분석", "읽어", "요약"],                      ["rfp", "이메일", "초안", "작성해"]),
]

_SAMPLE_VECTORS: Dict[str, List[List[float]]] | None = None
_SAMPLE_LOCK = threading.Lock()


# ── 헬퍼 ─────────────────────────────────────────────────────────

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _is_compound_request(text: str) -> bool:
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


def _has_recipient_hint(text: str) -> bool:
    if _EMAIL_RE.search(text or ""):
        return True
    if _contains_any(text, _DEPT_NAMES):
        return True
    if _PERSON_TITLE_RE.search(text or ""):
        return True
    return False


def _has_rfp_scope_hint(text: str) -> bool:
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


# ── 샘플 벡터 캐시 ────────────────────────────────────────────────

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


# ── 규칙 기반 라우터 (최후 폴백용) ───────────────────────────────

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


# ── 시맨틱 라우터 (임베딩 코사인 유사도) ─────────────────────────

def _semantic_route(user_input: str) -> Tuple[str, Dict[str, Any], List[float]]:
    """임베딩 코사인 유사도 기반 라우팅. 임계값 미달 시 'unknown' 반환."""

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

    vectors    = _load_sample_vectors()
    embeddings = get_embeddings()
    query_vec  = embeddings.embed_query(user_input)

    ranked: List[Tuple[str, float]] = []
    for task, task_vectors in vectors.items():
        if not task_vectors:
            continue
        score = max(_cosine(query_vec, svec) for svec in task_vectors)
        ranked.append((task, score))

    if not ranked:
        return "unknown", {"reason": "no_samples"}, []

    ranked.sort(key=lambda x: x[1], reverse=True)
    top1_task,  top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin     = top1_score - top2_score

    top1_min   = ROUTER_TOP1_MIN
    margin_min = ROUTER_MARGIN_MIN
    decision   = top1_task

    if top1_score < top1_min or margin < margin_min:
        decision = "unknown"

    if decision == "email_draft":
        hint_present = (
            _has_email_struct(user_input)
            or _contains_any(user_input, _EMAIL_WRITE_HINTS)
            or _contains_any(user_input, _EMAIL_EDIT_HINTS)
        )
        if not hint_present and top1_score <= EMAIL_DRAFT_HIGH_CONF_THRESHOLD:
            decision = "unknown"

    if decision == "file_extract":
        if not (_has_file_path_hint(user_input) and _contains_any(user_input, _FILE_EXTRACT_HINTS)):
            decision = "unknown"

    debug = {
        "mode": "semantic",
        "top1_task":        top1_task,
        "top1_score":       round(top1_score, 4),
        "top2_score":       round(top2_score, 4),
        "margin":           round(margin, 4),
        "decision":         decision,
        "threshold_top1":   top1_min,
        "threshold_margin": margin_min,
        "ranked":           [{"task": t, "score": round(s, 4)} for t, s in ranked[:4]],
    }
    return decision, debug, query_vec


# ── LLM 폴백 분류기 (시맨틱 unknown 시 2차 분류) ─────────────────

_LLM_FALLBACK_PROMPT = ChatPromptTemplate.from_template(
    "당신은 사용자 메시지의 의도를 분류하는 분류기입니다.\n"
    "아래 태스크 중 하나를 선택하고, 확신도(0.0~1.0)를 함께 반환하세요.\n\n"
    "태스크 종류:\n"
    "- knowledge_search: 사내 문서·정보 검색, 부서 문의처 조회, 사내 자료 탐색\n"
    "- ai_guide: 인사, AI 자기소개, 기능 안내, 도움말 요청\n"
    "- file_chat: 첨부 파일 분석, 업로드한 파일 내용 질문\n"
    "- email_draft: 이메일·메일 작성/수정 요청\n"
    "- file_extract: PDF·DOCX 등 파일 경로 지정 후 텍스트 추출\n"
    "- rfp_draft: RFP(제안요청서) 문서 작성\n"
    "- unknown: 위 태스크와 무관하거나 의도를 파악할 수 없는 경우\n\n"
    "규칙:\n"
    "- 확신도가 0.7 미만이면 반드시 unknown을 반환하세요.\n"
    "- 짧거나 대명사만 있는 입력('이거', '저거', '그거', '해줘')은 unknown으로 처리하세요.\n\n"
    "반드시 아래 형식으로만 출력:\n"
    "{{\"task\": \"<태스크>\", \"confidence\": <0.0~1.0>, \"reason\": \"<한 줄 이유>\"}}\n\n"
    "사용자 메시지: {user_input}"
)

_LLM_FALLBACK_ALLOWED     = {"knowledge_search", "ai_guide", "file_chat", "email_draft", "file_extract", "rfp_draft"}
_LLM_FALLBACK_CONFIDENCE  = 0.7


def llm_intent_fallback(user_input: str) -> Tuple[str, Dict[str, Any]]:
    """시맨틱 unknown 시 LLM 2차 분류. 결과를 intent_samples에 저장해 시맨틱 학습에 반영."""
    if LLM_FALLBACK_ENABLED.strip() in ("0", "false", "False", "NO", "no"):
        return "unknown", {"invoked": False, "reason": "disabled"}

    timeout_sec = LLM_FALLBACK_TIMEOUT_SEC

    def _call() -> Dict[str, Any]:
        response = (_LLM_FALLBACK_PROMPT | get_llm()).invoke({"user_input": user_input})
        content  = extract_text_content(response.content)
        return JsonOutputParser().parse(content) or {}

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            result = future.result(timeout=timeout_sec)

        raw_task   = (result.get("task") or "").strip().lower()
        confidence = float(result.get("confidence", 0.0))

        task = raw_task if (raw_task in _LLM_FALLBACK_ALLOWED and confidence >= _LLM_FALLBACK_CONFIDENCE) else "unknown"

        debug: Dict[str, Any] = {
            "invoked":    True,
            "source":     "llm_fallback",
            "raw_task":   raw_task,
            "confidence": confidence,
            "final_task": task,
            "reason":     result.get("reason", ""),
        }
        return task, debug

    except FuturesTimeoutError:
        return "unknown", {"invoked": True, "source": "llm_fallback", "error": "timeout"}
    except Exception as e:
        return "unknown", {"invoked": True, "source": "llm_fallback", "error": str(e)}


# ── 시맨틱 라우터 통합 진입점 ─────────────────────────────────────

def semantic_task_router(user_input: str, state: dict) -> Tuple[str, Dict[str, Any]]:
    """
    시맨틱 + LLM 폴백 + 후처리 라우팅 통합 함수.
    task_router_node에 통합하려면 이 함수를 참고해 재구성하세요.
    """
    task_args = state.get("task_args") or {}

    routed, debug, query_vec = _semantic_route(user_input)

    if routed == "unknown":
        llm_task, llm_debug = llm_intent_fallback(user_input)
        debug["llm_fallback"] = llm_debug
        if llm_task != "unknown":
            routed = llm_task
            debug["decision"]     = llm_task
            debug["final_source"] = "llm_fallback"
            added = add_sample(llm_task, user_input, source="llm_fallback")
            if added:
                invalidate_sample_cache()
        else:
            debug["final_source"] = "unknown"
    else:
        debug["final_source"] = "semantic"

    # 복합 요청
    if routed not in ("injection", "file_chat", "file_extract") and _is_compound_request(user_input):
        routed = "planner"
        debug.update({"decision": "planner", "final_source": "compound_request_detected"})

    # 상세 검색
    if routed in ("unknown", "knowledge_search"):
        prev_task     = (state.get("task_type") or "").strip()
        prev_messages = state.get("messages") or []
        if (
            prev_task in ("knowledge_search", "detail_search")
            and prev_messages
            and _contains_any(user_input, _DETAIL_HINTS)
        ):
            routed = "detail_search"
            debug.update({"decision": "detail_search", "final_source": "detail_search_detected"})

    # 슬롯 감지
    missing_slots: list[str] = []
    file_context_present = bool((state.get("file_context") or "").strip())
    file_path_present    = bool(task_args.get("file_path"))

    if routed == "file_extract" and not _has_file_path_hint(user_input) and not file_path_present:
        missing_slots = ["file_path"]
    elif routed == "file_chat" and not file_context_present and not file_path_present:
        missing_slots = ["file_context"]
    elif routed == "email_draft" and not _has_recipient_hint(user_input) and not task_args.get("to"):
        if not _contains_any(user_input, _EMAIL_EDIT_HINTS):
            missing_slots = ["to"]
    elif routed == "rfp_draft" and not _has_rfp_scope_hint(user_input) and not task_args.get("project_scope"):
        missing_slots = ["project_scope"]

    if missing_slots:
        routed = "clarification"
        debug.update({"decision": "clarification", "final_source": "slot_missing", "missing_slots": missing_slots})

    return routed, debug
