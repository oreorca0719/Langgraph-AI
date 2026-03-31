from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_llm
from app.core import trace_buffer
from app.core.history_utils import extract_text_content as _content_to_text
from app.graph.states.state import GraphState

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_EDIT_HINTS = ["수정", "변경", "바꿔", "고쳐", "초안에서", "이 초안", "그 초안", "주소만", "to만"]


def _try_parse_json_object(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0).strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_draft(draft: Dict[str, Any]) -> Dict[str, str]:
    to = draft.get("to", "")
    cc = draft.get("cc", "")
    subject = draft.get("subject", "")
    body = draft.get("body", "")
    to_s = "" if to is None else str(to)
    cc_s = "" if cc is None else str(cc)
    subject_s = "제목(자동생성)" if subject is None else str(subject)
    if isinstance(body, (list, dict)):
        body_s = _content_to_text(body)
        if not body_s:
            body_s = json.dumps(body, ensure_ascii=False)
    else:
        body_s = "" if body is None else str(body)
    return {"to": to_s, "cc": cc_s, "subject": subject_s, "body": body_s}


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _default_dept_email_map() -> Dict[str, str]:
    return {
        "인사팀": "hr@example.com",
        "hr": "hr@example.com",
        "it 지원팀": "it-help@example.com",
        "it지원팀": "it-help@example.com",
        "it": "it-help@example.com",
    }


def _load_dept_email_map() -> Dict[str, str]:
    raw = (os.getenv("DEPT_EMAIL_MAP") or "").strip()
    if not raw:
        return _default_dept_email_map()
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return _default_dept_email_map()
        merged = _default_dept_email_map()
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                merged[k.strip().lower()] = v.strip()
        return merged
    except Exception:
        return _default_dept_email_map()


def _extract_email(text: str) -> str:
    if not text:
        return ""
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else ""


def _infer_to_from_text(user_input: str, args: Dict[str, Any]) -> str:
    explicit = _extract_email(user_input)
    if explicit:
        return explicit
    args_to = _extract_email(str(args.get("to") or ""))
    if args_to:
        return args_to
    mapping = _load_dept_email_map()
    t = (user_input or "").lower()
    for key, addr in mapping.items():
        if key in t:
            return addr
    return ""


def _patch_existing_draft(prev_draft: Dict[str, Any], user_input: str, args: Dict[str, Any]) -> Dict[str, str] | None:
    if not prev_draft or not _contains_any(user_input, _EDIT_HINTS):
        return None
    patched = _normalize_draft(prev_draft)
    changed = False
    to_override = _infer_to_from_text(user_input, args)
    if to_override and to_override != patched.get("to"):
        patched["to"] = to_override
        changed = True
    if changed and _contains_any(user_input, ["주소만", "수신자만", "to만", "메일 주소만"]):
        return patched
    return patched if changed else None


def email_draft_node(state: GraphState) -> GraphState:
    print("--- [NODE] Email Draft ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    chat_history = state.get("messages", [])
    args = state.get("task_args") or {}
    prev_draft = state.get("draft_email") or {}
    planner_context = (state.get("planner_context") or "").strip()

    trace_buffer.push(trace_id, node="email_draft", event="enter", label="execute",
                      data={"input": user_input[:200], "has_prev_draft": bool(prev_draft)})

    # 결정적 수정 경로
    patched = _patch_existing_draft(prev_draft, user_input, args)
    if patched is not None:
        trace_buffer.push(trace_id, node="email_draft", event="exit", label="execute",
                          data={"mode": "deterministic_patch"})
        return {
            **state,
            "draft_email": patched,
            "draft_rfp": None,
            "current_task": "email_draft",
        }

    llm = get_llm()
    prefilled_to = _infer_to_from_text(user_input, args)
    prev_draft_norm = _normalize_draft(prev_draft) if prev_draft else {"to": "", "cc": "", "subject": "", "body": ""}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "당신은 업무 이메일 작성 에이전트입니다.\n"
                "사용자 요청과 단서를 바탕으로 이메일 초안을 작성하세요.\n"
                "반드시 JSON만 출력하고, 키는 to/cc/subject/body만 사용하세요.\n"
                "편집 요청인 경우 기존 초안의 유지 가능한 필드는 최대한 유지하세요.\n\n"
                "【작성 범위 원칙】\n"
                "• 이메일 내용은 사용자가 명시적으로 요청한 범위에만 한정하세요.\n"
                "• 사용자가 요청하지 않은 정보(상세 커리큘럼, 비용, 일정표 등)를 임의로 추가하지 마세요.\n"
                "• '간단히', '확정 안내', '알려줘' 등 단순 요청은 간결하게 핵심만 담아 작성하세요.\n"
                "• 추가 정보가 필요하면 이메일 본문에 포함하지 말고 수신자에게 별도 문의를 유도하세요."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "요청:\n{user_input}\n\n"
            "기존 초안:\n{prev_draft_json}\n\n"
            "추가 단서(task_args):\n{args_json}\n\n"
            "결정된 수신자(있으면 우선 반영): {prefilled_to}\n\n"
            "{planner_context_section}",
        ),
    ])

    planner_section = (
        f"[사전 검색 결과 - 이메일 작성에 활용하세요]\n{planner_context}"
        if planner_context else ""
    )
    resp = (prompt | llm).invoke({
        "chat_history": chat_history,
        "user_input": user_input,
        "prev_draft_json": json.dumps(prev_draft_norm, ensure_ascii=False),
        "args_json": json.dumps(args, ensure_ascii=False),
        "prefilled_to": prefilled_to,
        "planner_context_section": planner_section,
    })

    raw_text = _content_to_text(resp.content if hasattr(resp, "content") else resp)
    parsed = _try_parse_json_object(raw_text)

    if parsed is None:
        draft = {"to": prefilled_to, "cc": prev_draft_norm.get("cc", ""), "subject": "제목(자동생성)", "body": raw_text}
    else:
        draft = parsed

    draft = _normalize_draft(draft)

    to_override = _infer_to_from_text(user_input, args)
    if to_override:
        draft["to"] = to_override
    if not draft.get("to") and prev_draft_norm.get("to"):
        draft["to"] = prev_draft_norm["to"]

    trace_buffer.push(trace_id, node="email_draft", event="exit", label="execute",
                      data={"mode": "llm", "to": draft.get("to", ""), "subject": draft.get("subject", "")})
    return {
        **state,
        "draft_email": draft,
        "draft_rfp": None,
        "current_task": "email_draft",
    }
