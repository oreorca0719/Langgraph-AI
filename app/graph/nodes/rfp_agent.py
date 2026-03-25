from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_llm
from app.core import trace_buffer
from app.graph.states.state import GraphState


def _content_to_text(content: Any) -> str:
    """
    Gemini/LC 메시지 content가 다음 중 어떤 형태로 와도
    화면에 출력 가능한 '문자열'로 정규화합니다.

    - str
    - list[dict]  (예: [{"type":"text","text":"..."}])
    - dict
    - 기타
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t.strip()
        # dict인데 text가 없으면 json으로 fallback
        return json.dumps(content, ensure_ascii=False)

    return str(content).strip()


def rfp_draft_node(state: GraphState) -> GraphState:
    """RFP(제안요청서/요구사항 정의서) 초안 작성 노드."""
    print("--- [NODE] RFP Draft ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    chat_history = state.get("messages") or []
    args = state.get("task_args") or {}
    extracted = (state.get("extracted_text") or "").strip()

    trace_buffer.push(trace_id, node="rfp_draft", event="enter", label="execute",
                      data={"input": user_input[:200], "has_extracted": bool(extracted)})

    # draft 재조회 요청 처리: task_router가 draft_recall 모드로 라우팅한 경우
    existing_rfp = (state.get("draft_rfp") or "").strip()
    _RECALL_HINTS = ["이전", "작성한", "찾아줘", "보여줘", "다시", "아까", "초안", "방금"]
    if existing_rfp and any(k in user_input for k in _RECALL_HINTS):
        trace_buffer.push(trace_id, node="rfp_draft", event="exit", label="execute",
                          data={"mode": "draft_recall", "draft_len": len(existing_rfp)})
        return {**state, "draft_rfp": existing_rfp, "draft_email": None, "pending_task": "rfp_draft"}

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "당신은 RFP(제안요청서/요구사항 정의서) 작성 에이전트입니다.\n"
                    "사용자 요청과(있다면) 첨부문서의 추출 텍스트를 근거로 실무형 문서를 작성하세요.\n"
                    "한국어로, 아래 목차를 반드시 포함하세요:\n"
                    "1. 배경/목적\n"
                    "2. 범위(Scope)\n"
                    "3. 요구사항(기능/비기능)\n"
                    "4. 데이터/연동/보안\n"
                    "5. 일정/마일스톤\n"
                    "6. 산출물\n"
                    "7. 평가 기준\n"
                    "8. 가정/제약\n"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "요청:\n{user_input}\n\n단서(task_args):\n{args_json}\n\n(있다면) 첨부문서 추출 텍스트:\n{extracted}",
            ),
        ]
    )

    resp = (prompt | llm).invoke(
        {
            "chat_history": chat_history,
            "user_input": user_input,
            "args_json": json.dumps(args, ensure_ascii=False),
            "extracted": extracted[:20000],
        }
    )

    raw_content = resp.content if hasattr(resp, "content") else resp
    draft_text = _content_to_text(raw_content)

    trace_buffer.push(trace_id, node="rfp_draft", event="exit", label="execute",
                      data={"draft_len": len(draft_text)})
    # ✅ 중요: 상태 누적 환경에서 반대 draft 키를 명시적으로 제거
    return {
        **state,
        "draft_rfp": draft_text,
        "draft_email": None,
        "pending_task": "rfp_draft",
    }