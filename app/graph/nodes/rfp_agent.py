from __future__ import annotations

import json
from typing import Any, Dict

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import get_llm
from app.core.history_utils import extract_text_content as _content_to_text
from app.graph.states.state import GraphState
from app.graph.nodes.knowledge_search import _search_hybrid, _format_docs
from app.security.content_sanitizer import sanitize_docs


_AMBIGUOUS_REFS = re.compile(
    r"^(이걸|이거|이것|위\s*내용|앞\s*내용|저거|그거|그것|방금|아까|해당\s*내용)",
    re.I,
)


def _resolve_search_query(user_input: str, messages: list) -> str:
    """
    user_input이 모호한 지시어로 시작하면 직전 AIMessage에서 주제를 추출해
    검색에 적합한 쿼리로 재구성. 그렇지 않으면 user_input 그대로 반환.
    """
    if not _AMBIGUOUS_REFS.search(user_input):
        return user_input

    prev_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            prev_ai = str(msg.content)[:1000]
            break

    if not prev_ai:
        return user_input

    try:
        response = get_llm().invoke([
            SystemMessage(content=(
                "이전 AI 답변을 읽고, RFP 문서 검색에 사용할 핵심 주제 키워드를 한 문장으로 추출하세요.\n"
                "검색 쿼리만 출력하고 다른 텍스트는 출력하지 마세요."
            )),
            HumanMessage(content=f"[이전 AI 답변]\n{prev_ai}\n\n[현재 요청]\n{user_input}"),
        ])
        query = _content_to_text(response.content)
        return query if query else user_input
    except Exception:
        return user_input


# ── 에이전트 1: rfp_research_node ────────────────────────────

def rfp_research_node(state: GraphState) -> Dict[str, Any]:
    """
    RFP 작성에 필요한 사내 문서·컨텍스트를 수집합니다.
    - 사용자 요청에서 핵심 키워드 추출
    - ChromaDB에서 관련 문서 검색
    - 검색 결과를 rfp_research 필드에 저장
    """
    print("--- [NODE] RFP Research ---")

    trace_id = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    extracted = (state.get("extracted_text") or "").strip()
    messages = list(state.get("messages") or [])

    planner_context = (state.get("planner_context") or "").strip()
    search_query = _resolve_search_query(user_input, messages)


    human_content = f"사용자 요청:\n{user_input}"
    if planner_context:
        human_content += f"\n\n[Planner 사전 검색 결과]\n{planner_context}"
    if search_result:
        human_content += f"\n\n사내 문서 검색 결과:\n{search_result}"
    if extracted:
        human_content += f"\n\n첨부 문서 내용:\n{extracted[:5000]}"

    response = get_llm().invoke([
        SystemMessage(content=research_prompt),
        HumanMessage(content=human_content),
    ])
    research_result = _content_to_text(response.content)


    human_parts = [f"요청:\n{user_input}"]
    if research:
        human_parts.append(f"사전 조사 결과:\n{research}")
    if review_notes:
        human_parts.append(f"검토 의견 (반드시 반영):\n{review_notes}")
    if existing_draft:
        human_parts.append(f"기존 초안 (수정 기준):\n{existing_draft[:10000]}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "\n\n".join(human_parts)),
    ])

    resp = (prompt | llm).invoke({
        "chat_history": chat_history,
    })

    draft_text = _content_to_text(resp.content if hasattr(resp, "content") else resp)


    try:
        from langchain_core.output_parsers import JsonOutputParser
        chain = get_llm() | JsonOutputParser()
        result = chain.invoke([
            SystemMessage(content=review_prompt),
            HumanMessage(content=f"사용자 요청:\n{user_input}\n\nRFP 초안:\n{draft[:8000]}"),
        ])
        is_pass = bool(result.get("pass", True))
        notes = str(result.get("notes", "")).strip()
    except Exception:
        is_pass = True
        notes = ""

