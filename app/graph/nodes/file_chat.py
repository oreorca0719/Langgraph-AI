from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from app.core.config import get_llm
from app.core.history_utils import HISTORY_MAX_MESSAGES, filter_history_by_relevance as _filter_history_by_relevance
from app.graph.states.state import GraphState
from app.security.content_sanitizer import sanitize
from langgraph.prebuilt import create_react_agent


def file_chat_node(state: GraphState) -> Dict[str, Any]:
    """
    첨부 파일 Q&A 전용 노드 (file_chat).
    get_attached_file 단일 도구만 사용. 사내 RAG 검색 없음.
    """
    print("--- [NODE] File Chat ---")

    trace_id          = (state.get("trace_id") or "")
    user_input        = (state.get("input_data") or "").strip()
    raw_history       = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
    chat_history      = _filter_history_by_relevance(raw_history, user_input)
    file_context      = (state.get("file_context") or "").strip()
    file_context_name = (state.get("file_context_name") or "첨부 파일").strip()


    agent = create_react_agent(get_llm(), [get_attached_file])
    agent_input = {
        "messages": [SystemMessage(content=system_content)]
        + chat_history
        + [HumanMessage(content=user_input)]
    }

    result = agent.invoke(agent_input)
    final_message = result["messages"][-1]

    finish_reason = (
        (getattr(final_message, "response_metadata", None) or {}).get("finish_reason", "") or ""
    ).upper()
    if finish_reason == "MAX_TOKENS":
        final_message = AIMessage(
            content="요청하신 내용이 처리 가능한 텍스트 범위를 초과하였습니다. 질문을 더 구체적으로 나누어 입력해 주세요."
        )

    response_len = len(str(final_message.content))
