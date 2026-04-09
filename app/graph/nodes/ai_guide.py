from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_llm
from app.core.history_utils import (
    HISTORY_MAX_MESSAGES,
    extract_text_content,
    filter_history_by_relevance as _filter_history_by_relevance,
)
from app.graph.states.state import GraphState


def ai_guide_node(state: GraphState) -> Dict[str, Any]:
    """
    AI 자기소개 / 기능 안내 전용 노드.
    도구 없이 LLM 직접 호출. 정의된 5가지 기능만 안내.
    """
    print("--- [NODE] AI Guide ---")

    trace_id   = (state.get("trace_id") or "")
    user_input = (state.get("input_data") or "").strip()
    raw_history  = list(state.get("messages") or [])[-HISTORY_MAX_MESSAGES:]
    chat_history = _filter_history_by_relevance(raw_history, user_input)


    messages = (
        [SystemMessage(content=system_content)]
        + chat_history
        + [HumanMessage(content=user_input)]
    )
    response = get_llm().invoke(messages)

    response_len = len(extract_text_content(response.content))
